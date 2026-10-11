import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel, AutoProcessor

from .config import free_device

MODEL, REVISION = "google/siglip2-large-patch16-384", "1b426889ea62b5a72bf9839009a1b184bfc9c178"
TEMPLATES = (
    "a photo of a {}.",
    "a photo of the {}.",
    "a cropped photo of a {}.",
    "a close-up photo of a {}.",
    "a photo of a {} in a room.",
)
BATCH = 16
COLOR_SHARE = 0.5  # share of pixels the dominant colour word needs

# Room contents plus every SceneDiff label, without near-synonyms.
OBJECTS = tuple(
    "chair, armchair, office chair, rocking chair, folding chair, stool, bar stool, ottoman, bench, "  # noqa: SIM905
    "sofa, table, coffee table, side table, desk, nightstand, dresser, cabinet, bookshelf, shelf, "
    "wardrobe, bed, mattress, ladder, coat rack, cart, crib, bag, backpack, handbag, purse, duffle "
    "bag, tote bag, suitcase, paper bag, plastic bag, shopping bag, briefcase, wallet, box, cardboard "
    "box, shoebox, storage bin, basket, laundry basket, crate, tissue box, jar, container, bucket, "
    "trash can, toolbox, jewelry box, gift box, envelope, laptop, tablet, phone, monitor, television, "
    "keyboard, computer mouse, speaker, headphones, game controller, remote control, camera, charger, "
    "power strip, cable, router, printer, fan, heater, air purifier, humidifier, vacuum cleaner, "
    "robot vacuum, clock, radio, game console, hard drive, circuit board, calculator, projector, "
    "microphone, flashlight, mug, cup, glass, wine glass, plate, bowl, pan, pot, kettle, teapot, "
    "coffee maker, toaster, cutting board, knife, fork, spoon, water bottle, bottle, wine bottle, "
    "thermos, lunch box, tray, jug, blender, microwave, sponge, bread, apple, banana, orange, fruit, "
    "cookie, snack, cereal box, soda can, candy, cake, pizza box, coat, jacket, sweater, shirt, "
    "hoodie, pants, jeans, shorts, dress, skirt, pajamas, sock, shoe, sneaker, boot, slipper, sandal, "
    "hat, cap, scarf, glove, belt, towel, clothes hanger, pillow, cushion, blanket, duvet, bed sheet, "
    "rug, doormat, curtain, stuffed animal, teddy bear, toy, toy car, doll, ball, basketball, soccer "
    "ball, tennis ball, tennis racket, board game, puzzle, toy blocks, toy snake, skateboard, yoga "
    "mat, dumbbell, kettlebell, resistance band, jump rope, guitar, ukulele, drum, potted plant, "
    "flower, vase, flower pot, picture frame, painting, poster, mirror, candle, lamp, floor lamp, "
    "desk lamp, lantern, sculpture, figurine, globe, trophy, artificial fruit, toothbrush, "
    "toothpaste, hairbrush, comb, hair dryer, razor, soap, shampoo bottle, lotion bottle, cream jar, "
    "perfume, makeup bag, medicine bottle, pill bottle, spray bottle, wet wipes, toilet paper, paper "
    "towel, bathroom scale, eyeglasses, sunglasses, watch, necklace, keychain, key, umbrella, book, "
    "notebook, magazine, newspaper, sheet of paper, folder, binder, pen, pencil, marker, scissors, "
    "tape, stapler, calendar, pencil case, ruler, whiteboard, broom, mop, dustpan, detergent bottle, "
    "iron, ironing board, drying rack, hammer, screwdriver, drill, light bulb, fire extinguisher, "
    "bicycle, helmet, scooter, stroller, pet bed, person, cat, dog".split(", ")
)
# Raised supports only; the floor is found by geometry.
SURFACES = tuple(
    "table, coffee table, dining table, side table, desk, sofa, armchair, bed, chair, stool, ottoman, "  # noqa: SIM905
    "bench, shelf, bookshelf, cabinet, dresser, nightstand, counter, windowsill, "
    "tv stand, radiator".split(", ")
)


def crop(image: np.ndarray, mask: np.ndarray | None, box_xyxy: tuple, pad: float = 0.15) -> np.ndarray:
    """RGB crop for SigLIP. mask is (H, W) bool over the whole image, box_xyxy in its pixels."""
    h, w = image.shape[:2]
    x0, y0, x1, y1 = box_xyxy
    px, py = pad * (x1 - x0), pad * (y1 - y0)
    x0, y0 = max(0, round(x0 - px)), max(0, round(y0 - py))
    x1, y1 = min(w, round(x1 + px)), min(h, round(y1 + py))
    # The box's aspect is kept; the processor squashes the crop to 384 x 384.
    out = image[y0:y1, x0:x1].astype(np.float32)
    if mask is not None:
        # Dim the rest 30 % towards grey to show which object is meant.
        out = np.where(mask[y0:y1, x0:x1, None], out, 0.7 * out + 0.3 * 128)
    return out.round().astype(np.uint8)


def color_name(pixels: np.ndarray) -> str | None:
    """Dominant colour word of (N, 3) RGB pixels, or None if no word reaches COLOR_SHARE."""
    if len(pixels) == 0:
        return None
    hsv = Image.fromarray(np.asarray(pixels, np.uint8).reshape(-1, 1, 3)).convert("HSV")
    h, s, v = (np.asarray(hsv, np.float32).reshape(-1, 3) / 255).T
    hue = np.array(["red", "orange", "yellow", "green", "blue", "purple", "pink", "red"])
    name = hue[np.digitize(h * 360, [15, 40, 70, 165, 255, 290, 340])]
    warm = (name == "orange") | (name == "yellow")
    name[warm & (v < 0.6)] = "brown"
    beige = warm & (s >= 0.08) & (s < 0.35) & (v > 0.45)
    name[beige] = "beige"
    name[(name == "red") & (s < 0.6) & (v > 0.7)] = "pink"
    plain = (s < 0.2) & ~beige
    name[plain] = np.where(v[plain] > 0.65, "white", "grey")
    name[(v < 0.15) | ((v < 0.3) & (s < 0.4))] = "black"
    words, counts = np.unique(name, return_counts=True)
    k = counts.argmax()
    return str(words[k]) if counts[k] >= COLOR_SHARE * len(name) else None


def label(objects: list[list[np.ndarray]], surfaces: list[list[np.ndarray]], device: torch.device):
    """Best word and its probability per object (OBJECTS) and per support surface (SURFACES), averaged
    over each item's crops."""
    model = AutoModel.from_pretrained(MODEL, revision=REVISION, dtype=torch.float32).to(device).eval()
    processor = AutoProcessor.from_pretrained(MODEL, revision=REVISION)
    out = [
        classify(model, processor, crops, vocabulary, device)
        for crops, vocabulary in ((objects, OBJECTS), (surfaces, SURFACES))
    ]
    del model
    free_device(device)
    return out


def classify(model, processor, crops: list[list[np.ndarray]], vocabulary, device) -> list[tuple[str, float]]:
    if not crops:
        return []
    # SigLIP was trained on lowercase text padded to 64 tokens.
    prompts = [t.format(w).lower() for w in vocabulary for t in TEMPLATES]
    images = [Image.fromarray(c) for views in crops for c in views]
    with torch.inference_mode():
        text = torch.cat(
            [
                model.get_text_features(
                    **processor(
                        text=prompts[i : i + BATCH], padding="max_length", max_length=64, return_tensors="pt"
                    ).to(device)
                ).pooler_output
                for i in range(0, len(prompts), BATCH)
            ]
        )
        # Prompt ensembling as in CLIP: average the normalised template embeddings.
        text = F.normalize(
            F.normalize(text, dim=-1).view(len(vocabulary), len(TEMPLATES), -1).mean(1), dim=-1
        )
        image = torch.cat(
            [
                model.get_image_features(
                    **processor(images=images[i : i + BATCH], return_tensors="pt").to(device)
                ).pooler_output
                for i in range(0, len(images), BATCH)
            ]
        )
        # Softmax over the vocabulary instead of SigLIP's per-word sigmoid; the bias cancels.
        logits = model.logit_scale.exp() * F.normalize(image, dim=-1) @ text.T  # (views, vocabulary)
        probs = logits.softmax(-1).cpu().numpy()
    out, start = [], 0
    for views in crops:
        p = probs[start : start + len(views)].mean(0)
        start += len(views)
        out.append((vocabulary[int(p.argmax())], float(p.max())))
    return out
