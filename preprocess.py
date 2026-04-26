import os
from PIL import Image

INPUT_DIR = "data/initial_images"
OUTPUT_DIR = "data/images"
CAPTION_DIR = "data/captions"
CAPTION_TEXT = "a photo of <zendaya> person"
TARGET_SIZE = 512

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CAPTION_DIR, exist_ok=True)

image_files = sorted([
    f for f in os.listdir(INPUT_DIR)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
])

for idx, filename in enumerate(image_files):
    img = Image.open(os.path.join(INPUT_DIR, filename)).convert("RGB")

    # Center crop to square
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    img = img.crop((left, top, left + side, top + side))

    # Resize to 512x512
    img = img.resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)

    # Save image
    out_name = f"image_{idx+1:02d}"
    img.save(os.path.join(OUTPUT_DIR, f"{out_name}.png"))

    # Save caption
    with open(os.path.join(CAPTION_DIR, f"{out_name}.txt"), "w") as f:
        f.write(CAPTION_TEXT)

    print(f"Processed {filename} -> {out_name}.png")

print(f"\nDone. {len(image_files)} images saved to {OUTPUT_DIR}/")
