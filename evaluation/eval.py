import os
import argparse
import numpy as np
from PIL import Image
import torch
import open_clip


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated_dir", type=str, required=True, help="Directory of generated images")
    parser.add_argument("--training_dir", type=str, required=True, help="Directory of training images")
    parser.add_argument("--prompts", type=str, nargs="+", help="Prompts used for generation (for CLIP score)")
    return parser.parse_args()


def load_images(directory):
    paths = sorted([
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])
    return paths


def get_arcface_embeddings(image_paths):
    import insightface
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(512, 512))

    embeddings = []
    skipped = []
    for path in image_paths:
        img = np.array(Image.open(path).convert("RGB"))
        faces = app.get(img)
        if len(faces) == 0:
            skipped.append(path)
            continue
        # Use the largest detected face
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        embeddings.append(face.normed_embedding)

    if skipped:
        print(f"Warning: no face detected in {len(skipped)} image(s): {[os.path.basename(p) for p in skipped]}")

    return np.array(embeddings) if embeddings else np.array([])


def arcface_similarity(gen_embeddings, train_embeddings):
    if len(gen_embeddings) == 0 or len(train_embeddings) == 0:
        return None
    # Average cosine similarity between each generated image and all training images
    scores = []
    for gen_emb in gen_embeddings:
        sims = np.dot(train_embeddings, gen_emb)
        scores.append(np.mean(sims))
    return float(np.mean(scores))


def get_clip_scores(image_paths, prompts, device):
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model = model.to(device).eval()

    scores = []
    for path, prompt in zip(image_paths, prompts):
        image = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
        text = tokenizer([prompt]).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            score = (image_features @ text_features.T).item()

        scores.append(score)

    return float(np.mean(scores))


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gen_paths = load_images(args.generated_dir)
    train_paths = load_images(args.training_dir)

    print(f"Generated images: {len(gen_paths)}")
    print(f"Training images:  {len(train_paths)}")

    # ArcFace identity similarity
    print("\nComputing ArcFace embeddings...")
    gen_embeddings = get_arcface_embeddings(gen_paths)
    train_embeddings = get_arcface_embeddings(train_paths)

    arc_score = arcface_similarity(gen_embeddings, train_embeddings)
    if arc_score is not None:
        print(f"ArcFace Identity Similarity: {arc_score:.4f}")
    else:
        print("ArcFace: could not compute (no faces detected)")

    # CLIP text-image similarity
    if args.prompts:
        if len(args.prompts) != len(gen_paths):
            print(f"\nWarning: {len(args.prompts)} prompts but {len(gen_paths)} images — skipping CLIP score")
        else:
            print("\nComputing CLIP scores...")
            clip_score = get_clip_scores(gen_paths, args.prompts, device)
            print(f"CLIP Text-Image Similarity: {clip_score:.4f}")
    else:
        print("\nNo prompts provided — skipping CLIP score")

    print("\nDone.")


if __name__ == "__main__":
    main()
