# CS5788 Final Project: Identity Preservation in Diffusion Models

**Team:** Nicole Liao, Dhanushikka Ravichandiran

Comparing LoRA and Textual Inversion for identity-preserving image generation using Stable Diffusion v1.5.

## Project Structure

```
gm-final-project/
├── data/
│   ├── initial_images/   # Raw source images (not tracked by git)
│   ├── images/           # Preprocessed 512x512 training images
│   └── captions/         # Per-image caption .txt files
├── lora/                 # LoRA training loop (Dhanushikka)
├── textual_inversion/    # Textual Inversion training loop (Nicole)
├── evaluation/           # ArcFace + CLIP evaluation scripts
├── results/
│   ├── lora/             # Generated images from LoRA
│   ├── textual_inversion/
│   └── failed_runs/      # Examples of failed/overfit training runs
├── notebooks/            # Colab notebooks
└── preprocess.py         # Image preprocessing script
```

## Subject: Zendaya

Training subject is Zendaya (19 images). Token: `<zendaya>`.

### Why we switched from a private individual to a celebrity

Initial training was done on a private individual (Muskan) with the token `<muskan>`. The first run used incorrect hyperparameters — 50 epochs and learning rate 1e-4 — which caused severe overfitting. The model collapsed entirely: instead of learning the subject's face, it generated a distorted bearded man with hollow black eyes (see `results/failed_runs/muskan_overfit_example.png`). This is a known failure mode when LoRA training runs too long at too high a learning rate on a small dataset.

We switched to Zendaya for two reasons:
1. We did not want to continue generating disturbing images of a real private individual.
2. Zendaya's photos offer better variety (19 images across different hair styles, lighting, and outfits), which produces a more robust training set.

**Limitation:** SD v1.5 was pretrained on LAION-5B, which likely includes images of Zendaya. This means the base model has prior knowledge of her appearance, which we acknowledge as a confound in our evaluation. The LoRA vs. Textual Inversion comparison remains valid, but absolute identity similarity scores may be inflated relative to an unknown subject.

## Hyperparameters (LoRA)

| Parameter | Value |
|---|---|
| Base model | runwayml/stable-diffusion-v1-5 |
| Epochs | 20 |
| Learning rate | 5e-5 |
| Resolution | 512x512 |
| Batch size | 1 |
| Mixed precision | fp16 |

## Setup

```bash
pip install diffusers transformers accelerate torch torchvision peft
pip install insightface onnxruntime open_clip_torch
```

## Running LoRA Training

```bash
accelerate launch diffusers/examples/dreambooth/train_dreambooth_lora.py \
  --pretrained_model_name_or_path=runwayml/stable-diffusion-v1-5 \
  --instance_data_dir=data/images \
  --instance_prompt="a photo of <zendaya> person" \
  --output_dir=results/lora \
  --train_batch_size=1 \
  --num_train_epochs=20 \
  --learning_rate=5e-5 \
  --mixed_precision=fp16 \
  --resolution=512
```

## Running Textual Inversion

```bash
python textual_inversion/train.py \
    --data_dir data/images \
    --placeholder_token "<zendaya>" \
    --output_dir results/textual_inversion
```

## Evaluation

```bash
python evaluation/eval.py \
    --generated_dir results/lora \
    --training_dir data/images
```
