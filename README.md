# CS5788 Final Project: Identity Preservation in Diffusion Models

**Team:** Nicole Liao, Dhanushikka Ravichandiran

Comparing LoRA and Textual Inversion for identity-preserving image generation using Stable Diffusion v1.5.

## Project Structure

```
gm-final-project/
├── data/
│   ├── initial_images/      # Raw source images (not tracked by git)
│   ├── images/              # Preprocessed 512x512 Zendaya training images
│   └── captions/            # Per-image caption .txt files
├── niffle-data/
│   ├── images/              # Niffle hand-drawn character images
│   └── captions/            # Per-image caption .txt files
├── lora/
│   └── train.py             # Custom LoRA training loop (Dhanushikka)
├── textual_inversion/       # Textual Inversion training (Nicole)
├── evaluation/
│   └── eval.py              # ArcFace + CLIP evaluation script
├── notebooks/
│   └── ti_main.ipynb        # Textual Inversion Colab notebook (Nicole)
├── lora_testing.ipynb       # LoRA Colab notebook (Dhanushikka)
├── results/
│   ├── lora/generated/      # Generated images from LoRA (baseline + lora)
│   ├── textual_inversion/   # Generated images from TI
│   └── failed_runs/         # Overfit failure example
└── preprocess.py            # Image preprocessing script
```

## Subjects

- **Zendaya** — 19 photographs, token `<zendaya>`, uniform caption `"a photo of <zendaya> person"`
- **Niffle** — 16 hand-drawn illustrations of an original giraffe character, per-image descriptive captions

## Running on Google Colab (T4 GPU required)

### Step 1: Setup

```python
# Clone diffusers from source (required for training script)
!git clone https://github.com/huggingface/diffusers.git
!pip install /content/diffusers
!pip install peft accelerate transformers
!pip install --upgrade torchao

# Clone this repo
!git clone https://github.com/rdhanushikka/gm-final-project.git /content/gm-final-project

# Mount Google Drive (for saving outputs)
from google.colab import drive
drive.mount('/content/drive')
```

### Step 2: Preprocessing

```python
!python /content/gm-final-project/preprocess.py
```

This reads images from `data/initial_images/`, center-crops to square, resizes to 512×512, and writes outputs to `data/images/` and `data/captions/`.

### Step 3: LoRA Training (Custom)

```python
!python /content/gm-final-project/lora/train.py \
  --data_dir /content/gm-final-project/data/images \
  --caption_dir /content/gm-final-project/data/captions \
  --output_dir /content/drive/MyDrive/results/lora_custom \
  --num_epochs 20 \
  --learning_rate 5e-5 \
  --lora_rank 16
```

### Step 4: Inference

```python
from diffusers import StableDiffusionPipeline
import torch, os

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float32,
    safety_checker=None
).to("cuda")

prompts = [
    "a close-up portrait of <zendaya> person smiling",
    "a close-up portrait of <zendaya> person with curly hair",
    "a close-up portrait of <zendaya> person outdoors in sunlight",
    "a studio headshot of <zendaya> person",
    "a close-up portrait of <zendaya> person with a neutral expression",
]

os.makedirs("/content/drive/MyDrive/results/lora_custom/generated", exist_ok=True)

# Baseline — no LoRA
for i, prompt in enumerate(prompts):
    image = pipe(prompt, negative_prompt="full body, cropped face, low quality, blurry",
                 num_inference_steps=30, guidance_scale=7.5).images[0]
    image.save(f"/content/drive/MyDrive/results/lora_custom/generated/baseline_{i+1}.png")

# With LoRA
pipe.load_lora_weights("/content/drive/MyDrive/results/lora_custom")
for i, prompt in enumerate(prompts):
    image = pipe(prompt, negative_prompt="full body, cropped face, low quality, blurry",
                 num_inference_steps=30, guidance_scale=7.5).images[0]
    image.save(f"/content/drive/MyDrive/results/lora_custom/generated/lora_{i+1}.png")
```

### Step 5: Evaluation

```python
!pip install insightface onnxruntime open_clip_torch

# ArcFace + CLIP on LoRA results
!python /content/gm-final-project/evaluation/eval.py \
  --generated_dir /content/drive/MyDrive/results/lora_custom/generated \
  --training_dir /content/gm-final-project/data/images \
  --prefix lora \
  --prompts \
    "a close-up portrait of <zendaya> person smiling" \
    "a close-up portrait of <zendaya> person with curly hair" \
    "a close-up portrait of <zendaya> person outdoors in sunlight" \
    "a studio headshot of <zendaya> person" \
    "a close-up portrait of <zendaya> person with a neutral expression"

# ArcFace + CLIP on baseline results
!python /content/gm-final-project/evaluation/eval.py \
  --generated_dir /content/drive/MyDrive/results/lora_custom/generated \
  --training_dir /content/gm-final-project/data/images \
  --prefix baseline \
  --prompts \
    "a close-up portrait of <zendaya> person smiling" \
    "a close-up portrait of <zendaya> person with curly hair" \
    "a close-up portrait of <zendaya> person outdoors in sunlight" \
    "a studio headshot of <zendaya> person" \
    "a close-up portrait of <zendaya> person with a neutral expression"
```

## Results (Zendaya)

| Method | ArcFace ↑ | CLIP ↑ |
|---|---|---|
| Baseline (SD v1.5) | 0.2495 | 0.3176 |
| Textual Inversion | 0.2141 | 0.3230 |
| LoRA | **0.2752** | **0.3186** |

## LoRA Hyperparameters

| Parameter | Value |
|---|---|
| Base model | runwayml/stable-diffusion-v1-5 |
| LoRA rank | 16 |
| Epochs | 20 |
| Learning rate | 5e-5 |
| Resolution | 512×512 |
| Batch size | 1 |
| Trainable params | ~797K / 860M (0.09%) |

## Why we switched from a private subject to a celebrity

Initial training on a private individual (Muskan) with 50 epochs at lr=1e-4 caused severe training instability. The model forgot the subject's face entirely and generated a distorted bearded man with hollow black eyes (see `results/failed_runs/muskan_overfit_example.png`). We reduced to 20 epochs and lr=5e-5, and switched to Zendaya for a more diverse training set.

**Note:** SD v1.5 was pretrained on LAION-5B which likely includes Zendaya's images. This is a known confound — see the Niffle experiments for a cleaner evaluation.
