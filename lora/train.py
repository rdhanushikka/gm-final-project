import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from peft import LoraConfig, get_peft_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to preprocessed images")
    parser.add_argument("--caption_dir", type=str, required=True, help="Path to caption .txt files")
    parser.add_argument("--output_dir", type=str, required=True, help="Where to save LoRA weights")
    parser.add_argument("--model_id", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--lora_rank", type=int, default=4)
    return parser.parse_args()


class FaceDataset(Dataset):
    def __init__(self, data_dir, caption_dir, tokenizer, resolution=512):
        self.data_dir = data_dir
        self.caption_dir = caption_dir
        self.tokenizer = tokenizer

        self.image_files = sorted([
            f for f in os.listdir(data_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])

        self.transform = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.image_files[idx])
        pixel_values = self.transform(Image.open(img_path).convert("RGB"))

        caption_file = os.path.splitext(self.image_files[idx])[0] + ".txt"
        with open(os.path.join(self.caption_dir, caption_file), "r") as f:
            caption = f.read().strip()

        input_ids = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt"
        ).input_ids[0]

        return {"pixel_values": pixel_values, "input_ids": input_ids}


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    tokenizer = CLIPTokenizer.from_pretrained(args.model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.model_id, subfolder="text_encoder").to(device)
    vae = AutoencoderKL.from_pretrained(args.model_id, subfolder="vae").to(device)
    unet = UNet2DConditionModel.from_pretrained(args.model_id, subfolder="unet").to(device)
    noise_scheduler = DDPMScheduler.from_pretrained(args.model_id, subfolder="scheduler")

    # Freeze VAE and text encoder — only LoRA weights will be trained
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # Inject LoRA into UNet attention layers
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.1,
    )
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()

    dataset = FaceDataset(args.data_dir, args.caption_dir, tokenizer, args.resolution)
    dataloader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, unet.parameters()),
        lr=args.learning_rate
    )

    unet.train()
    for epoch in range(args.num_epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)

            with torch.no_grad():
                # Encode images into latent space
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Get text conditioning
                encoder_hidden_states = text_encoder(input_ids)[0]

            # Sample random noise and timestep
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],), device=device
            ).long()

            # Forward diffusion: add noise to latents
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # UNet predicts the noise
            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

            loss = F.mse_loss(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{args.num_epochs} — Loss: {avg_loss:.4f}")

    unet.save_pretrained(args.output_dir)
    print(f"LoRA weights saved to {args.output_dir}")


if __name__ == "__main__":
    main()
