# Textual Inversion Fine-Tuning with Stable Diffusion v1.5

## Description

This notebook performs **Textual Inversion** training on **Stable Diffusion v1.5** to create two custom learned tokens:

1. `<web-page>` — trained on images from `dataset/`
2. `<burzuminator>` — trained on images from `dataset_burzuminator/`

---

## Training 1 — `<web-page>`

```bash
!accelerate launch textual_inversion.py \
  --mixed_precision="fp16" \
  --pretrained_model_name_or_path="stable-diffusion-v1-5/stable-diffusion-v1-5" \
  --train_data_dir="dataset" \
  --learnable_property="object" \
  --placeholder_token="<web-page>" \
  --initializer_token="web" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=3000 \
  --learning_rate=5.0e-04 \
  --scale_lr \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir="./textual_output"
````

**Training details:**

* Number of images: 700
* Total training steps: 3000
* Mixed precision: fp16
* Final loss: **0.00855**
* Checkpoints saved every 500 steps (`checkpoint-500`, `checkpoint-1000`, `checkpoint-1500`, `checkpoint-2000`, `checkpoint-2500`, `checkpoint-3000`)
* Output folder: `./textual_output/`

---

## Generation Example — `<web-page>`

```python
from diffusers import StableDiffusionPipeline
import torch

pipeline = StableDiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

pipeline.load_textual_inversion("textual_output")
image = pipeline("web page of cats <web-page>", num_inference_steps=500).images[0]
image.save("funco.png")
```

---

## Training 2 — `<burzuminator>`

```bash
!accelerate launch textual_inversion.py \
  --mixed_precision="fp16" \
  --pretrained_model_name_or_path="stable-diffusion-v1-5/stable-diffusion-v1-5" \
  --train_data_dir="dataset_burzuminator" \
  --learnable_property="object" \
  --placeholder_token="<burzuminator>" \
  --initializer_token="black" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=500 \
  --learning_rate=5.0e-04 \
  --scale_lr \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir="./burzuminator_output"
```

**Training details:**

* Number of images: 1000
* Total training steps: 500
* Mixed precision: fp16
* Final loss: **0.00644**
* Output folder: `./burzuminator_output/`
* Checkpoint: `checkpoint-500`

---

## Generation Example — `<burzuminator>`

```python
from diffusers import StableDiffusionPipeline
import torch

pipeline = StableDiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

pipeline.load_textual_inversion("burzuminator_output")
image = pipeline("Jesus cross god <burzuminator>", num_inference_steps=500).images[0]
image.save("burzuminator_jesus.png")
```

---

Both learned embeddings (`<web-page>` and `<burzuminator>`) can be loaded into any compatible Stable Diffusion v1.5 pipeline using `load_textual_inversion()`.
