# Fine-Tuning Stable Diffusion v1.5 with LoRA

This notebook demonstrates how to fine-tune the **Stable Diffusion v1.5** model using **LoRA (Low-Rank Adaptation)** on a small custom dataset of images. The resulting LoRA adapter enables the model to reproduce specific visual styles and concepts from the training data.

---

## 🧠 Overview

The goal of this work is to adapt the pre-trained text-to-image model **Stable Diffusion v1.5** to generate images of a particular subject — in this case, a **wolf** — by training a lightweight LoRA module instead of fine-tuning the entire model.
This approach is much more efficient in terms of memory and computation while achieving high-quality results.

---

## 🚀 Training Configuration

Training is launched via the Hugging Face `accelerate` CLI:

```bash
!accelerate launch train_text_to_image_lora.py \
  --mixed_precision="fp16" \
  --pretrained_model_name_or_path="stable-diffusion-v1-5/stable-diffusion-v1-5" \
  --train_data_dir="dataset" \
  --resolution=512 \
  --center_crop \
  --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=15000 \
  --learning_rate=1e-4 \
  --max_grad_norm=1 \
  --lr_scheduler="cosine" \
  --lr_warmup_steps=0 \
  --output_dir="./lora_output2" \
  --report_to=tensorboard \
  --checkpointing_steps=5000 \
  --seed=7845
```

### Key Parameters

* **Base model:** `stable-diffusion-v1-5`
* **Dataset:** `dataset/` (custom image-caption pairs)
* **Resolution:** 512×512
* **Total training steps:** 15,000
* **Batch size:** 1 (with gradient accumulation = 4)
* **Learning rate:** `1e-4`
* **Precision:** FP16
* **Logging:** TensorBoard (`./lora_output2/logs`)
* **Checkpoints:** every 5,000 steps

Checkpoints and final LoRA weights are saved in:

```
./lora_output2/checkpoint-5000/
./lora_output2/checkpoint-10000/
./lora_output2/checkpoint-15000/
./lora_output2/pytorch_lora_weights.safetensors
```

---

## 📉 Training Progress

* The model trained for **15,000 optimization steps** on **15 training examples**.
* Checkpoints were successfully saved every 5,000 steps.
* Loss values decreased steadily during training, showing proper convergence.
* TensorBoard logs can be visualized using:

  ```python
  %tensorboard --logdir='./lora_output2/logs'
  ```

---

## 🎨 Inference and Results

After training, the LoRA adapter is loaded into the Stable Diffusion pipeline:

```python
from diffusers import DiffusionPipeline
import torch

pipe_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
pipe = DiffusionPipeline.from_pretrained(pipe_id, torch_dtype=torch.float16).to("cuda")
pipe.load_lora_weights("lora_output2", weight_name="pytorch_lora_weights.safetensors", adapter_name="wolf")
```

### Example Prompts

The model was tested with several text prompts:

* `"a wolf with sickles high quality"`
* `"a wolf looks forward"`
* `"a wolf with an evil look"`

Each prompt was generated twice — with and without LoRA — to compare results:

| Prompt                             | LoRA Enabled (`lora_scale=1`) | LoRA Disabled (`lora_scale=0`) |
| ---------------------------------- | ----------------------------- | ------------------------------ |
| `a wolf with sickles high quality` | Adapted visual style          | Base model output              |
| `a wolf looks forward`             | Personalized subject          | Generic wolf                   |
| `a wolf with an evil look`         | Stylized emotion retained     | Neutral appearance             |

---

## 🧾 Conclusions

* The LoRA fine-tuning successfully adapted **Stable Diffusion v1.5** to generate custom imagery matching the dataset.
* The adapter captured fine-grained visual traits (style, pose, emotion) specific to the training data.
* LoRA offers an efficient way to personalize diffusion models with minimal GPU resources.

---

Would you like me to make this README **formatted for GitHub** (with emojis, headers, and code blocks aligned nicely for markdown preview)? I can polish it for copy-paste into your repo.
