# Image Captioning with DINOv2 and GPT-2

This notebook implements an **image captioning pipeline** that extracts visual features using a pretrained **DINOv2** vision transformer and generates text captions using a modified **GPT-2** language model with cross-attention.
The project demonstrates the process of dataset preparation, feature extraction, multimodal training, and inference for caption generation.

---

## 📁 Project Overview

### 1. Feature Extraction

* Used **COCO 2017** dataset (`train2017`) and corresponding captions.
* Extracted image embeddings using **facebook/dinov2-base**.
* Saved features and tokenized captions into an HDF5 file (`together_dataset_12_tokens.hdf5`).
* Limited to ~25% of COCO dataset for computational efficiency.

### 2. Dataset Preparation

* Each image has **5 captions**.
* Implemented a dataset loader that randomly selects one caption per image.
* Stored:

  * Visual features (`input`)
  * Tokenized captions (`text`)
  * Attention masks (`attn`)

### 3. Model Architecture

* Based on **GPT-2** (`openai-community/gpt2`), modified to include **cross-attention layers** for multimodal input.
* The image features from DINOv2 are fed as encoder hidden states into GPT-2.
* Implemented a custom PyTorch module:

  ```python
  class Model(torch.nn.Module):
      def __init__(self, gpt):
          super().__init__()
          self.gpt = gpt

      def forward(self, hidden, inp):
          return self.gpt(input_ids=inp, encoder_hidden_states=hidden).logits
  ```

### 4. Training

* Used **CrossEntropyLoss** with **Adam optimizer** (`lr=1e-4`).
* Trained for 5 epochs.
* Monitored metrics (accuracy approximation) and loss on both training and validation datasets.

#### Example results:

| Epoch | Train Metric | Train Loss | Val Metric | Val Loss |
| ----- | ------------ | ---------- | ---------- | -------- |
| 1     | 0.399        | 3.552      | 0.092      | 2.917    |
| 2     | 0.468        | 2.745      | 0.083      | 2.638    |
| 5     | 0.518        | 2.263      | 0.090      | 2.370    |

---

### 5. Inference

* Implemented **greedy decoding** and **beam search** caption generation.
* Input: image → DINOv2 embeddings → GPT-2 generates caption token by token.
* Example:

  ```
  "A man kneeling in the grass holding a bunch of vegetables."
  ```

#### Example usage:

```python
caption = inference(image, max_length=30)
print(caption)
```

---

## 🧠 Key Components

| Component                      | Description                                         |
| ------------------------------ | --------------------------------------------------- |
| `DINOv2`                       | Vision Transformer for feature extraction           |
| `GPT-2 (with cross-attention)` | Language model for caption generation               |
| `HDF5 dataset`                 | Stores preprocessed features and text               |
| `train()`                      | Training loop with evaluation                       |
| `inference()`                  | Greedy decoding caption generation                  |
| `beam search`                  | Alternative decoding strategy for improved captions |

---

## ⚙️ Dependencies

```
torch
torchvision
transformers
pandas
numpy
opencv-python
tables
matplotlib
tqdm
```

---

## 🧾 Files

| File                                    | Description                            |
| --------------------------------------- | -------------------------------------- |
| `notebook2.ipynb`                       | Main training and inference notebook   |
| `together_dataset_12_tokens.hdf5`       | Preprocessed dataset (features + text) |
| `THE_BEST_MODEL_EVER_EVER_12tokens.pth` | Trained multimodal model               |
| `test_images/`                          | Example test images for inference      |
| `README.md`                             | Project documentation (this file)      |

---

## 🧩 Summary

This project demonstrates the integration of **visual and textual modalities** using pretrained transformers.
It provides a minimal but complete example of **training an image captioning model** with:

* Vision backbone (DINOv2)
* Language model (GPT-2 with cross-attention)
* Custom dataset loader and training loop
* Inference using greedy and beam search decoding.
