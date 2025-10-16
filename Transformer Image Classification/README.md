# Image Classification with Pretrained Vision Transformer (ViT) Backbone

This project demonstrates an image classification pipeline using a pretrained Vision Transformer (ViT) model for feature extraction. A lightweight classifier is trained on top of the extracted features to classify images from a custom dataset. Additional analysis includes dimensionality reduction and similarity metrics between feature vectors.

---

## Table of Contents

* [Requirements](#requirements)
* [Dataset](#dataset)
* [Setup](#setup)
* [Feature Extraction](#feature-extraction)
* [Model Architecture](#model-architecture)
* [Training](#training)
* [Evaluation](#evaluation)
* [Dimensionality Reduction](#dimensionality-reduction)
* [Bonus: Feature Similarity](#bonus-feature-similarity)
* [Bonus: Image Retrieval](#bonus-image-retrieval)

---

## Requirements

* Python 3.12
* PyTorch
* torchvision
* transformers
* numpy
* pandas
* scikit-learn
* matplotlib
* tqdm
* Pillow

Install dependencies using:

```bash
pip install torch torchvision transformers numpy pandas scikit-learn matplotlib tqdm pillow
```

---

## Dataset

* The dataset should be placed in the `dataset/` folder.
* Images are organized into subfolders per class (ImageFolder format).
* Minimum of 10,000 samples recommended.
* Each dataset should not be shared by more than two students.

---

## Setup

```python
import os
import torch
import torchvision
from torchvision import transforms
from datasets import load_dataset
from transformers import AutoModel, AutoImageProcessor
```

Define constants:

```python
DATASET_FOLDER = 'dataset'
TRANSFORMER_IMG_SIZE = (224, 224)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
```

---

## Feature Extraction

1. Load dataset and apply preprocessing:

```python
transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Resize(TRANSFORMER_IMG_SIZE)
])
dataset = torchvision.datasets.ImageFolder(DATASET_FOLDER, transform=transforms)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64)
```

2. Load pretrained ViT backbone:

```python
backbone = AutoModel.from_pretrained('google/vit-base-patch16-224-in21k')
processor = AutoImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
```

3. Apply normalization and extract features:

```python
vectors, labels_list = [], []

for imgs, labels in dataloader:
    with torch.no_grad():
        outs = backbone(imgs).pooler_output
    vectors.extend(outs.tolist())
    labels_list.extend(labels.tolist())
```

4. Save features:

```python
import pickle
with open('X_vect.pkl', 'wb') as f: pickle.dump(vectors, f)
with open('y.pkl', 'wb') as f: pickle.dump(labels_list, f)
```

---

## Model Architecture

A lightweight classifier on top of the ViT features:

```python
class Model(torch.nn.Module):
    def __init__(self, n_classes, backbone):
        super().__init__()
        self.backbone = backbone
        self.L1 = torch.nn.Linear(768, 256)
        self.dropout = torch.nn.Dropout(0.5)
        self.classifier = torch.nn.LazyLinear(n_classes)

    def forward(self, x, inp_img=None):
        if inp_img is not None:
            x = self.backbone(inp_img).pooler_output
        x = torch.relu(self.L1(x))
        x = self.dropout(x)
        return self.classifier(x)
```

---

## Training

* Optimizer: Adam
* Loss: CrossEntropy
* Metrics: Accuracy

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()
```

Training loop includes per-epoch evaluation and learning rate scheduler.

```python
train(model, optimizer, criterion, train_loader, test_loader, epochs=50)
```

Model weights are saved as:

```python
torch.save(model.state_dict(), 'models/model.pth')
```

---

## Evaluation

* Load the trained model:

```python
model.load_state_dict(torch.load('models/model.pth'))
```

* Evaluate on test set:

```python
eval_loss, eval_metric = evaluate(model, criterion, test_loader)
print(f'EVAL: {eval_loss:.2f} loss | {eval_metric*100:.2f}% accuracy')
```

* Confusion matrix:

```python
from sklearn.metrics import ConfusionMatrixDisplay
y_preds = model(torch.Tensor(X_test).to(device), None).argmax(dim=-1)
ConfusionMatrixDisplay.from_predictions(y_test, y_preds.cpu())
```

---

## Dimensionality Reduction

* Visualize feature vectors using TSNE, PCA, and TruncatedSVD:

```python
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD

reduced_tsne = TSNE(n_components=2).fit_transform(vectors)
reduced_pca = PCA(n_components=2).fit_transform(vectors)
reduced_svd = TruncatedSVD(n_components=2).fit_transform(vectors)
```

* Plot results to check cluster formation by classes.

---

## Bonus: Feature Similarity

Calculate mean cosine similarity between features to analyze separability:

```python
def calculate_mean_cos_sim(a, b):
    a = a / np.linalg.norm(a, axis=-1)[:, None]
    b = b / np.linalg.norm(b, axis=-1)[:, None]
    return (a @ b.T).mean()
```

* Save similarity matrix visualization.

---

## Bonus: Image Retrieval

* Use the model to predict class for new images:

```python
from PIL import Image
img = Image.open('example.jpg')
cls = model(None, torch.Tensor(processor(img)['pixel_values'][0]).unsqueeze(0).to(device)).argmax().item()
print(dataset.classes[cls])
```

* Retrieve the closest image from the dataset using cosine similarity.
