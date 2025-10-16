# Emotion Classification with Pretrained Transformers

This project demonstrates a pipeline for **emotion classification** using a pretrained transformer backbone and PyTorch. The model classifies text into six emotion categories: `sadness`, `joy`, `love`, `anger`, `fear`, and `surprise`.

---

## Table of Contents

* [Installation](#installation)
* [Data](#data)
* [Backbone Model](#backbone-model)
* [Feature Extraction](#feature-extraction)
* [Data Preparation](#data-preparation)
* [Model Architecture](#model-architecture)
* [Training](#training)
* [Evaluation](#evaluation)
* [Text Classification](#text-classification)
* [Word Impact Analysis](#word-impact-analysis)

---

## Installation

Install the required packages:

```bash
pip install einops numpy pandas transformers tqdm scikit-learn matplotlib torch torchvision torchinfo datasets
```

---

## Data

We use the [DAIR AI Emotion dataset](https://huggingface.co/datasets/dair-ai/emotion) with the following splits:

```python
DatasetDict({
    train: 16000 samples,
    validation: 2000 samples,
    test: 2000 samples
})
```

Example of a text sample:

```text
'i am ever feeling nostalgic about the fireplace i will know that it is still on the property'
```

Mapping classes to indices:

```python
class2idx = {'sadness': 0, 'joy': 1, 'love': 2, 'anger': 3, 'fear': 4, 'surprise': 5}
idx2class = {v:k for k,v in class2idx.items()}
```

---

## Backbone Model

We use a pretrained embedding model from Hugging Face for feature extraction:

```python
backbone = transformers.AutoModel.from_pretrained('jinaai/jina-embeddings-v3').to(device)
tokenizer = transformers.AutoTokenizer.from_pretrained('jinaai/jina-embeddings-v3')
```

> The backbone is **frozen**, meaning we do not fine-tune it.

---

## Feature Extraction

Tokenize the dataset and extract pooled embeddings:

```python
tokenized_train = tokenizer(dataset['train']['text'], return_tensors='pt', truncation=True, padding='max_length', max_length=22)
vectorized_train = backbone(tokenized_train['input_ids']).pooler_output
```

The features are saved to disk using `pickle` for later use.

---

## Data Preparation

The extracted features and labels are split into training and validation datasets and wrapped into PyTorch `Dataset` and `DataLoader` objects:

```python
train_data = Dataset(X_train, y_train)
train_data_loader = DataLoader(train_data, batch_size=128, shuffle=True)
```

---

## Model Architecture

A simple classifier is built on top of the extracted features:

```python
class Model(torch.nn.Module):
    def __init__(self, n_classes, backbone):
        super(Model, self).__init__()
        self.backbone = backbone
        self.L1 = torch.nn.LazyLinear(512)
        self.dropout = torch.nn.Dropout(0.4)
        self.classifier = torch.nn.LazyLinear(n_classes)

    def forward(self, x, inp_text):
        if inp_text is not None:
            x = self.backbone(inp_text).pooler_output.to(dtype=torch.float)
        x = self.dropout(x)
        x = self.L1(x)
        x = torch.nn.functional.relu(x)
        x = self.dropout(x)
        out = self.classifier(x)
        return out
```

---

## Training

The model is trained using **cross-entropy loss** and **Adam optimizer**:

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss().to(device)
```

Training loop:

```python
for epoch in range(epochs):
    train_loss, train_metric = train_epoch(model, optimizer, criterion, train_data_loader)
    val_loss, val_metric = evaluate(model, criterion, val_data_loader)
```

> Achieved ~80% accuracy on the training set and ~68% on validation.

---

## Evaluation

Evaluate the model on the test set:

```python
test_loss, test_metric = evaluate(model, criterion, test_data_loader)
print(f'TEST DATA: loss = {test_loss:.2f}, accuracy = {test_metric*100:.2f}%')
```

Confusion matrix:

```python
y_preds = model(torch.Tensor(vectorized_test).to(device), None).argmax(dim=-1)
ConfusionMatrixDisplay.from_predictions(test_y, y_preds.cpu())
```

---

## Text Classification

Classify a new text using the trained model:

```python
def classify_text(text: str) -> tuple[int | str, np.ndarray]:
    model.eval()
    text = tokenizer(text, return_tensors='pt')['input_ids'].to(device)
    res = model(None, text)
    return res.argmax(dim=-1).item()
```

Example:

```python
classify_text("I love you")  # Returns label: love
```

---

## Word Impact Analysis (Bonus)

Analyze the impact of each word on the predicted label:

```python
def get_words_impact(text: str) -> list[tuple[str, np.ndarray]]:
    words = text.split()
    probs_orig = model(None, tokenizer(text, return_tensors='pt')['input_ids'].to(device))
    ...
```

Example:

```python
get_words_impact("Hello honey:)")
# Returns per-word impact on each emotion class
```

This can help interpret why a model predicts a specific emotion.
