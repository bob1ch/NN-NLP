# Emotion Classification with RNNs

This project implements multilabel emotion classification using **RNN-based models** (LSTM and GRU) on the [GoEmotions dataset](https://github.com/google-research/google-research/tree/master/goemotions). It supports multiple architectures, including stacked and bidirectional RNNs, and allows training on datasets with **fixed** or **variable sentence lengths**.

---

## Table of Contents

* [Project Overview](#project-overview)
* [Setup and Requirements](#setup-and-requirements)
* [Dataset](#dataset)
* [Model Architecture](#model-architecture)
* [Training](#training)
* [Evaluation](#evaluation)
* [Inference](#inference)
* [Results](#results)

---

## Project Overview

The goal of this project is to train and evaluate RNN-based models for predicting **28 emotion labels** from text. The models support:

* LSTM or GRU cells
* Stacked layers
* Bidirectional feature extraction

Both **fixed-length** and **variable-length** tokenized datasets are supported.

---

## Setup and Requirements

```bash
# Python 3.12
pip install torch torchvision torcheval torchmetrics
pip install transformers datasets
pip install scikit-learn matplotlib tqdm
```

> GPU is recommended for training due to dataset size and model complexity.

---

## Dataset

* Dataset used: **GoEmotions** (`datasets.load_dataset('google-research-datasets/go_emotions')`)
* Number of labels: 28

```python
emotions = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval', 
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 
    'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]
```

* Two tokenization approaches:

  1. **Fixed-length sentences** (padding/truncating to 32 tokens)
  2. **Variable-length sentences** (dynamic padding in DataLoader)

---

## Model Architecture

The `Model` class supports:

* **Embedding layer** → RNN → Fully connected output
* RNN options: **LSTM** or **GRU**
* Options: `stacked` layers, `bidirectional` or unidirectional
* Output layer predicts 28 emotion labels

```python
model = get_model(
    units=128,
    n_tokens=len(tokenizer.vocab),
    n_labels=len(emotions),
    n_stacks=2,
    bidirectional=True,
    name='LSTM Stacked Bidirectional',
    cell_type=torch.nn.LSTM
)
```

---

## Training

* Loss function: **BCEWithLogitsLoss** (suitable for multilabel classification)
* Metric: **Multilabel AUROC (macro)**
* Optimizer: **Adam**

```python
train_loss, train_metric = train(model, loss_fn, optimizer, train_dataloader)
test_loss, test_metric = test(model, loss_fn, test_dataloader)
```

### Trained Model Variants:

* Simple RNN
* Stacked RNN
* Bidirectional RNN
* Stacked Bidirectional RNN

For both LSTM and GRU, trained on **fixed-length** and **variable-length** datasets.

---

## Evaluation

* Metrics: **ROC AUC per label**
* Plots: ROC curves for each emotion

```python
plot_roc_curve(test_dataloader, model, ax=ax)
```

* Mean ROC AUC scores can be computed to compare models and architectures.
* Stacked bidirectional models tend to achieve the highest ROC AUC.

---

## Inference

Predict emotions for new texts using `label_text`:

```python
texts = [
    'Tomorrow I will go to school',
    'Thanks for the help!',
    'Love you honey'
]

for labels in label_text(texts, var_models[-1], threshold=0.5, max_length=16).tolist():
    print(np.array(emotions)[labels])
```

* Outputs: list of predicted emotions above a probability threshold.
* Optionally visualize **emotion probabilities**:

```python
plot_emotion_scores("What do you call a happy cowboy? A jolly rancher.", var_models[-1], 16, ax, emotions)
```

---

## Results

* Training on variable-length datasets generally produces slightly better results due to preservation of sentence context.
* GRU-based models often slightly outperform LSTM for smaller datasets, but differences are small.
* Best performing model: **Stacked Bidirectional GRU** on variable-length dataset
* Sample predictions are **sensible** and match intuitive emotion labels.

---

## Notes

* Ensure that your environment supports **CUDA** for faster training.
* Adjust `max_length` and `BATCH_SIZE` depending on GPU memory.
* ROC AUC and AUROC metrics are computed **macro-averaged** across labels.
