# Seq2Seq Russian-to-English Translator with PyTorch

This project implements a sequence-to-sequence (Seq2Seq) neural network for Russian-to-English translation using PyTorch. The model includes options for LSTM, GRU, and attention-based architectures.

---

## Table of Contents

* [Installation](#installation)
* [Dataset](#dataset)
* [Tokenizer](#tokenizer)
* [DataLoader](#dataloader)
* [Model](#model)
* [Training](#training)
* [Translation / Testing](#translation--testing)
* [Results](#results)

---

## Installation

Install the required packages using pip:

```bash
pip install torch transformers torchmetrics tqdm numpy pandas scikit-learn matplotlib
```

> Note: Ensure CUDA is available if using GPU acceleration.

---

## Dataset

The dataset should contain parallel Russian-to-English sentences. It can be a TSV file structured like:

| id_1 | rus            | id_2 | eng           |
| ---- | -------------- | ---- | ------------- |
| 1    | Привет мир!    | 1    | Hello world!  |
| 2    | Я люблю пиццу. | 2    | I love pizza. |

Load the dataset:

```python
data = pd.read_csv('seq2seq_dataset.tsv', sep='\t', names=['id_1', 'rus', 'id_2', 'eng'])
data = data[['rus', 'eng']]
```

---

## Tokenizer

A custom tokenizer is used to add BOS and EOS tokens:

```python
from transformers import GPT2Tokenizer

class Tokenizer(GPT2Tokenizer):
    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        if token_ids_1 is None:
            return [self.bos_token_id, *token_ids_0, self.eos_token_id]
        return [self.bos_token_id, *token_ids_0, self.bos_token_id, *token_ids_1, self.eos_token_id]

tokenizer = Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token_id = tokenizer.eos_token_id
```

Tokenize your data:

```python
def to_tokenize(data):
    return tokenizer(data, max_length=32, truncation=True, padding=True)['input_ids']

X, y = to_tokenize(data['rus'].tolist()), to_tokenize(data['eng'].tolist())
```

---

## DataLoader

Create a DataLoader for training:

```python
def get_dataloader(batch_size, X, y):
    X = torch.LongTensor(np.array(X)).to('cuda')
    out_BOS = torch.LongTensor(y).to('cuda')[:, :-1]
    out_EOS = torch.LongTensor(y).to('cuda')[:, 1:]
    masking_PAD = (out_BOS != tokenizer.pad_token_id).to('cuda')
    masking_PAD[:, 0] = True
    dataset = torch.utils.data.TensorDataset(X, out_BOS, out_EOS, masking_PAD)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

train_loader = get_dataloader(256, X_train, y_train)
test_loader = get_dataloader(256, X_test, y_test)
```

---

## Model

The project supports multiple architectures:

* **LSTM / GRU Seq2Seq**

```python
class EncoderRNN(nn.Module): ...
class DecoderRNN(nn.Module): ...
class Seq2seq(nn.Module): ...
```

* **Attention-based Seq2Seq**

```python
class AttnEncoderRNN(nn.Module): ...
class AttnDecoderRNN(nn.Module): ...
class AttnSeq2seq(nn.Module): ...
```

Instantiate a model:

```python
model_LSTM = Seq2seq(nn.LSTM, tokenizer.vocab_size, 128, 'cuda', True, 2)
model_GRU = Seq2seq(nn.GRU, tokenizer.vocab_size, 128, 'cuda', True, 2)
model_attn = AttnSeq2seq(nn.GRU, tokenizer.vocab_size, 128, 'cuda', True, 2)
```

---

## Training

Train the model using teacher forcing:

```python
def train_epoch(dataloader, model, optimizer, criterion, teacher_forcing=True): ...
def train(train_loader, model, n_epochs=15, learning_rate=0.001): ...
```

Example:

```python
train(train_loader, model_LSTM, n_epochs=15)
torch.save(model_LSTM.state_dict(), 'models/seq2seq_lstm.pth')
```

---

## Translation / Testing

Translate sentences using the trained model:

```python
def translate(text, tokenizer, model, max_len=20, attention=False):
    input_ids = tokenizer(text, return_tensors="pt")['input_ids'].to('cuda')
    encoder_outputs, encoder_hidden = model.encoder(input_ids)
    decoder_input = torch.tensor([[tokenizer.bos_token_id]]).to('cuda')
    decoder_hidden = encoder_hidden
    generated_tokens = []
    
    for _ in range(max_len):
        embedded = model.decoder.embedding(decoder_input)
        decoder_output, decoder_hidden = model.decoder.rnn(embedded, decoder_hidden)
        if attention:
            out, _ = model.decoder.attention(decoder_output, encoder_outputs, encoder_outputs)
            output = torch.cat((decoder_output, out), dim=-1)
            output_token_logits = model.decoder.out(output[:, -1, :])
        else:
            output_token_logits = model.decoder.out(decoder_output[:, -1, :])
        output_token = torch.argmax(output_token_logits, dim=-1)
        if output_token.item() == tokenizer.eos_token_id:
            break
        decoder_input = output_token.unsqueeze(0)
        generated_tokens.append(output_token.item())
    return tokenizer.decode(generated_tokens, skip_special_tokens=True)
```

---

## Example Translation

```python
corpus = [
    'Доброе утро.',
    'Привет мир!',
    'Кот сидит на столе.'
]

for text in corpus:
    print(translate(text, tokenizer, model_attn, max_len=30, attention=True))
```

---

## Results

* LSTM / GRU without attention produce decent translations.
* Attention-based GRU significantly improves translation quality.
* Model performance improves over ~15 epochs with teacher forcing.

---

## Notes

* GPU acceleration is recommended for training.
* Dataset sequences are padded to a fixed maximum length (32 tokens).
* The attention mechanism used is `MultiheadAttention` with 1 head.
