# reviews-classification

A Russian-language text classifier for e-commerce marketplace reviews. Detects three content classes — **OK**, **TOXIC**, and **SPAM** — using a FastText supervised model, with a TF-IDF + LinearSVC baseline for comparison.

## Problem

Marketplace reviews are classified into three content categories:

| Label | Description |
|-------|-------------|
| `OK` (0) | Normal review |
| `TOXIC` (1) | Offensive, abusive, or threatening content |
| `SPAM` (2) | Ads for third-party sites, repetitive filler, rating manipulation |

## Dataset

Three sources are combined and balanced:

- **Toxic Russian Comments** — [`alexandersemiletov/toxic-russian-comments`](https://www.kaggle.com/datasets/alexandersemiletov/toxic-russian-comments) via KaggleHub; NORMAL → `OK`, everything else → `TOXIC`
- **E-commerce Reviews** — internal dataset with manually labeled toxic reviews and sampled positive/neutral ones
- **Synthetic Spam** — ~3 000 examples generated with GPT-4o (GitHub Models / Azure) covering ad links, repetition, keyword stuffing

Final dataset: shuffled, null-dropped, ~35 000+ samples across three classes.

## Pipeline

```
data-preparation  →  baseline-training  →  fasttext-training
```

| Notebook / Script | What it does |
|---|---|
| `notebooks/data-preparation.ipynb` | Downloads datasets, parses toxic labels, samples e-commerce data, generates spam with GPT-4o, encodes labels (OK=0, TOXIC=1, SPAM=2), saves `data/encoded_df.parquet` |
| `notebooks/baseline-training.ipynb` | TF-IDF + LinearSVC pipeline; train/test split, `classification_report` |
| `notebooks/fasttext-training.ipynb` | Converts parquet → FastText format (`data/train.txt`, `data/test.txt`), trains and evaluates the final model |

Scripts in the root (`data-preparation.py`, `baseline-training.py`, `fasttext-training.py`) are the Marimo source files backing the notebooks.

## Results

Evaluated on a 25% held-out test split (random_state=42).

### Baseline — TF-IDF + LinearSVC

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| OK    | 0.94 | 0.98 | 0.96 |
| TOXIC | 0.91 | 0.75 | 0.82 |
| SPAM  | 0.95 | 0.90 | 0.93 |
| **weighted avg** | **0.93** | **0.94** | **0.93** |

### FastText

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| OK    | 0.97 | 0.99 | 0.98 |
| TOXIC | 0.97 | 0.96 | 0.97 |
| SPAM  | 0.98 | 0.88 | 0.93 |
| **weighted avg** | **0.97** | **0.97** | **0.97** |

FastText achieves **+4 pp accuracy** (97% vs 93%) and closes most of the gap on TOXIC recall (96% vs 75%).

## FastText model

Trained with `fasttext.train_supervised`:

```python
model = fasttext.train_supervised(
    input="data/train.txt",
    epoch=25,
    lr=0.5,
    wordNgrams=2,
    dim=100,
    loss="softmax",
    minCount=1,
)
model.save_model("models/classifier.bin")
```

The saved model is `models/classifier.bin`. To load and run inference:

```python
import fasttext

model = fasttext.load_model("models/classifier.bin")
label, prob = model.predict("отличный товар, рекомендую!")
print(label[0].replace("__label__", ""))  # OK
```

## Setup

```bash
pip install fasttext marimo polars scikit-learn kagglehub matplotlib seaborn openai
```

For spam synthesis you need a GitHub Models / Azure token:

```bash
export GITHUB_TOKEN=<your_token>
```

## Running

```bash
# interactive Marimo notebooks
marimo edit notebooks/data-preparation.ipynb
marimo edit notebooks/baseline-training.ipynb
marimo edit notebooks/fasttext-training.ipynb

# or headless
python data-preparation.py
python baseline-training.py
python fasttext-training.py
```

## Tech stack

- [FastText](https://fasttext.cc/) — supervised text classification
- [Marimo](https://marimo.io/) — reactive Python notebooks
- [Polars](https://pola.rs/) — dataframe processing
- [scikit-learn](https://scikit-learn.org/) — TF-IDF baseline and metrics
- GPT-4o via GitHub Models — synthetic spam generation
