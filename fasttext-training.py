import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import polars as pl

    import fasttext

    return fasttext, pl


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Data preprocessing
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Class definition
    - 0: OK
    - 1: TOXIC
    - 2: SPAM
    """)
    return


@app.cell
def _():
    from sklearn.model_selection import train_test_split

    return (train_test_split,)


@app.cell
def _(pl):
    df = pl.read_parquet("data/encoded_df.parquet")
    df.sample(5)
    return (df,)


@app.cell
def _(df):
    df.shape
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Train / Test split
    """)
    return


@app.cell
def _(df):
    y = df.drop_in_place("label")
    X = df.clone()
    return X, y


@app.cell
def _(X, train_test_split, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    X_train = X_train["text"].to_list()
    X_test = X_test["text"].to_list()
    return X_test, X_train, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Prepare FastText format
    """)
    return


@app.cell
def _():
    import re

    return (re,)


@app.cell
def _(X_train, re, y_train):
    with open("data/train.txt", "w") as f1:
        for train_text, train_label in zip(X_train, y_train):
            train_text = re.sub(r"[\r\n]+", " ", train_text).strip()
            f1.write(f"__label__{train_label} {train_text}\n")
    return


@app.cell
def _(X_test, re, y_test):
    with open("data/test.txt", "w") as f2:
        for test_text, test_label in zip(X_test, y_test):
            test_text = re.sub(r"[\r\n]+", " ", test_text).strip()
            f2.write(f"__label__{test_label} {test_text}\n")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # FastText training
    """)
    return


@app.cell
def _():
    import os
    from pathlib import Path
    from huggingface_hub import hf_hub_download
    import shutil

    return Path, hf_hub_download, os, shutil


@app.cell
def _(Path, os):
    MODEL_DIR = Path(os.getcwd()) / "models/fasttext"
    os.makedirs(MODEL_DIR, exist_ok=True)

    model_path = MODEL_DIR / "classifier.bin"
    return (model_path,)


@app.cell
def _(fasttext, hf_hub_download, model_path, os, shutil):
    def load_model():
        hf_token = os.getenv("HF_TOKEN", None)

        model_cache_path = hf_hub_download(
            repo_id="ZenbiteXYZ/reviews-classification",
            filename="classifier.bin",
            repo_type="model",
            token=hf_token
        )
    
        shutil.copy(model_cache_path, model_path)
        return fasttext.load_model(str(model_path))


    def get_model(retrain=False):
        if retrain:
            model = fasttext.train_supervised(
                input="data/train.txt",
                epoch=25,
                lr=0.5,
                wordNgrams=2,
                dim=100,
                loss="softmax",
                minCount=1,
            )
            model.save_model(str(model_path))
        else:
            model = load_model()
        return model

    model = get_model(retrain=False)
    return (model,)


@app.cell
def _(model):
    test_result = model.test("data/test.txt", k=1)
    print(f"Precision: {test_result[1]:.3f}, Recall: {test_result[2]:.3f}")
    return


@app.cell
def _(model):
    from sklearn.metrics import classification_report

    texts, true_labels = [], []
    with open("data/test.txt") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(" ", 1)
            if len(parts) < 2:
                continue
            true_labels.append(parts[0].replace("__label__", ""))
            texts.append(parts[1])

    pred_labels = [model.predict(t)[0][0].replace("__label__", "") for t in texts]

    print(classification_report(true_labels, pred_labels))
    return


if __name__ == "__main__":
    app.run()
