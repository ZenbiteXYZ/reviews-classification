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
    import numpy as np

    return (pl,)


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
    df.describe()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Class Distribution
    """)
    return


@app.cell
def _(df):
    df["label"].value_counts()
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
    ## Train Baseline SVM
    """)
    return


@app.cell
def _():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline
    from sklearn.svm import LinearSVC

    tfidf_vectorizer = TfidfVectorizer()

    svc = LinearSVC(max_iter=1000, class_weight="balanced")
    return Pipeline, svc, tfidf_vectorizer


@app.cell
def _(Pipeline, svc, tfidf_vectorizer):
    pipeline = Pipeline([
        ("tf-idf", tfidf_vectorizer),
        ("svc", svc)
    ])
    return (pipeline,)


@app.cell
def _(X_train, pipeline, y_train):
    pipeline.fit(X_train, y_train)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Test baseline
    """)
    return


@app.cell
def _():
    from sklearn.metrics import classification_report

    return (classification_report,)


@app.cell
def _(X_test, pipeline):
    y_pred = pipeline.predict(X_test)
    return (y_pred,)


@app.cell
def _(classification_report, y_pred, y_test):
    print(classification_report(y_test, y_pred))
    return


@app.cell
def _():
    class_mapping = {0: "OK", 1: "TOXIC", 2: "SPAM"}
    return (class_mapping,)


@app.cell
def _(class_mapping, pipeline):
    test_cases = [
        "отличный товар рекомендую всем пять звезд",
        "купи дешевле на сайте xxx.ru акция до конца месяца",
        "продавец м***к верните деньги уроды",
        "хороший товар хороший товар хороший товар хороший товар",
        "товар не пришел очень расстроена",
        "Блузка хорошая, пошив просто класс",
        "Дабуди дабудай"
    ]

    predictions = pipeline.predict(test_cases)
    for text, pred in zip(test_cases, predictions):
        print(f"{class_mapping[pred]} | {text}")
    return


if __name__ == "__main__":
    app.run()
