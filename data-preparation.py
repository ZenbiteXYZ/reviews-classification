import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Prerequisite
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import polars as pl
    import matplotlib.pyplot as plt
    import seaborn as sns

    return pl, plt, sns


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Data preparation
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Toxic comments dataset
    """)
    return


@app.cell
def _():
    import kagglehub

    # Download latest version
    toxic_comments_path = kagglehub.dataset_download("alexandersemiletov/toxic-russian-comments")

    print("Path to dataset files:", toxic_comments_path)
    return (toxic_comments_path,)


@app.cell
def _(pl, toxic_comments_path):
    toxic_comments = (
        pl.scan_csv(
            toxic_comments_path+"/dataset.txt",
            has_header=False,
            new_columns=["raw"],
            separator="\n",
            quote_char=None,
        )
        .with_columns([
            # все лейблы в список: ["INSULT", "THREAT"]
            pl.col("raw")
              .str.extract_all(r"__label__[A-Z]+")
              .list.eval(pl.element().str.replace("__label__", ""))
              .alias("labels"),

            # текст — всё после последнего лейбла
            pl.col("raw")
              .str.replace_all(r"(__label__[A-Z]+,?\s*)+", "")
              .str.strip_chars()
              .alias("text"),
        ])
        .drop("raw")
        .collect()
    )

    print(toxic_comments.sample(3))
    return (toxic_comments,)


@app.cell
def _(toxic_comments):
    toxic_comments
    return


@app.cell
def _(toxic_comments):
    toxic_comments['labels'].explode().value_counts()
    return


@app.cell
def _(pl, toxic_comments):
    clean_toxic_comments = toxic_comments.with_columns(
        pl.col("labels").list.contains("NORMAL").not_().cast(dtype=pl.Int8).alias("toxic")
    ).drop("labels")
    clean_toxic_comments
    return (clean_toxic_comments,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Comments class distribution
    """)
    return


@app.cell
def _(clean_toxic_comments, pl):
    clean_toxic_comments["toxic"].value_counts().with_columns(
        (pl.col("count") / pl.col("count").sum() * 100)
            .round(2)
            .alias("percent")
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Comments size distribution
    """)
    return


@app.cell
def _(clean_toxic_comments, pl):
    comments_length_df = clean_toxic_comments.with_columns(
        pl.col("text").str.len_chars().alias("length")
    )
    return (comments_length_df,)


@app.cell
def _(comments_length_df, plt, sns):
    sns.histplot(data=comments_length_df, x="length")
    plt.title("Reviews length distribution")
    plt.grid()
    plt.show()
    return


@app.cell
def _(comments_length_df):
    comments_length_df["length"].describe()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Toxic vs length
    """)
    return


@app.cell
def _(comments_length_df, plt, sns):
    sns.violinplot(data=comments_length_df, x="toxic", y="length")
    plt.yscale("log")
    plt.grid()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## E-commerce reviews dataset
    """)
    return


@app.cell
def _(pl):
    ecommerce_df = pl.read_csv("data/ecommerce-data.csv", quote_char='"')
    ecommerce_df.head()
    return (ecommerce_df,)


@app.cell
def _(pl):
    ecommerce_reviewed_df = (
        pl.read_csv("data/ecommerce-data-reviewed.csv", columns=["id", "review", "toxic"])
        .rename({"toxic": "label"})
    )
    ecommerce_reviewed_df
    return (ecommerce_reviewed_df,)


@app.cell
def _(ecommerce_df, ecommerce_reviewed_df):
    joined_ecommerce_df = ecommerce_reviewed_df.join(ecommerce_df, on="id", how="inner")
    joined_ecommerce_df
    return


@app.cell
def _(ecommerce_df, ecommerce_reviewed_df, pl):
    positive_ecommerce_df = (
        ecommerce_df
        .filter(pl.col("sentiment") == "positive")
        .sample(1000)
        .with_columns(pl.lit("OK").alias("label"))
    )

    neautral_ecommerce_df = (
        ecommerce_df
        .filter(pl.col("sentiment") == "neautral")
        .sample(500)
        .with_columns(pl.lit("OK").alias("label"))
    )


    clean_ecommerce_df = pl.concat([
        ecommerce_reviewed_df.select(["review", "label"]),
        positive_ecommerce_df.select(["review", "label"]),
        neautral_ecommerce_df.select(["review", "label"]),
    ])
    return (clean_ecommerce_df,)


@app.cell
def _(clean_ecommerce_df):
    clean_ecommerce_df.sample(10)
    return


@app.cell
def _(clean_ecommerce_df):
    clean_ecommerce_df.write_csv("clean_ecommerce_df.csv")
    return


@app.cell
def _(clean_ecommerce_df, pl):
    clean_ecommerce_df.with_columns([
        pl.col("label").replace("", "OK")
    ]).rename({"review": "text"}).write_csv("ecommerce-data.csv")
    return


@app.cell
def _():
    return


@app.cell
def _(pl):
    loaded_ecommerce_df = pl.read_csv("data/ecommerce-data.csv")
    loaded_ecommerce_df.head()
    return (loaded_ecommerce_df,)


@app.cell
def _(loaded_ecommerce_df):
    loaded_ecommerce_df["label"].value_counts()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Spam reviews synthesis
    """)
    return


@app.cell
def _():
    import re
    import json

    def parse_json_response(content: str) -> list[str]:
        # убираем markdown
        content = re.sub(r"```json|```", "", content).strip()

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # если JSON обрезан - вытащим строки regex'ом
            matches = re.findall(r'"([^"\\]*(?:\\.[^"\\]*)*)"', content)
            return matches

    return (parse_json_response,)


@app.cell
def _():
    import os

    return (os,)


@app.cell
def _(os):
    GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
    return (GITHUB_TOKEN,)


@app.cell(disabled=True)
def _(GITHUB_TOKEN, parse_json_response):
    from openai import OpenAI

    client = OpenAI(
        base_url="https://models.inference.ai.azure.com",
        api_key=GITHUB_TOKEN,
    )

    spam_texts = []
    target_spam_texts_lenght = 3000

    while len(spam_texts) < target_spam_texts_lenght:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": """Сгенерируй 200 примеров спам-отзывов на русском языке для маркетплейса.

        Типы спама:
        - реклама сторонних сайтов ("купи дешевле на xxx.ru")
        - повторяющийся текст ("хороший товар хороший товар хороший товар")
        - бессмысленный набор слов
        - накрутка ("5 звезд лучший товар лучший продавец лучший магазин")
        - вместо xxx.ru должны быть реальные домены маркетплейсов

        Верни ТОЛЬКО валидный JSON массив строк без преамбулы и markdown:
        ["текст1", "текст2", ...]"""
            }],
            max_tokens=16384,
        )
        content = response.choices[0].message.content
        spam_texts.extend(parse_json_response(content))
    return (spam_texts,)


@app.cell
def _(spam_texts):
    print(len(spam_texts))
    return


@app.cell
def _(pl, spam_texts):
    spam_df = pl.DataFrame({
        "text": spam_texts,
        "label": "SPAM"
    })
    spam_df
    return (spam_df,)


@app.cell
def _(spam_df):
    spam_df.write_parquet("data/spam-data.parquet")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Joining datasets
    """)
    return


@app.cell
def _(loaded_ecommerce_df, pl):
    selected_ecommerce = loaded_ecommerce_df.filter(
        pl.col("label").is_in(["OK", "TOXIC"])
    )
    selected_ecommerce
    return (selected_ecommerce,)


@app.cell
def _(clean_toxic_comments, pl):
    selected_comments = clean_toxic_comments.sample(30_000).with_columns(
        pl.col("toxic")
        .map_elements(lambda x: "TOXIC" if x else "OK")
        .alias("label")
    ).drop("toxic")
    selected_comments
    return (selected_comments,)


@app.cell
def _(spam_df):
    selected_spam = spam_df
    return (selected_spam,)


@app.cell
def _(pl, selected_comments, selected_ecommerce, selected_spam):
    concated_df = pl.concat([selected_comments, selected_ecommerce, selected_spam])
    concated_df["label"].value_counts()
    return (concated_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Droping nulls
    """)
    return


@app.cell
def _(concated_df):
    concated_df.null_count()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Shuffling
    """)
    return


@app.cell
def _(concated_df):
    final_df = concated_df.drop_nulls().sample(fraction=1.0, shuffle=True)
    return (final_df,)


@app.cell
def _(final_df):
    final_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Target encoding
    """)
    return


@app.cell
def _(final_df, pl):
    class_mapping = {"OK": 0, "TOXIC": 1, "SPAM": 2}

    encoded_df = final_df.with_columns(
        pl.col("label").replace(class_mapping).cast(pl.Int8).alias("label")
    )
    return (encoded_df,)


@app.cell
def _(encoded_df):
    encoded_df.write_parquet("data/encoded_df.parquet")
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


if __name__ == "__main__":
    app.run()
