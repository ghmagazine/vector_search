#!/usr/bin/env python

from sentence_transformers import SentenceTransformer

import os
import pandas as pd


# 基本的なベクトル化モデルとして、Sentence TransformersのMiniLM-L6を読み込む関数
def get_basic_vectorization_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# 本書デフォルトのベクトル化の引数
DEFAULT_ARGS = {"show_progress_bar": True}


# 与えられたモデルと引数で、「別に与えられたテキストのリストをベクトル化する関数」を返す関数
def vectorize_with(model, args=DEFAULT_ARGS):

    # 与えられたテキストのリストをベクトル化する関数
    def vectorize(texts):
        # 高速化のため、対象のテキストを重複排除する
        unique_texts = sorted(set(texts))

        # モデルを推論する。このとき引数も与える
        vectors = model.encode(unique_texts, **args)

        # もとのテキスト列に対応するベクトル列に戻す
        text_to_vector = {text: vector for text, vector in zip(unique_texts, vectors)}
        return [text_to_vector[text] for text in texts]

    return vectorize


# 基本的なベクトル化モデルでベクトル化したデータの保存先。このまま実行した場合は、
# 本書のサンプルコードのディレクトリ code 以下、tmp/basic-vectorized.parquet に保存する
BASIC_VECTORIZED_PARQUET_PATH = os.path.join(
    os.path.dirname(__file__), "tmp", "basic-vectorized.parquet"
)


# ベクトル化したデータを保存する関数
def write_basic_vectorized_data(data):
    data.to_parquet(
        BASIC_VECTORIZED_PARQUET_PATH,
        index=False,
        engine="pyarrow",  # 1.0.1: .parquet ファイルを扱う際、engine を明示しました。
    )


# ベクトル化したデータを読み込む関数
def read_basic_vectorized_data():
    # ファイルが存在すれば読み込む
    if os.path.isfile(BASIC_VECTORIZED_PARQUET_PATH):
        return pd.read_parquet(
            BASIC_VECTORIZED_PARQUET_PATH,
            engine="pyarrow",  # 1.0.1: .parquet ファイルを扱う際、engine を明示しました。
        )

    # 存在しなければ例外をあげる
    raise ValueError("事前にベクトル化したデータがありません（第2章を参照）")


# データからベクトルの次元数を取得するヘルパー関数
def get_dimension_number_of(data):
    # 単に最初 (iloc[0]) のクエリベクトルの次元数を取得する
    return len(data["query_vector"].iloc[0])


# データをクエリとドキュメントに分割するヘルパー関数
def split_into_query_and_document(data):
    return (
        data.drop_duplicates("query_id"),
        data.drop_duplicates("product_id"),
    )


# このコードを直に実行した場合のみ、以下のコードを実行する
if __name__ == "__main__":
    from ch01_data_preparation import read_jp_data

    # 基本的なベクトル化モデルと本書デフォルトの引数で、
    # 「別に与えられたテキストのリストをベクトル化する関数」vectorize を準備する
    basic_vectorization_model = get_basic_vectorization_model()
    vectorize = vectorize_with(basic_vectorization_model)

    # 適当なテキストをベクトル化する
    texts = [
        "HDMIケーブル",
        "電話機",
        "液晶ディスプレイ",
        "液晶テレビ",
    ]
    vectors = vectorize(texts)

    # ベクトルの件数を表示する
    print(len(vectors))

    # 各ベクトルの次元数を表示する
    for vector in vectors:
        print(len(vector))

    # 題材のデータセットから1パーセントサンプリングしつつ読み込む
    jp_data = read_jp_data(sample_rate=0.01)

    # クエリと製品タイトルをベクトル化し、新たな列として保存する
    jp_data["query_vector"] = vectorize(jp_data["query"])
    jp_data["title_vector"] = vectorize(jp_data["product_title"])

    # ベクトル化したデータを保存する
    write_basic_vectorized_data(jp_data)
