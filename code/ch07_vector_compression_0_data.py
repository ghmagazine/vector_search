#!/usr/bin/env python

import os
import pandas as pd


# ファインチューニング後のモデルでベクトル化したデータの保存先。このまま実行した場合は、
# 本書のサンプルコードのディレクトリ code 以下、tmp/tuned-vectorized.parquet に保存する
TUNED_VECTORIZED_PARQUET_PATH = os.path.join(
    os.path.dirname(__file__), "tmp", "tuned-vectorized.parquet"
)


# ベクトル化したデータを保存する関数
def write_tuned_vectorized_data(data):
    data.to_parquet(
        TUNED_VECTORIZED_PARQUET_PATH,
        index=False,
        engine="pyarrow",  # 1.0.1: .parquet ファイルを扱う際、engine を明示しました。
    )


# ベクトル化したデータを読み込む関数
def read_tuned_vectorized_data():
    # ファイルが存在すれば読み込む
    if os.path.isfile(TUNED_VECTORIZED_PARQUET_PATH):
        return pd.read_parquet(
            TUNED_VECTORIZED_PARQUET_PATH,
            engine="pyarrow",  # 1.0.1: .parquet ファイルを扱う際、engine を明示しました。
        )

    # 存在しなければ例外をあげる
    raise ValueError("事前にベクトル化したデータがありません（第7章を参照）")


# このコードをじかに実行した場合のみ、以下のコードを実行する
if __name__ == "__main__":
    from argparse import ArgumentParser
    from ch01_data_preparation import read_jp_data
    from ch02_basic_vectorization import vectorize_with
    from ch05_advanced_vectorization_0_tune import get_tuned_vectorization_model

    # コマンドライン引数から、データのサンプル率を読み込む
    argument_parser = ArgumentParser()
    argument_parser.add_argument(
        "--sample-rate", default=0.01, type=float, choices=[0.01, 1.0]
    )
    args = argument_parser.parse_args()

    # 題材のデータセットを読み込む
    jp_data = read_jp_data(sample_rate=args.sample_rate)

    # ファインチューニング後のモデルを読み込む
    model = get_tuned_vectorization_model()

    # クエリとドキュメント（製品タイトル）をベクトル化し、新たな列として保存する
    vectorize = vectorize_with(model)
    jp_data["query_vector"] = vectorize(jp_data["query"])
    jp_data["title_vector"] = vectorize(jp_data["product_title"])

    # ベクトル化したデータを保存する
    write_tuned_vectorized_data(jp_data)
