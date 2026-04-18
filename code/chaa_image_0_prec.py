#!/usr/bin/env python

from ch02_basic_vectorization import DEFAULT_ARGS
from ch03_vector_search_engines_0_numpy import cos
from io import BytesIO
from sentence_transformers import models, SentenceTransformer

import datasets
import pandas as pd
import PIL


# 題材の画像データセットを読み込む関数
def read_image_data(split, sample_percent=100):
    # Hugging Face Datasetとして読み込み、pandas DataFrameに変換する
    image_data = datasets.load_dataset(
        "zalando-datasets/fashion_mnist",
        # 訓練またはテストデータに絞り、かつ指定のサンプル率で読み込む
        split=f"{split}[:{sample_percent}%]",
    ).to_pandas()

    # 以下の形式のデータを画像（PILのImageFileのインスタンス）としてデコードする関数
    # {'bytes': b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHD...
    def decode(image_dict):
        return PIL.Image.open(BytesIO(image_dict["bytes"]))

    # 画像をデコードする
    image_data["image"] = image_data["image"].apply(decode)

    return image_data


# 題材の画像データセットのラベル（リストのインデックス）と説明文（リストの要素）との
# 対応の宣言。本書では説明文をクエリのように扱う
IMAGE_LABEL_TO_QUERY = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


# 画像にも対応するベクトル化モデルとして、openai/clip-vit-base-patch32 を読み込み、
# SentenceTransformerのインスタンスにラップして返す関数
def get_text_or_image_vectorization_model():
    return SentenceTransformer(
        modules=[models.CLIPModel("openai/clip-vit-base-patch32")]
    )


# 与えられたモデルと引数で、「別に与えられたリストの要素をベクトル化する関数」を返す関数
def vectorize_texts_or_images_with(model, args=DEFAULT_ARGS):

    # 与えられたリストの要素をベクトル化する関数
    def vectorize(texts_or_images):
        return list(model.encode(texts_or_images, **args))

    return vectorize


# 単一のクエリに対して、スコアおよび適合率の計算を行う関数
def test_image(image_data, expected_label, query_vector, k=10):
    # クエリとドキュメント（画像）の両ベクトルのスコア（コサイン類似度）を計算する
    scores = pd.DataFrame(
        {
            "query_vector": len(image_data) * [query_vector],
            "image_vector": image_data.image_vector,
        }
    ).apply(cos, axis=1)

    # スコアの降順にソートした場合の順位を計算する
    ranks = scores.rank(ascending=False, method="first")

    # ランキング結果の上位k件の適合率を計算する
    return sum(image_data[ranks <= k]["label"] == expected_label) / k


# ベクトル化、スコアの計算、適合率のクエリ間平均の計算と表示を一気に行う関数
def evaluate(model, data):
    # クエリとドキュメントをベクトル化する
    vectorize = vectorize_texts_or_images_with(model=model)
    query_vectors = vectorize(IMAGE_LABEL_TO_QUERY)
    # 1.1.1: `sentence-transformers` のバージョンによっては `pandas.Series` を渡すと例外が上がる
    data["image_vector"] = vectorize(data["image"].to_list())

    # 適合率のクエリ間平均を計算し表示する
    precisions = [
        test_image(data, expected_label, query_vector)
        for expected_label, query_vector in enumerate(query_vectors)
    ]
    print(f"Mean Precision: {sum(precisions) / len(precisions):.03f}")


# このコードを直に実行した場合のみ、以下のコードを実行する
if __name__ == "__main__":
    # テストデータを1パーセントサンプリングと画像のデコードをしつつ読み込む
    image_test_data = read_image_data(split="test", sample_percent=1)

    # 画像にも対応するベクトル化モデルを読み込み、適合率のクエリ間平均を計算し表示する
    evaluate(get_text_or_image_vectorization_model(), image_test_data)
