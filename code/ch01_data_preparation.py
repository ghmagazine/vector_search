#!/usr/bin/env python

import os
import pandas as pd


# 題材のデータセットをメモリに読み込む関数。クエリ単位でサンプリングする機能もある
def read_jp_data(sample_rate=1.0, split=None, read_product_detail=False):

    # 製品（products）とその他の情報（examples）の各ファイルのパスを組み立てる
    examples_path, products_path = (
        os.path.join(
            os.path.dirname(__file__),
            "esci-data",
            "shopping_queries_dataset",
            f"shopping_queries_dataset_{suffix}.parquet",
        )
        for suffix in ("examples", "products")
    )

    # 本書で扱うサブセットに絞り込む
    example_filters = [("small_version", "==", 1), ("product_locale", "==", "jp")]

    # 訓練データまたはテストデータの指定があれば、さらに絞り込む
    if split is not None:
        example_filters.append(("split", "==", split))

    # 必要に応じてクエリ単位でサンプリングする
    if sample_rate < 1.0:
        # サンプリングが必要。まずクエリID列だけを読み込む
        query_ids = pd.read_parquet(
            examples_path,
            columns=["query_id"],
            filters=example_filters,
            engine="pyarrow",  # 1.0.1: .parquet ファイルを扱う際、engine を明示しました。
        )["query_id"]
        query_ids = set(query_ids)

        # たとえば1パーセントであれば、100で割った余りが0であるクエリIDだけを取り出す
        denominator = int(1.0 / sample_rate)
        query_ids = filter(lambda query_id: query_id % denominator == 0, query_ids)

        # のちに実際に使うデータセットを読み込むときに、クエリIDで絞り込む
        example_filters.append(("query_id", "in", query_ids))

    # 製品のテーブルの読み込む列も絞り込む。サイズが大きいが本書の大部分では使わないため
    product_columns_to_read = ["product_id", "product_title"]
    if read_product_detail:
        product_columns_to_read += [
            "product_brand",
            "product_color",
            "product_description",
            "product_bullet_point",
        ]

    # 製品のテーブルとその他の情報のテーブルを読み込み、製品ID列をキーとして結合して返す
    return pd.merge(
        pd.read_parquet(
            examples_path,
            columns=["query", "query_id", "product_id", "esci_label", "split"],
            filters=example_filters,
            engine="pyarrow",  # 1.0.1: .parquet ファイルを扱う際、engine を明示しました。
        ),
        pd.read_parquet(
            products_path,
            columns=product_columns_to_read,
            filters=[("product_locale", "==", "jp")],
            engine="pyarrow",  # 1.0.1: .parquet ファイルを扱う際、engine を明示しました。
        ),
        on="product_id",
    )


# このコードを直に実行した場合のみ、以下のコードを実行する
# つまり、このコードをほかのコードにimportした場合は、以下のコードを実行しない
if __name__ == "__main__":

    # クエリ数とデータの行数が正しいか確認する関数
    def assert_counts(sample_rate, split, query_count, row_count):
        if sample_rate == 1.0:
            # サンプリングなしの場合、データセットのREADMEに記載のとおりになるはず
            if split is None:
                assert (query_count, row_count) == (10407, 297883)
            if split == "train":
                assert (query_count, row_count) == (7284, 209094)
            if split == "test":
                assert (query_count, row_count) == (3123, 88789)
        else:
            # 1パーセントサンプリングした場合、以下のとおりになるはず
            if split is None:
                assert (query_count, row_count) == (97, 2681)
            if split == "train":
                assert (query_count, row_count) == (66, 1811)
            if split == "test":
                assert (query_count, row_count) == (31, 870)

    # 実際に確認する
    for sample_rate in [1.0, 0.01]:
        for split in [None, "train", "test"]:
            jp_data = read_jp_data(sample_rate=sample_rate, split=split)
            query_count = len(set(jp_data["query_id"]))
            row_count = len(jp_data)
            assert_counts(sample_rate, split, query_count, row_count)

    # 特定のクエリについて、データの例を表示する
    jp_data = read_jp_data(sample_rate=0.01, split="test")
    jp_data = jp_data[jp_data.query_id == 119300]
    print(
        jp_data.to_string(
            columns=["esci_label", "query", "product_title"],
            index=False,
            max_colwidth=30,
        )
    )
