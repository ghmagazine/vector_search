#!/usr/bin/env python

from opensearchpy import OpenSearch
from tqdm.asyncio import tqdm_asyncio

import asyncio
import json
import multiprocessing
import os
import pandas as pd


# 本書デフォルトの、インデックスのオプションを返す関数
def get_default_index_options(dimension):
    return {
        "mappings": {
            "properties": {
                # title_vector フィールドの値は、ベクトルとして扱う指定
                "title_vector": {
                    "type": "knn_vector",
                    # ベクトルの次元数
                    "dimension": dimension,
                }
            }
        }
    }


# 本書デフォルトの、ドキュメントデータの行をOpenSearchドキュメントに変換する関数
def format_document(row):
    return {
        # 製品タイトルのテキスト
        "product_title": row.product_title,
        # そのベクトル
        "title_vector": row.title_vector,
    }


# 本書デフォルトの、クエリデータの行をOpenSearchクエリに変換する関数
def format_query(row):
    return {
        "script_score": {
            # すべてのドキュメントを抽出する（総当たり）指定
            "query": {"match_all": {}},
            # ドキュメントに、ベクトルにもとづきスコアをつける指定
            "script": {
                "source": "knn_score",
                "lang": "knn",
                "params": {
                    # ドキュメントベクトルのフィールド名
                    "field": "title_vector",
                    # クエリベクトルそのもの
                    "query_value": row.query_vector,
                    # 両ベクトルの内積（またはドット積）を計算する指定
                    "space_type": "innerproduct",
                },
            },
        }
    }


# 本書のサンプルコードでOpenSearchを扱うためのクラス
class OpenSearchTester:

    # インスタンスを作成する特殊メソッド
    def __init__(self, index_name, concurrency=multiprocessing.cpu_count()):
        # すべてのメソッドで共通のインデックスを扱うので、その名前を保存しておく
        self.index_name = index_name

        # OpenSearchへのリクエストの並列度も保存しておく。デフォルトではCPU数と同じ
        self.concurrency = concurrency

        # OpenSearchへは：
        # - 別のサービスからはDocker Composeのサービス名（opensearch）でアクセスする
        # - Dockerを実行するマシンからは自身（localhost）のポートを介してアクセスする
        # このコードが /code 以下にあればコンテナで実行されているとみなす
        host = "opensearch" if __file__.startswith("/code/ch") else "localhost"

        # 内部で使うOpenSearchクライアントをインスタンス化し、これも保存しておく
        self.open_search = OpenSearch(
            # ホスト名は前述のとおり、ポート番号はデフォルト
            hosts=[{"host": host, "port": 9200}],
            # デフォルトの設定ではSSLが有効だが、証明書は広く通用するものではない
            use_ssl=True,
            verify_certs=False,
            ssl_show_warn=False,
            # 管理者パスワードの設定が必須で、サンプルコードなので常に管理者としてアクセスする
            http_auth=("admin", "Vect0rSe@rchEngine"),
        )

    # インデックスを作成するメソッド
    def create_index(self, options):
        # 同名のインデックスが存在するなら削除する
        if self.open_search.indices.exists(index=self.index_name):
            self.open_search.indices.delete(index=self.index_name)
        # 作成する
        self.open_search.indices.create(index=self.index_name, body=options)

    # 多数のドキュメントを整形し、入力するメソッド
    def input_documents(self, data, formatter=format_document):
        # 並列度を制限するためのセマフォをインスタンス化する
        semaphore = asyncio.Semaphore(self.concurrency)

        # 少数のドキュメントを整形し、まとめて入力する関数（非同期）
        async def input_group(group_by_query):
            # セマフォを獲得してから処理を進めることで、実際に並列度を制限する
            async with semaphore:
                # ドキュメントをまとめる先のリスト
                body = [None] * 2 * len(group_by_query)
                # 0はじまりで偶数番目の要素にはインデックス名とドキュメントIDを入れる
                body[::2] = group_by_query["product_id"].apply(
                    lambda product_id: {
                        "index": {"_index": self.index_name, "_id": product_id}
                    },
                )
                # 奇数番目の要素にはドキュメントそのものを整形して入れる
                body[1::2] = group_by_query.apply(formatter, axis=1)
                # リストを入力する（bulk）。高速化のためスレッドで非同期に行う
                await asyncio.to_thread(self.open_search.bulk, body=body)

        # 多数のドキュメントを整形し、入力する関数（非同期）
        async def input_documents_async():
            # 非同期なので完了を待つ（asyncio.gather）。このとき進捗を表示する（tqdm）
            await tqdm_asyncio.gather(
                # 非同期でも1つずつは遅いので、紐づくクエリ（ID）ごとにまとめて入力する
                *[
                    input_group(group_by_query)
                    for _, group_by_query in data.groupby("query_id")
                ]
            )

        # 実際に多数のドキュメントを整形し、入力する（非同期）
        asyncio.run(input_documents_async())

        # 入力したすべてのドキュメントを検索可能にする
        self.open_search.indices.refresh(index=self.index_name)

    # 多数のクエリを整形し、入力し、結果を表示したり保存したりするメソッド
    def input_queries(self, data, size, formatter=format_query):
        # 並列度を制限するためのセマフォをインスタンス化する
        semaphore = asyncio.Semaphore(self.concurrency)

        # 単一のクエリを処理する関数（非同期）
        async def input_query(row):
            # セマフォを獲得してから処理を進めることで、実際に並列度を制限する
            async with semaphore:
                # クエリを入力、つまり検索する（search)。高速化のためスレッドで非同期に行う
                search_result = await asyncio.to_thread(
                    self.open_search.search,
                    index=self.index_name,
                    body={
                        # 正確なヒット件数を数える指定
                        "track_total_hits": True,
                        # そのうちレスポンスに含めるドキュメント数
                        "size": size,
                        # クエリそのものを整形して入れる
                        "query": formatter(row),
                        # レスポンスからドキュメントベクトルを除外する指定
                        "_source": {"exclude": "title_vector"},
                    },
                )

            # 例のクエリについては、検索結果をそのまま表示
            if row.query_id == 119300:
                print(json.dumps(search_result, ensure_ascii=False, indent=4))

            # 処理にかかった時間とヒット件数を返す
            return search_result["took"], search_result["hits"]["total"]["value"]

        # 多数のクエリを処理する関数（非同期）
        async def input_queries_async():
            # 非同期なので完了を待つ（asyncio.gather)。このとき進捗を表示する（tqdm)
            results = await tqdm_asyncio.gather(
                # 1行が1つのクエリに対応するので、行を走査する（iterrows)
                *[input_query(row) for _, row in data.iterrows()]
            )

            # ヒット件数とレイテンシを保存する。このまま実行した場合は、本書のサンプルコードの
            # ディレクトリ code 以下、tmp/（インデックス名）-took-hits.parquet に保存する
            pd.DataFrame(results, columns=["took", "hits"]).to_parquet(
                os.path.join(
                    os.path.dirname(__file__),
                    "tmp",
                    f"{self.index_name}-took-hits.parquet",
                ),
                index=False,
                engine="pyarrow",  # 1.0.1: .parquet ファイルを扱う際、engine を明示しました。
            )

        # 実際に多数のクエリを処理する（非同期）
        asyncio.run(input_queries_async())


# このコードを直に実行した場合のみ、以下のコードを実行する
if __name__ == "__main__":
    from ch02_basic_vectorization import (
        get_dimension_number_of,
        read_basic_vectorized_data,
        split_into_query_and_document,
    )

    # ベクトル化したデータセットをメモリに読み込み、クエリとドキュメントに分割する
    query_data, document_data = split_into_query_and_document(
        read_basic_vectorized_data()
    )

    # OpenSearchを扱うためのインスタンスを作成する
    tester = OpenSearchTester("ch03")

    # データからベクトルの次元数を取得し、OpenSearchインデックスを作成する
    tester.create_index(
        get_default_index_options(get_dimension_number_of(query_data))
    )

    # ドキュメントを整形し入力する
    tester.input_documents(document_data)

    # クエリを整形し入力（つまり検索）する
    tester.input_queries(query_data, 10)
