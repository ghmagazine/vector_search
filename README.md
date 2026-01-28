# 「ベクトル検索実践入門」サポートページ

## リポジトリの構成

- `code`: サンプルコード本体
    - `esci-data`: 本文で題材とするデータセットへの参照（Gitサブモジュール）
    - `output`: 章別の出力例
    - `script`: 章別の細かいサンプルコード（コマンドラインなど）
    - `tmp`: 一時的なデータやモデルの保存先
    - `Dockerfile`: サンプルコードの実行環境の設定
    - `*.py`: 主要なサンプルコード
    - `docker-compose.yaml`: 実行環境やOpenSearchを立ち上げるための設定
    - `docker-compose.yaml.gpu.patch`: その設定にあて、実行環境からGPUを利用するためのパッチ
    - `requirements.txt`: サンプルコードで利用するライブラリの一覧
- `.gitmodules`: Gitサブモジュールの設定
- `README.md`: このファイル

## リリースノート

### 1.0.2

- 依存する `transformers` のバージョンを5未満に固定しました。
    - ほかの依存コンポーネントが動作しなくなるのを確認したためです。


### 1.0.1

- `.parquet` ファイルを扱う際、`engine` を明示しました。これにより、`pyarrow` がインストールされていない場合に  `code/ch01_data_preparation.py` の上げる例外が分かりやすくなると期待されます。
    - 1.0.0の例：
        ```
        File "/code/ch01_data_preparation.py", line 80, in assert_counts
            assert (query_count, row_count) == (10407, 297883)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        AssertionError
        ```
    - 1.0.1の例：
        ```
        ImportError: Missing optional dependency 'pyarrow'. pyarrow is required for parquet support. Use pip or conda to install pyarrow.
        ```
- 関連して、不要と思われる `fastparquet` 依存を削除しました。
- 環境によってはインストールされない挙動を確認したため、`pillow` 依存を明示しました。


### 1.0.0

- 出版時の状態
