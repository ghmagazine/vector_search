#!/usr/bin/env python

from sentence_transformers import (
    models,
    SentenceTransformer,
    SentenceTransformerTrainingArguments,
)

import os


# LINEのDistilBERTでSentence-BERTを組む関数
def get_sentence_transformer():
    # LINEのDistilBERTをダウンロードし、メモリに読み込む
    transformer_module = models.Transformer(
        "line-corporation/line-distilbert-base-japanese",
        tokenizer_args={
            # 独自のPythonコードを実行する許可を与える。特定のリビジョンに絞る
            "trust_remote_code": True,
            "revision": "c32d05753d0d4be0f60482291a9452ec7e6f7d68",  # 1.1.0: LINEのDistilBERTを更新しました。
        },
    )

    return SentenceTransformer(
        # ベクトル化モデルの処理の流れを書く
        modules=[
            # LINEのDistilBERT
            transformer_module,
            # 平均プーリング
            models.Pooling(transformer_module.get_word_embedding_dimension()),
            # ベクトル正規化
            models.Normalize(),
        ]
    )


# ファインチューニング後のベクトル化モデルの保存先
# このまま実行した場合は、本書のサンプルコードのディレクトリ code 以下、
# tmp/tuned-vectorization-model に保存する
TUNED_VECTORIZATION_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "tmp", "tuned-vectorization-model"
)


# ファインチューニング後のベクトル化モデルを保存する関数
def save_tuned_vectorization_model(model):
    model.save(TUNED_VECTORIZATION_MODEL_PATH)


# ファインチューニング後のベクトル化モデルを読み込む関数
def get_tuned_vectorization_model():
    # ディレクトリが存在すれば読み込む
    if os.path.isdir(TUNED_VECTORIZATION_MODEL_PATH):
        return SentenceTransformer(
            TUNED_VECTORIZATION_MODEL_PATH, trust_remote_code=True
        )

    # 存在しなければ例外をあげる
    raise ValueError(
        "ファインチューニング後のベクトル化モデルがありません（第5章を参照）"
    )


# 本書デフォルトのファインチューニングのパラメータ
DEFAULT_ARGS = SentenceTransformerTrainingArguments(
    # わかりやすさのため、チェックポイントを無効化する
    # したがってoutput_dirは不要だが、単に必須のため指定した
    save_strategy="no",
    output_dir=".",
    # ハイパーパラメータ
    num_train_epochs=1.0,
    per_device_train_batch_size=16,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    # 高速化のための設定でデフォルトで有効だが、省メモリ化のために無効にする
    dataloader_pin_memory=False,
)


# このコードを直に実行した場合のみ、以下のコードを実行する
if __name__ == "__main__":
    from argparse import ArgumentParser
    from ch01_data_preparation import read_jp_data
    from ch04_evaluate_search_results_0_ndcg import esci_label_to_ndcg_gain
    from datasets import Dataset
    from sentence_transformers import losses, SentenceTransformerTrainer

    # コマンドライン引数から、訓練データのサンプル率を読み込む
    argument_parser = ArgumentParser()
    argument_parser.add_argument(
        "--sample-rate", default=0.01, type=float, choices=[0.01, 1.0]
    )
    args = argument_parser.parse_args()

    # 訓練データを読み込む
    jp_train_data = read_jp_data(split="train", sample_rate=args.sample_rate)

    # ファインチューニング前のベクトル化モデルを組む
    model = get_sentence_transformer()

    # ファインチューニングする
    SentenceTransformerTrainer(
        # クエリ、ドキュメント（ここでは製品タイトル）、正解のスコア（nDCGの利得）を与える
        train_dataset=Dataset.from_dict(
            {
                "query": jp_train_data["query"],
                "product_title": jp_train_data["product_title"],
                "score": jp_train_data["esci_label"].apply(esci_label_to_ndcg_gain),
            }
        ),
        # もちろん対象のモデルも与える
        model=model,
        # パラメータも与える
        args=DEFAULT_ARGS,
        # 損失関数も与える。ここではコサイン類似度と正解のスコアの大小関係が一致するように指定
        loss=losses.CoSENTLoss(model),
    ).train()

    # ファインチューニング後のベクトル化モデルを保存する
    save_tuned_vectorization_model(model)
