from datasets import load_dataset
import os
from transformers import AutoTokenizer
import argparse

from typing import Tuple
from datasets import load_dataset, Dataset




def get_args():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.abspath(os.path.join(current_dir, ".."))

    parser = argparse.ArgumentParser(description="LLMs output text detector project.")
    # Hello-SimpleAI/HC3 andythetechnerd03/AI-human-text
    parser.add_argument("--dataset_name", type=str, default="Hello-SimpleAI/HC3", help="Name of the dataset to use.")
    parser.add_argument("--data_dir", type=str, default=os.path.join(project_dir, "data"), help="Directory of the dataset.")
    parser.add_argument("--split", type=str, default="wiki_csai", help="Dataset split or HC3 config (e.g., 'all', 'english').")
    parser.add_argument("--max_length", type=int, default=256, help="Maximum sequence length for tokenization.")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate for the optimizer.")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument(
        "--method",
        type=str,
        default="rnn",
        choices=[
            "tfidf_lr",
            "bert",
            "rnn",
            "lstm",
            "model_with_entropy",
            "gpt2_entropy",
            "gpt2_ppl",
        ],
        help="Which model family to train/evaluate.",
    )
    return parser.parse_args()



def load_and_preprocess_data(
    dataset_name: str = "Hello-SimpleAI/HC3",
    split: str = "train",          # 对 top2 有 train/test；对 HC3 我们再处理
    max_length: int = 256,
) -> Tuple[Dataset, AutoTokenizer]:
    """
    Load dataset (supports:
      - top1: Hello-SimpleAI/HC3
      - top2: andythetechnerd03/AI-human-text)
    and tokenize into model inputs with labels.

    返回:
      tokenized_ds: HuggingFace Dataset，包含 input_ids, attention_mask, labels
      tokenizer: AutoTokenizer
    """

    # --------- 1. 载入原始数据 ---------
    requested_split = split

    # 为了兼容，你以后要扩展别的 dataset 也容易
    if dataset_name == "Hello-SimpleAI/HC3":
        # HC3 是 script-based dataset，datasets>=3 会有报错，这里保留你原本的异常处理逻辑
        hc3_common_splits = {"all", "wiki_csai", "finance", "medicine", "dev"}
        try:
            # name=config 是它的子配置（如 open_qa, medicine 等）
            config = requested_split if requested_split in hc3_common_splits else "all"
            ds = load_dataset(dataset_name, name=config)
        except Exception as e:
            msg = str(e)
            if "Dataset scripts are no longer supported" in msg and "HC3.py" in msg:
                raise RuntimeError(
                    "Your environment has datasets>=3 which no longer supports script-based "
                    "datasets like HC3. Install 'datasets==2.19.0' or load local data files."
                ) from e
            raise

        # HC3 返回的是 DatasetDict
        if isinstance(ds, dict):
            # 尝试用用户指定 split；如果没有，就默认用 'train'，再不行就拿第一个
            if requested_split in ds:
                dataset = ds[requested_split]
            elif "train" in ds:
                dataset = ds["train"]
            else:
                first_key = next(iter(ds.keys()))
                dataset = ds[first_key]
        else:
            dataset = ds

        # --------- 2.1 HC3: 构造 text + label ---------
        # 思路：每条 question 生成多条样本：
        #   - question + 每个 human_answer -> label = 0
        #   - question + 每个 chatgpt_answer -> label = 1
        def build_hc3_text_and_label(batch):
            texts = []
            labels = []
            questions = batch.get("question", [])
            human_answers = batch.get("human_answers", [])
            chatgpt_answers = batch.get("chatgpt_answers", [])

            n = len(questions)
            for i in range(n):
                q = questions[i] or ""
                h_list = human_answers[i] if i < len(human_answers) and human_answers[i] is not None else []
                c_list = chatgpt_answers[i] if i < len(chatgpt_answers) and chatgpt_answers[i] is not None else []

                # human -> label 0
                for h in h_list:
                    if h is None:
                        continue
                    texts.append((q + "\n\n" + h).strip())
                    labels.append(0)

                # chatgpt -> label 1
                for c in c_list:
                    if c is None:
                        continue
                    texts.append((q + "\n\n" + c).strip())
                    labels.append(1)

            return {"text": texts, "label": labels}

        dataset = dataset.map(
            build_hc3_text_and_label,
            batched=True,
            remove_columns=dataset.column_names,  # 把原始列全部替换掉
        )

    elif dataset_name == "andythetechnerd03/AI-human-text":
        # --------- top2: Koala-like AI-human text dataset ---------
        # 这个数据集已经有:
        #   - text: 文本
        #   - generated: 0 / 1 (0=human, 1=LLM)
        ds = load_dataset(dataset_name)

        # DatasetDict -> 选 split
        if isinstance(ds, dict):
            if requested_split in ds:
                dataset = ds[requested_split]
            else:
                # 默认用 'train'
                dataset = ds.get("train", next(iter(ds.values())))
        else:
            dataset = ds

        # 统一成 label 列
        if "generated" in dataset.column_names and "label" not in dataset.column_names:
            dataset = dataset.rename_column("generated", "label")

        # 确保有 text / label
        assert "text" in dataset.column_names, f"{dataset_name} must have a 'text' column."
        assert "label" in dataset.column_names, f"{dataset_name} must have a 'label' or 'generated' column."

    else:
        # 你以后扩展别的 dataset 可以在这里加分支
        raise ValueError(
            f"Unsupported dataset_name={dataset_name}. "
            "Currently only support 'Hello-SimpleAI/HC3' and 'andythetechnerd03/AI-human-text'."
        )

    # --------- 3. Tokenizer ---------
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    # --------- 4. Tokenization (加上 labels) ---------
    def preprocess(examples):
        enc = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        # 把 label 一并带进去，方便直接喂给 AutoModelForSequenceClassification
        enc["labels"] = examples["label"]
        return enc

    tokenized_ds = dataset.map(preprocess, batched=True)

    return tokenized_ds, tokenizer
