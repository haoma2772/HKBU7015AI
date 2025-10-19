from datasets import load_dataset
import os
from transformers import AutoTokenizer
import argparse


def get_args():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.abspath(os.path.join(current_dir, ".."))

    parser = argparse.ArgumentParser(description="LLMs output text detector project.")
    parser.add_argument("--dataset_name", type=str, default="Hello-SimpleAI/HC3", help="Name of the dataset to use.")
    parser.add_argument("--data_dir", type=str, default=os.path.join(project_dir, "data"), help="Directory of the dataset.")
    parser.add_argument("--split", type=str, default="wiki_csai", help="Dataset split or HC3 config (e.g., 'all', 'english').")
    parser.add_argument("--max_length", type=int, default=256, help="Maximum sequence length for tokenization.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate for the optimizer.")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs.")
    return parser.parse_args()


def load_and_preprocess_data(dataset_name: str = "Hello-SimpleAI/HC3", split: str = "all", max_length: int = 256):
    """
    Load dataset (supports HC3) and tokenize into model inputs.

    Notes for HC3 and datasets>=3.0:
    - datasets 3.x removed script-based datasets. HC3 on the Hub is script-based (HC3.py).
      If your env has datasets>=3, loading HC3 will raise:
      "Dataset scripts are no longer supported, but found HC3.py".
      In that case, either:
        * install: datasets==2.19.0, or
        * load from local files via data_files (not covered here).
    - HC3 does not have a "text" column; we construct one from available fields.
    - The function returns a single split Dataset (not a DatasetDict) so callers can index with ints.
    """

    requested_split = split
    hc3_common_splits = {"all", "wiki_csai", "finance", "medicine", "dev"}

    try:
        if dataset_name == "Hello-SimpleAI/HC3":
         
            config = requested_split if requested_split not in hc3_common_splits else "all"
            ds = load_dataset(dataset_name, name=config)
        else:
            ds = load_dataset(dataset_name, split=requested_split)
    except Exception as e:
        msg = str(e)
        if "Dataset scripts are no longer supported" in msg and "HC3.py" in msg:
            raise RuntimeError(
                "Your environment has datasets>=3 which no longer supports script-based datasets like HC3. "
                "Fix by installing 'datasets==2.19.0' or switch to loading local data files."
            ) from e
        raise

    # If DatasetDict, pick a concrete split
    if isinstance(ds, dict):
        if requested_split in ds:
            dataset = ds[requested_split]
        elif "train" in ds:
            dataset = ds["train"]
        else:
            first_key = next(iter(ds.keys()))
            dataset = ds[first_key]
    else:
        dataset = ds

    # Ensure a 'text' column exists
    def ensure_text(batch):
        if "text" in batch:
            return {"text": batch["text"]}
        n = len(next(iter(batch.values()))) if batch else 0

        def first_or_empty(x):
            return x[0] if isinstance(x, list) and len(x) > 0 else ""

        texts = []
        for i in range(n):
            q = batch.get("question", [""] * n)[i]
            ans = ""
            if "human_answers" in batch:
                ans = first_or_empty(batch["human_answers"][i])
            if not ans and "chatgpt_answers" in batch:
                ans = first_or_empty(batch["chatgpt_answers"][i])
            texts.append((q + "\n\n" + ans).strip())
        return {"text": texts}

    dataset = dataset.map(ensure_text, batched=True)

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def preprocess(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=max_length)

    tokenized_ds = dataset.map(preprocess, batched=True)
    return tokenized_ds, tokenizer

