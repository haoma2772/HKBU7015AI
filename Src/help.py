from typing import Tuple, List

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from model import (
    TfidfLRDetector,
    GPT2EntropyDetector,
    GPT2PerplexityDetector,
    RNNModel,
    LSTMModel,
)


def split_tokenized_dataset(
    dataset: Dataset,
    test_size: float = 0.2,
    seed: int = 42,
) -> Tuple[Dataset, Dataset]:
    """
    Split a HuggingFace tokenized Dataset into train / test.
    """
    split = dataset.train_test_split(test_size=test_size, seed=seed)
    return split["train"], split["test"]


def _build_dataloader(ds: Dataset, batch_size: int, shuffle: bool) -> DataLoader:
    """
    Build a DataLoader from a HuggingFace Dataset, avoiding NumPy 2.0 copy=False issues
    by converting directly to torch.Tensor.
    """
    input_ids = torch.tensor(ds["input_ids"], dtype=torch.long)
    attention_mask = torch.tensor(ds["attention_mask"], dtype=torch.long)
    labels = torch.tensor(ds["labels"], dtype=torch.long)
    tensor_ds = TensorDataset(input_ids, attention_mask, labels)
    return DataLoader(tensor_ds, batch_size=batch_size, shuffle=shuffle)


def train_torch_model(
    model: torch.nn.Module,
    train_ds: Dataset,
    batch_size: int,
    lr: float,
    num_epochs: int,
    device: torch.device,
    model_kind: str,
) -> Tuple[torch.nn.Module, dict]:
    """
    Generic training loop for torch models (BERT, RNN, LSTM, ModelWithEntropy).

    model_kind:
        - "bert"              -> AutoModelForSequenceClassification (has .loss)
        - "rnn" / "lstm"      -> RNNModel / LSTMModel (return logits, we compute CE)
        - "model_with_entropy"-> ModelWithEntropy (need entropy feature; here we use length as a proxy)
    """
    model.to(device)
    model.train()
    

    train_loader = _build_dataloader(train_ds, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    history = []
    last_metrics = {"loss": None, "accuracy": None}

    for epoch in range(num_epochs):
        total_loss = 0.0
        n_steps = 0
        correct = 0
        total = 0

        for batch in tqdm(
            train_loader,
            desc=f"Train[{model_kind}] epoch {epoch + 1}/{num_epochs}",
            leave=False,
        ):
            optimizer.zero_grad()

            input_ids, attention_mask, labels = [b.to(device) for b in batch]

            if model_kind == "bert":
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss
                logits = outputs.logits
            elif model_kind in {"rnn", "lstm"}:
                logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                loss = criterion(logits, labels)
            elif model_kind == "model_with_entropy":
                # Simple entropy proxy: normalized length of the sequence
                seq_len = attention_mask.sum(dim=1).float() if attention_mask is not None else (
                    torch.full((input_ids.size(0),), input_ids.size(1), device=device).float()
                )
                entropy_feature = (seq_len / seq_len.max()).detach()
                logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    entropy=entropy_feature,
                )
                loss = criterion(logits, labels)
            else:
                raise ValueError(f"Unsupported model_kind: {model_kind}")

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_steps += 1
            # accuracy
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / max(1, n_steps)
        acc = correct / max(1, total)
        last_metrics = {"loss": avg_loss, "accuracy": acc}
        history.append(last_metrics)
        print(f"[{model_kind}] Epoch {epoch + 1}/{num_epochs} - train loss: {avg_loss:.4f}, acc: {acc:.4f}")

    return model, {"last_epoch": last_metrics, "history": history}


@torch.no_grad()
def evaluate_torch_model(
    model: torch.nn.Module,
    eval_ds: Dataset,
    batch_size: int,
    device: torch.device,
    model_kind: str,
) -> dict:
    """
    Evaluate a torch model on a tokenized Dataset, returning accuracy.
    """
    model.to(device)
    model.eval()

    eval_loader = _build_dataloader(eval_ds, batch_size=batch_size, shuffle=False)

    all_preds: List[int] = []
    all_labels: List[int] = []
    total_loss = 0.0
    n_steps = 0
    criterion = torch.nn.CrossEntropyLoss()

    for batch in tqdm(
        eval_loader,
        desc=f"Eval[{model_kind}]",
        leave=False,
    ):
        input_ids, attention_mask, labels = [b.to(device) for b in batch]

        if model_kind == "bert":
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            logits = outputs.logits
        elif model_kind in {"rnn", "lstm"}:
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        elif model_kind == "model_with_entropy":
            seq_len = attention_mask.sum(dim=1).float() if attention_mask is not None else (
                torch.full((input_ids.size(0),), input_ids.size(1), device=device).float()
            )
            entropy_feature = (seq_len / seq_len.max()).detach()
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                entropy=entropy_feature,
            )
        else:
            raise ValueError(f"Unsupported model_kind: {model_kind}")

        loss = criterion(logits, labels)
        total_loss += loss.item()
        n_steps += 1

        probs = torch.softmax(logits, dim=-1)
        preds = probs.argmax(dim=-1)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    avg_loss = total_loss / max(1, n_steps)
    print(f"[{model_kind}] Eval loss: {avg_loss:.4f}, accuracy: {acc:.4f}")
    return {"loss": avg_loss, "accuracy": acc}


def train_eval_tfidf(
    texts: List[str],
    labels: List[int],
    test_size: float = 0.2,
    seed: int = 42,
    num_epochs: int = 10,
    batch_size: int = 128,
    lr: float = 1e-2,
    device: torch.device = torch.device("cpu"),
) -> Tuple[TfidfLRDetector, dict]:
    """
    Train and evaluate TF-IDF + Logistic Regression baseline.
    Returns accuracy on the test split.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=test_size, random_state=seed, stratify=labels
    )

    model = TfidfLRDetector(device=device)
    model.fit(X_train, y_train, num_epochs=num_epochs, batch_size=batch_size, lr=lr)
    probs = model.predict_proba(X_test)
    preds = np.argmax(probs, axis=1)
    acc = accuracy_score(y_test, preds)
    print(f"[tfidf_lr] Eval accuracy: {acc:.4f}")
    return model, {"accuracy": acc}


def train_eval_gpt2_entropy(
    texts: List[str],
    labels: List[int],
    test_size: float = 0.2,
    seed: int = 42,
) -> Tuple[GPT2EntropyDetector, dict]:
    """
    Train and evaluate GPT-2 entropy-based detector.
    Returns accuracy on the test split.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=test_size, random_state=seed, stratify=labels
    )

    det = GPT2EntropyDetector()
    det.fit(X_train, y_train)
    probs = det.predict_proba(X_test)
    preds = np.argmax(probs, axis=1)
    acc = accuracy_score(y_test, preds)
    print(f"[gpt2_entropy] Eval accuracy: {acc:.4f}")
    return det, {"accuracy": acc}


def train_eval_gpt2_ppl(
    texts: List[str],
    labels: List[int],
    test_size: float = 0.2,
    seed: int = 42,
) -> Tuple[GPT2PerplexityDetector, dict]:
    """
    Train and evaluate GPT-2 perplexity-based detector (no entropy features).
    Returns accuracy on the test split.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=test_size, random_state=seed, stratify=labels
    )

    det = GPT2PerplexityDetector()
    det.fit(X_train, y_train)
    probs = det.predict_proba(X_test)
    preds = np.argmax(probs, axis=1)
    acc = accuracy_score(y_test, preds)
    print(f"[gpt2_ppl] Eval accuracy: {acc:.4f}")
    return det, {"accuracy": acc}
