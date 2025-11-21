from typing import List

import numpy as np
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
)



class TfidfLRDetector:
    """
    TF-IDF feature extractor + PyTorch linear classifier, trained on GPU if available.

    Training is controlled by num_epochs / batch_size / lr in fit().
    """

    def __init__(
        self,
        max_features: int = 100000,
        ngram_range=(1, 2),
        device: torch.device | None = None,
    ):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            lowercase=True,
        )
        self.model: nn.Module | None = None
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

    def fit(
        self,
        texts: List[str],
        labels: List[int],
        num_epochs: int = 10,
        batch_size: int = 128,
        lr: float = 1e-2,
    ):
       
        X_sparse = self.vectorizer.fit_transform(texts)
        X = torch.tensor(X_sparse.toarray(), dtype=torch.float32, device=self.device)
        y = torch.tensor(labels, dtype=torch.long, device=self.device)

        
        num_features = X.size(1)
        self.model = nn.Linear(num_features, 2).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model.train()
        for _ in range(num_epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                logits = self.model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

        return self

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """
        Return 2-class probabilities with shape (n_samples, 2):
        [:, 0] -> class 0 (e.g. human), [:, 1] -> class 1 (e.g. LLM).
        """
        if self.model is None:
            raise RuntimeError("TfidfLRDetector has not been fitted yet.")

        X_sparse = self.vectorizer.transform(texts)
        X = torch.tensor(X_sparse.toarray(), dtype=torch.float32, device=self.device)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(X)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        return probs

    def predict(self, texts: List[str], threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(texts)
        return (probs[:, 1] >= threshold).astype(int)





class RNNModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        hidden_dim: int = 256,
        num_classes: int = 2,
        num_layers: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.1,
        pad_token_id: int = 0,
    ):
        super(RNNModel, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_token_id)

        self.rnn = nn.RNN(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.fc = nn.Linear(out_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)  # (B, T, E)

        if attention_mask is None:
            lengths = torch.full(
                (input_ids.size(0),),
                input_ids.size(1),
                dtype=torch.long,
                device=input_ids.device,
            )
        else:
            lengths = attention_mask.sum(dim=1).cpu()

        packed_input = torch.nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )

        packed_out, hidden = self.rnn(packed_input)
        _out, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)

        if self.rnn.bidirectional:
            feat = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            feat = hidden[-1]

        feat = self.dropout(feat)
        logits = self.fc(feat)
        return logits


class LSTMModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        hidden_dim: int = 256,
        num_classes: int = 2,
        num_layers: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.1,
        pad_token_id: int = 0,
    ):
        super(LSTMModel, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_token_id)

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.fc = nn.Linear(out_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)  # (B, T, E)

        if attention_mask is None:
            lengths = torch.full(
                (input_ids.size(0),),
                input_ids.size(1),
                dtype=torch.long,
                device=input_ids.device,
            )
        else:
            lengths = attention_mask.sum(dim=1).cpu()

        packed_input = torch.nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )

        packed_out, (hidden, _cell) = self.lstm(packed_input)
        _out, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)

        if self.lstm.bidirectional:
            feat = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            feat = hidden[-1]

        feat = self.dropout(feat)
        logits = self.fc(feat)
        return logits


class ModelWithEntropy(nn.Module):
    """
    BERT encoder + extra entropy feature(s), followed by a linear classifier.
    """

    def __init__(
        self,
        base_model_name: str = "distilbert-base-uncased",
        entropy_dim: int = 1,
        num_classes: int = 2,
    ):
        super(ModelWithEntropy, self).__init__()
        self.encoder = AutoModel.from_pretrained(base_model_name)
        hidden_size = self.encoder.config.hidden_size
        self.entropy_dim = entropy_dim
        self.fc = nn.Linear(hidden_size + entropy_dim, num_classes)

    def forward(self, input_ids, attention_mask=None, entropy=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = outputs.last_hidden_state.mean(dim=1)  # (B, H)

        if entropy is not None:
            if entropy.dim() == 1:
                entropy = entropy.unsqueeze(1)
            # ensure entropy has correct last dim
            if entropy.size(1) != self.entropy_dim:
                entropy = entropy[:, : self.entropy_dim]
            combined = torch.cat([hidden_state, entropy], dim=1)
        else:
            combined = hidden_state

        logits = self.fc(combined)
        return logits





class GPT2EntropyDetector:
    """
    Use GPT-2 to compute entropy-based features, then feed into a Logistic Regression classifier.
    Output is 2-class probability (human vs LLM).
    """

    def __init__(self, model_name: str = "gpt2", max_length: int = 512):
        self.tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.max_length = max_length
        self.clf = LogisticRegression(max_iter=1000)

    @torch.no_grad()
    def _entropy_for_text(self, text: str) -> float:
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        input_ids = enc.input_ids.to(self.device)
        outputs = self.model(input_ids)
        logits = outputs.logits[:, :-1, :]  # predict next token
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log(probs + 1e-12)
        token_ent = -(probs * log_probs).sum(dim=-1)  # (B, T-1)
        avg_ent = token_ent.mean().item()
        return avg_ent

    def _features(self, texts: List[str]) -> np.ndarray:
        feats = []
        for t in texts:
            ent = self._entropy_for_text(t)
            length = len(self.tokenizer.encode(t, add_special_tokens=False))
            feats.append([ent, float(length)])
        return np.asarray(feats, dtype=np.float32)

    def fit(self, texts: List[str], labels: List[int]):
        X = self._features(texts)
        y = np.asarray(labels, dtype=np.int64)
        self.clf.fit(X, y)
        return self

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        X = self._features(texts)
        return self.clf.predict_proba(X)

    def predict(self, texts: List[str], threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(texts)
        return (probs[:, 1] >= threshold).astype(int)


class GPT2PerplexityDetector:
    """
    GPT-2 based detector using average token-level perplexity (and length)
    as features, without explicit entropy features. Used for ablation
    against GPT2EntropyDetector.
    """

    def __init__(self, model_name: str = "gpt2", max_length: int = 512):
        self.tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.max_length = max_length
        self.clf = LogisticRegression(max_iter=1000)

    @torch.no_grad()
    def _ppl_for_text(self, text: str) -> float:
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        input_ids = enc.input_ids.to(self.device)
        outputs = self.model(input_ids, labels=input_ids)
        loss = outputs.loss  # average NLL per token
        ppl = torch.exp(loss).item()
        return ppl

    def _features(self, texts: List[str]) -> np.ndarray:
        feats = []
        for t in texts:
            ppl = self._ppl_for_text(t)
            length = len(self.tokenizer.encode(t, add_special_tokens=False))
            feats.append([ppl, float(length)])
        return np.asarray(feats, dtype=np.float32)

    def fit(self, texts: List[str], labels: List[int]):
        X = self._features(texts)
        y = np.asarray(labels, dtype=np.int64)
        self.clf.fit(X, y)
        return self

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        X = self._features(texts)
        return self.clf.predict_proba(X)

    def predict(self, texts: List[str], threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(texts)
        return (probs[:, 1] >= threshold).astype(int)




def get_model(
    model_type: str = "distilbert-base-uncased",
    tokenizer=None,
    embed_dim: int = 256,
    hidden_dim: int = 256,
    num_classes: int = 2,
    dropout: float = 0.1,
    entropy_dim: int = 1,
):
    """
    Build model by type.

    Supported:
      - 'tfidf_lr'                -> TfidfLRDetector (traditional ML)
      - 'distilbert-base-uncased' -> AutoModelForSequenceClassification
      - 'bert-base-uncased'       -> AutoModelForSequenceClassification
      - 'roberta-base'            -> AutoModelForSequenceClassification
      - 'rnn'                     -> RNNModel
      - 'lstm'                    -> LSTMModel
      - 'model_with_entropy'      -> ModelWithEntropy (BERT + entropy feature)
      - 'gpt2_entropy'            -> GPT2EntropyDetector
      - 'gpt2_ppl'                -> GPT2PerplexityDetector
    """
    if model_type == "tfidf_lr":
        return TfidfLRDetector()

    if model_type in {"distilbert-base-uncased", "bert-base-uncased", "roberta-base"}:
        return AutoModelForSequenceClassification.from_pretrained(
            model_type,
            num_labels=num_classes,
        )

    if model_type == "rnn":
        if tokenizer is None:
            raise ValueError("Tokenizer is required for RNN models.")
        vocab_size = tokenizer.vocab_size
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        return RNNModel(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
            pad_token_id=pad_token_id,
        )

    if model_type == "lstm":
        if tokenizer is None:
            raise ValueError("Tokenizer is required for LSTM models.")
        vocab_size = tokenizer.vocab_size
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        return LSTMModel(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
            pad_token_id=pad_token_id,
        )

    if model_type == "model_with_entropy":
        return ModelWithEntropy(
            base_model_name="distilbert-base-uncased",
            entropy_dim=entropy_dim,
            num_classes=num_classes,
        )

    if model_type == "gpt2_entropy":
        return GPT2EntropyDetector()

    if model_type == "gpt2_ppl":
        return GPT2PerplexityDetector()

    raise ValueError(
        "Invalid model_type. Supported: "
        "'tfidf_lr', 'distilbert-base-uncased', 'bert-base-uncased', "
        "'roberta-base', 'rnn', 'lstm', 'model_with_entropy', "
        "'gpt2_entropy', 'gpt2_ppl'."
    )
