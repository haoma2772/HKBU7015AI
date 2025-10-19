import torch.nn as nn
from transformers import AutoModelForSequenceClassification
import torch


class ModelWithEntropy(nn.Module):
    def __init__(self, model, entropy_dim=1):
        super(ModelWithEntropy, self).__init__()
        self.bert = model
        self.fc = nn.Linear(model.config.hidden_size + entropy_dim, 2)  # 拼接特征后进行分类
    
    def forward(self, input_ids, attention_mask, entropy):
        # 获取 BERT 的 embedding
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        hidden_state = outputs.last_hidden_state.mean(dim=1)  # 获取 [CLS] token 的向量（可以选择其他聚合方式）
        
        # 将 BERT 输出与熵特征拼接
        combined = torch.cat((hidden_state, entropy.unsqueeze(1)), dim=1)
        
        # 分类头
        logits = self.fc(combined)
        return logits


# ------------------------------
# Baseline 1: TF-IDF + Logistic Regression
# ------------------------------
from typing import List, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


class TfidfLRDetector:
    def __init__(self, max_features: int = 100000, ngram_range=(1, 2), C: float = 1.0):
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, lowercase=True)),
            ("clf", LogisticRegression(max_iter=1000, C=C, n_jobs=None))
        ])

    def fit(self, texts: List[str], labels: List[int]):
        self.pipeline.fit(texts, labels)
        return self

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        if hasattr(self.pipeline.named_steps["clf"], "predict_proba"):
            return self.pipeline.predict_proba(texts)[:, 1]
        # Fallback to decision_function if probas unavailable
        clf = self.pipeline.named_steps["clf"]
        scores = clf.decision_function(self.pipeline.named_steps["tfidf"].transform(texts))
        # Min-max normalize to [0,1]
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
        return scores

    def predict(self, texts: List[str], threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(texts)
        return (probs >= threshold).astype(int)


# ------------------------------
# Baseline 2: GPT-2 Perplexity Detector
# ------------------------------
from transformers import GPT2LMHeadModel, GPT2TokenizerFast


class GPT2PerplexityDetector:
    def __init__(self, model_name: str = "gpt2", stride: int = 512, max_length: int = 1024):
        self.model_name = model_name
        self.stride = stride
        self.max_length = max_length
        self.tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    @torch.no_grad()
    def perplexities(self, texts: List[str]) -> np.ndarray:
        ppl_list = []
        for text in texts:
            enc = self.tokenizer(text, return_tensors="pt")
            input_ids = enc.input_ids.to(self.device)
            nlls = []
            for i in range(0, input_ids.size(1), self.stride):
                begin = i
                end = min(i + self.max_length, input_ids.size(1))
                trg_len = end - i
                input_ids_slice = input_ids[:, begin:end]
                target_ids = input_ids_slice.clone()
                target_ids[:, :-trg_len] = -100
                outputs = self.model(input_ids_slice, labels=target_ids)
                nll = outputs.loss * trg_len
                nlls.append(nll)
                if end == input_ids.size(1):
                    break
            ppl = torch.exp(torch.stack(nlls).sum() / input_ids.size(1))
            ppl_list.append(ppl.item())
        return np.array(ppl_list)

    def predict(self, texts: List[str], threshold: float, ai_when_lower_ppl: bool = True) -> np.ndarray:
        ppls = self.perplexities(texts)
        if ai_when_lower_ppl:
            return (ppls <= threshold).astype(int)
        return (ppls >= threshold).astype(int)


# ------------------------------
# Baseline 3: DistilBERT Classifier (fine-tuning)
# ------------------------------
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


class _TextLabelDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer: AutoTokenizer, max_length: int = 256):
        self.texts = texts
        self.labels = labels
        self.tok = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        t = self.texts[idx]
        enc = self.tok(t, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


class TransformerClassifierDetector:
    def __init__(self, model_name: str = "distilbert-base-uncased", max_length: int = 256):
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def fit(self, texts: List[str], labels: List[int], epochs: int = 3, batch_size: int = 16, lr: float = 2e-5):
        ds = _TextLabelDataset(texts, labels, self.tokenizer, self.max_length)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

        optim = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        self.model.train()
        for _ in range(epochs):
            for batch in dl:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                out = self.model(**batch)
                loss = out.loss
                optim.zero_grad()
                loss.backward()
                optim.step()
        return self

    @torch.no_grad()
    def predict_proba(self, texts: List[str]) -> np.ndarray:
        self.model.eval()
        ds = _TextLabelDataset(texts, [0] * len(texts), self.tokenizer, self.max_length)
        dl = DataLoader(ds, batch_size=32)
        probs = []
        for batch in dl:
            batch = {k: v.to(self.device) for k, v in batch.items() if k != "labels"}
            out = self.model(**batch)
            p = torch.softmax(out.logits, dim=-1)[:, 1]
            probs.append(p.detach().cpu())
        return torch.cat(probs, dim=0).numpy()

    def predict(self, texts: List[str], threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(texts)
        return (probs >= threshold).astype(int)
