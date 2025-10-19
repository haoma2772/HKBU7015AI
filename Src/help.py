import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

def compute_entropy(text, gpt2_model, tokenizer):
    # GPT-2 tokenizer
    inputs = tokenizer(text, return_tensors="pt")
    
    # 获取每个 token 的 log probability
    with torch.no_grad():
        outputs = gpt2_model(**inputs, labels=inputs["input_ids"])
    
    # 使用负对数损失来估算熵
    log_probs = -outputs.loss.item() * len(inputs["input_ids"][0])
    avg_entropy = -log_probs / len(inputs["input_ids"][0])  # 计算平均熵
    return avg_entropy




import torch.nn as nn
from transformers import AutoModelForSequenceClassification

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



def collate_fn_with_entropy(batch, tokenizer, gpt2_model):
    texts = [example['text'] for example in batch]
    labels = torch.tensor([example['label'] for example in batch])
    
    # 计算每个文本的熵特征
    entropies = torch.tensor([compute_entropy(text, gpt2_model, tokenizer) for text in texts])
    
    # 使用tokenizer处理文本
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    
    return {
        'input_ids': inputs['input_ids'],
        'attention_mask': inputs['attention_mask'],
        'entropy': entropies,
        'labels': labels
    }



from transformers import Trainer, TrainingArguments

def train_model(model, train_dataset, eval_dataset, tokenizer, gpt2_model, output_dir="./results"):
    # 设置训练参数
    args = TrainingArguments(
        output_dir=output_dir,              # 保存结果的目录
        evaluation_strategy="epoch",        # 每一轮后评估
        learning_rate=2e-5,                 # 学习率
        per_device_train_batch_size=16,     # 每个设备的批量大小
        num_train_epochs=3,                 # 训练周期
        weight_decay=0.01                   # 权重衰减
    )
    
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=lambda batch: collate_fn_with_entropy(batch, tokenizer, gpt2_model)
    )

    # 开始训练
    trainer.train()
