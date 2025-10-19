from transformers import GPT2LMHeadModel, GPT2TokenizerFast, AutoModelForSequenceClassification
from utils import get_args, load_and_preprocess_data


def test():
    # 获取命令行参数
    args = get_args()

    # 加载并预处理数据
    tokenized_ds, tokenizer = load_and_preprocess_data(args.dataset_name, args.split, args.max_length)
    
    # 打印出一些数据看看
    print(f"Loaded dataset: {args.dataset_name}")
    print(f"Number of examples in train set: {len(tokenized_ds)}")
    
    # 获取示例文本并打印
    example_text = tokenized_ds[0]  # 获取第一条样本数据
    print(f"Example tokenized text: {example_text['input_ids']}")  # 打印 tokenized 的 ID 序列
    print(f"Original text: {tokenizer.decode(example_text['input_ids'])}")  # 使用 tokenizer 解码为原始文本


def main():
    # 获取命令行参数
    args = get_args()

    # 加载并预处理数据
    tokenized_ds, tokenizer = load_and_preprocess_data(args.dataset_name, args.split, args.max_length)
    
    # 打印出一些数据看看
    print(f"Loaded dataset: {args.dataset_name}")
    print(f"Number of examples in train set: {len(tokenized_ds)}")
    
    # 获取示例文本并打印
    example_text = tokenized_ds[0]  # 获取第一条样本数据
    print(f"Example tokenized text: {example_text['input_ids']}")  # 打印 tokenized 的 ID 序列
    print(f"Original text: {tokenizer.decode(example_text['input_ids'])}")  # 使用 tokenizer 解码为原始文本

if __name__ == "__main__":
    main()


