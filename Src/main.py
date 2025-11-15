import os

import torch

from utils import get_args, load_and_preprocess_data
from help import (
    split_tokenized_dataset,
    train_torch_model,
    evaluate_torch_model,
    train_eval_tfidf,
    train_eval_gpt2_entropy,
    train_eval_gpt2_ppl,
)
from model import get_model


def main():
    args = get_args()
    tokenized_ds, tokenizer = load_and_preprocess_data(
        args.dataset_name, args.split, args.max_length
    )

    print(f"Loaded dataset: {args.dataset_name}")
    print(f"Number of examples: {len(tokenized_ds)}")

    texts = tokenized_ds["text"]
    labels = tokenized_ds["label"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.method == "tfidf_lr":
        model, eval_metrics = train_eval_tfidf(
            texts,
            labels,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            lr=args.learning_rate,
            device=device,
        )
        os.makedirs("models", exist_ok=True)
        tfidf_path = os.path.join("models", "tfidf_lr_model.joblib")
        try:
            import joblib

            joblib.dump(model, tfidf_path)
            print(f"[tfidf_lr] Saved model to {tfidf_path}")
        except Exception as e:
            print(f"[tfidf_lr] Failed to save model: {e}")
        print(f"[tfidf_lr] Final metrics: {eval_metrics}")
        return

    if args.method == "gpt2_entropy":
        model, eval_metrics = train_eval_gpt2_entropy(texts, labels)
        os.makedirs("models", exist_ok=True)
        gpt2_path = os.path.join("models", "gpt2_entropy_model.joblib")
        try:
            import joblib

            joblib.dump(model, gpt2_path)
            print(f"[gpt2_entropy] Saved model to {gpt2_path}")
        except Exception as e:
            print(f"[gpt2_entropy] Failed to save model: {e}")
        print(f"[gpt2_entropy] Final metrics: {eval_metrics}")
        return

    if args.method == "gpt2_ppl":
        model, eval_metrics = train_eval_gpt2_ppl(texts, labels)
        os.makedirs("models", exist_ok=True)
        gpt2_ppl_path = os.path.join("models", "gpt2_ppl_model.joblib")
        try:
            import joblib

            joblib.dump(model, gpt2_ppl_path)
            print(f"[gpt2_ppl] Saved model to {gpt2_ppl_path}")
        except Exception as e:
            print(f"[gpt2_ppl] Failed to save model: {e}")
        print(f"[gpt2_ppl] Final metrics: {eval_metrics}")
        return

    # Torch-based models: BERT / RNN / LSTM / ModelWithEntropy
    train_ds, test_ds = split_tokenized_dataset(tokenized_ds, test_size=0.2, seed=42)

    if args.method == "bert":
        model = get_model(model_type="distilbert-base-uncased")
        model, train_metrics = train_torch_model(
            model,
            train_ds,
            batch_size=args.batch_size,
            lr=args.learning_rate,
            num_epochs=args.num_epochs,
            device=device,
            model_kind="bert",
        )
        eval_metrics = evaluate_torch_model(
            model,
            test_ds,
            batch_size=args.batch_size,
            device=device,
            model_kind="bert",
        )
        os.makedirs("models", exist_ok=True)
        save_path = os.path.join("models", "bert_model.pt")
        torch.save(model.state_dict(), save_path)
        print(f"[bert] Saved model to {save_path}")
        print(f"[bert] Train metrics: {train_metrics}")
        print(f"[bert] Eval metrics: {eval_metrics}")

    elif args.method == "rnn":
        model = get_model(
            model_type="rnn",
            tokenizer=tokenizer,
        )
        model, train_metrics = train_torch_model(
            model,
            train_ds,
            batch_size=args.batch_size,
            lr=args.learning_rate,
            num_epochs=args.num_epochs,
            device=device,
            model_kind="rnn",
        )
        eval_metrics = evaluate_torch_model(
            model,
            test_ds,
            batch_size=args.batch_size,
            device=device,
            model_kind="rnn",
        )
        os.makedirs("models", exist_ok=True)
        save_path = os.path.join("models", "rnn_model.pt")
        torch.save(model.state_dict(), save_path)
        print(f"[rnn] Saved model to {save_path}")
        print(f"[rnn] Train metrics: {train_metrics}")
        print(f"[rnn] Eval metrics: {eval_metrics}")

    elif args.method == "lstm":
        model = get_model(
            model_type="lstm",
            tokenizer=tokenizer,
        )
        model, train_metrics = train_torch_model(
            model,
            train_ds,
            batch_size=args.batch_size,
            lr=args.learning_rate,
            num_epochs=args.num_epochs,
            device=device,
            model_kind="lstm",
        )
        eval_metrics = evaluate_torch_model(
            model,
            test_ds,
            batch_size=args.batch_size,
            device=device,
            model_kind="lstm",
        )
        os.makedirs("models", exist_ok=True)
        save_path = os.path.join("models", "lstm_model.pt")
        torch.save(model.state_dict(), save_path)
        print(f"[lstm] Saved model to {save_path}")
        print(f"[lstm] Train metrics: {train_metrics}")
        print(f"[lstm] Eval metrics: {eval_metrics}")

    elif args.method == "model_with_entropy":
        model = get_model(
            model_type="model_with_entropy",
            entropy_dim=1,
            num_classes=2,
        )
        model, train_metrics = train_torch_model(
            model,
            train_ds,
            batch_size=args.batch_size,
            lr=args.learning_rate,
            num_epochs=args.num_epochs,
            device=device,
            model_kind="model_with_entropy",
        )
        eval_metrics = evaluate_torch_model(
            model,
            test_ds,
            batch_size=args.batch_size,
            device=device,
            model_kind="model_with_entropy",
        )
        os.makedirs("models", exist_ok=True)
        save_path = os.path.join("models", "model_with_entropy.pt")
        torch.save(model.state_dict(), save_path)
        print(f"[model_with_entropy] Saved model to {save_path}")
        print(f"[model_with_entropy] Train metrics: {train_metrics}")
        print(f"[model_with_entropy] Eval metrics: {eval_metrics}")
    else:
        raise ValueError(f"Unsupported method: {args.method}")


if __name__ == "__main__":
    main()
