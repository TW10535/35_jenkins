# train.py
import os
import json
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim

from model import TinyCharRNN


def ensure_corpus(path: str) -> str:
    """
    Ensure there is a small text corpus at `path`.
    If missing, create a tiny default one.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if not os.path.exists(path):
        default_text = """
Jenkins ML text demo.

This is a very small text corpus for training a tiny character-level language model.
The goal is not to get good language quality, but to make sure the CI pipeline runs
fast and successfully. You can replace this text with your own domain data: logs,
meeting notes, product descriptions, bug reports or documentation.

Once the model is trained, we will generate a short paragraph of text given a prompt.
"""
        with open(path, "w", encoding="utf-8") as f:
            f.write(default_text.strip())

    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def build_vocab(text: str):
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return chars, stoi, itos


def encode(text: str, stoi: dict) -> torch.Tensor:
    return torch.tensor([stoi[c] for c in text], dtype=torch.long)


def get_batch(data: torch.Tensor, batch_size: int, block_size: int, device: str):
    """
    Sample random subsequences of length block_size for training.
    """
    n = data.size(0)
    # Make sure we have enough tokens
    max_start = max(1, n - block_size - 1)
    ix = torch.randint(max_start, (batch_size,), device=device)
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_path", type=str, default="data/corpus.txt")
    parser.add_argument("--block_size", type=int, default=64, help="sequence length")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_steps", type=int, default=200, help="training steps (keep small for CI)")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output_dir", type=str, default="artifacts")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1) Load / create corpus
    text = ensure_corpus(args.corpus_path)
    print(f"Loaded corpus from {args.corpus_path}, length={len(text)} characters")

    # 2) Build vocab + encode
    chars, stoi, itos = build_vocab(text)
    vocab_size = len(chars)
    print(f"Vocab size: {vocab_size} characters")

    data = encode(text, stoi).to(device)

    # 3) Create model
    model = TinyCharRNN(vocab_size=vocab_size, embed_dim=64, hidden_dim=128).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    # 4) Training loop (tiny, for CI)
    model.train()
    for step in range(1, args.max_steps + 1):
        x, y = get_batch(data, args.batch_size, args.block_size, device)
        logits, _ = model(x)  # [B, T, V]

        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % 50 == 0 or step == 1 or step == args.max_steps:
            print(f"Step {step}/{args.max_steps} - loss: {loss.item():.4f}")

    # 5) Save model weights
    model_path = os.path.join(args.output_dir, "tiny_char_rnn.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Saved model to {model_path}")

    # 6) Save vocab + config
    meta = {
        "chars": chars,
        "stoi": stoi,
        "itos": itos,
        "block_size": args.block_size,
    }
    meta_path = os.path.join(args.output_dir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f)
    print(f"Saved vocab/meta to {meta_path}")


if __name__ == "__main__":
    main()
