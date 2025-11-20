# predict.py
import os
import json
import argparse
import torch

from model import TinyCharRNN


def generate(model, idx, max_new_tokens, block_size, temperature=1.0):
    """
    Autoregressive char-level generation.
    idx: [1, T] tensor of token indices
    """
    model.eval()
    device = next(model.parameters()).device

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        probs = torch.softmax(logits, dim=-1)
        next_idx = torch.multinomial(probs, num_samples=1)  # [1, 1]
        idx = torch.cat([idx, next_idx], dim=1)
    return idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="artifacts/tiny_char_rnn.pt")
    parser.add_argument("--meta_path", type=str, default="artifacts/meta.json")
    parser.add_argument("--prompt", type=str, default="Jenkins text demo:")
    parser.add_argument("--max_new_tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
    if not os.path.exists(args.meta_path):
        raise FileNotFoundError(f"Meta file not found: {args.meta_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load vocab + config
    with open(args.meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    chars = meta["chars"]
    stoi = meta["stoi"]
    itos = {int(k): v for k, v in meta["itos"].items()} if isinstance(next(iter(meta["itos"].keys())), str) else meta["itos"]
    block_size = meta["block_size"]
    vocab_size = len(chars)

    def encode(s: str):
        return [stoi.get(ch, 0) for ch in s]  # unknown chars → 0

    def decode(indices):
        return "".join(itos[int(i)] for i in indices)

    # Recreate model + load weights
    model = TinyCharRNN(vocab_size=vocab_size, embed_dim=64, hidden_dim=128).to(device)
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state)

    # Prepare prompt
    prompt_indices = torch.tensor([encode(args.prompt)], dtype=torch.long, device=device)

    # Generate
    with torch.no_grad():
        out_idx = generate(
            model,
            prompt_indices,
            max_new_tokens=args.max_new_tokens,
            block_size=block_size,
            temperature=args.temperature,
        )

    generated_text = decode(out_idx[0].tolist())
    print("\n=== Generated Text ===\n")
    print(generated_text)
    print("\n======================\n")


if __name__ == "__main__":
    main()
