# MiniLM — Minimal Neural Language Model from Scratch

A tiny, from-scratch language model built on **PyTorch**. A minimal single-head
Transformer block (self-attention + feed-forward head) — small enough to read end
to end, real enough to grow. Bilingual (Hebrew + English). The model *learns*
behavior instead of relying on hardcoded algorithms.

> **Note:** This project began as a pure-NumPy implementation with a hand-written
> backprop. It has since been migrated to PyTorch: `autograd` derives the
> gradients, a standard optimizer updates the weights, and GPU (CUDA/Intel XPU)
> is supported — so the model can scale up in the future without rewriting the core.

## Quick Start

### Prerequisites
- Python 3.9+
- PyTorch (`torch>=2.2`)
- Tkinter (usually included with Python)

### Install
```bash
git clone <repo>
cd ai-model

# CPU build (works everywhere):
pip install torch

# — or — Intel GPU (XPU) build (what this project was developed against):
pip install torch --index-url https://download.pytorch.org/whl/xpu

# — or — NVIDIA CUDA build:
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

Training runs on **CPU by default** — deliberately: the model is tiny, so GPU
per-op overhead (and Intel XPU's ~47s one-time kernel warmup) costs more than it
saves at this size. When you scale the model up, opt into a GPU with `DEVICE=xpu`
or `DEVICE=cuda`.

### Run
```bash
# Full app: train (if needed) + open GUI
python run.py

# CLI mode (single query, no GUI)
python ask.py "What is 5 times 3?"

# Train only
python train.py
```

## What It Does

The model:
- **Learns** to understand Hebrew and English user queries
- **Decides** when to use math, code generation, or knowledge lookup
- **Delegates** computation to deterministic tools (always exact)
- **Replies** in the language you used, with fallback to "I don't know"

### Examples

```
You: שלום, מה שלומך?
AI: שלום! איך אפשר לעזור?

You: What is 7 plus 3?
AI: <calc> +
[Model emits signal; tool computes 7 + 3 = 10]

You: Who wrote Python?
AI: Guido van Rossum created Python in 1989...
[Model triggers knowledge lookup; wiki search finds the answer]
```

## How It Works

### Architecture

A minimal single-head Transformer block, built from `torch.nn` layers:

```
Input (word IDs)
    ↓
Token Embedding + Positional Embedding      (each word → dense vector + position)
    ↓
Self-Attention:  Q, K, V  →  softmax(QKᵀ/√D)·V  →  output projection
    ↓
Residual: x + attention                     (word representations "talk" to each other)
    ↓
Take last position only                     (it has seen the whole context)
    ↓
Linear → ReLU → Linear                       (feed-forward head)
    ↓
Logits over vocabulary → next-word probabilities
```

Unlike a plain flatten-and-project network, self-attention lets every word in the
context window dynamically weigh every other word — the same core idea as GPT.

**Sizes (default):**
- Vocab: ~5,000–10,000 words (depends on training data)
- Embedding: 64-dim vectors
- Hidden: 128 neurons
- Context window: 8 tokens

### Training

Mini-batch SGD with PyTorch `autograd`. `loss.backward()` derives every gradient,
`torch.optim.SGD` updates the weights, and `clip_grad_norm_` keeps the attention
block numerically stable. Runs on CPU or GPU (CUDA/Intel XPU) automatically.

**Smart incremental learning:** The trainer hashes each article. If an article hasn't changed since the last run, it's skipped. New/changed articles are trained once; the model is updated in-place. Next run is instant if there are no changes.

```bash
# First run: trains everything from scratch
python train.py

# Second run: instant (no changes)
python train.py  # Done in ~1 second

# After adding data: retrains only new/changed articles
python train.py
```

### Inference & Tools

When you send a message:

1. **Quick replies** (instant, no model):
   - Greetings ("Hello") → hardcoded responses
   - Identity ("Who are you?") → trained answer
   - Action triggers (code, lookup) → tool activation

2. **Model generation** (if no quick reply):
   - Feed context through the neural net
   - Generate tokens word-by-word
   - Stop at `<EOS>` or max length

3. **Tool dispatch**:
   - **Math**: If model outputs `<calc> +`, extract operands from user input, compute exactly
   - **Knowledge**: If no tool match, search Wikipedia or local sources
   - **Fallback**: "I don't know"

This design separates *learning* (model decides "this is math") from *execution* (deterministic tool does the computation—always correct, any size).

## Training Data

### Built-in
- `training_data.json` — Hand-written dialogue and facts (Hebrew + English)
- `math_training_data.json` — Arithmetic exemplars (generated)
  - Model learns to recognize operation intent across 20+ Hebrew/English phrasings
  - Operands vary so the operator is the only predictable signal
- `better_code_examples.json`, `simple_code.json` — Code snippets

### Optional (fetch externally)
```bash
# Wikipedia articles
python collect_corpus.py --source wikipedia --out api_training_data.json

# GitHub code (requires GITHUB_TOKEN env var for high rate limits)
python collect_corpus.py --source github --github-limit 20

# Both
python collect_corpus.py --source both --merge-into training_data.json
```

### Add your own
Edit `training_data.json` or append a new JSON file:
```json
[
  {"id": "unique_id_1", "text": "Your training text here."},
  {"id": "unique_id_2", "text": "Another example."}
]
```

Then add the path to `DATA_PATHS` in `train.py` and run:
```bash
python train.py
```

## Configuration

### Environment Variables
```bash
# Hyperparameters (train.py)
CONTEXT_LEN=16     # How many prior tokens the model sees
EMBED_DIM=64       # Embedding dimension
HIDDEN_DIM=128     # Feed-forward head size
EPOCHS=40          # Training passes
LR=0.01            # Learning rate (scaled linearly by batch size)
LOG_EVERY=10       # Print loss every N epochs
BATCH_SIZE=256     # Mini-batch size
DEVICE=cpu         # Force device: cpu | cuda | xpu (default: auto-detect)

# Example:
CONTEXT_LEN=12 HIDDEN_DIM=256 EPOCHS=200 python train.py
```

### Saved Files
- `model.pt` — Trained weights (PyTorch checkpoint: config + `state_dict`)
- `tokenizer.json` — Vocabulary (word↔index mappings)
- `trained_hashes.json` — Change tracking (for incremental training)

## Project Philosophy

**Goal:** Build an AI that *learns* instead of following hardcoded rules.

**Current progress:**
- ✅ **Math**: Model learned to emit `<calc> +` for arithmetic; tool executes
- 🔄 **Code generation**: Partially learned; some templates still hardcoded
- 🔄 **Knowledge routing**: Tool-driven; model learning to emit signals
- ✅ **Greeting/identity**: Trained on examples

**Future:** Remove all hardcoded behavior. The model learns *what to do*; tools handle *how to do it*.

## Debugging & Development

### Unit Tests (smoke tests)
```bash
python model.py      # Forward pass, backprop
python tokenizer.py  # Encode/decode
```

### Inspect Weights
```python
import torch
from model import MiniLM
model = MiniLM.load("model.pt")
print(model)                       # config + parameter count
print(list(model.state_dict()))    # ['tok_emb.weight', 'pos_emb', 'Wq.weight', ...]
```

### See Loss During Training
```bash
LOG_EVERY=1 python train.py  # Print loss every epoch
```

### Test Inference
```bash
python ask.py "What's 2 + 2?"
```

## Limitations

- **Tiny model**: single attention head, one feed-forward head. Not built for state-of-the-art quality.
- **Word-level tokens**: No subword (BPE) generalization. Unknown words become `<UNK>`.
- **Small context**: 16 tokens ≈ a couple of sentences. Long conversations lose history.
- **Vocab changes force a full retrain**: embeddings are word-specific, so a new word means training from scratch.

These limits are intentional—the goal is clarity and simplicity. The PyTorch core
means you *can* scale up (more heads, more layers, GPU) when you want to.

## Files Overview

| File | Purpose |
|------|---------|
| `model.py` | Neural network core — `MiniLM(nn.Module)` (attention + FF head) |
| `tokenizer.py` | Word tokenization & vocab |
| `train.py` | PyTorch training loop with incremental learning |
| `chat.py` | Tkinter GUI & inference |
| `ask.py` | CLI mode (single query) |
| `assistant_tools.py` | External tools (math, knowledge, code) |
| `training_data.json` | Hand-written examples |
| `math_training_data.json` | Arithmetic exemplars |
| `CLAUDE.md` | Developer guide for Claude Code |

## Contributing

The project is designed for easy extension:
- **New training data**: Drop a JSON file in the project; add path to `DATA_PATHS` in train.py
- **New tools**: Add to `assistant_tools.py`; emit a signal from the model; wire it in `chat.py`'s `send()` function
- **Hyperparameter tuning**: Use environment variables

## License

MIT (or your choice)

## Notes

- This is an educational project. It's not intended to be a production AI assistant.
- Training is deterministic (seed 42) but can be made stochastic by removing the seed.
- Vocab changes (new word in training data) force a full retrain; unchanged articles are skipped.
- Hebrew and English are handled bidirectionally; language detection is automatic.

---

**Questions?** See [CLAUDE.md](CLAUDE.md) for a deeper architecture guide.
