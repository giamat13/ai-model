# MiniLM — Minimal Neural Language Model from Scratch

A tiny, from-scratch language model built in pure NumPy. No frameworks. Bilingual (Hebrew + English). The model *learns* behavior instead of relying on hardcoded algorithms.

## Quick Start

### Prerequisites
- Python 3.8+
- NumPy
- Tkinter (usually included with Python)

### Install
```bash
git clone <repo>
cd ai-model
pip install numpy
```

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

```
Input (word IDs)
    ↓
Embedding Layer (each word → dense vector)
    ↓
Flatten context window
    ↓
Hidden Layer (ReLU) + Linear transform
    ↓
Softmax over vocabulary
    ↓
Output (next word probabilities)
```

**Sizes (default):**
- Vocab: ~5,000–10,000 words (depends on training data)
- Embedding: 64-dim vectors
- Hidden: 128 neurons
- Context window: 8 tokens

### Training

Pure SGD in NumPy, ~14 seconds per epoch on 6.4k training samples.

**Smart incremental learning:** The trainer hashes each article. If an article hasn't changed since the last run, it's skipped. New/changed articles are trained once; the model is updated in-place. Next run is instant if there are no changes.

```bash
# First run: trains everything
python train.py  # ~30 min for 120 epochs

# Second run: instant (no changes)
python train.py  # Done in 1 second

# After adding data: retrains only new articles
python train.py  # ~5 min (changed articles only)
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
CONTEXT_LEN=8      # How many prior tokens the model sees
EMBED_DIM=64       # Embedding dimension
HIDDEN_DIM=128     # Hidden layer size
EPOCHS=120         # Training passes
LR=0.01            # Learning rate
LOG_EVERY=10       # Print loss every N epochs

# Example:
CONTEXT_LEN=12 HIDDEN_DIM=256 EPOCHS=200 python train.py
```

### Saved Files
- `model.npz` — Trained weights (E, W1, b1, W2, b2)
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
import numpy as np
model_dict = np.load("model.npz", allow_pickle=True)
print(model_dict.files)  # ['E', 'W1', 'b1', 'W2', 'b2']
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

- **No GPU**: Pure CPU NumPy. ~14 seconds per epoch on 6.4k samples.
- **No batching**: Processes one token at a time during training.
- **Word-level tokens**: No subword (BPE) generalization. Unknown words become `<UNK>`.
- **Small context**: 8 tokens ≈ 1–2 sentences. Long conversations lose history.
- **No attention/transformer**: Simple 2-layer feedforward. Can't handle very complex patterns.

These limits are intentional—the goal is clarity and simplicity, not state-of-the-art performance.

## Files Overview

| File | Purpose |
|------|---------|
| `model.py` | Neural network core (forward, backward, SGD) |
| `tokenizer.py` | Word tokenization & vocab |
| `train.py` | Training loop with incremental learning |
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
