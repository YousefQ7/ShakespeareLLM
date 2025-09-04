ShakespeareLLM

ShakespeareLLM is a custom, character-level Transformer language model built and trained from scratch in PyTorch. It is trained on the WikiText-103 dataset and demonstrates the core principles of modern large language models (LLMs), including self-attention, positional embeddings, and autoregressive text generation.

This project highlights the foundations of LLM design without relying on high-level frameworks like Hugging Face’s Transformers — instead, it implements the architecture directly in PyTorch for maximum learning value.

🚀 Features

Custom GPT-style Transformer implemented from scratch.

Character-level tokenization for fine-grained text generation.

Training loop with live loss visualization using Matplotlib.

Checkpoint saving and resuming for long training runs.

Autoregressive text generation with temperature, top-k, and nucleus (top-p) sampling.

Trained on WikiText-103, a large collection of Wikipedia articles.

🧠 Model Architecture

Embedding layers for characters + positions.

Multi-Head Self-Attention mechanism.

Feedforward layers with GELU activation.

LayerNorm + residual connections.

Configurable hyperparameters:

n_embd = 256

n_head = 8

n_layer = 8

block_size = 256 (context window)
