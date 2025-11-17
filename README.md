# 🧠 TinyViT — Simple Vision Transformer in Less Than 80 Lines

This repository contains a **minimal yet functional Vision Transformer (ViT)** implementation in PyTorch, designed to be **easy to understand**, **well-documented**, and **extremely concise**. The core transformer model is implemented in **under 80 lines of clean, readable code**—perfect for learning, experimentation, and educational purposes.

---

## 🚀 Highlights

- ✅ **Pure PyTorch** implementation — no external dependencies beyond standard libraries.
- ✨ **Core Transformer in <80 lines** — the self-attention and MLP block are as simple as they get.
- 🧩 Includes patch embedding via convolution for easy spatial tokenization.
- 🎯 Trains on CIFAR-10 with validation and training loop included.
- 📊 Automatically plots and saves training/validation accuracy and loss.

---

## 📦 Model Architecture

The Vision Transformer (ViT) follows a modular structure:

- **Patch Embedding**: Converts image into a sequence of flattened patches via a single convolutional layer.
- **Transformer Blocks**: Includes multi-head self-attention and MLP, wrapped with residual connections and layer normalization.
- **Classification Token**: A learnable CLS token is prepended to the sequence.
- **Final MLP Head**: Maps the CLS token to class logits.

---

## 🧠 Code Simplicity

The transformer implementation itself (excluding training and plotting) is located in a single file and spans fewer than 80 lines:

```bash
EmbeddingLayer   → 10 lines  
TransformerBlock → 26 lines  
Transformer      → 22 lines  
