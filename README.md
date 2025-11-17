# 🧠 TinyViT — Simple Vision Transformer in Less Than 80 Lines

This repository contains a **minimal yet functional Vision Transformer (ViT)** implementation in PyTorch, designed to be **easy to understand**, **well-documented**, and **extremely concise**. The core transformer model is implemented in **under 80 lines of clean, readable code**—perfect for learning, experimentation, and educational purposes.

---

## 🚀 Highlights

- ✅ **Pure PyTorch** implementation — no external dependencies beyond standard libraries.
- ✨ **Core Transformer in <80 lines** — the self-attention and MLP block are as simple as they get.
-    **Transformer Improvements** - such as Multi-head Attention easily comparable to the core transformer via [pull requests](https://github.com/PetrDvoracek/dead-simple-vit/pull/2/files).
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

Each class is **clearly structured and documented** to highlight the essential components of the Vision Transformer.

---

## 🏁 Quick Start

### 1. Clone this repository

```bash
git clone https://github.com/your-username/tiny-vit.git
cd tiny-vit
```

### 2. Adjust Configurations

Open the script and modify the following variables if needed:

```python
dataset_root = "/path/to/torch_datasets"
device = "cuda"  # or "mps" / "cpu"
```

### 3. Run the Training Script

```bash
python vit.py
```

The script will:

* Train the model on CIFAR-10.
* Display progress with `tqdm`.
* Save a plot of loss and accuracy per epoch.

---

## 📈 Output Example

At the end of training, a performance graph will be saved:

```
1_block_e-40_acc-XX.XX.png
```

It includes:

* Training vs. validation loss
* Training vs. validation accuracy

---

## 🔍 Files Overview

| File            | Description                               |
| --------------- | ----------------------------------------- |
| `vit.py`        | Full model and training loop              |
| `README.md`     | You're reading it                         |
| *(Coming soon)* | Notebook version for easy experimentation |

---

## 📚 Educational Value

If you're looking to **learn how transformers work in vision tasks**, this repo is a great starting point. All complexity has been stripped away to **highlight the core ideas** without distraction.

---

## 🛠 TODO

* [ ] Add support for different datasets.
* [x] Add Multi-head attention to different branch. See changes required to turn regular self-attention into multi-head self attention [in this pull request](https://github.com/PetrDvoracek/dead-simple-vit/pull/2/files)


---

## 🧾 License

This project is open-source and available under the [MIT License](LICENSE).

---

## 🙌 Acknowledgements

Inspired by the original [ViT paper](https://arxiv.org/abs/2010.11929) and simplified implementations across the PyTorch community.

