# Sustainable Science Mapping: Green AI vs. Transformers 🌿

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red)](https://pytorch.org/)
[![Gradio](https://img.shields.io/badge/Gradio-Demo-orange)](https://gradio.app/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> **Official Repository for the paper:** "Sustainable Science Mapping: Benchmarking Green AI against Transformers for Cross-Disciplinary Abstract Classification using arXiv"

## 📄 Abstract
This study proposes a resource-efficient deep learning methodology to categorize academic abstracts from **arXiv** into three domains: **AI, Economics, and Psychology**. We conducted a systematic comparative analysis of RNNs (LSTM, GRU) and Transformers (BERT, RoBERTa).

Our results demonstrate that a **GRU model with GloVe embeddings** matches the predictive performance of BERT variants (**F1: 0.968**) while being **20x faster** and consuming **3x less energy**, aligning with the **Green AI** paradigm.

## 🚀 Key Results

| Model | Val. Accuracy | Parameters (M) | Inference Time (ms) | Energy (kWh) |
|-------|---------------|----------------|---------------------|--------------|
| **GRU (GloVe)** | **96.8%** | **1.06** | **0.36** | **0.15** |
| BERT (Base) | 94.4% | 109.5 | 7.22 | 0.50 |
| RoBERTa | 93.4% | 125.0 | 7.80 | 0.52 |

## 🛠️ Installation

1. Clone the repository:
```bash
git clone [https://github.com/mrkn7/sustainable-science-mapping.git](https://github.com/mrkn7/sustainable-science-mapping.git)
cd sustainable-science-mapping
