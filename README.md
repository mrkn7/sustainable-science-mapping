# Sustainable Science Mapping: Green AI vs. Transformers 🌿

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red)](https://pytorch.org/)
[![Gradio](https://img.shields.io/badge/Gradio-Demo-orange)](https://gradio.app/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> **Official Repository for the paper:** "Sustainable Science Mapping: Benchmarking Green AI against Transformers for Cross-Disciplinary Abstract Classification using arXiv"

## 📄 Abstract
This study proposes a resource-efficient deep learning methodology to categorize academic abstracts, scaling from coarse-grained domains to high-cardinality, fine-grained disciplinary hierarchies. We conducted a systematic comparative analysis of Recurrent Neural Networks (Attention-GRU) and Transformer-based architectures (BERT, SciBERT).

Extensive experiments on massive benchmarks, including the **WOS-46985 dataset with 134 sub-disciplines**, reveal a groundbreaking finding: Our proposed **Attention-based GRU model** utilizing static GloVe embeddings achieved a **Macro-F1 score of 0.920**.This significantly outperforms domain-specific state-of-the-art models like **SciBERT (F1: 0.867)**. Furthermore, this superior accuracy was achieved with approximately **14x faster training times** and significantly lower energy consumption compared to Transformer variants.

## 📊 Datasets Evaluated
1. **arXiv Dataset:** 3 broad categories (AI, Economics, Psychology) for baseline interdisciplinary overlap analysis.
2. **WOS-11967:** 11,967 abstracts spanning 35 fine-grained sub-disciplines (Level-2).
3. **WOS-46985:** 46,985 abstracts spanning 134 fine-grained sub-disciplines (Level-2).

## 🚀 Key Results

### 1. State-of-the-Art (SOTA) Comparison on Web of Science
Our Green AI model outperforms heavy, domain-specific Transformers on complex scientific taxonomies.

| Model | WOS-11967 (35 Classes) F1 | WOS-46985 (134 Classes) F1 | Training Time |
|-------|---------------------------|----------------------------|---------------|
| BERT-Base | 0.903 | 0.850 | ~ Hours |
| BioBERT | 0.903 | 0.856 | ~ Hours |
| SciBERT (SOTA) | 0.921 | 0.867 | ~ Hours |
| **Attention-GRU (Ours)** | **0.953** | **0.920** | **~10min** |

*(Transformer baseline results derived from recent literature benchmarks*

### 2. Efficiency & Green AI Metrics (arXiv Benchmark)
The proposed architecture matches predictive performance while drastically reducing computational overhead.

| Model | Val. Accuracy | Parameters (M) | Inference Time (ms) | Energy (kWh) |
|-------|---------------|----------------|---------------------|--------------|
| **Attention-GRU** | **96.8%** | **1.06** | **0.36** | **0.15** |
| BERT (Base) | 94.4% | 109.5 | 7.22 | 0.50 |
| RoBERTa | 93.4% | 125.0 | 7.80 | 0.52 |

## 🧠 Proposed Architecture: Attention-GRU
Our architecture utilizes a **Bidirectional GRU** combined with a **Soft Attention Mechanism** and **Frozen GloVe 300d embeddings**. This design specifically leverages the "semantic stability" of scientific terminology, avoiding the quadratic computational complexity of Transformer attention maps.


## 🛠️ Installation & Quick Start

1. Clone the repository:
```bash
git clone [https://github.com/mrkn7/sustainable-science-mapping.git](https://github.com/mrkn7/sustainable-science-mapping.git)
cd sustainable-science-mapping
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download Pre-trained GloVe Embeddings:
```bash
wget [http://nlp.stanford.edu/data/glove.6B.zip](http://nlp.stanford.edu/data/glove.6B.zip)
unzip glove.6B.zip
```

4. Running Inference:
   
```bash
from inference import predict_abstract
abstract_text = "The exponential growth of scholarly literature necessitates automated systems."
class_id, confidence = predict_abstract(abstract_text)
print(f"Predicted Class ID: {class_id} | Confidence: {confidence:.4f}")
```
