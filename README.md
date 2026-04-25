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
git clone https://github.com/mrkn7/sustainable-science-mapping.git
cd sustainable-science-mapping
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Gradio app locally:
```bash
python app.py
```
The app loads the pre-trained **Attention-GRU WOS-46985 (134-class)** weights from `models/attention_gru/attention_gru_wos.pth`. No GloVe download is needed at inference time — embeddings are baked into the checkpoint.

4. Programmatic inference:
```python
from models.attention_gru.inference import predict_abstract, predict_abstract_topk

abstract = "The exponential growth of scholarly literature necessitates automated systems."
class_id, confidence = predict_abstract(abstract)
print(f"Predicted Class ID: {class_id} | Confidence: {confidence:.4f}")

# Or top-5 with labels
for cid, name, conf in predict_abstract_topk(abstract, k=5):
    print(f"{cid:3d}  {name:<30s}  {conf:.4f}")
```

## 🚀 Deploy as a Live Web App (HuggingFace Spaces)

The repository is pre-configured as a **HuggingFace Space**. To deploy:

1. Create a new Space at https://huggingface.co/new-space — choose **Gradio** as the SDK.
2. Push this repo to the Space:
   ```bash
   git remote add space https://huggingface.co/spaces/<your-username>/sustainable-science-mapping
   git push space main
   ```
   (Large files like `attention_gru_wos.pth` are tracked via Git LFS or HF's native large-file support.)
3. The Space will build automatically from the YAML header in this README and `requirements.txt`. After ~2 minutes the app is live at `https://huggingface.co/spaces/<your-username>/sustainable-science-mapping`.

### Optional: showing real discipline names
The app reads `models/attention_gru/labels.json` to map class IDs → discipline names. The shipped file contains placeholders (`Class 0`, `Class 1`, ...). Replace each value with the correct WOS-46985 sub-discipline name to display real labels in the UI.
