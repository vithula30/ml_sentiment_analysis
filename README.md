# Word Vectorizer Benchmark — Play Store Reviews

**Subject:** Machine Learning in CYS (24CYS214)

Sentiment classification on Google Play Store app reviews using 13 different vectorization methods — from simple bag-of-words baselines to fine-tuned transformer models.

---

## Dataset

Google Play Store app reviews, binary classification: **positive** vs **negative** sentiment.

---

## Methods Compared

| # | Model | Type |
|---|-------|------|
| 1 | Bag of Words + Naive Bayes | Conventional |
| 2 | Bag of Words + Logistic Regression | Conventional |
| 3 | Bag of Words + SVM | Conventional |
| 4 | TF-IDF + Naive Bayes | Conventional |
| 5 | TF-IDF + Logistic Regression | Conventional |
| 6 | TF-IDF + SVM | Conventional |
| 7 | N-Gram (BoW) + Naive Bayes | Conventional |
| 8 | TF-IDF + N-Gram + Naive Bayes | Conventional |
| 9 | TF-IDF + N-Gram + Logistic Regression | Conventional |
| 10 | TF-IDF + N-Gram + SVM | Conventional |
| 11 | BiLSTM | Deep Learning |
| 12 | RoBERTa (roberta-base) | Transformer |
| 13 | Char-CNN | Deep Learning |

---

## Results

| Model | Accuracy | F1-Score | Time (s) |
|-------|----------|----------|----------|
| RoBERTa | 0.9040 | 0.9035 | 126.83 |
| TF-IDF + N-Gram + SVM | 0.8810 | 0.8787 | 0.05 |
| TF-IDF + Logistic Regression | 0.8790 | 0.8765 | 0.11 |
| TF-IDF + N-Gram + LR | 0.8775 | 0.8736 | 0.78 |
| BoW + Logistic Regression | 0.8695 | 0.8717 | 0.11 |
| BiLSTM | 0.8520 | 0.8517 | 5.61 |
| Char-CNN | 0.8525 | 0.8601 | 4.72 |

RoBERTa achieves the best F1-score. TF-IDF + N-Gram + SVM is the best conventional method at a fraction of the compute cost.

---

## Setup

```bash
pip install transformers torch scikit-learn pandas numpy matplotlib seaborn
```

Run in Google Colab with GPU runtime for RoBERTa and deep learning models.

---

## Structure

```
notebook.ipynb       # Main experiment notebook
README.md            # This file
*.png                # Generated plots (saved on run)
```

---

## Key Findings

- **RoBERTa** achieves the highest F1 (0.9035) but requires ~127 seconds and GPU.
- **TF-IDF + SVM** is the fastest at 0.03s with competitive accuracy.
- **Char-CNN** handles informal text and OOV words better than static embedding methods.
- Conventional TF-IDF methods remain strong baselines, especially with n-gram features.

---

## References

- Joulin et al. (2017). Bag of Tricks for Efficient Text Classification.
- Liu et al. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach.
- Zhang et al. (2015). Character-level Convolutional Networks for Text Classification.
