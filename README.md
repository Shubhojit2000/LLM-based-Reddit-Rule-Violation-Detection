# 🛡️ LLM-based Reddit Rule Violation Detection

This project implements a binary classification system to predict whether a Reddit comment breaks a specific subreddit rule. Developed for the Kaggle competition regarding automated content moderation.

## 🏆 Project Evolution & Performance

We approached this problem in two stages, moving from traditional Transformer encoders to Generative LLMs.

| Approach | Model Architecture | ROC AUC Score
| :--- | :--- | :--- |
| **Baseline** | BERT Sequence Classification (Ensemble) | **0.620** 
| **Final** | **Qwen 2.5 (0.5B) + LoRA Fine-Tuning** | **0.701**

---

## 📌 Problem Overview
**Context-Aware Moderation:**
Your task is to create a classifier that predicts whether a Reddit comment broke a specific rule. The dataset comes from a large collection of moderated comments, encompassing diverse subreddit norms, tones, and community expectations.

**Background:**
Identifying rule-breaking comments automatically is complex because:
*   **Context:** A comment might be safe in *r/gaming* but break rules in *r/science*.
*   **Nuance:** Rules are often subjective and depend on moderator precedents.
*   **Data:** The dataset is derived from older, unlabeled content, requiring robust generalization from a small set of labeled examples (Few-Shot learning).

**Evaluation:**
Submissions are evaluated on **Column-Averaged ROC AUC**.

---

## ⚙️ Methodology

### 🚀 Phase 2: The LLM Approach (Qwen 2.5)
To improve upon the BERT baseline, we leveraged a Small Language Model (SLM) optimized for instruction following. This method significantly improved the model's ability to understand the nuance between a rule and a comment.

*   **Model:** `Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4` (Quantized).
*   **Technique:** Parameter-Efficient Fine-Tuning (PEFT) using **LoRA** (Low-Rank Adaptation).
*   **Prompt Engineering:** We utilized a Few-Shot prompting strategy where the model is fed:
    1.  The Subreddit Name.
    2.  The specific Rule.
    3.  Randomized Positive & Negative examples.
    4.  The Target Comment.
*   **Inference:** Constrained decoding (forcing the model to output probability of "Yes" vs "No") using `vLLM` for high-speed processing.

**Files:**
*   `QWEN_LORA_FINETUNE.ipynb`: The training pipeline using `SFTTrainer`.
*   `QWEN_LORA_FINETUNE_INFERENCE.ipynb`: The inference pipeline for generating submission files.

### 🏛️ Phase 1: The BERT Baseline
Our initial approach established a solid baseline using traditional BERT architecture.

*   **Text Preprocessing:** Concatenated the input as: `"body [RULE] rule"`.
*   **Architecture:** `BertForSequenceClassification` with a binary output head.
*   **Strategy:** 5-Fold Cross-Validation Ensemble. Predictions from all 5 folds were averaged to reduce variance.

**File:**
*   `jigsaw.ipynb`: The original baseline notebook.

---

## 📂 Repository Structure

```text
├── QWEN_LORA_FINETUNE.ipynb             # [New] LoRA Fine-tuning code
├── QWEN_LORA_FINETUNE_INFERENCE.ipynb   # [New] Inference & Submission code
├── jigsaw.ipynb                         # [Legacy] BERT Baseline code
└── README.md                            # Project Documentation
