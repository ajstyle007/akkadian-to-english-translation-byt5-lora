# Akkadian to English Translation using ByT5 + LoRA

### 🚀 Overview
- Kaggle competition: Deep Past Challenge
- Goal: Akkadian → English translation (low-resource NLP problem)
- Approach: Fine-tuning ByT5-small using LoRA

### 🔗 Resources
For full training artifacts and experimentation details:
- 📦 Kaggle Dataset / Model: https://www.kaggle.com/datasets/crazyelon/byt5-small-new
- 📓 Kaggle Notebook: https://www.kaggle.com/code/crazyelon/notebook94d8eed9fa

### ⚙️ Tech Stack
- Transformers 🤗
- PEFT (LoRA)
- PyTorch
- SacreBLEU

1. Model
- google/byt5-small (byte-level model, useful for rare languages)

2. Fine-tuning
- Used LoRA for memory efficiency
- Significantly reduced trainable parameters

`model.print_trainable_parameters()`
trainable params: 2,375,680 || all params: 302,013,440 || trainable%: 0.7866

3. Training Setup
- Batch size: 2 (gradient accumulation = 8)
- Epochs: 12
- FP16 training
- Label smoothing: 0.1

4. Evaluation Metrics
- BLEU
- chrF++
- Final Score = √(BLEU × chrF)

### ⚠️ Challenges

This section is very powerful:
- Low-resource language
- Noisy labels
- Long sequence handling
- GPU memory constraints

### 💡 Learnings

This impresses recruiters:
- LoRA significantly reduces memory cost
- Byte-level models work well for unknown scripts
- Evaluation in translation is tricky (BLEU ≠ actual quality)

### 🔄 Data Preprocessing & Tokenization

The dataset was preprocessed and tokenized using the Hugging Face ecosystem to ensure efficient and reproducible training.

- The original dataset was loaded from disk and split into training and validation sets (90% / 10% split).
- A byte-level tokenizer (ByT5) was used, which is particularly effective for low-resource and ancient languages like Akkadian.
- Both source (Akkadian) and target (English) texts were tokenized with truncation and fixed-length padding to maintain consistency across batches.
- The tokenized datasets were processed in batches for better performance and memory efficiency.
- To avoid repeated preprocessing and speed up experimentation, the tokenized datasets were saved to disk.
- Sample decoded outputs were manually inspected to verify the correctness of the preprocessing pipeline.

#### 💡 Key Benefits
- Faster training due to preprocessed data
- Improved reproducibility
- Efficient handling of large datasets
- Reduced computational overhead during experimentation
