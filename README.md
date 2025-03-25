# Fine-Tuning GPT-2 for Text Summarization

## Overview
This repository contains an implementation of fine-tuning GPT-2 (Generative Pre-trained Transformer 2) for abstractive text summarization. The objective is to train GPT-2 to generate coherent and concise summaries based on input text.

## Dataset
We use a public text summarization dataset from Kaggle:
[Text Summarization Dataset](https://www.kaggle.com/code/lusfernandotorres/text-summarization-with-large-language-models/input)

## Model Architecture
GPT-2 is a transformer-based autoregressive model that generates text by predicting the next word in a sequence. For abstractive summarization, GPT-2 is fine-tuned to generate a summary that captures the essence of the original text.

### Fine-Tuning Process
- **Preprocessing:**
  - Tokenization using Hugging Face's `GPT2Tokenizer`.
  - Padding and truncation of text sequences.

- **Training:**
  - Model: `gpt2` from Hugging Face Transformers.
  - Loss Function: Cross-Entropy Loss.
  - Optimizer: AdamW.
  - Learning Rate Scheduler: Linear decay with warm-up steps.
  - Training with teacher forcing for better sequence generation.

- **Evaluation:**
  - Loss tracking during training.
  - Summarization performance comparison using ROUGE-N and ROUGE-L scores.

## Installation
### Prerequisites
Ensure you have the following dependencies installed:
```bash
pip install transformers datasets torch nltk rouge-score
```

## Running the Notebook
1. Clone the repository:
```bash
git clone https://github.com/yourusername/gpt2-text-summarization.git
cd gpt2-text-summarization
```
2. Open the Jupyter Notebook:
```bash
jupyter notebook GPT2.ipynb
```
3. Follow the steps in the notebook to preprocess the data, fine-tune GPT-2, and evaluate the model.

## Results
- **Loss Analysis:**
  - Training loss is reported for each epoch.
- **Sample Summaries:**
  - Example original text and generated summaries are displayed.
- **Comparison:**
  - Summary quality is compared using ROUGE scores.

## Future Improvements
- Implement hyperparameter tuning for better performance.
- Experiment with larger GPT-2 variants (GPT-2 Medium, Large, or XL).
- Deploy the trained model as an API for real-world usage.

## Output
![Sudoku Solver Screenshot](https://github.com/Muradhameed921/Sudoku-Puzzle-Solver/blob/main/O1.jpg)
