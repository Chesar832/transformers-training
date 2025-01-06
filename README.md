## Transformers Training
![banner](https://www.valterlongo.com/wp-content/uploads/2018/08/neuron-banner.jpg)

📖 This repository is based on *"Transformers for machine learning - A deep dive"* book from Uday Kamath, Kenneth L. Graham and Wael Emara.

👩🏻‍💻 Most of the notebooks were created and run on Kaggle because it enables the use of GPUs, which allow parallelization and lower training times. But, some of them were delevoped or copied on Google Colab because Kaggle's notebooks has a default environment which has so many libraries already installed and is complicated to customize it for dependencies conflicts.

🔎 Additionally, each of these notebooks uses Kaggle's Datasets. That's why this repo does not include data files such as CSVs. However, the replicated notebooks contain their respective data, either in a file or from Hugging Face's Datasets package.

#### Contents
1. **Attention is all You Need**
    - Analysis and explanation [here](papers/attention_is_all_you_need.md).
2. **BERT and its Variants (RoBERTa, ALBERT, DistilBERT and others)**
    - Review and explanation of the models [here](transformers-deep-dive-book/chapter-03/BERT%20and%20its%20variants.md).
    - Replicated notebook to finetune BERT for sentiment classification [here](kaggle-notebooks/BERT-single-lingual-torch-hf-text-classification/BERT-single-lingual-torch-hf-text-classification.ipynb) with the *Reviews Dataset* [data here](kaggle-notebooks/BERT-single-lingual-torch-hf-text-classification/reviews.csv).
    - Notebook for Text Classification using Pytorch and HF(🤗) [here](kaggle-notebooks/ALBERT-single-lingual-torch-hf-text-classification.ipynb) with the *Coronavirus tweets NLP Dataset* [data here](https://www.kaggle.com/datasets/datatattle/covid-19-nlp-text-classification).
    - Notebook for Name Entity Recognition (NER) using Pytorch and HF(🤗) [here](kaggle-notebooks/RoBERTa-single-lingual-torch-hf-ner.ipynb) with the *Named Entity Recognition (NER) Corpus Dataset* [data here](https://www.kaggle.com/datasets/naseralqaydeh/named-entity-recognition-ner-corpus).
3. **Multilingual transformers**
    - Basic Multilingual Transformer, Single-Encoder Multilingual NLU, Dual-Encoder NLU and Multilingual NLG review and explanation of the models [here](transformers-deep-dive-book/chapter-04/Multilingual%20transformer%20architectures.md).
    - Replicated notebook to mUSE for sentiment classification [here](colab-notebooks/mUSE-multilingual-tf-torch-text-classification.ipynb) with data extracted directly from HF(🤗) Datasets library.
    - Notebook for Textual Similarity Detection using Tensorflow Hub (now Kaggle Models Hub), Pytorch Lightning and HF(🤗) [here](colab-notebooks/USE-single-lingual-tf-torch-sentence-similarity-classification.ipynb) with the *Quora Question Pairs* dataset [data here](https://www.kaggle.com/datasets/naseralqaydeh/named-entity-recognition-ner-corpus).
4. **Transformer Modifications**
    - Explanation of Trnasformer Modifications that aim to make more efficient the computational cost or to expand model capabilities like Transformer-XL which modifies the way to compute the hidden states and challenge the original positional encoding presented in Attention is All You Need. All those concepts are explained [here](transformers-deep-dive-book/chapter-05/Transformer%20modifications.md).
---
🧠 Made in Chesar's bedroom