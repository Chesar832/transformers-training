## Transformers Training
![banner](https://www.valterlongo.com/wp-content/uploads/2018/08/neuron-banner.jpg)

📖 This repository is based on *"Transformers for machine learning - A deep dive"* book from Uday Kamath, Kenneth L. Graham and Wael Emara.

👩🏻‍💻 Most of the notebooks were created and run on Kaggle because it enables the use of GPUs, which allow parallelization and lower training times.

🔎 Additionally each one of these notebooks use Kaggle's Datasets, that's why this repo does not have data files such as csv's , but replicated notebooks do have its respective data.

#### Contents
1. **Attention is all You Need**
    - Analysis and explanation [here](papers/attention_is_all_you_need.md).
2. **BERT and its Variants (RoBERTa, ALBERT, DistilBERT and others)**
    - Review and explanation of the models [here](transformers-deep-dive-book/chapter-03/BERT%20and%20its%20variants.md).
    - Replicated notebook to finetune BERT for sentiment classification [here](transformers-deep-dive-book/chapter-03/fine-tune-bert-for-sentiment-classification.ipynb) with the *Reviews Dataset* [data here](transformers-deep-dive-book/chapter-03/reviews.csv).
    - Notebook for Text Classification using Pytorch and HF(🤗) [here](kaggle-notebooks/single-lingual-torch-hf-text-classification.ipynb) with the *Coronavirus tweets NLP Dataset* [data here](https://www.kaggle.com/datasets/datatattle/covid-19-nlp-text-classification).
    - Notebook for Name Entity Recognition (NER) using Pytorch and HF(🤗) [here](kaggle-notebooks/single-lingual-torch-hf-ner.ipynb) with the *Named Entity Recognition (NER) Corpus Dataset* [data here](https://www.kaggle.com/datasets/naseralqaydeh/named-entity-recognition-ner-corpus).
3. **Multilingual transformers**
    - Basic Multilingual Transformer, Single-Encoder Multilingual NLU, Dual-Encoder NLU and Multilingual NLG review and explanation of the models [here](transformers-deep-dive-book/chapter-04/Multilingual%20transformer%20architectures.md).

---
🧠 Made in Chesar's bedroom