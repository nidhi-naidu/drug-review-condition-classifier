# Condition Classification from Drug Reviews using NLP & Deep Learning

**Authors:** Nidhi Naidu & Hritik Majgaonkar

**Course:** EMSE 6575 — Applied ML for Analytics, George Washington University

## Contributors

- [Nidhi Naidu](https://github.com/nidhi-naidu)
- [Hritik Majgaonkar](https://github.com/HritikMajgaonkar)

## Overview

Online drug reviews contain valuable firsthand insight into patient experiences, side effects, and treatment outcomes — but manually reading thousands of reviews is slow, inconsistent, and hard to scale. This project builds an automated NLP pipeline that classifies patient drug reviews by the medical condition they discuss (Birth Control, Depression, Pain, Anxiety), enabling faster and more consistent analysis of large-scale patient feedback.

## Dataset

- **Source:** [Drugs.com](https://www.drugs.com/) review dataset (~160,000 reviews)
- **Features:** drug name, review text, condition, rating, date, useful count
- **Focus:** the four most frequent conditions — Birth Control, Depression, Pain, Anxiety

## Approach

1. **Preprocessing** — HTML removal, regex cleanup, lowercasing, stop-word removal, and WordNet lemmatization; 80/20 train-test split with label encoding.
2. **Feature engineering** — Bag-of-Words (`CountVectorizer`) and TF-IDF with unigram/bigram/trigram ranges; word clouds to explore top tokens per condition.
3. **Classical ML models** — Multinomial Naive Bayes and Passive-Aggressive Classifier, trained on both BoW and TF-IDF features.
4. **Deep learning** — a Bidirectional LSTM (Embedding → BiLSTM → Dense) trained on padded token sequences with sparse categorical cross-entropy loss.

## Results

| Model | Features | Accuracy |
|---|---|---|
| Multinomial Naive Bayes | Bag-of-Words | baseline |
| Passive-Aggressive Classifier | TF-IDF (uni+bi+trigrams) | **96.5%** |
| Bidirectional LSTM | Embedding + BiLSTM | 94.8% |

The Passive-Aggressive classifier with TF-IDF trigrams was the top performer and fastest to train, making it well-suited for efficient deployment. The BiLSTM captured deeper sequential patterns in longer, more complex reviews.

## Key Insights

- N-gram features significantly boosted detection of medical phrases like "severe side effects."
- Passive-Aggressive + TF-IDF offered the best accuracy-to-speed tradeoff.
- The BiLSTM handled longer, more nuanced reviews with stronger sequence understanding, at higher compute cost.

## Future Work

- Extend to sentiment analysis alongside condition classification
- Fine-tune with pre-trained embeddings (GloVe, Word2Vec)
- Explore transformer-based models (e.g., BERT)
- Support multi-label classification for reviews mentioning multiple conditions
- Build a real-time web app for instant review classification

## Repo Contents

- `Drugs_NLP.ipynb` — full notebook: preprocessing, feature engineering, model training and evaluation
- `Final_Presentation.pdf` — final project presentation slides
- `requirements.txt` — Python dependencies

## Links

- 📓 [Colab Notebook](https://colab.research.google.com/drive/15y7GNxfC7eesbgK406Iit7RWGCe7B0CX?usp=sharing)
- ✍️ [Medium Article: From Raw Drug Reviews to Production-Ready Multiclass Text Classifier](https://medium.com/@hritik.majgaonkar/from-raw-drug-reviews-to-production-ready-multiclass-text-classifier-c04ede86e85a)

## Tech Stack

Python, pandas, scikit-learn, TensorFlow/Keras, NLTK, BeautifulSoup, WordCloud, matplotlib, seaborn
