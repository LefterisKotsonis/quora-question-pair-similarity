# Quora Question Pair Similarity

This project addresses the Quora Question Pairs problem: determining whether two questions have the same underlying intent and should therefore be considered duplicates.

The project applies a combination of Natural Language Processing (NLP) and traditional machine learning techniques. The questions are first preprocessed using lowercasing, stopword removal, and Porter stemming. Several features are then extracted to quantify the similarity between question pairs, including question length, word count, common words, Jaccard similarity, and TF-IDF-based cosine similarity.

For the textual representation, TF-IDF vectorization with unigram and bigram features is used. These features are combined with the engineered similarity features and used to train and evaluate four classification models:

Logistic Regression
Random Forest
XGBoost
Support Vector Machine (SVM)

The models are evaluated using Accuracy, F1-Score, and Log-Loss, with particular attention given to Log-Loss as suggested by the project specification.

The implementation is written in Python using libraries such as Pandas, NLTK, Scikit-learn, SciPy, and XGBoost.
