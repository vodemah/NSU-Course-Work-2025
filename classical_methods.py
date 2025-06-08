# Файл для применения классических методов классификации текстов

from collections import Counter
import json

import gensim.downloader
from nltk.tokenize import WordPunctTokenizer
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB


def preprocess_labels(y):
    # Функция, переопределяющая лейблы: 1 -> 0; [2, 5] -> 1; [6, 8] -> 2; [9, 10] -> 3
    for i in range(len(y)):
        if y[i] == 1:
            y[i] = 0
        elif 2 <= y[i] <= 5:
            y[i] = 1
        elif 6 <= y[i] <= 8:
            y[i] = 2
        elif 9 <= y[i] <= 10:
            y[i] = 3
    return y


def build_vocabulary(texts):
    k = 10000
    tokenizer = WordPunctTokenizer()
    vocabulary = Counter()

    for text in texts:
        vocabulary.update(tokenizer.tokenize(text.lower()))

    return [i[0] for i in vocabulary.most_common()[:k]]


def text_to_bow(vocabulary, text):
    tokenizer = WordPunctTokenizer()
    features = np.zeros(len(vocabulary), dtype='float32')
    for token in tokenizer.tokenize(text.lower()):
        if token in vocabulary:
            index = vocabulary.index(token)
            features[index] += 1
    return features


class MultinomialNaiveBayesModel:
    def __init__(self):
        self.classifier = MultinomialNB(alpha=1.0, fit_prior=True)

    def fit(self, train_bows, train_labels):
        self.classifier.fit(train_bows, train_labels)

    def predict(self, test_bows):
        return self.classifier.predict(test_bows)


class TFIDFModel:
    def __init__(self, vocabulary, N):
        self.bow_vocabulary = vocabulary
        self.classifier = LogisticRegression(max_iter=1000)
        self.idf_dict = None
        self.N = N

    def _compute_idf(self, train_bows):
        train_bows_summed = np.zeros(len(self.bow_vocabulary), dtype='float32')
        for i in range(len(train_bows_summed)):
            for text_index in range(len(train_bows)):
                train_bows_summed[i] += train_bows[text_index][i]

        self.idf_dict = {self.bow_vocabulary[i]: train_bows_summed[i] for i in range(len(self.bow_vocabulary))}

        for word in self.bow_vocabulary:
            self.idf_dict[word] = np.log(self.N / (self.idf_dict[word] + 1))

    def _text_to_tfidf(self, text):
        features = text_to_bow(self.bow_vocabulary, text)
        # Умножаем на IDF веса
        return features * np.array([self.idf_dict[word] for word in self.bow_vocabulary])

    def fit(self, train_bows, train_labels, train_texts):
        self._compute_idf(train_bows)
        X_train_vectors = np.stack([self._text_to_tfidf(text) for text in train_texts])
        self.classifier.fit(X_train_vectors, train_labels)

    def predict(self, test_texts):
        X_test_vectors = np.stack([self._text_to_tfidf(text) for text in test_texts])
        return self.classifier.predict(X_test_vectors)


class WordVectorsModel:
    def __init__(self):
        self.embeddings = gensim.downloader.load('fasttext-wiki-news-subwords-300')
        self.tokenizer = WordPunctTokenizer()
        self.classifier = LogisticRegression(max_iter=1000)

    def vectorize_sum(self, text):
        embedding_dim = self.embeddings.vector_size
        features = np.zeros(embedding_dim, dtype='float32')
        tokens = self.tokenizer.tokenize(text.lower())
        for token in tokens:
            if token in self.embeddings:
                features += self.embeddings[token]
        return features

    def fit(self, train_texts, train_labels):
        X_train_vectors = np.stack([self.vectorize_sum(text) for text in train_texts])
        self.classifier.fit(X_train_vectors, train_labels)

    def predict(self, test_texts):
        X_test_vectors = np.stack([self.vectorize_sum(text) for text in test_texts])
        return self.classifier.predict(X_test_vectors)


def save_model(model, vocab, model_name, accuracy):
    vocab_path = f'saved_models/{model_name}_vocab.json'
    with open(vocab_path, 'w') as f:
        json.dump(vocab, f)

    if model_name == 'MultinomialNaiveBayes':
        params = {
            'class_log_prior': model.classifier.class_log_prior_.tolist(),
            'feature_log_prob': model.classifier.feature_log_prob_.tolist()
        }
        model_path = f'saved_models/{model_name}_params.json'
        with open(model_path, 'w') as f:
            json.dump(params, f)

    elif model_name == "TFIDF":
        params = {
            'idf_dict': model.idf_dict,
            'N': model.N,
            'coef': model.classifier.coef_.tolist(),
            'intercept': model.classifier.intercept_.tolist()
        }
        model_path = f'saved_models/{model_name}_params.json'
        with open(model_path, 'w') as f:
            json.dump(params, f)

    elif model_name == "WordVectors":
        params = {
            'coef': model.classifier.coef_.tolist(),
            'intercept': model.classifier.intercept_.tolist()
        }
        model_path = f'saved_models/{model_name}_params.json'
        with open(model_path, 'w') as f:
            json.dump(params, f)

    metadata = {
        'model_name': model_name,
        'accuracy': accuracy,
        'vocab_path': vocab_path,
        'model_path': model_path,
    }

    metadata_path = f'saved_models/{model_name}_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f)

    print(f'Model {model_name} with accuracy: {100*accuracy:.4f}% has just been saved!')


if __name__ == '__main__':
    data_train = pd.read_csv('train_reviews.csv', sep=',')
    data_test = pd.read_csv('test_reviews.csv', sep=',')

    y_train = preprocess_labels(data_train['rating'].values)
    X_train = data_train['text'].values
    length_of_X_train = len(X_train)

    y_test = preprocess_labels(data_test['rating'].values)
    X_test = data_test['text'].values

    bow_vocabulary = build_vocabulary(X_train)
    train_bow_vectors = np.stack([text_to_bow(bow_vocabulary, text) for text in X_train])
    test_bow_vectors = np.stack([text_to_bow(bow_vocabulary, text) for text in X_test])

    nb_model = MultinomialNaiveBayesModel()
    nb_model.fit(train_bow_vectors, y_train)
    nb_prediction = nb_model.predict(test_bow_vectors)

    tfidf_model = TFIDFModel(vocabulary=bow_vocabulary, N=length_of_X_train)
    tfidf_model.fit(train_bow_vectors, y_train, X_train)
    tfidf_prediction = tfidf_model.predict(X_test)

    wv_model = WordVectorsModel()
    wv_model.fit(X_train, y_train)
    wv_prediction = wv_model.predict(X_test)

    # Определяем точность каждой модели
    nb_score = accuracy_score(y_test, nb_prediction)
    tfidf_score = accuracy_score(y_test, tfidf_prediction)
    wv_score = accuracy_score(y_test, wv_prediction)

    save_model(nb_model, bow_vocabulary, 'MultinomialNaiveBayes', nb_score)
    save_model(tfidf_model, bow_vocabulary, 'TFIDF', tfidf_score)
    save_model(wv_model, None, 'WordVectors', wv_score)

    for model, score in [
        ('NaiveBayes with Manual BOW:', nb_score),
        ('LogisticRegression with Manual TF-IDF:', tfidf_score),
        ('LogisticRegression with WordVectors:', wv_score)
    ]:
        print(f'Model: {model}. Accuracy: {100 * score:.4f}%\n')
