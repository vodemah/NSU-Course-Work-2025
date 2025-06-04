# Файл для применения классических методов классификации текстов

import gensim.downloader
from nltk.tokenize import WordPunctTokenizer
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB


def preprocess_labels(y) -> list:
    # Функция, переопределяющая лейблы: 1 -> 1; [2, 5] -> 2; [6, 8] -> 3; [9, 10] -> 4
    for i in range(len(y)):
        if 2 <= y[i] <= 5:
            y[i] = 2
        elif 6 <= y[i] <= 8:
            y[i] = 3
        elif 9 <= y[i] <= 10:
            y[i] = 4
    return y


class Vocabulary:
    def __init__(self):
        self.k = 10000
        self.tokenizer = WordPunctTokenizer()

    def build_vocabulary(self, texts):
        vocabulary = {}
        for text in texts:
            tokens = self.tokenizer.tokenize(text.lower())
            for token in tokens:
                if token in vocabulary:
                    vocabulary[token] += 1
                else:
                    vocabulary[token] = 1
        entries = sorted(vocabulary.items(), key=lambda item: item[1], reverse=True)
        return [entry[0] for entry in entries][:self.k]


class MultinomialNaiveBayesModel:
    def __init__(self, vocabulary):
        self.bow_vocabulary = vocabulary
        self.classifier = MultinomialNB(alpha=1.0, fit_prior=True)
        self.tokenizer = WordPunctTokenizer()

    def _text_to_bow(self, text):
        features = np.zeros(len(self.bow_vocabulary), dtype='float32')
        tokens = self.tokenizer.tokenize(text.lower())
        for token in tokens:
            if token in self.bow_vocabulary:
                index = self.bow_vocabulary.index(token)
                features[index] += 1
        return features

    def prediction(self, train_texts, train_labels, test_texts):
        # Векторизация
        X_train_vectors = np.stack([self._text_to_bow(text) for text in train_texts])
        X_test_vectors = np.stack([self._text_to_bow(text) for text in test_texts])

        # Обучение и предсказание
        self.classifier.fit(X_train_vectors, train_labels)
        return self.classifier.predict(X_test_vectors)


class TFIDFModel:
    def __init__(self, vocabulary):
        self.bow_vocabulary = vocabulary
        self.classifier = LogisticRegression(max_iter=1000)
        self.idf_dict = None
        self.tokenizer = WordPunctTokenizer()

    def _compute_idf(self, texts):
        N = len(texts)
        a = 1
        self.idf_dict = {word: 0 for word in self.bow_vocabulary}

        for word in self.bow_vocabulary:
            for text in texts:
                tokens = self.tokenizer.tokenize(text.lower())
                if word in tokens:
                    self.idf_dict[word] += 1

        for word in self.bow_vocabulary:
            self.idf_dict[word] = np.log(N / (self.idf_dict[word] + a))

    def _text_to_tfidf(self, text):
        features = np.zeros(len(self.bow_vocabulary), dtype='float32')
        tokens = self.tokenizer.tokenize(text.lower())
        for token in tokens:
            if token in self.bow_vocabulary:
                index = self.bow_vocabulary.index(token)
                features[index] += 1

        # Умножаем на IDF веса
        return features * np.array([self.idf_dict[word] for word in self.bow_vocabulary])

    def prediction(self, train_texts, train_labels, test_texts):
        # Вычисление IDF
        self._compute_idf(train_texts)

        # Векторизация
        X_train_vectors = np.stack([self._text_to_tfidf(text) for text in train_texts])
        X_test_vectors = np.stack([self._text_to_tfidf(text) for text in test_texts])

        # Обучение и предсказание
        self.classifier.fit(X_train_vectors, train_labels)
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

    def prediction(self, train_texts, train_labels, test_texts):
        X_train_vectors = np.stack([self.vectorize_sum(text) for text in train_texts])
        X_test_vectors = np.stack([self.vectorize_sum(text) for text in test_texts])

        self.classifier.fit(X_train_vectors, train_labels)
        return self.classifier.predict(X_test_vectors)


if __name__ == '__main__':
    data_train = pd.read_csv('train_reviews.csv', sep=',')
    data_test = pd.read_csv('test_reviews.csv', sep=',')

    y_train = preprocess_labels(data_train['rating'].values)
    X_train = data_train['text'].values

    y_test = preprocess_labels(data_test['rating'].values)
    X_test = data_test['text'].values

    bow_vocabulary = Vocabulary().build_vocabulary(X_train)

    nb_model = MultinomialNaiveBayesModel(vocabulary=bow_vocabulary)
    nb_prediction = nb_model.prediction(X_train, y_train, X_test)

    tfidf_model = TFIDFModel(vocabulary=bow_vocabulary)
    tfidf_prediction = tfidf_model.prediction(X_train, y_train, X_test)

    wv_model = WordVectorsModel()
    wv_prediction = wv_model.prediction(X_train, y_train, X_test)

    # Определяем точность каждой модели
    nb_score = accuracy_score(y_test, nb_prediction)
    tfidf_score = accuracy_score(y_test, tfidf_prediction)
    wv_score = accuracy_score(y_test, wv_prediction)

    for model, score in [
        ('NaiveBayes with Manual BOW', nb_score),
        ('LogisticRegression with Manual TF-IDF', tfidf_score),
        ('LogisticRegression with WordVectors', wv_score)
    ]:
        print(f'Model: {model}. Accuracy: {100 * score:.4f}%\n')
