# Файл для применения лучшей нейронной модели

from classical_methods import MultinomialNaiveBayesModel, TFIDFModel, WordVectorsModel, text_to_bow
from neural_networks import RNNModel, LSTMModel, GRUModel, TextCNN

import json

from nltk.tokenize import WordPunctTokenizer
import numpy as np
import torch


def load_model(model_name):
    # Загрузка метаданных
    with open(f'saved_models/{model_name}_metadata.json', encoding='UTF-8') as f:
        metadata = json.load(f)

    # Загрузка словаря
    with open(metadata['vocab_path'], 'r', encoding='UTF-8') as f:
        vocabulary = json.load(f)
    try:
        vocabulary_len = len(vocabulary)
    except TypeError:
        pass  # У WordVectors vocab == None

    try:
        with open(metadata['model_path'], 'r', encoding='UTF-8') as f:
            params = json.load(f)
    except UnicodeDecodeError:
        pass

    # Инициализация модели
    if model_name == 'RNN':
        model = RNNModel(vocabulary_len)

    elif model_name == 'LSTM':
        model = LSTMModel(vocabulary_len)

    elif model_name == 'GRU':
        model = GRUModel(vocabulary_len)

    elif model_name == 'TextCNN':
        model = TextCNN(vocabulary_len)

    elif model_name == 'MultinomialNaiveBayes':
        model = MultinomialNaiveBayesModel()
        model.classifier.class_log_prior_ = np.array(params['class_log_prior'])
        model.classifier.feature_log_prob_ = np.array(params['feature_log_prob'])
        model.classifier.classes_ = np.array([0, 1, 2, 3])

    elif model_name == 'TFIDF':
        model = TFIDFModel(vocabulary=vocabulary, N=params['N'])
        model.idf_dict = params['idf_dict']
        model.classifier.coef_ = np.array(params['coef'])
        model.classifier.intercept_ = np.array(params['intercept'])
        model.classifier.classes_ = np.array([0, 1, 2, 3])

    elif model_name == 'WordVectors':
        print('Ожидайте. Выполняется загрузка предобученных векторов...\n')
        model = WordVectorsModel()
        model.classifier.coef_ = np.array(params['coef'])
        model.classifier.intercept_ = np.array(params['intercept'])
        model.classifier.classes_ = np.array([0, 1, 2, 3])

    if model_name in ('RNN', 'LSTM', 'GRU', 'TextCNN'):
        model.load_state_dict(torch.load(metadata['model_path']))
        model.eval()

    if 'max_len' in metadata:
        return model, vocabulary, metadata['max_len']
    else:
        return model, vocabulary, None


def preprocess_text(text, vocab, max_len, tokenizer=WordPunctTokenizer()):
    tokens = tokenizer.tokenize(text.lower())[:max_len]
    sequence = [vocab.get(token, vocab['<UNK>']) for token in tokens]
    padded_sequence = sequence + [vocab['<PAD>']] * (max_len - len(sequence))
    return torch.tensor([padded_sequence], dtype=torch.long)


while True:
    text = input('Введите одну цифру для модели, которую желаете использовать:\n'
                 'RNN - 1, LSTM - 2, GRU - 3, TextCNN - 4,\n'
                 'MultinomialNaiveBayes - 5, TFIDF - 6, WordVectors - 7\n')
    assert len(text) == 1 and text in (str(i) for i in range(1, 8)), print('Вы ввели неправильное число')

    models = {
        '1': 'RNN', '2': 'LSTM', '3': 'GRU', '4': 'TextCNN',
        '5': 'MultinomialNaiveBayes', '6': 'TFIDF', '7': 'WordVectors'
    }
    model_name = models[text]

    model, vocab, max_len = load_model(model_name)

    text = input(f'Введите текст отзыва для классификации на основе {model_name}.\n'
                 'Если желаете прервать выполнение программы, введите "0"\n')

    if len(text) > 0 and text != '0':
        if max_len is not None:
            input_tensor = preprocess_text(text, vocab, max_len)
            prediction = model(input_tensor)
            predicted_class = torch.argmax(prediction, dim=1).item()
        elif model_name == 'WordVectors':
            predicted_class = model.predict(text)
        elif model_name == 'MultinomialNaiveBayes':
            features = text_to_bow(vocab, text)
            predicted_class = model.predict([features])
        else:
            predicted_class = model.predict([text])

    elif len(text) == 0:
        print('Вы не ввели текст.\n')
        continue
    elif text == '0':
        print('Завершение программы.')
        break

    if predicted_class == 0:
        predicted_class = '1'
    elif predicted_class == 1:
        predicted_class = '2-5'
    elif predicted_class == 2:
        predicted_class = '6-8'
    elif predicted_class == 3:
        predicted_class = '9-10'

    print(f'\nВероятнее всего, этот текст имеет оценку {predicted_class}\n')
