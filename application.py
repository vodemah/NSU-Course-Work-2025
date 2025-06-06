# Файл для применения лучшей нейронной модели

from neural_networks import RNNModel, LSTMModel, GRUModel, TextCNN

import json

from nltk.tokenize import WordPunctTokenizer
import torch


def load_model(model_name):
    # Загрузка метаданных
    with open(f'saved_models/{model_name}_metadata.json') as f:
        metadata = json.load(f)

    # Загрузка словаря
    with open(metadata['vocab_path']) as f:
        vocabulary = json.load(f)
    vocabulary_len = len(vocabulary)

    # Инициализация модели
    if metadata['model_name'] == 'RNN':
        model = RNNModel(vocabulary_len)
    elif metadata['model_name'] == 'LSTM':
        model = LSTMModel(vocabulary_len)
    elif metadata['model_name'] == 'GRU':
        model = GRUModel(vocabulary_len)
    elif metadata['model_name'] == 'TextCNN':
        model = TextCNN(vocabulary_len)

    # Загрузка весов
    model.load_state_dict(torch.load(metadata['model_path']))
    model.eval()

    return model, vocabulary, metadata['max_len']


def preprocess_text(text, vocab, max_len, tokenizer=WordPunctTokenizer()):
    tokens = tokenizer.tokenize(text.lower())[:max_len]
    sequence = [vocab.get(token, vocab['<UNK>']) for token in tokens]
    padded_sequence = sequence + [vocab['<PAD>']] * (max_len - len(sequence))
    return torch.tensor([padded_sequence], dtype=torch.long)


model, vocab, max_len = load_model('TextCNN')

while True:
    text = input('Введите текст отзыва для классификации на основе CNN.\n'
                 'Если желаете прервать выполнение программы, введите "0"\n')
    if len(text) > 0 and text != '0':
        input_tensor = preprocess_text(text, vocab, max_len)
    elif len(text) == 0:
        print('Вы не ввели текст.\n')
        continue
    elif text == '0':
        print('Завершение программы.')
        break
    prediction = model(input_tensor)
    predicted_class = torch.argmax(prediction, dim=1).item()
    if predicted_class == 0:
        predicted_class = '1'
    elif predicted_class == 1:
        predicted_class = '2-5'
    elif predicted_class == 2:
        predicted_class = '6-8'
    elif predicted_class == 3:
        predicted_class = '9-10'
    print(f'\nВероятнее всего, этот текст имеет оценку {predicted_class}\n')
