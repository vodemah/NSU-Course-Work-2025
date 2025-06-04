# Файл для применения нейронных методов классификации текста

from collections import Counter
import json

import matplotlib.pyplot as plt
from nltk.tokenize import WordPunctTokenizer
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
import torch
import torch.nn as nn
from torch.nn.functional import max_pool1d, relu
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


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


class ReviewsDataset(Dataset):
    def __init__(self, texts, targets, vocab=None, max_len=150):
        self.texts = texts
        self.targets = targets
        self.max_len = max_len
        self.tokenizer = WordPunctTokenizer()

        if vocab is None:
            self.build_vocab()
        else:
            self.vocab = vocab

    def build_vocab(self):
        word_counts = Counter()
        for text in self.texts:
            word_counts.update(self.tokenizer.tokenize(text.lower()))

        self.vocab = {
            '<PAD>': 0,  # Токен для дополнения с индексом 0
            '<UNK>': 1,  # Токен для неизвестных слов с индексом 1
            **{word: i + 2 for i, word in enumerate(word_counts)}  # Токены с индексами с 2
        }

    def text_to_sequence(self, text):
        # Конвертирует текст в список индексов, обрезая до max_len токенов и заменяя неизвестные токены на индекс <UNK>
        tokens = self.tokenizer.tokenize(text.lower())[:self.max_len]
        return [self.vocab.get(token, self.vocab['<UNK>']) for token in tokens]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        sequence = self.text_to_sequence(self.texts[index])
        # Дополнение до max_len токенами <PAD>
        padded_sequence = sequence + [self.vocab['<PAD>']] * (self.max_len - len(sequence))

        return {
            'text': torch.tensor(padded_sequence, dtype=torch.long),
            'target': torch.tensor(self.targets[index], dtype=torch.long)
        }


class RNNModel(nn.Module):
    def __init__(self, vocabulary_size, embedding_dim=300, hidden_dim=512,
                 num_layers=2, num_classes=4, dropout=0.5, bidirectional=True):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim, padding_idx=0)
        self.bn = nn.BatchNorm1d(embedding_dim)
        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            nonlinearity='relu',
            bidirectional=bidirectional
        )
        self.dropout = nn.Dropout(dropout)
        output_size = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(output_size, num_classes)

    def forward(self, x):
        # [batch, seq_len] -> [batch, seq_len, emb_dim]
        embedded = self.embedding(x)
        embedded = self.bn(embedded.transpose(1, 2)).transpose(1, 2)
        rnn_out, _ = self.rnn(embedded)
        last_hidden = rnn_out[:, -1, :]
        return self.fc(self.dropout(last_hidden))


class LSTMModel(nn.Module):
    def __init__(self, vocabulary_size, embedding_dim=300, hidden_dim=512,
                 num_layers=2, num_classes=4, dropout=0.5, bidirectional=True):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim, padding_idx=0)
        self.bn = nn.BatchNorm1d(embedding_dim)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        self.dropout = nn.Dropout(dropout)
        output_size = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(output_size, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        embedded = self.bn(embedded.transpose(1, 2)).transpose(1, 2)
        lstm_out, _ = self.lstm(embedded)
        last_hidden = lstm_out[:, -1, :]
        return self.fc(self.dropout(last_hidden))


class GRUModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=300, hidden_dim=512,
                 num_layers=2, num_classes=4, dropout=0.5, bidirectional=True):
        super(GRUModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.bn = nn.BatchNorm1d(embedding_dim)
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        self.dropout = nn.Dropout(dropout)
        self.bidirectional = bidirectional
        output_size = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(output_size, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        embedded = self.bn(embedded.transpose(1, 2)).transpose(1, 2)
        gru_out, _ = self.gru(embedded)
        last_hidden = gru_out[:, -1, :]
        return self.fc(self.dropout(last_hidden))


class TextCNN(nn.Module):
    def __init__(self, vocabulary_size, embedding_dim=300, num_filters=200,
                 filter_sizes=[2, 3, 4, 5], num_classes=4, dropout=0.5):
        super(TextCNN, self).__init__()

        self.embedding = nn.Embedding(vocabulary_size, embedding_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim,
                      out_channels=num_filters,
                      kernel_size=fs)
            for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def forward(self, x):
        embedded = self.embedding(x)  # [batch, seq, emb]
        embedded = embedded.permute(0, 2, 1)  # [batch, emb, seq]

        pooled_outputs = []
        for conv in self.convs:
            convolved = relu(conv(embedded))  # [batch, filters, new_seq_len]
            pooled = max_pool1d(convolved, convolved.shape[2]).squeeze(2)  # Глобальный макспулинг
            pooled_outputs.append(pooled)

        cat = self.dropout(torch.cat(pooled_outputs, dim=1))  # Объединение фич от разных фильтров
        return self.fc(cat)


def train_and_evaluate(model, train_loader, test_loader, model_name, vocab, max_len, class_weights=None, num_epochs=20):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    if class_weights is not None:
        class_weights = class_weights.to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()  # Функция потерь
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)  # Оптимизатор

    # Цикл обучения
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            texts = batch['text'].to(device)
            targets = batch['target'].to(device)

            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, targets)

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'{model_name} - Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader):.4f}')

    # Оценка модели
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in test_loader:
            texts = batch['text'].to(device)
            targets = batch['target'].to(device)

            outputs = model(texts)
            _, preds = torch.max(outputs, 1)

            all_predictions.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    accuracy = accuracy_score(all_targets, all_predictions)

    print(f'\n{model_name} Classification Report:')
    print(classification_report(all_targets, all_predictions))
    print(f'Accuracy: {100*accuracy:.4f}%')

    # Сохраняем текущую модель
    save_model(model, vocab, max_len, model_name, accuracy)

    return accuracy


def save_model(model, vocab, max_len, model_name, accuracy):
    # Сохраняем словарь
    vocab_path = f'saved_models/{model_name}_vocab.json'
    with open(vocab_path, 'w') as f:
        json.dump(vocab, f)

    # Сохраняем состояние модели
    model_path = f'saved_models/{model_name}_model.pth'
    torch.save(model.state_dict(), model_path)

    # Сохраняем метаинформацию
    metadata = {
        'model_name': model_name,
        'accuracy': accuracy,
        'vocab_path': vocab_path,
        'model_path': model_path,
        'max_len': max_len
    }

    with open(f'saved_models/{model_name}_metadata.json', 'w') as f:
        json.dump(metadata, f)

    print(f'Model "{model_name}" with accuracy: {100*accuracy:.4f}% has just been saved')


if __name__ == '__main__':
    data_train = pd.read_csv('train_reviews.csv', delimiter=',')
    data_test = pd.read_csv('test_reviews.csv', delimiter=',')

    y_train = preprocess_labels(data_train['rating'].values)
    X_train = data_train['text'].values

    y_test = preprocess_labels(data_test['rating'].values)
    X_test = data_test['text'].values

    # Создание Dataset и DataLoader
    train_dataset = ReviewsDataset(X_train, y_train)
    test_dataset = ReviewsDataset(X_test, y_test, vocab=train_dataset.vocab)

    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    vocab = train_dataset.vocab
    vocab_size = len(vocab)
    max_len = train_dataset.max_len

    # Вычисляем веса классов
    class_counts = torch.bincount(torch.tensor(y_train))
    class_weights = 1.0 / class_counts.float()
    class_weights = class_weights / class_weights.sum()  # Нормализуем

    # Инициализация, обучение и оценка моделей
    models = {
        'RNN': RNNModel(vocab_size),
        'LSTM': LSTMModel(vocab_size),
        'GRU': GRUModel(vocab_size),
        'TextCNN': TextCNN(vocab_size)
    }

    results = {}
    for name, model in models.items():
        print(f'\nTraining {name} model')
        accuracy = train_and_evaluate(model, train_loader, test_loader, name, vocab, max_len, class_weights)
        results[name] = accuracy

    # Визуализация результатов
    models_names = list(results.keys())
    accuracies = list(results.values())

    plt.figure(figsize=(10, 6))
    bars = plt.bar(models_names, accuracies, color=['blue', 'green', 'orange', 'purple'])
    plt.title('Model Comparison')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.4f}',
                 ha='center', va='bottom')

    plt.savefig('saved_models/models_comparison.png')  # Сохраняем график
    plt.show()

    # Определяем лучшую модель
    best_model = max(results, key=results.get)
    best_accuracy = results[best_model]

    print(f'\nBest model: "{best_model}" with accuracy {100*best_accuracy:.4f}%')
