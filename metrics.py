# Файл для оценки показателей полученного датасета

import csv

from nltk.tokenize import WordPunctTokenizer


def metrics(file):
    tokenizer = WordPunctTokenizer()
    assert file == 'train' or 'test', 'Функция приняла некорректное значение'
    file_name = file + '_reviews.csv'
    reviews_amount = {str(i): 0 for i in range(1, 11)}
    highest = 0
    ranges = {'0-100': 0, '101-200': 0, '201-300': 0, '301-500': 0, '500+': 0}

    with open(file_name, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rating = row['rating']
            reviews_amount[rating] += 1
            text = tokenizer.tokenize(row['text'].lower())
            if len(text) > highest:
                highest = len(text)
            if 0 <= len(text) <= 100:
                ranges['0-100'] += 1
            elif 101 <= len(text) <= 200:
                ranges['101-200'] += 1
            elif 201 <= len(text) <= 300:
                ranges['201-300'] += 1
            elif 301 <= len(text) <= 500:
                ranges['301-500'] += 1
            else:
                ranges['500+'] += 1

    return reviews_amount, highest, ranges


for x in ('train', 'test'):
    metadata = metrics(x)

    print(f'{x.capitalize()} ratings: {metadata[0]}\n'
          f'The longest text among {x} ones: {metadata[1]}\n'
          f'The ranges of {x} texts lengths: {metadata[2]}\n\n')
