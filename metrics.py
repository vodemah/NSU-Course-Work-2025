# Файл для оценки показателей полученного датасета

import csv

from nltk.tokenize import WordPunctTokenizer

train_amounts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0}
test_amounts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0}

highest_train = 0
highest_test = 0

train_ranges = {'0-100': 0, '101-200': 0, '201-300': 0, '301-500': 0, '500+': 0}
test_ranges = {'0-100': 0, '101-200': 0, '201-300': 0, '301-500': 0, '500+': 0}


tokenizer = WordPunctTokenizer()

# Обработка train_reviews.csv
with open('train_reviews.csv', mode='r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    for row in reader:
        rating = int(row['rating'])
        if rating in train_amounts:
            train_amounts[rating] += 1
        text = tokenizer.tokenize(row['text'].lower())
        if len(text) > highest_train:
            highest_train = len(text)
        if 0 <= len(text) <= 100:
            train_ranges['0-100'] += 1
        elif 101 <= len(text) <= 200:
            train_ranges['101-200'] += 1
        elif 201 <= len(text) <= 300:
            train_ranges['201-300'] += 1
        elif 301 <= len(text) <= 500:
            train_ranges['301-500'] += 1
        else:
            train_ranges['500+'] += 1

# Обработка test_reviews.csv
with open('test_reviews.csv', mode='r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    for row in reader:
        rating = int(row['rating'])
        if rating in test_amounts:
            test_amounts[rating] += 1
        text = tokenizer.tokenize(row['text'].lower())
        if len(text) > highest_test:
            highest_test = len(text)
        if 0 <= len(text) <= 100:
            test_ranges['0-100'] += 1
        elif 101 <= len(text) <= 200:
            test_ranges['101-200'] += 1
        elif 201 <= len(text) <= 300:
            test_ranges['201-300'] += 1
        elif 301 <= len(text) <= 500:
            test_ranges['301-500'] += 1
        else:
            test_ranges['500+'] += 1

print('Train ratings:', train_amounts)
print('Test ratings:', test_amounts)
print('The longest text among train ones:', highest_train)
print('The longest text among test ones:', highest_test)
print('The ranges of train texts lengths:', train_ranges)
print('The ranges of test texts lengths:', test_ranges)
