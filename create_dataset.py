# Файл для создания датасета

import csv
import time

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException, ElementClickInterceptedException
from selenium.webdriver.common.action_chains import ActionChains


def setup_driver():
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                         'Chrome/91.0.4472.124 Safari/537.36')
    options.add_experimental_option('excludeSwitches', ['enable-logging'])
    driver = webdriver.Chrome(options=options)
    return driver


def get_reviews_pages(driver, url) -> list:
    # Функция для сбора ссылок на сериалы/фильмы со страниц с их списками
    driver.get(url)
    time.sleep(2)
    elements = driver.find_elements(By.CSS_SELECTOR, 'a.ipc-title-link-wrapper')
    links = [element.get_attribute("href") for element in elements]
    for i in range(len(links)):
        links[i] = ''.join(x + '/' for x in links[i].split('/')[:-1]) + 'reviews'
    driver.quit()
    return links


def expand_spoilers(driver):
    # Функция для раскрытия всех спойлеров на текущей итерации загрузки страницы
    try:
        spoiler_buttons = WebDriverWait(driver, 1).until(
            EC.presence_of_all_elements_located(
                (By.CSS_SELECTOR, 'button.review-spoiler-button')
            )
        )
        for button in spoiler_buttons:
            driver.execute_script("arguments[0].scrollIntoView();", button)
            try:
                button.click()
            except ElementClickInterceptedException:
                pass
    except TimeoutException:
        pass


def scroll_to_element(driver, element):
    # Функция для прокрутки к элементу
    ActionChains(driver).scroll_to_element(element).perform()


def write_into_train(review_rating, review_text):
    # Функция для записи новой строки в train-файле
    with open('train_reviews.csv', 'a', newline='', encoding='utf-8') as file:
        wr = csv.DictWriter(file, fieldnames=['rating', 'text'])
        wr.writerow({'rating': review_rating, 'text': review_text})


def write_into_test(review_rating, review_text):
    # Функция для записи новой строки в test-файле
    with open('test_reviews.csv', 'a', newline='', encoding='utf-8') as file:
        wr = csv.DictWriter(file, fieldnames=['rating', 'text'])
        wr.writerow({'rating': review_rating, 'text': review_text})


def check_if_full() -> bool:
    # Функция для проверки заполненности датасета
    global reviews_amount_train, reviews_amount_test
    for key in reviews_amount_train.keys():
        if reviews_amount_train[key] < 2000 or reviews_amount_test[key] < 2000:
            return False
    return True


def parse_reviews(driver, url) -> bool:
    # Основная функция для парсинга веб-страниц с отзывами
    global reviews_amount_train, reviews_amount_test
    reviews_found = 0
    processed_reviews_hrefs = set()  # Для отслеживания уже обработанных отзывов

    driver.get(url)
    last_height = driver.execute_script("return document.body.scrollHeight")

    # Переменная для равномерного заполнения файлов
    flag = True
    while True:
        expand_spoilers(driver)
        reviews = driver.find_elements(
            By.CSS_SELECTOR, 'article.user-review-item'
        )
        for review in reviews:
            scroll_to_element(driver, review)
            link_wrapper = review.find_element(By.CSS_SELECTOR, 'a.ipc-title-link-wrapper')
            review_href = link_wrapper.get_attribute("href").split('/')[-2]
            if review_href in processed_reviews_hrefs:
                continue  # Пропускаем уже обработанные
            else:
                processed_reviews_hrefs.add(review_href)
                reviews_found += 1

            # Парсинг рейтинга и текста
            try:
                rating = review.find_element(
                    By.CSS_SELECTOR,
                    'span.ipc-rating-star--rating'
                ).text.strip()

                text = review.find_element(
                    By.CSS_SELECTOR,
                    'div.ipc-html-content-inner-div'
                ).text.strip()

            except NoSuchElementException:
                continue  # Пропускаем отзывы без рейтинга

            rev_train = reviews_amount_train[rating]
            rev_test = reviews_amount_test[rating]

            if flag:
                if rev_train < 2000:
                    write_into_train(review_rating=rating, review_text=text)
                    reviews_amount_train[rating] += 1
                    # Меняем флаг для равномерного заполнения файлов
                    flag = False
                elif rev_test < 2000:
                    write_into_test(review_rating=rating, review_text=text)
                    reviews_amount_test[rating] += 1
            else:
                if rev_test < 2000:
                    write_into_test(review_rating=rating, review_text=text)
                    reviews_amount_test[rating] += 1
                    # Меняем флаг для равномерного заполнения файлов
                    flag = True
                elif rev_train < 2000:
                    write_into_train(review_rating=rating, review_text=text)
                    reviews_amount_train[rating] += 1

            if check_if_full():
                # Если мы закончили формировать датасет, прекращаем выполнение программы, возвращая False
                return False

        try:
            # Ищем кнопку More внизу страницы
            more_button = WebDriverWait(driver, 1).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "button.ipc-see-more__button"))
            )
            scroll_to_element(driver, more_button)
            more_button.click()

        except TimeoutException:
            print('Не найдена кнопка more, больше отзывов нет')
            break  # Больше нет отзывов

        # Прокручиваем страницу вниз
        driver.execute_script("window.scrollBy(0, 600);")
        time.sleep(1.5)  # Ждем загрузки новых отзывов

        # Проверяем, достигли ли конца страницы
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break  # Достигли самого конца страницы
        last_height = new_height

    driver.quit()
    # Если мы дошли до этого момента, значит, нужны ещё отзывы, поэтому возвращаем True для продолжения работы
    return True


if __name__ == '__main__':
    with open('train_reviews.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['rating', 'text'])
        writer.writeheader()

    with open('test_reviews.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['rating', 'text'])
        writer.writeheader()

    # Сбор URL худших сериалов
    worst_serials = get_reviews_pages(driver=setup_driver(), url='https://www.imdb.com/list/ls063837343/')
    # Сбор URL лучших сериалов
    best_serials = get_reviews_pages(driver=setup_driver(), url='https://www.imdb.com/chart/toptv/')
    # Сбор URL худших фильмов
    worst_movies = get_reviews_pages(driver=setup_driver(), url='https://www.imdb.com/chart/bottom/')

    all_links = worst_serials + best_serials + worst_movies

    reviews_amount_train = {'1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, '8': 0, '9': 0, '10': 0}
    reviews_amount_test = {'1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, '8': 0, '9': 0, '10': 0}

    for link in all_links:
        print(link)
        go_next = parse_reviews(driver=setup_driver(), url=link)
        print(f'Train: {reviews_amount_train},\nTest: {reviews_amount_test}\n___________________')
        if not go_next:
            print('Формирование датасета закончено')
            break

    print('Работа программы закончена. Удалось получить отзывов:\n'
          f'Для обучения: {reviews_amount_train},\n'
          f'Для оценки: {reviews_amount_test}\n'
          )
