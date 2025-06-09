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


def write_into_file(file, review_rating, review_text):
    # Функция для записи новой строки в train-файле
    with open(f'{file}_reviews.csv', 'a', newline='', encoding='utf-8') as f:
        wr = csv.DictWriter(f, fieldnames=['rating', 'text'])
        wr.writerow({'rating': review_rating, 'text': review_text})


def get_reviews(driver, url) -> list:
    driver.get(url)
    processed_reviews_hrefs = set()  # Для отслеживания уже обработанных отзывов
    last_height = driver.execute_script("return document.body.scrollHeight")
    url_reviews = []

    while True:
        expand_spoilers(driver)
        review_elements = driver.find_elements(
            By.CSS_SELECTOR, 'article.user-review-item'
        )
        for review in review_elements:
            scroll_to_element(driver, review)
            link_wrapper = review.find_element(By.CSS_SELECTOR, 'a.ipc-title-link-wrapper')
            review_href = link_wrapper.get_attribute("href").split('/')[-2]
            if review_href in processed_reviews_hrefs:
                continue  # Пропускаем уже обработанные
            processed_reviews_hrefs.add(review_href)

            # Парсинг рейтинга и текста
            try:
                review_rating = review.find_element(
                    By.CSS_SELECTOR,
                    'span.ipc-rating-star--rating'
                ).text.strip()

                review_text = review.find_element(
                    By.CSS_SELECTOR,
                    'div.ipc-html-content-inner-div'
                ).text.strip()

            except NoSuchElementException:
                continue  # Пропускаем отзывы без рейтинга

            url_reviews.append((review_rating, review_text))

        try:
            # Ищем кнопку More внизу страницы
            more_button = WebDriverWait(driver, 1).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "button.ipc-see-more__button"))
            )
            scroll_to_element(driver, more_button)
            more_button.click()

        except TimeoutException:
            print('Не найдена кнопка more, больше отзывов нет')
            return url_reviews  # Больше нет отзывов

        # Прокручиваем страницу вниз
        driver.execute_script("window.scrollBy(0, 600);")
        time.sleep(1.5)  # Ждем загрузки новых отзывов

        # Проверяем, достигли ли конца страницы
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            driver.quit()  # Достигли самого конца страницы
            return url_reviews
        else:
            last_height = new_height


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

    collected_reviews = {str(i): [] for i in range(1, 11)}
    total_collected = 0
    driver = setup_driver()

    LIMIT_PER_RATING = 4000

    for link in all_links:
        if all(len(collected_reviews[str(i)]) >= LIMIT_PER_RATING for i in range(1, 11)):
            print('Формирование датасета закончено')
            break
        print(link)
        reviews = get_reviews(driver=driver, url=link)
        for (rating, text) in reviews:
            if len(collected_reviews[rating]) >= LIMIT_PER_RATING:
                continue
            else:
                collected_reviews[rating].append(text)

    # Равномерная запись в файлы
    for rating in collected_reviews:
        for i in range(len(collected_reviews[rating])):
            if i % 2 == 0:
                write_into_file(file='train', review_rating=rating, review_text=collected_reviews[rating][i])
            else:
                write_into_file(file='test', review_rating=rating, review_text=collected_reviews[rating][i])
