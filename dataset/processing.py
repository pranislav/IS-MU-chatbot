from typing import Generator, NamedTuple, Optional
from pathlib import Path

import bs4
import diskcache
import json
import requests


BASE_URL = 'https://is.muni.cz'
HELP_URL_CZ = f'{BASE_URL}/help/?lang=cz'
HELP_URL_EN = f'{BASE_URL}/help/?lang=en'

PROJECT_DIR = Path(__file__).parent
CZ_FILE = PROJECT_DIR / 'data_cz.json'
EN_FILE = PROJECT_DIR / 'data_en.json'


CACHE = diskcache.Cache('cache')


class Topic(NamedTuple):
    class Question(NamedTuple):
        url: str
        title: str
        answer: str
        has_image: bool

    category: str
    topic: str
    questions: list[Question]


def load_page_body(url: str) -> Optional[bs4.Tag]:
    if '.pdf' in url:
        return None

    if url in CACHE:
        page = CACHE[url]
    else:
        page = requests.get(url)
        CACHE[url] = page

    assert isinstance(page, requests.Response)

    page_html = bs4.BeautifulSoup(page.text, 'html.parser').body
    result = page_html if isinstance(page_html, bs4.Tag) else None

    return result


def find_categories(tag: bs4.Tag) -> Generator[tuple[str, bs4.Tag]]:
    categories = tag.find('div', class_='napoveda_index')
    assert isinstance(categories, bs4.Tag)

    for category in categories.find_all('div', class_='napoveda_index_box'):
        assert isinstance(category, bs4.Tag)

        name_tag = category.find('h2')
        assert isinstance(name_tag, bs4.Tag)

        yield name_tag.text, category


def find_topics(tag: bs4.Tag) -> Generator[tuple[str, bs4.Tag, str]]:
    topics = tag.find('ul')
    assert isinstance(topics, bs4.Tag)

    for topic in topics.find_all('li'):
        assert isinstance(topic, bs4.Tag)

        name_tag = topic.find('a')
        assert isinstance(name_tag, bs4.Tag)

        url = f'{BASE_URL}{name_tag['href']}'

        try:
            body = load_page_body(url)
        except requests.ConnectionError:
            print(f'[ERROR] Could not load page: {url}.')
            continue

        if body is None:
            print(f'[ERROR] Invalid page: {url}.')
            continue

        yield name_tag.text, body, url




def find_questions(tag: bs4.Tag, base_url: str) -> Generator[Topic.Question]:
    questions = tag.find_all('li', class_='accordion-item')

    if len(questions) == 0:
        print(f'[WARNING] Page missing questions: {base_url}.')

    for question in questions:
        assert isinstance(question, bs4.Tag)

        header_tag = tag.find('a', class_='accordion-title')
        assert isinstance(header_tag, bs4.Tag)

        number_tag = header_tag.find('span')
        assert isinstance(number_tag, bs4.Tag)

        answer_tags = list(tag.find_all('div', class_='accordion-content'))

        url = f'{base_url}#{question['id']}'
        question = header_tag.text.removeprefix(number_tag.text)
        has_image = question.find('img') != None
        answer = '\n'.join(a.get_text(separator='\n') for a in answer_tags)

        if len(answer_tags) != 1:
            print(f'[INFO] Multiple content elements: {url}.')

        yield Topic.Question(url=url,
                             title=question,
                             answer=answer,
                             has_image=has_image)


def process_help(url: str) -> Generator[Topic]:
    help_page_html = load_page_body(url)

    if help_page_html is None:
        print(f'[ERROR] Missing categories: {url}.')
        return

    for category_name, category_tag in find_categories(help_page_html):
        for topic_name, topic_tag, topic_url in find_topics(category_tag):
            yield Topic(category=category_name,
                              topic=topic_name,
                              questions=list(find_questions(topic_tag,
                                                            topic_url)))


def main() -> None:
    for file, url in ((CZ_FILE, HELP_URL_CZ), (EN_FILE, HELP_URL_EN)):
        topics = list(process_help(url))

        data = [
            {
                'category': topic.category,
                'topic': topic.topic,
                'questions': list(map(lambda x: x._asdict(), topic.questions))
            }
            for topic in topics
        ]

        file.write_text(json.dumps(data, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
