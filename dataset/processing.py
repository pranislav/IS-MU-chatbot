from typing import Generator, NamedTuple, Optional
from pathlib import Path

import bs4
import diskcache
import json
import logging
import requests
import shutil


BASE_URL = 'https://is.muni.cz'
HELP_URL_CZ = f'{BASE_URL}/help/?lang=cz'
HELP_URL_EN = f'{BASE_URL}/help/?lang=en'

PROJECT_DIR = Path(__file__).parent
CACHE_DIR = PROJECT_DIR / 'cache'
CZ_FILE = PROJECT_DIR / 'data_cz.json'
EN_FILE = PROJECT_DIR / 'data_en.json'


LOGGER = logging.getLogger(__name__)


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
    cache = diskcache.Cache(CACHE_DIR.name)

    if '.pdf' in url:
        return None

    if url in cache:
        page = cache[url]
    else:
        page = requests.get(url)
        cache[url] = page

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
            logging.error(f'Could not load page: {url}.')
            continue

        if body is None:
            logging.error(f'Invalid page: {url}.')
            continue

        yield name_tag.text, body, url




def find_questions(tag: bs4.Tag, base_url: str) -> Generator[Topic.Question]:
    questions = tag.find_all('li', class_='accordion-item')

    if len(questions) == 0:
        logging.warning(f'Page missing questions: {base_url}.')

    for question in questions:
        assert isinstance(question, bs4.Tag)

        header_tag = tag.find('a', class_='accordion-title')
        assert isinstance(header_tag, bs4.Tag)

        number_tag = header_tag.find('span')
        assert isinstance(number_tag, bs4.Tag)

        answer_tag = question.find('div', class_='accordion-content')
        assert isinstance(answer_tag, bs4.Tag)

        url = f'{base_url}#{question['id']}'
        question = header_tag.text.removeprefix(number_tag.text)
        has_image = (answer_tag.find('img') != None)

        yield Topic.Question(url=url,
                             title=question,
                             answer=answer_tag.text,
                             has_image=has_image)


def process_help(url: str) -> Generator[Topic]:
    help_page_html = load_page_body(url)

    if help_page_html is None:
        logging.error(f'Missing categories: {url}.')
        return

    for category_name, category_tag in find_categories(help_page_html):
        for topic_name, topic_tag, topic_url in find_topics(category_tag):
            yield Topic(category=category_name,
                              topic=topic_name,
                              questions=list(find_questions(topic_tag,
                                                            topic_url)))


def user_choice(question: str) -> bool:
    for _ in range(3):
        match input(question + ' [y/n]: ').lower():
            case 'y': return True
            case 'n': return False
            case _: continue

    LOGGER.error('User did not choose an option, defaulting to False.')
    return False


def main() -> None:
    logging.basicConfig(filename='processing.log',
                        filemode='w',
                        level=logging.INFO)

    if (CACHE_DIR.is_dir()
        and user_choice('There are cached IS Help pages, should I delete them?')):
        shutil.rmtree(CACHE_DIR)
        logging.info('Removing cached IS Help pages.')

    logging.info('Found cached data, will use them.'
                 if CACHE_DIR.is_dir() else
                 'No cached data, will download and cache pages.')

    for file, url in ((CZ_FILE, HELP_URL_CZ), (EN_FILE, HELP_URL_EN)):
        data = [
            {
                'category': topic.category,
                'topic': topic.topic,
                'questions': list(map(lambda x: x._asdict(), topic.questions))
            }
            for topic in process_help(url)
        ]

        file.write_text(json.dumps(data, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
