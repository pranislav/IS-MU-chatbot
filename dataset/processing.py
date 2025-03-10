# Don't mind this nasty code, too lazy to think now. ^^
import json
from pathlib import Path

import bs4

ROOT_DIR = Path(__file__).parent
SCRAP_DIR = ROOT_DIR / 'scrap'
RAW_FILE = ROOT_DIR / 'raw.json'


def find_question(tag: bs4.Tag) -> tuple[int, str]:
    for header in tag.find_all('a', class_='accordion-title'):
        if not isinstance(header, bs4.Tag):
            continue

        number_elem = header.find('span')

        if not isinstance(number_elem, bs4.Tag):
            continue

        number = int(number_elem.text[:-1])
        question = header.text.removeprefix(number_elem.text)

        return number, question

    assert False, f'Failed to parse {tag}'


def find_content(tag: bs4.Tag) -> str:
    content_elements = list(tag.find_all('div', class_='accordion-content'))

    if len(content_elements) != 1:
        print(f'[WARNING] Multiple content elements: "{content_elements}"')

    return '\n'.join(c.get_text(separator='\n') for c in content_elements)


def find_image(tag: bs4.Tag) -> bool:
    return tag.find('img') != None


def main():
    data = []

    for dir in SCRAP_DIR.iterdir():
        for x in dir.rglob('*.html'):
            name = x.name.removesuffix('.html')
            questions = []
            page = {
                    'category': dir.name,
                    'topic': name,
                    'questions': questions
            }

            # print(f'[INFO] Parsing page: "{name}"')

            soup = bs4.BeautifulSoup(x.read_text(), 'html.parser')

            for item in soup.find_all('li', class_='accordion-item'):
                assert isinstance(item, bs4.Tag)

                number, question = find_question(item)
                # print(f'[INFO] Parsing question: ({number}) {question}')

                content = find_content(item)
                has_image = find_image(item)

                questions.append({
                        'number': number,
                        'question': question,
                        'answer': content,
                        'has_image': has_image
                })

            data.append(page)

    RAW_FILE.write_text(json.dumps(data, indent=4))


if __name__ == '__main__':
    main()
