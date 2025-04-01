from pathlib import Path

import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('datafile')
    args = parser.parse_args()
    data_file = Path(args.datafile)

    if not data_file.exists():
        print('Given file does not exists, terminating.')
        return

    data = json.loads(data_file.read_text())

    questions_count = sum((len(cat['questions'])
                           for cat in data))
    images_count = sum((que['has_image']
                        for cat in data
                        for que in cat['questions']))
    print(f'Number of questions: {questions_count}')
    print(f'Number of questions with images: {images_count}')


if __name__ == '__main__':
    main()
