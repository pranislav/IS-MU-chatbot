# Dataset
All pages were scraped from [Help](https://is.muni.cz/napoveda/?lang=en).

## Data Filtering
There were numerous problems with provided data that must be resolved to
obtain usable training data for our model:
 - Some pages were not translated and were kept in czech.
 - One page is in pdf file format and does not follow the same structure.
 - Quetions can also contain some image or use some html styling.

## Format
Raw data are stored in a JSON file `raw.json` with following format:
```json
[
    {
        "category": "student",
        "topic": "Study Planner",
        "questions" : [
            {
                "number": 1,
                "question": "What is the Study Planner?",
                "answer": "Study Planner is an ...",
                "has_image": false
            }
        ]
    }
]```

## TODO
 - [ ] Extract Q&A dataset from the pages.
 - [ ] Translate extracted data.
