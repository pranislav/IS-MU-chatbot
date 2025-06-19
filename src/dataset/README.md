# Dataset (WIP)
All pages were scraped from [Help](https://is.muni.cz/napoveda/?lang=en).
There are numerous cavities wich must be resolved during data cleansing:
 - One page is in a different format, so we cannot load it using this script.
 - Some pages marked as English are in Czech.
 - Some answers include images.
 - Some answers have some visual styling.

## Raw Format
Final JSON format of raw parsed dataset is:
```json
[
    {
        "category": "Student",
        "topic": "Study Planner",
        "questions": [
            {
                "url": "https://is.muni.cz/...",
                "title": "What is the Study Planner?",
                "answer": "Study Planner is an...",
                "has_image": false
            }
        ]
    }
]
```
