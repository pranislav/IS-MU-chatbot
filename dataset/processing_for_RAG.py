import json

with open("data_cz.json", "r", encoding="utf-8") as f:
    raw = json.load(f)

entries = []
i = 0
for item in raw:
    category = item["category"]
    topic = item["topic"]
    for q in item["questions"]:
        entries.append({
            "text": (
                f"Kategorie: {category}\n"
                f"Téma: {topic}\n"
                f"Otázka: {q['title']}\n"
                f"Odpověď: {q['answer']}"
            ),
            "metadata": {
                "id": i,
                "category": category,
                "topic": topic,
                "title": q["title"],
                "url": q["url"]
            }
        })
        i += 1

with open("transformed_for_llamaindex.json", "w", encoding="utf-8") as f:
    json.dump(entries, f, ensure_ascii=False, indent=2)
