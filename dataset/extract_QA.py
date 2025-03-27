import json

def extract_qa(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    qa_list = []
    for item in data:
        for q in item["questions"]:
            qa_list.append({
                "question": q["question"],
                "answer": q["answer"],
            })
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(qa_list, f, indent=4, ensure_ascii=False)

        
extract_qa("raw.json", "raw_QA.json")
