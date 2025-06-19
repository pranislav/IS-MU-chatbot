import json
import matplotlib.pyplot as plt
import argparse

def load_data(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
        return json.load(f)

def compute_lengths(data):
    question_lengths = [len(qa["question"]) for qa in data]
    answer_lengths = [len(qa["answer"]) for qa in data]
    return question_lengths, answer_lengths

def plot_histogram(lengths, title, xlabel, filename):
    plt.figure()
    plt.hist(lengths, bins=20, edgecolor='black', alpha=0.7)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.title(title)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(filename)
    plt.show()

def main(json_file):
    data = load_data(json_file)
    question_lengths, answer_lengths = compute_lengths(data)
    
    plot_histogram(question_lengths, "Question Length Distribution", "Question Length", "question_len_histogram.png")
    plot_histogram(answer_lengths, "Answer Length Distribution", "Answer Length", "answer_len_histogram.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot histograms for question and answer lengths.")
    parser.add_argument("json_file", help="Path to the JSON file containing question-answer pairs.")
    args = parser.parse_args()
    main(args.json_file)
