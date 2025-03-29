import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings
warnings.filterwarnings("ignore", message="Found missing adapter keys while loading the checkpoint") #TODO


# Chat function with chat template
def chat(model, tokenizer, question):
    messages = [{"role": "user", "content": question}]
    formatted_text = tokenizer.apply_chat_template(messages, tokenize=False)

    inputs = tokenizer(formatted_text, return_tensors="pt").to("cuda")
    output = model.generate(**inputs, max_new_tokens=100)
    return tokenizer.decode(output[0], skip_special_tokens=True)

def load_model(adapter_path=None):
    # Load model and tokenizer
    model_name = "google/gemma-3-4b-it"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load the model
    model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")

    # If an adapter path is provided, load the adapter
    if adapter_path:
        model.load_adapter(adapter_path, load_as="lora")
        model.set_active_adapters("lora")

    return model, tokenizer

def main():
    parser = argparse.ArgumentParser(description="Chat with the model. Provide path to LoRA adapter or leave empty for original model.")
    parser.add_argument("--adapter_path", type=str, default=None, help="Path to the LoRA adapter.")
    args = parser.parse_args()

    model, tokenizer = load_model(args.adapter_path)

    print("Start chatting with the model (type 'quit' to exit)...")

    while True:
        question = input()
        if question.lower() == 'quit':
            break
        answer = chat(model, tokenizer, question)
        print(answer)


if __name__ == "__main__":
    main()
