from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.readers.json import JSONReader
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.prompts import PromptTemplate


documents = JSONReader().load_data("dataset/transformed_for_llamaindex.json")
PERSIST_DIR = "./dataset/index"

model_name = "google/gemma-3-4b-it"


class E5Embedding(HuggingFaceEmbedding):
    def _get_query_embedding(self, query: str):
        return super()._get_query_embedding(f"query: {query}")

    def _get_text_embedding(self, text: str):
        return super()._get_text_embedding(f"passage: {text}")


def load_or_create_index(PERSIST_DIR, documents):
    if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        print("🔄 Loading existing index...")
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        embed_model = E5Embedding(model_name="intfloat/multilingual-e5-base")
        index = load_index_from_storage(storage_context, embed_model=embed_model)
    else:
        print("✨ Creating new index...")
        index = VectorStoreIndex.from_documents(
            documents,
            embed_model=embed_model,
        )
        index.storage_context.persist(persist_dir=PERSIST_DIR)
        print("index created")
    return index


def format_prompt(query, context_str, tokenizer):
    system_msg = '''Jsi nápomocný chatbot Masarykovy univerzity. Tvým úkolem je pomáhat uživatelům orientovat se v Informačním systému (IS MU) a poskytovat rady, jak provést požadované akce v systému.
    Níže jsou oficiální dokumenty nápovědy IS MU, které mohou obsahovat užitečné informace:
    {context_str}
    Pokud informace ve zdrojích nejsou dostatečné, řekni to upřímně.'''
    
    messages = [
        {"role": "system", "content": system_msg.format(context_str=context_str)},
        {"role": "user", "content": query}
    ]
    
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def query_is_muni(query, index, tokenizer, pipeline):
    # Retrieve documents (get context)
    retriever = index.as_retriever(similarity_top_k=3)

    retrieved_nodes = retriever.retrieve(query)
    context_str = "\n\n".join([n.node.get_content() for n in retrieved_nodes])
    
    # Format the prompt
    formatted_prompt = format_prompt(query, context_str, tokenizer)
    
    # Generate response
    response = pipeline(formatted_prompt, max_length=100, do_sample=True)["generated_text"]
    
    return response


tokenizer = AutoTokenizer.from_pretrained(model_name)
pipeline = pipeline("text-generation", model=model_name)
index = load_or_create_index(PERSIST_DIR, documents)
query = "Kde najdu muj rozvrh?"

response = query_is_muni(query, index, tokenizer, pipeline)
print(response)
