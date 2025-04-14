from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.readers.json import JSONReader
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.prompts import PromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


model_name = "google/gemma-3-4b-it"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")


# Connect to LlamaIndex
llm = HuggingFaceLLM(
    model=model,
    tokenizer=tokenizer,
    context_window=8192,
    max_new_tokens=512,
    generate_kwargs={"do_sample": False, "top_k": None, "top_p": None},
    # system_prompt="You are a helpful university assistant."
)


class E5Embedding(HuggingFaceEmbedding):
    def _get_query_embedding(self, query: str):
        return super()._get_query_embedding(f"query: {query}")

    def _get_text_embedding(self, text: str):
        return super()._get_text_embedding(f"passage: {text}")

embed_model = E5Embedding(model_name="intfloat/multilingual-e5-base")

PERSIST_DIR = "./dataset/index"

if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
    print("🔄 Loading existing index...")
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context, embed_model=embed_model)
else:
    print("✨ Creating new index...")
    documents = JSONReader().load_data("dataset/transformed_for_llamaindex.json")
    index = VectorStoreIndex.from_documents(
        documents,
        embed_model=embed_model,
    )
    index.storage_context.persist(persist_dir=PERSIST_DIR)
    print("index created")

cz_prompt = PromptTemplate(
    "Jsi nápomocný chatbot Masarykovy univerzity. Tvým úkolem je pomáhat uživatelům orientovat se v Informačním systému (IS MU) a poskytovat rady, jak provést požadované akce v systému.\n\n"
    "Níže jsou oficiální dokumenty nápovědy IS MU, které mohou obsahovat užitečné informace:\n"
    "{context_str}\n\n"
    "Dotaz uživatele: {query_str}\n\n"
    "Odpověz stručně, přesně a česky. Pokud informace ve zdrojích nejsou dostatečné, řekni to upřímně."
)

query_engine = index.as_query_engine(
    llm=llm,
    text_qa_template=cz_prompt)
response = query_engine.query("Kde najdu muj rozvrh?")
print(response)
