from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_huggingface.llms import HuggingFacePipeline
from transformers import pipeline
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import argparse
# re-create embedding
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# reload vectorstore
vector_store = FAISS.load_local("faiss_index_mlbook", embeddings=embedding_model, allow_dangerous_deserialization=True)

# LLM local
generator = pipeline("text2text-generation", model="google/flan-t5-base", max_new_tokens=512)
llm = HuggingFacePipeline(pipeline=generator)

# Prompt
custom_prompt_template = """Use the following information to answer user questions.
If you don't know the answer, just say that you don't know. Be concise and informative.
Answer in English.

Context: {context}
Question: {question}
"""
parser = argparse.ArgumentParser(description="Ask a question to the AI Agent")
parser.add_argument("question", type=str, help="Your question for the model")
args = parser.parse_args()
query = args.question
prompt = PromptTemplate(input_variables=["context", "question"], template=custom_prompt_template)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
    chain_type_kwargs={
        "prompt": prompt,
        "document_variable_name": "context"
    },
    return_source_documents=False  # Không trả về source
)


result = qa_chain.invoke({"query": query})
print("===Question===")
print(query)
print("=== Answer ===")
print(result["result"])

