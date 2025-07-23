# from langchain_community.document_loaders import UnstructuredPDFLoader
# from langchain_community.vectorstores import FAISS
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
# import os
# import nltk
# nltk.download('punkt_tab')
# nltk.download('averaged_perceptron_tagger_eng')

# # ocr
# pdf_path = "./data/PCRNet_ Parent–Child Relation Network for automatic polyp segmentation.pdf"
# loader = UnstructuredPDFLoader(pdf_path, mode="elements")  # mode="elements" để chia nhỏ nội dung theo đoạn
# documents = loader.load()

# # text_split
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=500,
#     chunk_overlap=50,
#     length_function=len
# )

# chunks = text_splitter.split_documents(documents)
# clean_chunks = [chunk.page_content for chunk in chunks if chunk.page_content.strip() != ""]

# # embedding
# embedding_model = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-MiniLM-L6-v2"
# )

# # faiss vector store
# vector_store = FAISS.from_texts(clean_chunks, embedding=embedding_model)

# # save
# vector_store.save_local("faiss_index_mlbook")

# print("✅ Vector store đã được tạo và lưu thành công!")

import os
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 1. Tải tất cả PDF trong thư mục ./data/
pdf_folder = "./data"
pdf_files = [os.path.join(pdf_folder, f) for f in os.listdir(pdf_folder) if f.endswith(".pdf")]

# 2. Load và gộp nội dung tất cả PDF
all_docs = []
for file in pdf_files:
    loader = UnstructuredPDFLoader(file)
    docs = loader.load()
    all_docs.extend(docs)

# 3. Tách văn bản thành từng đoạn nhỏ (chunk)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = text_splitter.split_documents(all_docs)

# 4. Khởi tạo mô hình embedding
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 5. Tạo FAISS vector store và lưu lại
vector_store = FAISS.from_documents(split_docs, embedding=embedding_model)
vector_store.save_local("faiss_index_multi_pdf")

print(f"✅ Processed {len(pdf_files)} PDFs and saved vector store.")
