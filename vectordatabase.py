from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Corrected import
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os

load_dotenv()

# Upload & Load raw PDF(s)
pdfs_directory = "Data/"  # Save the data

def load_documents(folder_path):
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]
    documents = []
    for pdf in pdf_files:
        loader = PyPDFLoader(os.path.join(folder_path, pdf))
        documents.extend(loader.load())
    return documents

# Load documents before processing
documents = load_documents(pdfs_directory)  

# Create Chunks
def create_chunks(documents): 
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    return text_splitter.split_documents(documents)

text_chunks = create_chunks(documents)  # Now correctly defined

# Embedding model
embedding_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

# Index Documents & Store embeddings in FAISS (vector store)
FAISS_DB_PATH = "Vectorstore/db_faiss"
faiss_db = FAISS.from_documents(text_chunks,embedding=embedding_model)
faiss_db.save_local(FAISS_DB_PATH)
