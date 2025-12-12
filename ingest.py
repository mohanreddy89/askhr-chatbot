# ingest.py - uses FREE HuggingFace embeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os

load_dotenv()

print("Loading your 5 HR PDFs...")
loader = PyPDFDirectoryLoader("data/")
docs = loader.load()
print(f"Loaded {len(docs)} pages")

print("Splitting...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

print("Creating FREE embeddings & uploading to Pinecone...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
PineconeVectorStore.from_documents(splits, embeddings, index_name="hr-bot")

print("SUCCESS! Your AskHR bot is ready (100% FREE)!")