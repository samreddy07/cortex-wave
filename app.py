import streamlit as st
import faiss
import wikipediaapi
import numpy as np
import json
import os
import PyPDF2
from sentence_transformers import SentenceTransformer
from openai import AzureOpenAI
# === CONFIGURATION ===
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY", "85015946c55b4763bcc88fc4db9071dd")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "ada-002")
AZURE_OPENAI_COMPLETION_DEPLOYMENT = os.getenv("AZURE_OPENAI_COMPLETION_DEPLOYMENT", "gpt-4o-mini")
FAISS_INDEX_PATH = "faiss_index.bin"
FAISS_METADATA_PATH = "faiss_metadata.json"
MODEL_NAME = "all-MiniLM-L6-v2"
# === Load embedding model ===
# @st.cache_resource
def load_model():
   return SentenceTransformer(MODEL_NAME)
model = load_model()
# === Azure OpenAI Clients ===
# Initialize Azure OpenAI client for embeddings
client = AzureOpenAI(
  api_key=AZURE_OPENAI_KEY,
  api_version="2024-02-01",
  azure_endpoint="https://innovate-openai-api-mgt.azure-api.net/innovate-tracked/deployments/ada-002/embeddings?api-version=2024-02-01"
)
# Initialize Azure OpenAI client for chat completions
chat_client = AzureOpenAI(
  api_key=AZURE_OPENAI_KEY,
  api_version="2024-02-01",
  azure_endpoint="https://innovate-openai-api-mgt.azure-api.net/innovate-tracked/deployments/gpt-4o-mini/chat/completions?api-version=2024-02-01"
)
# === FAISS Store ===
class FAISSStore:
   def __init__(self, embedding_dim=1536):
       self.embedding_dim = embedding_dim
       self.index = faiss.IndexFlatL2(self.embedding_dim)
       self.metadata = []
       if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(FAISS_METADATA_PATH):
           self.load_index()
   def add_embeddings(self, texts, embeddings):
       if not embeddings:
           return
       vectors = np.array(embeddings).astype("float32")
       self.index.add(vectors)
       self.metadata.extend(texts)
       self.save_index()
   def search(self, query_embedding, top_k=3):
       query_vector = np.array(query_embedding).astype("float32").reshape(1, -1)
       distances, indices = self.index.search(query_vector, top_k)
       results = [self.metadata[idx] for idx in indices[0] if idx < len(self.metadata)]
       return results
   def save_index(self):
       faiss.write_index(self.index, FAISS_INDEX_PATH)
       with open(FAISS_METADATA_PATH, "w") as f:
           json.dump(self.metadata, f)
   def load_index(self):
       self.index = faiss.read_index(FAISS_INDEX_PATH)
       with open(FAISS_METADATA_PATH, "r") as f:
           self.metadata = json.load(f)
   def clear(self):
       self.index = faiss.IndexFlatL2(self.embedding_dim)
       self.metadata = []
       if os.path.exists(FAISS_INDEX_PATH):
           os.remove(FAISS_INDEX_PATH)
       if os.path.exists(FAISS_METADATA_PATH):
           os.remove(FAISS_METADATA_PATH)
# === Utilities ===
def extract_text_from_pdf(pdf_file):
   text = ""
   pdf_reader = PyPDF2.PdfReader(pdf_file)
   for page in pdf_reader.pages:
       page_text = page.extract_text()
       if page_text:
           text += page_text + "\n"
   return text
# === Helpers ===
def chunk_text(text, max_length=500):
   sentences = text.split(". ")
   chunks = []
   current_chunk = ""
   for sentence in sentences:
       if len(current_chunk) + len(sentence) < max_length:
           current_chunk += sentence + ". "
       else:
           chunks.append(current_chunk.strip())
           current_chunk = sentence + ". "
   if current_chunk:
       chunks.append(current_chunk.strip())
   return chunks
def fetch_wikipedia_content(title):
   wiki_wiki = wikipediaapi.Wikipedia("en")
   page = wiki_wiki.page(title)
   if not page.exists():
       return None
   return page.text
def get_embedding(text):
   response = client.embeddings.create(
       input=text,
       model=AZURE_OPENAI_EMBEDDING_DEPLOYMENT
   )
   return response.data[0].embedding
# === Streamlit App ===
st.set_page_config(page_title="Cortex Wave", layout="wide")
st.title("Cortex Wave: AI for Wiki and Document Exploration")
# FAISS init
if "faiss_store" not in st.session_state:
   st.session_state.faiss_store = FAISSStore()
# Sidebar
with st.sidebar:
   st.header("Data Loader")
   if st.button("🔄 Reset All"):
       st.session_state.faiss_store.clear()
       st.session_state.clear()
       st.rerun()
   uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
   if uploaded_file and not st.session_state.get("pdf_processed", False):
       with st.spinner("Processing PDF..."):
           text = extract_text_from_pdf(uploaded_file)
           chunks = chunk_text(text)
           embeddings = [get_embedding(chunk) for chunk in chunks]
           st.session_state.faiss_store.add_embeddings(chunks, embeddings)
           st.session_state.pdf_processed = True
           st.success("✅ PDF processed and added to FAISS!")
   wiki_url = st.text_input("Wikipedia URL")
   if wiki_url and not st.session_state.get("wiki_processed", False):
       with st.spinner("Processing Wikipedia..."):
           wiki_text = fetch_wikipedia_content(wiki_url)
           if wiki_text:
               wiki_chunks = chunk_text(wiki_text)
               wiki_embeddings = model.encode(wiki_chunks)
               st.session_state.faiss_store.add_embeddings(wiki_chunks, wiki_embeddings)
               st.session_state.wiki_processed = True
               st.success("✅ Wikipedia content added to FAISS!")
# Chat functionality
if "chat_history" not in st.session_state:
   st.session_state.chat_history = []
st.header("Ask me anything about the PDF or Wiki")
user_input = st.chat_input("Your question...")
if user_input:
   st.session_state.chat_history.append({"role": "user", "content": user_input})
   with st.spinner("Thinking..."):
       query_embedding = get_embedding(user_input)
       relevant_chunks = st.session_state.faiss_store.search(query_embedding, top_k=3)
       context = "\n".join(relevant_chunks)
       messages = [
           {"role": "system", "content": "You are a helpful assistant answering based on provided context."},
           {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_input}"}
       ]
       response = chat_client.chat.completions.create(
           model=AZURE_OPENAI_COMPLETION_DEPLOYMENT,
           messages=messages,
           temperature=0.3
       )
       answer = response.choices[0].message.content
       st.session_state.chat_history.append({"role": "assistant", "content": answer})
# Show chat history
for msg in st.session_state.chat_history:
   with st.chat_message(msg["role"]):
       st.markdown(msg["content"])
