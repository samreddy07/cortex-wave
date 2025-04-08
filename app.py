import streamlit as st
import faiss
import numpy as np
import json
import os
import PyPDF2
import requests
from bs4 import BeautifulSoup
from openai import AzureOpenAI  # Azure SDK
# === CONFIGURATION ===
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY", "85015946c55b4763bcc88fc4db9071dd")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "ada-002")
AZURE_OPENAI_COMPLETION_DEPLOYMENT = os.getenv("AZURE_OPENAI_COMPLETION_DEPLOYMENT", "gpt-4o-mini")
FAISS_INDEX_PATH = "faiss_index.bin"
FAISS_METADATA_PATH = "faiss_metadata.json"
# === Azure OpenAI Clients ===
client = AzureOpenAI(
   api_key=AZURE_OPENAI_KEY,
   api_version="2024-02-01",
   azure_endpoint="https://innovate-openai-api-mgt.azure-api.net/innovate-tracked/deployments/ada-002/embeddings?api-version=2024-02-01"
)
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
# Initialize FAISS
if "faiss_store" not in st.session_state:
   st.session_state.faiss_store = FAISSStore()
# === PDF PROCESSOR ===
def extract_text_from_pdf(pdf_file):
   text = ""
   pdf_reader = PyPDF2.PdfReader(pdf_file)
   for page in pdf_reader.pages:
       text += page.extract_text() + "\n"
   return text
# === WIKIPEDIA PROCESSOR ===
def extract_text_from_wikipedia(url):
   """Extracts text from a Wikipedia page given its URL."""
   try:
       response = requests.get(url)
       response.raise_for_status()
       soup = BeautifulSoup(response.text, "html.parser")
       # Remove unnecessary elements like scripts and styles
       for element in soup(["script", "style"]):
           element.decompose()
       text = soup.get_text(separator="\n")
       # Clean and normalize the text
       lines = [line.strip() for line in text.splitlines() if line.strip()]
       return "\n".join(lines)
   except Exception as e:
       st.error(f"Error fetching Wikipedia content: {e}")
       return ""
def get_embedding(text):
   response = client.embeddings.create(
       input=text,
       model=AZURE_OPENAI_EMBEDDING_DEPLOYMENT
   )
   return response.data[0].embedding
# === Streamlit UI ===
st.set_page_config(page_title="Cortex Wave", layout="wide")
# st.title("Cortex Wave: AI for Wiki and Document Exploration")
# === Sidebar: PDF Upload & Reset ===
with st.sidebar:
   st.header("Data store")
   # PDF Uploader
   uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
   if uploaded_file and not st.session_state.get("pdf_processed", False):
       with st.spinner("Processing PDF..."):
           text = extract_text_from_pdf(uploaded_file)
           chunks = [chunk.strip() for chunk in text.split("\n") if chunk.strip()]
           embeddings = [get_embedding(chunk) for chunk in chunks]
           st.session_state.faiss_store.add_embeddings(chunks, embeddings)
           st.session_state.pdf_processed = True
           st.success("✅ PDF processed and stored!")
    # Wikipedia URL input
   wiki_url = st.text_input("Enter Wikipedia URL")
   if wiki_url and not st.session_state.get("wiki_processed", False):
       with st.spinner("🔍 Processing Wikipedia content..."):
           wiki_text = extract_text_from_wikipedia(wiki_url)
           if wiki_text:
               wiki_chunks = [chunk for chunk in wiki_text.split("\n") if chunk.strip()]
               wiki_embeddings = [get_embedding(chunk) for chunk in wiki_chunks]
               faiss_store.add_embeddings(wiki_chunks, wiki_embeddings)
               st.session_state.wiki_processed = True
               st.success("✅ Wikipedia content processed and stored in FAISS!")
# === Initialize Chat History ===
if "chat_history" not in st.session_state:
   st.session_state.chat_history = []
# === Chat Input ===
st.header("Cortex Wave")
user_input = st.chat_input("Ask something about the PDF and wiki...")
if user_input:
   st.session_state.chat_history.append({"role": "user", "content": user_input})
   with st.spinner("Generating answer..."):
       query_embedding = get_embedding(user_input)
       relevant_chunks = st.session_state.faiss_store.search(query_embedding, top_k=3)
       context = "\n".join(relevant_chunks)
       messages = [
           {"role": "system", "content": "You are a helpful assistant."},
           {"role": "user", "content": f"Context: {context}\n\nQuestion: {user_input}"}
       ]
       response = chat_client.chat.completions.create(
           model=AZURE_OPENAI_COMPLETION_DEPLOYMENT,
           messages=messages,
           temperature=0.3,
       )
       answer = response.choices[0].message.content
       st.session_state.chat_history.append({"role": "assistant", "content": answer})
# === Display Chat History ===
for msg in st.session_state.chat_history:
   if msg["role"] == "assistant":
       st.chat_message("assistant").write(msg["content"])
   else:
       st.chat_message("user").write(msg["content"])
