import streamlit as st
import faiss
import numpy as np
import json
import os
import PyPDF2
from openai import AzureOpenAI
# === Set Page Configuration First ===
st.set_page_config(page_title="🦙💬 Cortex Waves")
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
# === Utilities ===
def extract_text_from_pdf(pdf_file):
   text = ""
   pdf_reader = PyPDF2.PdfReader(pdf_file)
   for page in pdf_reader.pages:
       page_text = page.extract_text()
       if page_text:
           text += page_text + "\n"
   return text
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
def clear_chat_history():
    st.session_state.faiss_store.clear()
    st.session_state.clear()
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
def get_embedding(text):
   response = client.embeddings.create(
       input=text,
       model=AZURE_OPENAI_EMBEDDING_DEPLOYMENT
   )
   return response.data[0].embedding
# === Initialize or Load FAISS Store ===
if "faiss_store" not in st.session_state:
   st.session_state.faiss_store = FAISSStore()
# === Initialize Chat History and Messages BEFORE Rendering Chat Messages ===
if "chat_history" not in st.session_state:
   st.session_state.chat_history = []
if "messages" not in st.session_state:
   st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
# === Now Render the Chat History and Messages ===
for msg in st.session_state.chat_history:
   with st.chat_message(msg["role"]):
       st.markdown(msg["content"])
for message in st.session_state.messages:
   with st.chat_message(message["role"]):
       st.write(message["content"])
# === Sidebar and PDF Upload Section ===
with st.sidebar:
   st.title('🦙💬 Cortex Waves Chatbot')
   st.write('This chatbot is created using the open-source Faiss LLM model from Meta.')
   uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
   if uploaded_file and not st.session_state.get("pdf_processed", False):
       with st.spinner("Processing PDF..."):
           text = extract_text_from_pdf(uploaded_file)
           chunks = chunk_text(text)
           embeddings = [get_embedding(chunk) for chunk in chunks]
           st.session_state.faiss_store.add_embeddings(chunks, embeddings)
           st.session_state.pdf_processed = True
           st.success("✅ PDF processed and added to FAISS!")
   st.markdown('📖 Learn how to build this app in this blog!')
   st.sidebar.button('Clear Chat History', on_click=clear_chat_history)
   # if st.sidebar.button('Clear Chat History'):
   #     st.session_state.faiss_store.clear()
   #     st.session_state.clear()
   #     st.rerun()
   #     st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
# === Response Generation ===
def generate_response(user_input):
   query_embedding = get_embedding(user_input)
   relevant_chunks = st.session_state.faiss_store.search(query_embedding, top_k=3)
   if relevant_chunks:
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
   else:
       answer = "No relevant information was found in the document. Please check your document or ask another question."
   st.session_state.chat_history.append({"role": "assistant", "content": answer})
   return answer
# === User Interaction with Chat Input ===
if prompt := st.chat_input(disabled=False):
   st.session_state.messages.append({"role": "user", "content": prompt})
   with st.chat_message("user"):
       st.write(prompt)
   with st.chat_message("assistant"):
       with st.spinner("Thinking..."):
           response = generate_response(prompt)
           placeholder = st.empty()
           full_response = ''
           for item in response:
               full_response += item
               placeholder.markdown(full_response)
           placeholder.markdown(full_response)
   st.session_state.messages.append({"role": "assistant", "content": full_response})
