# Creating an AI-Powered Chatbot with Retrieval-Augmented Generation (RAG) Using FAISS, LangChain, and Streamlit

In today’s AI-driven world, one exciting application is a chatbot capable of answering questions based on the content of uploaded PDF documents. This type of chatbot can be used in various industries—legal, educational, and business—where quick retrieval of information from large documents is essential.

In this article, we’ll walk through the creation of a question-answering chatbot using FAISS for efficient search, LangChain for embedding and language model interactions, Streamlit for an interactive interface, and PyPDF2 for PDF handling. The chatbot will allow users to upload a PDF document, ask questions about its content, and retrieve accurate answers.

### Architecture
<img src="rag.jpg" height="600" width="1200" >

### Key Libraries and Technologies

- **FAISS**: Facebook’s AI Similarity Search (FAISS) is a library designed for efficient similarity search and clustering of dense vectors, which is perfect for large-scale search tasks.
- **LangChain**: An open-source library for building language model-driven applications, LangChain simplifies the handling of embeddings, chains, and interactions with language models.
- **Streamlit**: An open-source Python library that lets you build interactive web applications quickly.
- **PyPDF2**: A Python library for extracting text from PDF files.

### Prerequisites

Before we dive into the code, ensure you have the following libraries installed:

```bash
pip install PyPDF2 streamlit langchain faiss-cpu langchain_openai python-dotenv
```

We’ll also need an OpenAI API key for generating embeddings and performing text processing, which we’ll keep secure by storing it in a `.env` file.

### Step 1: Set Up Project Files

1. **Create a `.env` File**: This file will store the OpenAI API key securely, which we’ll load in our code without exposing it in version control.

   ```plaintext
   OPENAI_API_KEY=your_openai_api_key
   ```

   Don’t forget to add `.env` to your `.gitignore` file to avoid committing sensitive information to your repository.

2. **Create the main Python file**: We’ll name this `pdf_chatbot.py` and write our code here.

### Step 2: Code Walkthrough

Here’s the complete code with explanations:

```python
import os
import PyPDF2
import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI

# Load environment variables from .env file
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Path to save/load FAISS index
FAISS_INDEX_PATH = "faiss_index"

# Initialize embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=OPENAI_API_KEY)

# Initialize Streamlit app
st.header("Retrieval-Augmented Generation (RAG) based AI-Powered PDF Question-Answering Chatbot")
with st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader("Upload a PDF file and start asking questions", type="pdf")

# Function to extract text from PDF
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page_num in range(len(reader.pages)):
        page = reader.pages[page_num]
        text += page.extract_text()
    return text

# Check if FAISS index exists
vector_store = None
if os.path.exists(FAISS_INDEX_PATH):
    # Load the existing FAISS index
    vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    st.write("Loaded existing FAISS index.")

# Process uploaded PDF
if file:
    text = extract_text_from_pdf(file)
    splitter = RecursiveCharacterTextSplitter(
        separators="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = splitter.split_text(text)
    st.write(chunks)
    st.write(f"Total chunks created: {len(chunks)}")

    # Create new FAISS index if not already loaded
    if vector_store is None:
        vector_store = FAISS.from_texts(chunks, embeddings)
        vector_store.save_local(FAISS_INDEX_PATH)
        st.write("Created and saved new FAISS index with uploaded PDF.")

# Allow question input if vector store is available
if vector_store is not None:
    question = st.text_input("Ask a question")

    # Perform similarity search when user asks a question
    if question:
        question_embedding = embeddings.embed_query(question)
        match = vector_store.similarity_search_by_vector(question_embedding)

        llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0, max_tokens=1000, model="gpt-3.5-turbo")
        qa_chain = load_qa_chain(llm, chain_type="stuff")
        answer = qa_chain.run(input_documents=match, question=question)
        st.write(answer)
else:
    st.write("Please upload a PDF to create or load the FAISS index.")

```

### Explanation of the Code

1. **Environment Setup**: We load the OpenAI API key from the `.env` file for embedding generation.

2. **Uploading and Reading the PDF**: 
   - Users can upload a PDF file, which will be processed with `PyPDF2` to extract text.
   - The extracted text is then split into manageable chunks using `RecursiveCharacterTextSplitter`.

3. **Embedding and FAISS Index**:
   - For a new PDF upload, embeddings are generated using `OpenAIEmbeddings` from LangChain, and a FAISS index is created and saved.
   - If the FAISS index file already exists, it is loaded, allowing users to ask questions even if they don’t upload a new document.

4. **Question Input and Search**:
   - Users can ask questions related to the uploaded PDF, which are converted into embeddings and queried against the FAISS index.
   - The top matches are retrieved and fed into the language model for answering, with responses displayed in Streamlit.

### Key Features

- **Persistent Storage**: The FAISS index is saved locally, allowing the chatbot to retrieve answers even across different sessions.
- **Secure API Key Management**: With `dotenv`, sensitive information like API keys is kept secure.
- **Streamlit Interface**: Streamlit’s user-friendly interface makes it easy for non-technical users to interact with the chatbot.

### Running the Chatbot

To run the chatbot, use:

```bash
streamlit run pdf_chatbot.py
```

This command will open a local web server where you can upload a PDF, ask questions, and see the answers based on the document’s content.

### Conclusion

This AI-powered PDF Q&A chatbot demonstrates how FAISS and LangChain can be combined with Streamlit to create a useful tool for information retrieval. It’s scalable, secure, and perfect for handling large documents with ease, making it a valuable asset for any organization that deals with significant amounts of text data. 

Explore further by adding features like support for multiple file formats, conversational memory, or multi-document querying to make your chatbot even more robust.

## How To Run
Creating a `requirements.txt` file in VS Code for a Python project is straightforward. Here's a step-by-step guide:

1. **Open Your Project in VS Code**:
   - Launch VS Code and open the folder containing your Python project.

2. **Open the Integrated Terminal**:
   - You can open the terminal in VS Code by selecting `Terminal` from the top menu and then `New Terminal`, or by using the shortcut ``Ctrl+` `` (backtick).

3. **Activate Your Virtual Environment** (if you have one):
   - If you’re using a virtual environment, make sure it’s activated. For example:
     - On Windows: `venv\Scripts\activate`
     - On macOS/Linux: `source venv/bin/activate`

4. **Generate `requirements.txt`**:
   - Run the following command in the terminal to generate the `requirements.txt` file based on your current environment's installed packages:
     ```bash
     pip3 freeze > requirements.txt
     ```

5. **Verify `requirements.txt`**:
   - Open the `requirements.txt` file in the VS Code editor to verify that it contains the list of packages and their versions.

6. **Edit `requirements.txt`** (if necessary):
   - You can manually edit `requirements.txt` in VS Code if you need to add or remove specific packages or versions.

That’s it! You’ve created and verified your `requirements.txt` file. This file can now be used to install the required packages in other environments by running:
```bash
pip3 install -r requirements.txt
```

7. **Create a `.env` File**:
   Add the API key in a `.env` file (e.g., named `.env`), which will not be committed to Git. Inside `.env`, add the following line:

   ```
   OPENAI_API_KEY=sk-proj-byLvK-Dummy-gA
   ```

8. **Run**:
```
python3 chatbot.py
streamlit run chatbot.py
```
Process
PDF Source -> Chunks -> Embeddings(openAI) -> Vector Store(faiss/vectordb/pinedb) -> ranked results
