
# Cortex Waves Chatbot with Retrieval-Augmented Generation (RAG) Using FAISS, AzureOpenAI, and Streamlit

Cortex Waves is a Streamlit application that leverages Azure OpenAI and FAISS to process PDF documents and provide chat functionality based on the document content.

### Solution Name
Cortex Waves Chatbot
### Solution Description
Cortex Waves is a question-answering chatbot designed to provide efficient and accurate responses to user queries based on the content of uploaded PDF documents. This solution addresses the challenge of quickly retrieving relevant information from large documents, making it easier for users to find the answers they need.
### Solution Features

- **PDF Upload**: Users can upload a PDF document.
- **Interactive Q&A**: Users can ask questions about the content of the uploaded PDF and receive accurate answers.
- **Chat History**: Maintains a history of user interactions for context-aware responses.
Clear Chat History: Option to clear the chat history and reset the chatbot.

### Technologies and Architecture Used

- **FAISS**: Facebook’s AI Similarity Search (FAISS) is a library designed for efficient similarity search and clustering of dense vectors, which is perfect for large-scale search tasks.
- **AzureOpenAI**: Employed for embedding and language model interactions..
- **Streamlit**: An open-source Python library that lets you build interactive web applications quickly.
- **PyPDF2**: A Python library for extracting text from PDF files.

### Architecture
<img src="rag.jpg" height="600" width="1200" >

- PDF Processing: Uploaded PDFs are processed using PyPDF2 to extract text.
- Text Chunking: Extracted text is chunked into manageable pieces.
- Embedding Generation: AzureOpenAI generates embeddings for each text chunk.
- FAISS Indexing: Embeddings are indexed using FAISS for efficient similarity search.
- Interactive Q&A: Users interact with the chatbot via Streamlit, asking questions and receiving context-aware answers.

### Code Purpose
The code is designed to create an interactive chatbot that can understand and respond to questions based on the content of uploaded PDF documents. It combines various technologies to ensure efficient processing, embedding, and retrieval of relevant information.


```
Process
PDF Source -> Chunks -> Embeddings(openAI) -> Vector Store(faiss/vectordb/pinedb) -> ranked results
