# Cortex Waves Chatbot with Retrieval-Augmented Generation (RAG) Using FAISS, AzureOpenAI, and Streamlit
This project demonstrates the creation of a question-answering chatbot. The chatbot leverages several technologies to provide efficient and accurate responses to user queries based on the content of uploaded PDF documents.
### Architecture
<img src="rag.jpg" height="600" width="1200" >

### Key Libraries and Technologies

- **FAISS**: Facebook’s AI Similarity Search (FAISS) is a library designed for efficient similarity search and clustering of dense vectors, which is perfect for large-scale search tasks.
- **AzureOpenAI**: Employed for embedding and language model interactions..
- **Streamlit**: An open-source Python library that lets you build interactive web applications quickly.
- **PyPDF2**: A Python library for extracting text from PDF files.

```
Process
PDF Source -> Chunks -> Embeddings(openAI) -> Vector Store(faiss/vectordb/pinedb) -> ranked results
