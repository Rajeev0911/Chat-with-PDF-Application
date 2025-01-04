# PDF Chat Assistant

The PDF Chat Assistant is a web-based tool built using **Streamlit**, **Hugging Face Transformers**, **Sentence Transformers**, and **Faiss**. It allows users to upload PDF documents, extract text, create embeddings, and interact with the documents using natural language questions. The assistant searches relevant chunks of text from the uploaded PDFs and provides answers, highlighting the relevant sections in the original PDF.

## Features

- Upload one or more PDF documents.
- Text extraction and chunking from PDFs.
- Search for similar content in the document using natural language queries.
- Display answers with relevant context from the documents.
- Highlight the relevant text in the original PDFs and allow downloading with highlights.

## Installation

To set up the project, follow the instructions below.

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/pdf-chat-assistant.git
   cd pdf-chat-assistant
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Setup API Key:

   ```bash
   API_KEY=your_api_key
   ```

4. Run the Streamlit App

   ```bash
   streamlit run main.py
   ```





