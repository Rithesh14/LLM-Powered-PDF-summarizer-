# ChatGroq with Llama 3 Demo

This is a Streamlit-based RAG (Retrieval-Augmented Generation) application that allows you to chat with your PDF documents using Groq's Llama 3 model and local embeddings via Ollama.

## Prerequisites

1.  **Python 3.8+** installed.
2.  **Ollama** installed locally for vector embeddings.
    *   Download from [ollama.com](https://ollama.com/).
    *   Pull the embedding model: `ollama pull nomic-embed-text`
3.  **Groq API Key**.
    *   Get your API key from [Groq Console](https://console.groq.com/).

## Installation

1.  Clone this repository.
2.  Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3.  Create a `.env` file in the root directory and add your Groq API key:

    ```env
    GROQ_API_KEY=your_groq_api_key_here
    ```

## Usage

1.  **Start Ollama Server**:
    Open a terminal and run:
    ```bash
    ollama serve
    ```
    *Keep this terminal running.*

2.  **Run the Application**:
    Open a new terminal in the project directory and run:
    ```bash
    streamlit run app2.py
    ```

3.  **Interact with the App**:
    *   **Upload PDF**: Use the file uploader at the top to select your PDF documents.
    *   **Embed Documents**: Click the "Documents Embedding" button. Wait for the "vector store db is ready" message.
    *   **Chat**: Type your questions in the chat input box at the bottom. The app will answer based on the content of your uploaded PDFs.

## Troubleshooting

*   **ConnectionError**: If you see an error about connecting to Ollama, make sure `ollama serve` is running in a separate terminal window.
*   **Empty Responses**: Ensure your PDFs contain selectable text and are not just scanned images.
