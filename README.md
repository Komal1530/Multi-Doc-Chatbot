# Document Chat with Ollama (PDF + Excel/CSV)

Chat with your documents locally using Ollama. The app supports:
- PDF files
- Spreadsheets (`.xlsx`, `.xls`, `.csv`)

Runs locally with no OpenAI API key required.

## Project Requirements

### System requirements
- Python `3.10+`
- [Ollama](https://ollama.com/) installed
- Ollama service running locally

### Python package requirements
From `requirements.txt`:

- `streamlit>=1.30.0`
- `langchain>=0.3.0`
- `langchain-classic>=0.3.0`
- `langchain-ollama>=0.2.0`
- `langchain-community>=0.3.0`
- `langchain-text-splitters>=0.3.0`
- `pypdf>=4.0.0`
- `faiss-cpu>=1.8.0`
- `python-dotenv>=1.0.0`
- `pandas>=2.0.0`
- `openpyxl>=3.1.0`

### Ollama model requirements
Pull these models before using the app:

- Chat model (default): `llama3.2`
- Embedding model (default): `nomic-embed-text`


Install them with:

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
ollama pull llava
```

## Setup

1. Clone/open this project folder.
2. Install Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Start Ollama (if not already running).
4. Pull required Ollama models (commands above).

## Run the project

Start Streamlit:

```bash
streamlit run app.py
```

If default port is busy, run on another port:

```bash
streamlit run app.py --server.port 8512
```

Then open the URL shown in terminal (for example `http://localhost:8512`).

## How to use

1. In the sidebar, choose:
   - Chat model
   - Embedding model
   - Vision model (for images)
2. Upload any mix of files:
   - PDFs
   - Excel/CSV
3. Click **Process**.
4. Ask questions in the chat input.

## Troubleshooting

- **Model not found (404)**
  - Pull the missing model using `ollama pull <model-name>`
  - Example: `ollama pull nomic-embed-text`

- **Port already in use**
  - Run Streamlit with a different port:
    - `streamlit run app.py --server.port 8513`

- **No response from models**
  - Check Ollama is running and mostreamlit run app.py
dels exist:
    - `ollama list`


