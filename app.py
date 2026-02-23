import streamlit as st
import pandas as pd
import base64
from io import BytesIO, StringIO
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_core.messages import HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from templates import css, bot_template, user_template

load_dotenv()


def extract_pdf_text(pdf_files):
    """Extract text from uploaded PDF files."""
    text = ""
    for pdf in pdf_files:
        reader = PdfReader(pdf)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text


def extract_spreadsheet_text(spreadsheet_files):
    """Extract tabular text from CSV/XLS/XLSX files."""
    if not spreadsheet_files:
        return ""

    collected_text = []

    for uploaded_file in spreadsheet_files:
        file_name = uploaded_file.name
        file_bytes = uploaded_file.getvalue()
        if not file_bytes:
            continue

        lower_name = file_name.lower()
        try:
            if lower_name.endswith(".csv"):
                csv_text = file_bytes.decode("utf-8", errors="ignore")
                dataframe = pd.read_csv(StringIO(csv_text), dtype=str).fillna("")
                preview = dataframe.head(200).to_csv(index=False)
                collected_text.append(f"Spreadsheet file: {file_name}\n{preview}")
            else:
                workbook = pd.read_excel(BytesIO(file_bytes), sheet_name=None, dtype=str)
                for sheet_name, dataframe in workbook.items():
                    preview = dataframe.fillna("").head(200).to_csv(index=False)
                    collected_text.append(
                        f"Spreadsheet file: {file_name} | Sheet: {sheet_name}\n{preview}"
                    )
        except Exception as err:
            collected_text.append(
                f"Could not parse spreadsheet '{file_name}': {err}"
            )

    return "\n\n".join(collected_text)


def extract_image_text(image_files, vision_model):
    """Extract image context using an Ollama vision-capable model."""
    if not image_files:
        return ""

    cleaned_vision_model = (vision_model or "").strip() or "llava"
    vision_llm = ChatOllama(model=cleaned_vision_model, temperature=0)
    collected_text = []

    for image in image_files:
        image_bytes = image.getvalue()
        if not image_bytes:
            continue

        mime_type = image.type or "image/png"
        encoded_image = base64.b64encode(image_bytes).decode("utf-8")
        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": "Extract all useful information from this image for document QA. Include visible text, table values, headings, labels, and key visual facts in plain concise text.",
                },
                {
                    "type": "image_url",
                    "image_url": f"data:{mime_type};base64,{encoded_image}",
                },
            ]
        )

        try:
            response = vision_llm.invoke([message])
            image_context = response.content if isinstance(response.content, str) else str(response.content)
            collected_text.append(f"Image file: {image.name}\n{image_context}")
        except Exception as err:
            collected_text.append(
                f"Could not parse image '{image.name}' with model '{cleaned_vision_model}': {err}"
            )

    return "\n\n".join(collected_text)


def split_text_into_chunks(text):
    """Split text into smaller chunks for processing."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = splitter.split_text(text)
    return [chunk.strip() for chunk in chunks if chunk and chunk.strip()]


def create_vectorstore(text_chunks, embedding_model):
    """Create a FAISS vector store from text chunks."""
    if not text_chunks:
        raise ValueError("No valid text chunks were generated from the PDFs.")

    cleaned_model = (embedding_model or "").strip() or "nomic-embed-text"
    embeddings = OllamaEmbeddings(model=cleaned_model)
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def create_conversation_chain(vectorstore, llm_model):
    """Create a conversational retrieval chain."""
    llm = ChatOllama(model=llm_model, temperature=0.7)
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True,
        output_key='answer'
    )
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        return_source_documents=True
    )
    return chain


def handle_user_input(user_question):
    """Process user question and display chat history."""
    if st.session_state.conversation is None:
        st.warning("Please upload and process documents first.")
        return
    
    response = st.session_state.conversation({"question": user_question})
    st.session_state.chat_history = response['chat_history']
    
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    st.set_page_config(
        page_title="PDF Chat with Ollama",
        page_icon="📚",
        layout="wide"
    )
    st.write(css, unsafe_allow_html=True)
    
    # Initialize session state
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    
    st.header("📚 Chat with your documents")
    st.caption("Powered by local Ollama models")
    
    # Sidebar for file upload
    with st.sidebar:
        st.subheader("Your Documents")
        llm_model = st.selectbox(
            "Chat model",
            ["llama3.2"],
            index=0
        )
        embedding_model = st.text_input(
            "Embedding model",
            value="nomic-embed-text"
        )
        vision_model = st.text_input(
            "Vision model (for images)",
            value="llava"
        )
        pdf_files = st.file_uploader(
            "Upload PDFs (optional)",
            type="pdf",
            accept_multiple_files=True
        )
        image_files = st.file_uploader(
            "Upload images (optional)",
            type=["png", "jpg", "jpeg", "webp"],
            accept_multiple_files=True
        )
        spreadsheet_files = st.file_uploader(
            "Upload Excel/CSV (optional)",
            type=["xlsx", "xls", "csv"],
            accept_multiple_files=True
        )
        
        if st.button("Process", type="primary"):
            if not pdf_files and not image_files and not spreadsheet_files:
                st.error("Please upload at least one file (PDF, image, or spreadsheet).")
            else:
                with st.spinner("Processing your documents..."):
                    embedding_model = embedding_model.strip() or "nomic-embed-text"
                    vision_model = vision_model.strip() or "llava"

                    raw_text_parts = []

                    # Extract PDF text
                    pdf_text = extract_pdf_text(pdf_files)
                    if pdf_text.strip():
                        raw_text_parts.append(pdf_text)

                    # Extract spreadsheet text
                    spreadsheet_text = extract_spreadsheet_text(spreadsheet_files)
                    if spreadsheet_text.strip():
                        raw_text_parts.append(spreadsheet_text)

                    # Extract image text
                    image_text = extract_image_text(image_files, vision_model)
                    if image_text.strip():
                        raw_text_parts.append(image_text)

                    raw_text = "\n\n".join(raw_text_parts)
                    
                    if not raw_text.strip():
                        st.error("Could not extract usable text from the uploaded files.")
                        return
                    
                    # Split into chunks
                    text_chunks = split_text_into_chunks(raw_text)
                    st.info(f"Created {len(text_chunks)} text chunks")

                    if not text_chunks:
                        st.error("No valid text content found after chunking. Try a different PDF.")
                        return
                    
                    try:
                        # Create vector store
                        vectorstore = create_vectorstore(text_chunks, embedding_model)

                        # Create conversation chain
                        st.session_state.conversation = create_conversation_chain(vectorstore, llm_model)
                    except Exception as err:
                        st.error(f"Model setup failed: {err}")
                        return
                    
                    st.success("Documents processed! You can now ask questions.")
    
    # Main chat interface
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_user_input(user_question)


if __name__ == "__main__":
    main()
