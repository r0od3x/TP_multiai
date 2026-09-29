import os

import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader

load_dotenv(override=True)

LLM_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "16"))
TOP_K = int(os.getenv("RAG_TOP_K", "5"))

prompt_template = """
Answer the following question based only on the provided context:
<context>
    {context}
</context>
<question>
    {input}
</question>
"""

llm = ChatOpenAI(model=LLM_MODEL, temperature=0)


def build_retriever(pdf_docs):
    content = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            content += page.extract_text() or ""

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_text(content)

    vector_store = Chroma.from_texts(
        chunks,
        OpenAIEmbeddings(model=EMBEDDING_MODEL),
        collection_name="data_collection",
    )
    return vector_store.as_retriever(search_kwargs={"k": TOP_K}), chunks


def main():
    st.set_page_config(page_title="RAG", layout="wide")
    st.subheader("Retrieval Augmented Generation", divider="blue")

    with st.sidebar:
        st.title("Data loader")
        pdf_docs = st.file_uploader(label="Load your PDFs", type="pdf", accept_multiple_files=True)
        if st.button("Submit"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF.")
            else:
                with st.spinner("Indexing documents..."):
                    retriever, chunks = build_retriever(pdf_docs)
                    st.session_state.retriever = retriever
                st.success(f"Indexed {len(chunks)} chunks.")
                with st.expander("Show chunks"):
                    st.write(chunks)

    st.subheader("Chatbot")
    user_question = st.text_input("Ask your question")
    if user_question:
        if "retriever" not in st.session_state:
            st.info("Upload and submit PDFs in the sidebar first.")
            return

        context_docs = st.session_state.retriever.invoke(user_question)
        context_text = "\n\n".join(d.page_content for d in context_docs)
        prompt = prompt_template.format(context=context_text, input=user_question)

        resp = llm.invoke(prompt)
        st.write(resp.content)


if __name__ == "__main__":
    main()
