import argparse
import os
import glob
from langchain.document_loaders import (
    DirectoryLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredPDFLoader,
    UnstructuredHTMLLoader,
    TextLoader
)
from langchain.text_splitter import TokenTextSplitter
from langchain.text_splitter import NLTKTextSplitter
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings  
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.storage import LocalFileStore
from langchain.vectorstores import Chroma

def main(offline_flag):
    documents_folder = "./research_documents"

    # md_loader = DirectoryLoader(documents_folder, glob="**/*.md", use_multithreading=True, loader_cls=UnstructuredMarkdownLoader)
    pdf_loader = DirectoryLoader(documents_folder, glob="**/*.pdf", use_multithreading=True, loader_cls=UnstructuredPDFLoader)
    doc_loader = DirectoryLoader(documents_folder, glob="**/*.doc", use_multithreading=True, loader_cls=UnstructuredWordDocumentLoader)
    docx_loader = DirectoryLoader(documents_folder, glob="**/*.docx", use_multithreading=True, loader_cls=UnstructuredWordDocumentLoader)
    html_loader = DirectoryLoader(documents_folder, glob="**/*.html", use_multithreading=True, loader_cls=UnstructuredHTMLLoader)
    txt_loader = DirectoryLoader(documents_folder, glob="**/*.txt", use_multithreading=True, loader_cls=TextLoader)
    # md_docs = md_loader.load()
    pdf_docs = pdf_loader.load()
    doc_docs = doc_loader.load()
    docx_docs = docx_loader.load()
    txt_docs = txt_loader.load()
    html_docs = html_loader.load()

    # text_splitter = TokenTextSplitter(chunk_size=650, chunk_overlap=0)
    text_splitter = NLTKTextSplitter.from_tiktoken_encoder(chunk_size=650, encoding_name="cl100k_base")
    # md_docs = text_splitter.split_documents(md_docs)
    pdf_docs = text_splitter.split_documents(pdf_docs)
    doc_docs = text_splitter.split_documents(doc_docs)
    docx_docs = text_splitter.split_documents(docx_docs)
    txt_docs = text_splitter.split_documents(txt_docs)
    html_docs = text_splitter.split_documents(html_docs)

    consolidated_docs = pdf_docs + txt_docs + html_docs + docx_docs + doc_docs
    print(f"pdf chunks: {len(pdf_docs)}\ntxt chunks: {len(txt_docs)}\nhtml chunks: {len(html_docs)}\ndocx chunks: {len(docx_docs)}\ndoc chunks: {len(doc_docs)}")

    # embeddings shit
    store = LocalFileStore("./embedding_cache")
    underlying_embedder = OpenAIEmbeddings()
    if offline_flag:
        underlying_embedder = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    cached_embedder = CacheBackedEmbeddings.from_bytes_store(underlying_embedder, store)

    persist_path = "./document_vector_store_db"

    return Chroma.from_documents(
        documents=consolidated_docs,
        embedding=cached_embedder,
        persist_directory=persist_path,  # persist_path and serializer are optional
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--offline', dest='offline_flag', action='store_true')
    parser.set_defaults(offline_flag=False)
    args = parser.parse_args()
    main(args.offline_flag)