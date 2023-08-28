import argparse
import glob
import os
from termcolor import colored, cprint
from langchain.chat_models import ChatOpenAI
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.callbacks import get_openai_callback
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings  
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.storage import LocalFileStore
from langchain.vectorstores import Chroma

SYSTEM_TEXT_FONT_COLOR = 'blue'
SYSTEM_TEXT_BG_COLOR = 'on_blue'
TRANSLATED_TEXT_COLOR = 'yellow'
FILENAME_COLOR = 'magenta'
ERROR_COLOR = 'red'
DIAGNOSTIC_COLOR = 'cyan'
ANSWER_COLOR = 'green'

HELP_TEXT = f"""
Command                         | Description
------------------------------- | ----------------------------------------------
quit / exit                     | Exits the program with a farewell message.
help                            | Displays this help text.
translate                       | Toggles the translation feature on or off.
offline                         | Toggles the offline mode on or off.
auto_select                     | Toggles the auto-select document feature on or off.
show_context                    | Toggles the document context showing feature on or off.
target_language                 | Allows you to enter a new target language.
update                          | Runs the `load_and_embed_documents.py` script to update document embeddings in vector store.
settings                        | Lists all setting values for translate, offline, auto_select, and target_language.
"""

def main(offline_flag, translate_flag, auto_select_specific_doc_flag, show_context_flag, target_language):
    """
    Main function to process the user query based on the provided flags and target language.

    Args:
        offline_flag (bool): Flag to indicate if offline text embeddings need to be used.
        translate_flag (bool): Flag to indicate if the query needs to be translated.
        auto_select_specific_doc_flag (bool): Flag to indicate if a specific document needs to be auto-selected based on the query.
        target_language (str): The target language for translation.

    Returns:
        None
    """
    documents_folder = "./research_documents"
    vector_store_path = "./document_vector_store_db"

    raw_document_filenames = glob.glob(f"{documents_folder}/*")
    document_filenames = [filename[2:] for filename in raw_document_filenames]

    store = LocalFileStore("./embedding_cache")
    underlying_embedder = OpenAIEmbeddings()
    if offline_flag:
        underlying_embedder = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    cached_embedder = CacheBackedEmbeddings.from_bytes_store(underlying_embedder, store)

    vector_store = Chroma(persist_directory=vector_store_path, embedding_function=cached_embedder)

    llm = ChatOpenAI(model='gpt-4', temperature=0)

    while True:
        query = input(colored("Enter your query (enter `help` for list of commands): ", color=SYSTEM_TEXT_FONT_COLOR))

        if len(query.lower().strip()) == 0:
            cprint("Query is empty, please try again.", on_color=ERROR_COLOR)
            continue
        elif query.lower().strip() == "quit" or query.lower().strip() == "exit" or query.lower().strip() == "seeya":
            cprint("SEEYAA~~~", on_color='on_light_green', attrs=['blink'])
            break
        elif query.lower().strip() == "help":
            cprint(HELP_TEXT, color=SYSTEM_TEXT_FONT_COLOR)
            continue
        elif query.lower().strip() == "offline":
            offline_flag = not offline_flag
            cprint(f"Offline embeddings now set to {colored(offline_flag, color=FILENAME_COLOR)}", color=SYSTEM_TEXT_FONT_COLOR)
            continue
        elif query.lower().strip() == "translate":
            translate_flag = not translate_flag
            cprint(f"Translate now set to {colored(translate_flag, color=FILENAME_COLOR)}", color=SYSTEM_TEXT_FONT_COLOR)
            continue
        elif query.lower().strip() == "auto_select":
            auto_select_specific_doc_flag = not auto_select_specific_doc_flag
            cprint(f"Auto-select document now set to {colored(auto_select_specific_doc_flag, color=FILENAME_COLOR)}", color=SYSTEM_TEXT_FONT_COLOR)
            continue
        elif query.lower().strip() == "show_context":
            show_context_flag = not show_context_flag
            cprint(f"Showing source document context is now set to {colored(show_context_flag, color=FILENAME_COLOR)}", color=SYSTEM_TEXT_FONT_COLOR)
            continue
        elif query.lower().strip() == "target_language":
            target_language = input(colored("Enter new target language: ", color=SYSTEM_TEXT_FONT_COLOR))
            cprint(f"Target language now set to {colored(target_language, color=FILENAME_COLOR)}", color=SYSTEM_TEXT_FONT_COLOR)
            continue
        elif query.lower().strip() == "update":
            cprint("Running load and embed documents", color=SYSTEM_TEXT_FONT_COLOR)
            os.system('python load_and_embed_documents.py')
            cprint("Documents have been loaded and embedded", color=SYSTEM_TEXT_FONT_COLOR)
            continue
        elif query.lower().strip() == "settings":
            cprint(f"Translate: {colored(translate_flag, color=FILENAME_COLOR)}", color=SYSTEM_TEXT_FONT_COLOR)
            cprint(f"Offline embeddings: {colored(offline_flag, color=FILENAME_COLOR)}", color=SYSTEM_TEXT_FONT_COLOR)
            cprint(f"Auto-select document: {colored(auto_select_specific_doc_flag, color=FILENAME_COLOR)}", color=SYSTEM_TEXT_FONT_COLOR)
            cprint(f"Target language: {colored(target_language, color=FILENAME_COLOR)}\n", color=SYSTEM_TEXT_FONT_COLOR)
            continue

        specific_doc_filename = None
        vector_query = query

        if translate_flag:
            vector_query = get_translated_query(query, target_language, llm)
            cprint(f"Translated: {vector_query}", color=TRANSLATED_TEXT_COLOR)
        if auto_select_specific_doc_flag:
            specific_doc_filename = get_specific_doc_filename(query, document_filenames, llm)
            cprint(f"Auto-selected file: {colored(specific_doc_filename, color=FILENAME_COLOR)}", color=SYSTEM_TEXT_FONT_COLOR)

        docs = run_vector_store_search(vector_query, vector_store, doc_filename=specific_doc_filename)

        unique_docs = []
        prev_doc = None
        for doc in docs:
            if prev_doc == doc:
                continue
            else:
                unique_docs.append(doc)
                prev_doc = doc

        if len(unique_docs) == 0:
            cprint("No hits on vector store results.", color=ERROR_COLOR)
        else:
            doc_sources = colored(list(set([doc.metadata['source'] for doc in unique_docs])), color=FILENAME_COLOR)
            cprint(f"Taking context from the following sources: {doc_sources}", color=SYSTEM_TEXT_FONT_COLOR)
            cprint("Refining answer...", color=SYSTEM_TEXT_FONT_COLOR)
            sources_chain = load_qa_with_sources_chain(llm, chain_type="refine")
            with get_openai_callback() as cb:
                answer = sources_chain({"input_documents": unique_docs, "question": query})
                cprint(f"tokens: {cb.total_tokens}, cost: {cb.total_cost}", color=DIAGNOSTIC_COLOR)
                
            cprint(f"\n{answer['output_text']}\n", color=ANSWER_COLOR)

def get_translated_query(query, target_language, llm):
    translator_template = (
        "You are a helpful assistant that translates the given query to {target_language}."
    )
    system_message_prompt = SystemMessagePromptTemplate.from_template(translator_template)
    human_template = "{query}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )
    translated_query = llm(
        chat_prompt.format_prompt(
            target_language=target_language, query=query
        ).to_messages()
    )
    translated_query = translated_query.content
    return translated_query

def get_specific_doc_filename(query, document_filenames, llm):
    filename_selector_template = (
        "You are a helpful assistant that outputs only the index of the most relevant filename given the query. If no relevant files are found, output -1. Here is the list of available filenames: {filenames}"
    )
    system_message_prompt = SystemMessagePromptTemplate.from_template(filename_selector_template)
    human_template = "{query}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )
    specific_doc_filename_index = llm(
        chat_prompt.format_prompt(
            filenames=document_filenames, query=query
        ).to_messages()
    )
    specific_doc_filename_index = int(specific_doc_filename_index.content)
    if specific_doc_filename_index < 0:
        return None
    return document_filenames[specific_doc_filename_index]

def run_vector_store_search(query, vector_store, doc_filename=None):
    if doc_filename:
       return vector_store.similarity_search(query, k=6, filter={"source": doc_filename})
    return vector_store.similarity_search(query, k=6)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--offline', dest='offline_flag', action='store_true')
    parser.set_defaults(offline_flag=False)
    parser.add_argument('--translate', dest='translate_flag', action='store_true')
    parser.set_defaults(translate_flag=False)
    parser.add_argument('--auto-select', dest='auto_select_specific_doc_flag', action='store_true')
    parser.set_defaults(auto_select_specific_doc_flag=False)
    parser.add_argument('--no-show-context', dest='show_context_flag', action='store_false')
    parser.set_defaults(show_context_flag=True)
    parser.add_argument('--target-language', dest='target_language', default='Korean')
    args = parser.parse_args()

    main(args.offline_flag, args.translate_flag, args.auto_select_specific_doc_flag, args.show_context_flag, args.target_language)