# Personal Knowledge Base Document Query (PKB DocQuery)

PKB DocQuery allows you to harness the power of Large Language Models (LLMs) to answer questions about your documents. DocQuery can be especially useful for pinpointing hard-to-read primary source documents such as legal texts and policy memos.

This tool is **language agnostic**, meaning you can upload documents in a different language (Thai for example). I haven't tested this with languages beyond Thai and Korean, but I've confirmed that this does a pretty good job parsing and accurately summarizing documents in those languages.

## Disclaimer

This software tool uses large language models to generate summaries. While it strives for accuracy, it may not always capture the full context or nuances of the original document. Therefore, it should be used as a supplementary resource rather than a standalone source of information. This tool is designed to assist users in gaining a better understanding of primary source documents by providing a refined summary. It is not intended to replace reading through the full text. 

Users are strongly encouraged to read the original text in its entirety to ensure a comprehensive understanding of the content. The developer of this tool will not be held responsible for any misinterpretations or inaccuracies that may arise from the use of this software tool.

By using this tool, you acknowledge and agree to these terms.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### OpenAI API Key

Ensure that your API key is stored in your `.bashrc` or `.zshrc` file as an environmental variable. If you don't have one already, you can get your API key [here](https://platform.openai.com/account/api-keys).

To set the environment variable, you can enter the following in your `.bashrc`/`.zshrc` file:
```
export OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

### Installing PKB DocQuery

To install PKB DocQuery, follow these steps:

1. Clone this repository
2. Navigate to the project directory
```
cd pkb_docquery
```
3. Install the required packages
```
pip install -r requirements.txt
```
4. Install [ollama](https://ollama.com/download).
5. Install `llama3` for offline use:
```
ollama run llama3
```

### Usage

To use PKB DocQuery, follow these steps:

1. Place your documents in the `./research_documents` directory.
   - Supported filetypes: .doc, .docx, .pdf, .html, and .txt files.

2. Run `load_and_embed_documents.py` script to load and embed your documents into the vector store database (stored in `document_vector_store_db`).
```
# Regular load and embed with OpenAI embeddings
python load_and_embed_documents.py

# Run with `intfloat/multilingual-e5-large-instruct` embedding model.
python load_and_embed_documents.py --offline
```
3. Run the `query.py` script to start the query interface. This script allows you to enter queries and get answers from the LLM. It also supports translation, offline embeddings, and auto-selection of documents based on the query.
```
python query.py
```
### Query Commands
Command                         | Description
------------------------------- | ----------------------------------------------
`quit`, `exit`                     | Exits the program with a farewell message.
`help`                            | Displays this help text.
`translate`                       | Toggles the translation feature on or off.
`offline`                         | Toggles the offline mode on or off.
`auto_select`                     | Toggles the auto-select document feature on or off.
`target_language`                 | Allows you to enter a new target language.
`update`                          | Runs the `load_and_embed_documents.py` script to update document embeddings in vectorstore.
`settings`                        | Lists all setting values for translate, offline, auto_select, and target_language.

## TODOs
* export query and selected documents as text
* switch between MMR and similarity search modes when not in auto-select document mode
* better text cleaning of PDFs
* add other summary generation types (stuff, map_reduce, map_rerank)
* research better methods of retrieving context other than cosine similarity