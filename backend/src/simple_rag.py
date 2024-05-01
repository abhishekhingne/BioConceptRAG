#Importing the python dependencies
from langchain_community.llms import Ollama
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate

class SimpleRAG:
    """
    A simple implementation of a Retrieval-Augmented Generation (RAG) system.

    Args:
        llm_url (str): The base URL for the Ollama language model.

    Attributes:
        url (str): The base URL for the Ollama language model.
        llm (Ollama): An instance of Ollama language model initialized with specific parameters.
        embeddings (OllamaEmbeddings): An instance of OllamaEmbeddings for text embeddings.

    Methods: 
        rag(query, retriever):
            Executes the Retrieval-Augmented Generation (RAG) process with a query.

    """
    def __init__(self, llm_url) -> None:
        """
        Initialize the SimpleRAG instance.

        Args:
            llm_url (str): The base URL for the Ollama language model.
        """
        self.url = llm_url
        self.llm = Ollama(base_url=self.url, model="llama3", temperature=0.0)
        self.embeddings = OllamaEmbeddings(base_url=self.url, model="nomic-embed-text")

    
    def rag(self, query, retriever):
        """
        Execute the Retrieval-Augmented Generation (RAG) process with a query.

        Args:
            query (str): The input query for the RAG system.
            retriever (Retriever): A retriever object configured for document retrieval.

        Returns:
            dict: Result of the RAG process including generated response and context.
        """
        raw_prompt = PromptTemplate.from_template(
            """ 
            <s>[INST] You are an assistant for question-answering tasks. 
                    Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. 
            [/INST] </s>
            [INST] {input}
                Context: {context}
                Answer:
            [/INST]
        """
        )
        document_chain = create_stuff_documents_chain(self.llm, raw_prompt)
        chain = create_retrieval_chain(retriever, document_chain)
        result = chain.invoke({"input": query})

        return result
