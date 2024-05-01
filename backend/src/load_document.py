#Importing the python dependencies
from langchain_community.llms import Ollama
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain.document_loaders import PyPDFLoader

class LoadDocument:
    """
    A class to load and process a document for embedding using Ollama.

    Attributes:
        llm_url (str): The base URL for the Ollama language model.
        file_path (str): The path to the document file to be processed.
        persist_directory (str, optional): Directory to persist generated embeddings. Default is 'db/'.
        embeddings (OllamaEmbeddings): An instance of OllamaEmbeddings for text embeddings.

    Methods:
        load_documents(chunk_size, chunk_overlap, use_existing):
            Loads and processes documents into chunks with embeddings.
        
        retrieve_documents(vector_store, k):
            Retrieves documents from the vector store based on similarity.
    """
    def __init__(self, llm_url, file_path, persist_directory="db/") -> None:
        """
        Initialize LoadDocument with the provided parameters.

        Args:
            llm_url (str): The base URL for the Ollama language model.
            file_path (str): The path to the document file to be processed.
            persist_directory (str, optional): Directory to persist generated embeddings. Default is 'db/'.
        """
        self.url = llm_url
        self.embeddings = OllamaEmbeddings(base_url=self.url, model="nomic-embed-text")
        self.file_path = file_path
        self.persist_directory = persist_directory

    def load_documents(self, chunk_size, chunk_overlap, use_existing):
        """
        Load and process documents into chunks with embeddings.

        Args:
            chunk_size (int): Size of each document chunk.
            chunk_overlap (int): Overlap size between consecutive chunks.
            use_existing (bool): Flag to use existing vector store if available.

        Returns:
            Chroma: A vector store containing document embeddings.
        """
        if not use_existing:
            document = PyPDFLoader(file_path=self.file_path).load()
            document_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                            chunk_overlap=chunk_overlap,
                                                            length_function=len,
                                                            )
            chunks = document_splitter.split_documents(document)
            vector_store = Chroma.from_documents(documents=chunks, 
                                                embedding=self.embeddings, 
                                                persist_directory=self.persist_directory)
        else:
            vector_store = Chroma(persist_directory=self.persist_directory, 
                                  embedding_function=self.embeddings)
        return vector_store

    def retrive_documents(self, vector_store, k):
        """
        Retrieve documents from the vector store based on similarity.

        Args:
            vector_store (Chroma): A vector store containing document embeddings.
            k (int): Number of documents to retrieve.

        Returns:
            Retriever: A retriever object configured for document retrieval.
        """
        retriever = vector_store.as_retriever(search_type="similarity",
            search_kwargs={
                "k": k,
                #"score_threshold": 0.1,
            },
        )
        return retriever