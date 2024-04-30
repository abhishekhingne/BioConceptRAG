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
        file_path (str): The path to the document file to be processed.
        persist_directory (str, optional): Directory to persist generated embeddings. Default is 'db/'.

    Attributes:
        url (str): The base URL for the Ollama language model.
        llm (Ollama): An instance of Ollama language model initialized with specific parameters.
        embeddings (OllamaEmbeddings): An instance of OllamaEmbeddings for text embeddings.
        file_path (str): The path to the document file.
        persist_directory (str): Directory to persist generated embeddings.

    Methods:
        load_documents(chunk_size, chunk_overlap, use_existing):
            Loads and processes documents into chunks with embeddings.
        
        retrieve_documents(vector_store, k):
            Retrieves documents from the vector store based on similarity.
        
        rag(query, retriever):
            Executes the Retrieval-Augmented Generation (RAG) process with a query.

    """
    def __init__(self, llm_url, file_path, persist_directory="db/") -> None:
        """
        Initialize the SimpleRAG instance.

        Args:
            llm_url (str): The base URL for the Ollama language model.
            file_path (str): The path to the document file to be processed.
            persist_directory (str, optional): Directory to persist generated embeddings. Default is 'db/'.
        """
        self.url = llm_url
        self.llm = Ollama(base_url=self.url, model="llama3", temperature=0.0)
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
            <s>[INST] You are a helpful assistant good at searching docuemnts. You also master in Biology. If you do not have an answer from the provided information say so. [/INST] </s>
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
