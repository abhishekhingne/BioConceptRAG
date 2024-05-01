#Importing the python dependencies
from langchain_community.llms import Ollama
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from typing_extensions import TypedDict
from typing import List
from langchain.schema import Document
from langgraph.graph import END, StateGraph


class AdvancedRAG:
    """
    Advanced Retrieval-Augmented Generation (RAG) system using Ollama.

    Attributes:
        llm_url (str): The base URL for the Ollama language model.
        vector_store (Chroma): A vector store containing document embeddings.

    Methods:
        __init__(llm_url, vector_store):
            Initializes the AdvancedRAG instance with the specified language model URL and vector store.

        retrieve(state):
            Retrieves documents relevant to a given question from the vector store.

        generate(state):
            Generates an answer to the question using the retrieved documents.

        grade_documents(state):
            Grades the relevance of retrieved documents to the question and filters out irrelevant ones.

        default_reply(state):
            Provides a default reply if no relevant documents are found or the question cannot be answered.

        grade_generation(state):
            Determines whether the generated answer is well-grounded in the retrieved documents and addresses the question.

        add_nodes():
            Adds nodes representing different stages of the RAG workflow to the state graph.

        build_graph():
            Builds and compiles the state graph representing the RAG workflow.

        execute_graph(question):
            Executes the RAG workflow on the given question and returns the final output.

    """
    def __init__(self, llm_url, vector_store) -> None:
        """
        Initializes the AdvancedRAG instance.

        Args:
            llm_url (str): The base URL for the Ollama language model.
            vector_store (Chroma): A vector store containing document embeddings.
        """
        self.url = llm_url
        self.llm = Ollama(base_url=self.url, model="llama3", temperature=0.0)
        self.retriever = vector_store.as_retriever(search_type="similarity",
                    search_kwargs={
                        "k": 5,
                        #"score_threshold": 0.1,
                    },
                )
        self.workflow = None

        # Prompt to grade the retrieved documents
        retriever_prompt = PromptTemplate(
            template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing relevance 
            of a retrieved document to a user question. If the document contains keywords related to the user question, 
            grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
            Provide the binary score as a JSON with a single key 'score' and no premable or explaination.
            <|eot_id|><|start_header_id|>user<|end_header_id|>
            Here is the retrieved document: \n\n {document} \n\n
            Here is the user question: {question} \n <|eot_id|><|start_header_id|>assistant<|end_header_id|>
            """,
            input_variables=["question", "document"],
        )

        self.retrieval_grader = retriever_prompt | self.llm | JsonOutputParser()

        # Prompt to generate final response
        rag_generate_prompt = PromptTemplate(
            template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an assistant for question-answering tasks. 
            Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. 
            Use three sentences maximum and keep the answer concise <|eot_id|><|start_header_id|>user<|end_header_id|>
            Question: {question} 
            Context: {context} 
            Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
            input_variables=["question", "document"],
        )

        self.rag_chain = rag_generate_prompt | self.llm | StrOutputParser()

        # Prompt to check final response is hallucinated or not
        hallucination_prompt = PromptTemplate(
            template=""" <|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether 
            an answer is grounded in / supported by a set of facts. Give a binary 'yes' or 'no' score to indicate 
            whether the answer is grounded in / supported by a set of facts. Provide the binary score as a JSON with a 
            single key 'score' and no preamble or explanation. <|eot_id|><|start_header_id|>user<|end_header_id|>
            Here are the facts:
            \n ------- \n
            {documents} 
            \n ------- \n
            Here is the answer: {generation}  <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
            input_variables=["generation", "documents"],
        )

        self.hallucination_grader = hallucination_prompt | self.llm | JsonOutputParser()
    
        # Prompt to check final response is aligned with the user question
        answer_grader_prompt = PromptTemplate(
            template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether an 
            answer is useful to resolve a question. Give a binary score 'yes' or 'no' to indicate whether the answer is 
            useful to resolve a question. Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
            <|eot_id|><|start_header_id|>user<|end_header_id|> Here is the answer:
            \n ------- \n
            {generation} 
            \n ------- \n
            Here is the question: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
            input_variables=["generation", "question"],
        )

        self.answer_grader = answer_grader_prompt | self.llm | JsonOutputParser()

    # Define State
    class GraphState(TypedDict):
        """
        Represents the state of our graph.

        Attributes:
            question: question
            generation: LLM generation
            documents: list of documents 
        """
        question : str
        generation : str
        documents : List[str]
    
    # Nodes
    def retrieve(self, state):
        """
        Retrieves documents relevant to a given question from the vector store.

        Args:
            state (dict): The current state of the graph.

        Returns:
            dict: Updated state containing retrieved documents and the question.
        """
        print("---RETRIEVE---")
        question = state["question"]

        # Retrieval
        documents = self.retriever.invoke(question)
        return {"documents": documents, "question": question}
    
    def generate(self, state):
        """
        Generates an answer to the question using the retrieved documents.

        Args:
            state (dict): The current state of the graph.

        Returns:
            dict: Updated state containing the generated answer, retrieved documents, and the question.
        """
        print("---GENERATE---")
        question = state["question"]
        documents = state["documents"]
        
        # RAG generation
        generation = self.rag_chain.invoke({"context": documents, "question": question})
        return {"documents": documents, "question": question, "generation": generation}
    
    def grade_documents(self, state):
        """
        Grades the relevance of retrieved documents to the question and filters out irrelevant ones.

        Args:
            state (dict): The current state of the graph.

        Returns:
            dict: Updated state containing relevant documents and the question.
        """
        print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        question = state["question"]
        documents = state["documents"]
        
        # Score each doc
        filtered_docs = []
        for d in documents:
            score = self.retrieval_grader.invoke({"question": question, "document": d.page_content})
            grade = score['score']
            # Document relevant
            if grade.lower() == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)
            # Document not relevant
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")
                continue
        return {"documents": filtered_docs, "question": question}

    def default_reply(self, state):
        """
        Provides a default reply if no relevant documents are found or the question cannot be answered.

        Args:
            state (dict): The current state of the graph.

        Returns:
            dict: Updated state containing a default reply.
        """
        print("---DEFAULT REPLY---")
        question = state["question"]
        documents = state["documents"]
        #documents = ["Sorry, I cannot answer this question. It is beyond my capability."]
        return {"documents": documents, "question": question, "generation": "Sorry, I cannot answer this question. It is beyond my capability."}

    def grade_generation(self, state):
        """
        Determines whether the generated answer is well-grounded in the retrieved documents and addresses the question.

        Args:
            state (dict): The current state of the graph.

        Returns:
            str: Decision for the next node in the workflow.
        """
        print("---CHECK HALLUCINATIONS---")
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]

        score = self.hallucination_grader.invoke({"documents": documents, "generation": generation})
        grade = score['score']

        # Check hallucination
        if grade == "yes":
            print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
            # Check question-answering
            print("---GRADE GENERATION vs QUESTION---")
            score = self.answer_grader.invoke({"question": question,"generation": generation})
            grade = score['score']
            if grade == "yes":
                print("---DECISION: GENERATION ADDRESSES QUESTION---")
                return "useful"
            else:
                print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
                return "not useful"
        else:
            print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
            return "not supported"
    
    def add_nodes(self):
        """
        Adds nodes representing different stages of the RAG workflow to the state graph.
        """
        self.workflow = StateGraph(self.GraphState)

        # Define the nodes
        self.workflow.add_node("retrieve", self.retrieve) # retrieve
        self.workflow.add_node("grade_documents", self.grade_documents) # grade documents
        self.workflow.add_node("generate", self.generate) # generatae

    def build_graph(self):
        """
        Builds and compiles the state graph representing the RAG workflow.

        Returns:
            StateGraph: Compiled state graph.
        """
        self.add_nodes()
        self.workflow.set_entry_point("retrieve")
        self.workflow.add_edge("retrieve", "grade_documents")
        self.workflow.add_edge("grade_documents", "generate")
        self.workflow.add_conditional_edges(
            "generate",
            self.grade_generation,
            {
                "not supported": "generate",
                "useful": END,
                "not useful": END,
            },
        )
        app = self.workflow.compile()
        return app

    def execute_graph(self, question):
        """
        Executes the RAG workflow on the given question and returns the final output.

        Args:
            question (str): The input question for the RAG system.

        Returns:
            dict: Final output containing the generated answer, relevant documents, and decision.
        """
        inputs = {"question": question}
        app = self.build_graph()
        for output in app.stream(inputs):
            for key, value in output.items():
                # Node
                print(f"Node '{key}':")
                # Optional: print full state at each node
                # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
            print("\n---\n")
        return value