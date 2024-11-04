import asyncio
from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document
from langgraph.graph import END, StateGraph, START
import streamlit as st
from dotenv import load_dotenv
import os
import pathlib
import time
import json
from datetime import datetime
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredExcelLoader,
    CSVLoader,
    TextLoader
)

# Load environment variables
load_dotenv('var.env')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_INDEX = "omer"
NAMESPACE = "wondervector5000"

# Initialize paths
ROOT_DIR = pathlib.Path(__file__).parent.absolute()
DATA_DIR = ROOT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# Set up OpenAI embeddings
embeddings = OpenAIEmbeddings()

# Data models


class ChunkMetadata(BaseModel):
    quote: str
    doc_id: str
    chunk_id: str
    source_type: str


class RetrievalMetadata(BaseModel):
    documents_reviewed: int
    relevant_chunks_found: int
    query_expansion_rounds: int
    retrieval_timestamp: str


class CuratedResponse(BaseModel):
    answer: str
    hallucination_grade: float


class RAGResponse(BaseModel):
    query: str
    retrieved_chunks: List[ChunkMetadata]
    retrieval_metadata: RetrievalMetadata
    curated_response: CuratedResponse


class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[Document]
    memory: ConversationSummaryMemory


# Prompts
SYSTEM_PROMPT = """You are an advanced RAG system that provides detailed, structured responses.
For each query, you must:
1. Analyze retrieved chunks thoroughly
2. Provide specific citations for each claim
3. Assess potential hallucinations
4. Generate a hallucination grade (0-100)
5. Structure output according to the specified JSON format

Always base your responses strictly on the provided context."""

RESPONSE_TEMPLATE = """
Based on the following information, generate a structured response:

Context: {context}
Question: {question}
Previous Context: {history}

Your response MUST follow this JSON structure:
{
    "query": "the original question",
    "retrieved_chunks": [
        {
            "quote": "exact quote from context",
            "doc_id": "document identifier",
            "chunk_id": "chunk identifier",
            "source_type": "type of source"
        }
    ],
    "retrieval_metadata": {
        "documents_reviewed": number,
        "relevant_chunks_found": number,
        "query_expansion_rounds": number,
        "retrieval_timestamp": "ISO timestamp"
    },
    "curated_response": {
        "answer": "detailed answer with citations [chunk_id]",
        "hallucination_grade": number between 0 and 100
    }
}"""


class RAGSystem:
    def __init__(self):
        self.initialize_system()
        self.setup_llm_and_prompts()
        self.setup_graph()

    def initialize_system(self):
        self.embeddings = embeddings
        empty_file_path = DATA_DIR / "empty.txt"
        if not empty_file_path.exists():
            with open(empty_file_path, "w", encoding="utf-8") as f:
                f.write("")
        self.initialize_vectorstore(str(empty_file_path))

    def initialize_vectorstore(self, file_path: str, documents: List[Document] = None):
        try:
            if documents is None:
                loader = TextLoader(file_path)
                documents = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
            )
            splits = text_splitter.split_documents(documents)

            # Add metadata to chunks
            for i, doc in enumerate(splits):
                doc.metadata.update({
                    "chunk_id": f"chunk_{i+1}",
                    "doc_id": getattr(doc.metadata, "source", f"doc_{i+1}"),
                    "source_type": self._determine_source_type(file_path)
                })

            self.vectorstore = PineconeVectorStore.from_documents(
                documents=splits,
                embedding=self.embeddings,
                index_name=PINECONE_INDEX,
                namespace=NAMESPACE
            )
            time.sleep(1)

            self.retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": 4}
            )
        except Exception as e:
            st.error(f"Error initializing vector store: {str(e)}")
            raise

    def _determine_source_type(self, file_path: str) -> str:
        ext = pathlib.Path(file_path).suffix.lower()
        source_types = {
            '.pdf': 'PDF Document',
            '.txt': 'Text Document',
            '.csv': 'CSV Data',
            '.xlsx': 'Excel Document',
            '.docx': 'Word Document'
        }
        return source_types.get(ext, 'Unknown')

    def setup_llm_and_prompts(self):
        self.llm = ChatOpenAI(model="gpt-4", temperature=0.7)

        self.prompt = PromptTemplate(
            input_variables=["context", "question", "history"],
            template=RESPONSE_TEMPLATE
        )

        self.system_prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", RESPONSE_TEMPLATE)
        ])

    def setup_graph(self):
        def retrieve(state):
            try:
                question = state["question"]
                documents = self.retriever.invoke(question)
                return {
                    "documents": documents,
                    "question": question,
                    "memory": state["memory"]
                }
            except Exception as e:
                st.error(f"Error in retrieve: {str(e)}")
                return state

        def generate(state):
            try:
                question = state["question"]
                documents = state["documents"]
                memory = state["memory"]
                history = memory.load_memory_variables({})["history"]

                # Prepare retrieved chunks
                retrieved_chunks = [
                    ChunkMetadata(
                        quote=doc.page_content,
                        doc_id=doc.metadata.get("doc_id", f"doc_{i+1}"),
                        chunk_id=doc.metadata.get("chunk_id", f"chunk_{i+1}"),
                        source_type=doc.metadata.get("source_type", "Unknown")
                    ).dict()
                    for i, doc in enumerate(documents)
                ]

                # Create context string
                context_str = "\n\n".join(
                    doc.page_content for doc in documents)

                # Get LLM response for curated answer and hallucination grade
                llm_response = self.llm.invoke(
                    f"""Based on this context: {context_str}
                    Question: {question}
                    Generate a detailed answer with citations and a hallucination grade (0-100).
                    Format your response as JSON:
                    {{
                        "answer": "detailed answer with [chunk_id] citations",
                        "hallucination_grade": number
                    }}"""
                )

                try:
                    curated_response = json.loads(llm_response.content)
                except:
                    curated_response = {
                        "answer": str(llm_response.content),
                        "hallucination_grade": 50.0
                    }

                # Construct final response
                final_response = RAGResponse(
                    query=question,
                    retrieved_chunks=retrieved_chunks,
                    retrieval_metadata=RetrievalMetadata(
                        documents_reviewed=len(documents),
                        relevant_chunks_found=len(retrieved_chunks),
                        query_expansion_rounds=1,
                        retrieval_timestamp=datetime.now().isoformat()
                    ).dict(),
                    curated_response=curated_response
                )

                # Save to memory and return
                memory.save_context(
                    {"input": question},
                    {"output": final_response.json()}
                )

                return {
                    "documents": documents,
                    "question": question,
                    "generation": final_response.json(),
                    "memory": memory
                }

            except Exception as e:
                st.error(f"Error in generate: {str(e)}")
                return state

        workflow = StateGraph(GraphState)
        workflow.add_node("retrieve", retrieve)
        workflow.add_node("generate", generate)
        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)
        self.app = workflow.compile()

    def process_file(self, file_content, file_type: str) -> bool:
        try:
            # Process based on file type
            if file_type == 'pdf':
                temp_path = DATA_DIR / "temp.pdf"
                with open(temp_path, "wb") as f:
                    f.write(file_content)
                loader = PyPDFLoader(str(temp_path))
                documents = loader.load()
                temp_path.unlink()

            elif file_type == 'txt':
                content = file_content.decode('utf-8')
                documents = [Document(page_content=content)]

            elif file_type == 'csv':
                content = file_content.decode('utf-8')
                documents = [Document(page_content=content)]

            elif file_type == 'xlsx':
                temp_path = DATA_DIR / "temp.xlsx"
                with open(temp_path, "wb") as f:
                    f.write(file_content)
                loader = UnstructuredExcelLoader(str(temp_path))
                documents = loader.load()
                temp_path.unlink()

            elif file_type == 'docx':
                temp_path = DATA_DIR / "temp.docx"
                with open(temp_path, "wb") as f:
                    f.write(file_content)
                loader = Docx2txtLoader(str(temp_path))
                documents = loader.load()
                temp_path.unlink()

            else:
                raise ValueError("Unsupported file type")

            if documents:
                self.initialize_vectorstore(file_type, documents)
                return True
            return False

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            return False

    def chat(self, question: str) -> str:
        try:
            inputs = {
                "question": question,
                "memory": st.session_state.memory
            }

            final_output = None
            for output in self.app.stream(inputs):
                final_output = output

            if final_output and "generate" in final_output:
                return final_output["generate"]["generation"]

            return json.dumps({
                "error": "Could not generate response",
                "timestamp": datetime.now().isoformat()
            })

        except Exception as e:
            st.error(f"Error processing question: {str(e)}")
            return json.dumps({
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })


def initialize_memory():
    return ConversationSummaryMemory(llm=ChatOpenAI(model="gpt-4", temperature=0.7))


def main():
    st.set_page_config(page_title="RAG System", layout="wide")
    st.title("RAG System with Structured Output")

    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = RAGSystem()

    if "memory" not in st.session_state:
        st.session_state.memory = initialize_memory()

    with st.sidebar:
        st.header("Document Upload")
        uploaded_file = st.file_uploader(
            "Upload a document",
            type=['pdf', 'txt', 'csv', 'xlsx', 'docx']
        )

        if uploaded_file:
            with st.spinner("Processing document..."):
                file_type = uploaded_file.name.split('.')[-1].lower()
                file_content = uploaded_file.read()

                success = st.session_state.rag_system.process_file(
                    file_content, file_type)
                if success:
                    st.success("Document processed successfully!")
                else:
                    st.error("Failed to process document.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                try:
                    response_data = json.loads(message["content"])
                    st.json(response_data)
                except json.JSONDecodeError:
                    st.markdown(message["content"])
            else:
                st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Processing..."):
                response = st.session_state.rag_system.chat(prompt)
                try:
                    parsed_response = json.loads(response)
                    st.json(parsed_response)
                except json.JSONDecodeError:
                    st.error("Failed to parse response")
                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )


if __name__ == "__main__":
    main()
