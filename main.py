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

# Enhanced Prompts
SYSTEM_PROMPT = """You are a helpful and knowledgeable assistant with expertise in analyzing and understanding documents.
You can engage in casual conversation while also providing detailed analysis when needed.

When analyzing documents or answering specific questions:
1. Provide clear, conversational responses
2. Include relevant facts and citations when appropriate
3. Maintain a friendly and helpful tone
4. Offer follow-up suggestions naturally
5. Balance detailed analysis with accessible explanations

Remember to:
- Be conversational and engaging
- Provide accurate information
- Keep responses clear and helpful
- Adapt your tone to the conversation
"""

PROMPT_TEMPLATE = """
Context: {context}

Question: {question}

Previous Conversation: {history}

Please provide a natural response that:
1. Addresses the question clearly
2. Includes relevant information from the context when applicable
3. Maintains a conversational tone
4. Suggests relevant follow-ups if appropriate

Feel free to be conversational while ensuring accuracy and helpfulness.
"""


class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[Document]
    memory: ConversationSummaryMemory


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

            text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=1000,
                chunk_overlap=200
            )
            splits = text_splitter.split_documents(documents)

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

    def setup_llm_and_prompts(self):
        self.llm = ChatOpenAI(model="gpt-4", temperature=0.7)

        self.prompt = PromptTemplate(
            input_variables=["context", "question", "history"],
            template=PROMPT_TEMPLATE
        )

        self.system_prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", PROMPT_TEMPLATE)
        ])

        self.rag_chain = (
            self.system_prompt
            | self.llm
            | StrOutputParser()
        )

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

                generation = self.rag_chain.invoke({
                    "context": "\n\n".join(doc.page_content for doc in documents),
                    "question": question,
                    "history": history
                })

                memory.save_context({"input": question}, {
                                    "output": generation})

                return {
                    "documents": documents,
                    "question": question,
                    "generation": generation,
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
                # Save temporarily and load
                temp_path = DATA_DIR / "temp.pdf"
                with open(temp_path, "wb") as f:
                    f.write(file_content)
                loader = PyPDFLoader(str(temp_path))
                documents = loader.load()
                temp_path.unlink()  # Remove temp file

            elif file_type == 'txt':
                content = file_content.decode('utf-8')
                documents = [Document(page_content=content)]

            elif file_type == 'csv':
                content = file_content.decode('utf-8')
                documents = [Document(page_content=content)]

            elif file_type == 'xlsx':
                # Save temporarily and load
                temp_path = DATA_DIR / "temp.xlsx"
                with open(temp_path, "wb") as f:
                    f.write(file_content)
                loader = UnstructuredExcelLoader(str(temp_path))
                documents = loader.load()
                temp_path.unlink()  # Remove temp file

            elif file_type == 'docx':
                # Save temporarily and load
                temp_path = DATA_DIR / "temp.docx"
                with open(temp_path, "wb") as f:
                    f.write(file_content)
                loader = Docx2txtLoader(str(temp_path))
                documents = loader.load()
                temp_path.unlink()  # Remove temp file

            else:
                raise ValueError("Unsupported file type")

            if documents:
                self.initialize_vectorstore(None, documents)
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
                generation = final_output["generate"]["generation"]
                return generation
            return "I couldn't generate a response. Please try again."

        except Exception as e:
            st.error(f"Error processing question: {str(e)}")
            return "I encountered an error processing your question. Please try again."


def initialize_memory():
    return ConversationSummaryMemory(llm=ChatOpenAI(model="gpt-4", temperature=0.7))


def main():
    st.set_page_config(page_title="AI Assistant", layout="wide")
    st.title("AI Assistant")

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

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Chat with me..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.rag_system.chat(prompt)
                st.markdown(response)
                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )


if __name__ == "__main__":
    main()
