from langchain_openai import ChatOpenAI
from langchain_experimental.sql import SQLDatabaseChain
from langchain_community.utilities import SQLDatabase
from langchain_community.cache import InMemoryCache  # Updated import
from langchain_community.agent_toolkits import create_sql_agent
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os
import sys
import pysqlite3
from langchain.agents import AgentExecutor, create_react_agent, load_tools
from langchain import hub
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import Tool
import sqlite3
    


sys.modules["sqlite3"] = pysqlite3


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set. Please check your .env file.")

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

llm = ChatOpenAI(model="gpt-4")

db = SQLDatabase.from_uri("sqlite:///energy_products.db")

def multi_table(query):
    """Query multiple tables in SQLite database using SQL agent."""
    agent_executor = create_sql_agent(
        llm=llm,
        db=db,
        agent_type="openai-tools",
        verbose=True
    )
    result = agent_executor.invoke({"input": query})
    return result["output"]

chroma_db_path = "chroma_db"  # Path where ChromaDB is stored
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_db = Chroma(persist_directory=chroma_db_path, embedding_function=embeddings)
retriever = vector_db.as_retriever(search_kwargs={"k": 3})  # Retrieve top 3 results

def hybrid_search(query):
    """
    Queries both SQL database (structured data) and Chroma vector database (unstructured data).
    """
    sql_result = multi_table(query)

    chroma_results = retriever.get_relevant_documents(query)

    structured_response = f"🔹 **Structured Data (SQL Results):**\n{sql_result}\n"
    unstructured_response = "🔹 **Unstructured Data (ChromaDB Results):**\n"

    for doc in chroma_results:
        unstructured_response += f"- {doc.page_content[:500]}...\n"  # Show snippet

    return structured_response + unstructured_response

def create_ticket(user_request):
    """
    Logs a support ticket inside `energy_products.db`.
    """
    try:
        conn = sqlite3.connect("energy_products.db")  # Connect to the existing database
        cursor = conn.cursor()

        # Insert ticket into database
        cursor.execute("INSERT INTO tickets (user_request) VALUES (?)", (user_request,))
        conn.commit()
        ticket_id = cursor.lastrowid  # Get the new ticket ID
        conn.close()

        return f"Ticket #{ticket_id} created successfully: '{user_request}'."
    except Exception as e:
        return f"Error creating ticket: {str(e)}"

class HybridQueryInput(BaseModel):
    query: str = Field(description="should be a query for QA")

retriever_tool = Tool(
    name="retriever_tool",
    description="Use this tool when you need to retrieve and analyze information from both structured (SQL database) and unstructured (Chroma vector database) sources. This tool is designed to answer questions about energy products, solar panel installations, energy efficiency, and related topics by combining insights from tabular data and document-based knowledge. Pass the natural language query directly to this tool for a comprehensive response.",
    func=hybrid_search,
    args_schema=HybridQueryInput
)

class TicketQueryInput(BaseModel):
    request: str = Field(description="A description of the issue or request for the ticket.")

ticket_tool = Tool(
    name="ticket_creation_tool",
    description="Use this tool to create a support ticket for customer requests related to energy products, solar panels, or energy efficiency. Pass a description of the request, and this tool will log it as a support ticket in the system.",
    func=create_ticket,
    args_schema=TicketQueryInput
)

tools = [retriever_tool, ticket_tool]
prompt = hub.pull("hwchase17/react")
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

query = "Hello what can you do for me?"
response = agent_executor.invoke({"input": query})

print(response)