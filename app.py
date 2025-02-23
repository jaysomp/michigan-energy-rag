import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_experimental.sql import SQLDatabaseChain
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import Tool
from langchain.callbacks import StreamlitCallbackHandler
import sqlite3
import os
from dotenv import load_dotenv
import sys
import pysqlite3
import time
from langchain.prompts import PromptTemplate


sys.modules["sqlite3"] = pysqlite3

# load_dotenv()
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# if not OPENAI_API_KEY:
#     raise ValueError("OPENAI_API_KEY is not set. Please check your .env file.")

# os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Check if the API key is already in session state
# Sidebar for entering the API key
st.sidebar.header("API Key Configuration")
api_key = st.sidebar.text_input("Enter your OpenAI API key:", value="", type="password")

# Update the session state and environment variable if a new key is entered
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key

# Check if the API key is set
if not api_key:
    st.sidebar.warning("Please enter your OpenAI API key to continue.")
    st.stop()

st.title("Michigan Imaginary Energy Assistant")


if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Initialize LLM and databases
llm = ChatOpenAI(model="gpt-4")
db = SQLDatabase.from_uri("sqlite:///energy_products.db")
chroma_db_path = "chroma_db"
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = Chroma(persist_directory=chroma_db_path, embedding_function=embeddings)
retriever = vector_db.as_retriever(search_kwargs={"k": 3})

# Define your functions (multi_table, hybrid_search, create_ticket)
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
    

def check_ticket(user_request):
    """
    Queries both SQL database to check status of tickets from ticket table
    """

    sql_result = multi_table(user_request + "(Only check from the tickets table provided the id for the ticket)")

    return sql_result


# Define your tool schemas and tools
class HybridQueryInput(BaseModel):
    query: str = Field(description="should be a query for QA")

retriever_tool = Tool(
    name="retriever_tool",
    description="Use this tool when you need to retrieve and analyze information from both structured (SQL database) and unstructured (Chroma vector database) sources. This tool is designed to answer questions about energy products, solar panel installations, energy efficiency, and related topics by combining insights from tabular data and document-based knowledge. Pass the natural language query directly to this tool for a comprehensive response. This can also be used to retreve the status of tickets created from the tickets table",
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

# class TicketStatInput(BaseModel):
#     request: str = Field(description="Description of what is needed from the tickets table")

# ticket_status = Tool(
#     name="ticket_status_tool",
#     description="Use this tool to check the status of an existing support ticket. Provide the ticket number, and this tool will return the current status and any available details about the ticket. This is useful for following up on customer inquiries about their previously submitted requests or issues related to energy products, solar panels, or energy efficiency.",
#     func=check_ticket,
#     args_schema=TicketStatInput
# )

# Create agent
tools = [retriever_tool, ticket_tool] 
# tools = [retriever_tool, ticket_tool, ticket_status]

prompt = hub.pull("hwchase17/react")

# custom_prompt = PromptTemplate(
#     input_variables=["input", "agent_scratchpad", "tools", "tool_names"],
#     template="""
# Answer the following questions as best you can. You are an AI assistant for Michigan Imaginary Energy, a power company. 
# Field agents frequently answer customer questions about the company’s products and offerings. Information is scattered 
# across different sources, but you can access the following tools to retrieve relevant details:

# {tools}

# Use the following structured approach:

# ---
# **Format:**
# Question: the input question you must answer  
# Thought: you should always think about what to do  
# Action: the action to take, should be one of [{tool_names}]  
# Action Input: the input to the action  
# Observation: the result of the action  
# ... (this Thought/Action/Action Input/Observation can repeat N times)  
# Thought: I now know the final answer  
# Final Answer: the final answer to the original input question  
# ---

# Begin!

# Question: {input}
# Thought: {agent_scratchpad}
# """
# )


agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    if message.get("visible", True):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

if prompt := st.chat_input("Type your message here..."):
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.messages.append({"role": "user", "content": prompt})

    conversation_history = "\n".join(
        [f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages]
    )

    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        
        start_time = time.time()
        
        response = agent_executor.invoke(
            {"input": conversation_history}, {"callbacks": [st_callback]}
        )
        
        end_time = time.time()
        response_time = end_time - start_time

        response_content = response["output"]

        st.session_state.messages.append({"role": "assistant", "content": response_content})

        st.markdown(response_content)