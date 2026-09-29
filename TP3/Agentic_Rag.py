import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_community.vectorstores import Chroma
from langchain_core.tools import create_retriever_tool
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings

load_dotenv()

LLM_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

# Small demo knowledge base (one document per employee profile)
chunks = [
    "Yassine is a software engineer with 5 years of experience and a salary of 5000 dollars.",
    "Sara is a data scientist with 3 years of experience and a salary of 4000 dollars.",
    "Ahmed is a product manager with 7 years of experience and a salary of 6000 dollars.",
    "Lina is a UX designer with 4 years of experience and a salary of 4500 dollars.",
    "Omar is a DevOps engineer with 6 years of experience and a salary of 5500 dollars.",
    "Nadia is a marketing specialist with 2 years of experience and a salary of 3500 dollars.",
    "Hassan is a sales manager with 8 years of experience and a salary of 7000 dollars.",
]

vector_store = Chroma.from_texts(
    texts=chunks,
    collection_name="cv_profile",
    embedding=OpenAIEmbeddings(),
)

retriever_tool = create_retriever_tool(
    vector_store.as_retriever(),
    name="cv_retriever",
    description="Search employee profiles (role, years of experience, salary).",
)


@tool
def get_employee_info(name: str):
    """
    Get information about an employee (mock HR system).
    """
    return {"name": name, "salary": 5000, "seniority": 5}


@tool
def send_email(email: str, subject: str, content: str):
    """
    Send an email to a recipient (simulated: printed to the console).
    """
    print(f"Email sent to {email} with subject '{subject}' and content '{content}'")
    return f"Email sent successfully to {email} with subject '{subject}'"


llm = ChatOpenAI(model=LLM_MODEL, temperature=0)

agent = create_agent(
    model=llm,
    tools=[retriever_tool, get_employee_info, send_email],
    system_prompt="Answer user queries using the provided tools.",
)


if __name__ == "__main__":
    from langchain_core.messages import HumanMessage

    resp = agent.invoke({"messages": [HumanMessage("Quel est le salaire de Sara ?")]})
    print(resp["messages"][-1].content)
