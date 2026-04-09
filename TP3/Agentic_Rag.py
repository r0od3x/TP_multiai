from langchain.agents import create_agent
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from  langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.tools import create_retriever_tool


load_dotenv()


chunks = [
    "Yassine is a software engineer with 5 years of experience and a salary of 5000 dollars.",
    "Sara is a data scientist with 3 years of experience and a salary of 4000 dollars.",
    "Ahmed is a product manager with 7 years of experience and a salary of 6000 dollars."
    "Lina is a UX designer with 4 years of experience and a salary of 4500 dollars."
    "Omar is a DevOps engineer with 6 years of experience and a salary of 5500 dollars."
    "Nadia is a marketing specialist with 2 years of experience and a salary of 3500 dollars."
    "Hassan is a sales manager with 8 years of experience and a salary of 7000 dollars."
]
embedding_model = OpenAIEmbeddings()

vector_store = Chroma.from_texts(
    texts= chunks,
    collection_name="cv_profile",
    embedding=embedding_model)

retriever = vector_store.as_retriever()
retriever_tool = create_retriever_tool(retriever, name="cv_retriever", description="get information abt me from cv")



@tool
def get_employee_info(name:str):
    """ 
    Get information about an employee .
    """
    return {"name":name , "salary": 5000, "seniority":5}

@tool
def send_email(email:str, subject:str,content:str):
    """
    Send an email to a recipient.
    """
    print(f"Email sent to {email} with subject '{subject}' and content '{content}'")
    
    return f"Email sent successfully to {email} with subject '{subject}' and content '{content}'"


llm = ChatOpenAI(model="gpt-4o", temperature=0)

agent = create_agent(
    model=llm,
    tools=[get_employee_info, send_email],
    system_prompt="answer to user querie using provided tools")

# resp=agent.invoke(input={
#     "messages":[HumanMessage("Quel est le salaire de yassine")] })

# print(resp['messages'][-1].content)

