secret_key = "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"

from langchain.document_loaders import TextLoader

loader = TextLoader("data.txt")

from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS

documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(texts, embeddings)


retriever = db.as_retriever()

from langchain.agents.agent_toolkits import create_retriever_tool

tool = create_retriever_tool(
    retriever,
    "search_data",
    "Searches and returns documents from data file.",
)
tools = [tool]

from langchain.agents.agent_toolkits import create_conversational_retrieval_agent

from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(temperature=0)

agent_executor = create_conversational_retrieval_agent(llm, tools, verbose=True)


result = agent_executor({"input": "hi, im Russell"})

print(result["output"])


result = agent_executor(
    {
        "input": "briefly summarize the story"
    }
)

print(result["output"])

