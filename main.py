from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.schema import Document

import gradio as gr
import pandas as pd
import requests
from bs4 import BeautifulSoup


load_dotenv()

lang_map = {
    6: "Hindi",
    7: "Kannada",
    11: "Malayalam",
    20: "Tamil",
    21: "Telugu",
    24: "English"
}

df = pd.read_csv("data.csv")

def map_languages(value):
    if pd.isna(value):
        return ""
    codes = str(value).split(",")
    langs = [lang_map.get(int(c.strip()), f"Unknown({c})") for c in codes if c.strip().isdigit()]
    return ", ".join(langs)

docs = []
for idx, row in df.iterrows():
    content_lines = []
    for col, val in row.items():
        if col == "Released Languages":
            mapped_val = map_languages(val)
            content_lines.append(f"{col}: {mapped_val}")
        else:
            content_lines.append(f"{col}: {val}")
    page_content = "\n".join(content_lines)
    metadata = {"source": "bw_courses - Sheet1.csv", "row": idx}
    docs.append(Document(page_content=page_content, metadata=metadata))


splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=OpenAIEmbeddings(model='text-embedding-3-small'),
    collection_name='Boss_Wallah'
)

retriever = vector_store.as_retriever()


prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Answer ONLY from the provided transcript context."
               "Always answer in the **same language** as the question."
               "If the context is insufficient, just say you don't know."),
    ("system", "Context:\n{context}"),   
    ("human", "{question}")
])

model = ChatOpenAI(model='gpt-4')

parser=StrOutputParser()


def format_docs(all_docs):
    return "\n\n".join(doc.page_content for doc in all_docs)

parallel_chain = RunnableParallel({
    'context': RunnableLambda(lambda x: x["question"]) | retriever | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
}) 

final_chain = parallel_chain | prompt | model | parser

# RAG Tool 
def rag_search(query: str) -> str:
    return final_chain.invoke({'question': query})

rag_tool = Tool(
    name="Boss Wallah Dataset",
    func=rag_search,
    description="Useful for answering questions about the Boss Wallah CSV dataset (courses, languages, etc)."
)

# Web Search Tool 
def web_search(query: str) -> str:
    """Search the web using DuckDuckGo's HTML endpoint"""
    url = "https://duckduckgo.com/html/"
    resp = requests.get(url, params={"q": query}, headers={"User-Agent": "Mozilla/5.0"})
    if resp.status_code != 200:
        return "Search failed."
    soup = BeautifulSoup(resp.text, "html.parser")
    results = [a.get_text() for a in soup.select(".result__a")[:5]]
    return "\n".join(results) or "No results found."

web_tool = Tool(
    name="Web Search",
    func=web_search,
    description="Useful for finding real-world info like store locations, farming inputs, etc."
)


memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

system_instructions = """
You are Boss Wallah's support agent.

RULES:
- If the query is about Dataset related:
  -> ONLY use the 'Boss Wallah Dataset' tool. Do not use web search.
- If the query is unrelated to the dataset (like farming, real-world locations, stores):
  -> ONLY use the 'Web Search' tool.
- Never mix answers from both sources.
- Always answer in the **same language** as the question.
"""

agent = initialize_agent(
    tools=[rag_tool, web_tool],
    llm=model,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    handle_parsing_errors=True,
    verbose=True,
    agent_kwargs={"system_message" : system_instructions}
)

# Chatbot Function
def chatbot_fn(message, history):
    result = agent.invoke({"input": message})
    return result["output"]

# Gradio UI 
gr.ChatInterface(
    fn=chatbot_fn,
    title="Boss Wallah Agentic Chatbot",
    description="Ask questions about the Boss Wallah CSV dataset or real-world info (via web search).",
    chatbot=gr.Chatbot(type="messages"),
    type="messages"
).launch()
