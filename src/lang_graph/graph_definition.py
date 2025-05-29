from typing_extensions import TypedDict
from langchain_community.chat_models import ChatOllama
from ..web_search.serp import get_top_results
from ..web_search.faiss_stuff import retrieve_relevant_docs, create_index_with_text
from langchain.prompts import PromptTemplate
from langgraph.graph import END, StateGraph, START

class GraphState(TypedDict):
    query: str
    response: str
    retrieved: str


llm = ChatOllama(model="llama3.2", format="json", temperature=0)

def retrieve_context(state):
    web_docs = get_top_results(state["query"], 5)
    index, text_chunks = create_index_with_text(web_docs)
    top_docs = retrieve_relevant_docs(state["query"], index, text_chunks, 3)
    state["retrieved"] = " ".join(top_docs)
    return state

prompt_template = PromptTemplate(
    input_variables = ["query", "retrieved"],
        template="""
        You are an AI assistant. Use the following extracted passages to answer the question.

        Context:
        {retrieved}

        Question:
        {query}

        Answer:
        """
)

def call_llm(state):
    prompt = prompt_template.format(**state)
    output = llm.call_as_llm(prompt)
    state["response"] = output
    return state

def build_lang_graph():
    graph = StateGraph(GraphState)
    graph.add_node("llm_node", call_llm)
    graph.add_node("web_node", retrieve_context)

    graph.add_edge(START, "web_node")
    graph.add_edge("web_node", "llm_node")
    graph.add_edge("llm_node", END)

    app = graph.compile()
    return app

app = build_lang_graph()
input = GraphState(query="What is azure content understanding?")
output = app.invoke(input)
print(output)
