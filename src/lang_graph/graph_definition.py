from typing_extensions import TypedDict
from langchain_ollama import ChatOllama
from ..web_search.serp import get_top_results
from ..web_search.faiss_stuff import retrieve_relevant_docs, create_index_with_text, IndexStruct
from langchain.prompts import PromptTemplate
from langgraph.graph import END, StateGraph, START
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager



class GraphState(TypedDict):
    query: str
    history: list[BaseMessage]
    retrieved: str
    index: IndexStruct


llm = ChatOllama(model="llama3.2", temperature=0, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))

def retrieve_context(state: GraphState):
    web_docs = get_top_results(state["query"], 5)
    # index, text_chunks = create_index_with_text(web_docs)
    # top_docs = retrieve_relevant_docs(state["query"], index, text_chunks, 3)
    state["websearch_result"] = " ".join(web_docs)
    return state

def build_rag_index(state: GraphState):
    with open("corpus.txt", "r", encoding="utf-8") as file:
        corpus = file.read()
        index, chunks = create_index_with_text([corpus])
        state["index"] = IndexStruct(chunks, index)
        print("Built RAG index")
    return state

def rag_search(state: GraphState):
    index_struct = state["index"]
    top_docs = retrieve_relevant_docs(state["query"], index_struct.index, index_struct.chunks, 5)
    state["retrieved"] = " ".join(top_docs)
    return state

def get_user_input(state: GraphState):
    state["query"] = input("... ")
    return state

prompt_template = PromptTemplate(
    input_variables = ["retrieved"],
        template="""
        You are an AI assistant. Use the following extracted passages to help you answer the question.

        Context:
        {retrieved}
        """
)

def router(state: GraphState):
    prompt = SystemMessage(
        "You are a smart router.\n\n"
        "1️⃣ If you can answer the user's query directly from your knowledge, respond with exactly:\n"
        "   {\n"
        "     \"tool\": \"none\",\n"
        "     \"answer\": \"<your answer here>\"\n"
        "   }\n\n"
        "2️⃣ If you think a web search is needed to get the best, up-to-date answer, respond with exactly:\n"
        "   {\n"
        "     \"tool\": \"web_search\",\n"
        "     \"search_query\": \"<search engine optimized query>\"\n"
        "   }\n\n"
        "3. If it's a question about AI Agents, LangChain or LLMs, use the RAG system. It has information from LangChain's documentation. Respond with exactly:\n"
        "   {\n"
        "     \"tool\": \"rag\",\n"
        "     \"search_query\": \"<RAG system optimized query>\"\n"
        "   }\n\n"
        "Make the JSON valid, with no extra keys."
    )
    user_message = HumanMessage(state["query"])
    state["history"].append(user_message)
    
    raw_output = llm.invoke([prompt] + state["history"])
    print(raw_output)
    parser = JsonOutputParser()
    parsed = parser.parse(raw_output.content)
    if parsed["tool"] == "none":
        ai_message = AIMessage(parsed["answer"])
        state["history"].append(ai_message)
    else:
        state["query"] = parsed["search_query"]
    return parsed["tool"]

def augmented_llm(state: GraphState):
    prompt = prompt_template.format(retrieved=state["retrieved"])
    output = llm.invoke([prompt] + state["history"])
    ai_message = AIMessage(output.content)
    state["history"].append(ai_message)
    return state

def build_lang_graph():
    graph = StateGraph(GraphState)
    graph.add_node("user_input", get_user_input)
    graph.add_node("augmented_llm_node", augmented_llm)
    graph.add_node("build_index", build_rag_index)
    graph.add_node("rag_search_node", rag_search)
    graph.add_node("web_node", retrieve_context)
    graph.add_edge(START, "build_index")
    graph.add_edge("build_index", "user_input")
    graph.add_conditional_edges(
        "user_input",
        router,
        {
            "web_search": "web_node",
            "none": "user_input",
            "rag": "rag_search_node"
        }
    )
    graph.add_edge("web_node", "augmented_llm_node")
    graph.add_edge("rag_search_node", "augmented_llm_node")
    graph.add_edge("augmented_llm_node", "user_input")

    app = graph.compile()
    return app

app = build_lang_graph()
input_state = GraphState(history=[])
output = GraphState(history=[""])
output = app.invoke(input_state)
