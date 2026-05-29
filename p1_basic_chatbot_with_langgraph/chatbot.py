from aup_config import aup_setup
aup_setup()

from typing import Annotated
from typing_extensions import TypedDict
from IPython.display import display

from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

class State(TypedDict):
    """
    Messages have the type 'list'.
    The `add_messages` function defines how this state key should be updated
    (in this case, it appends messages to the list, rather than overwriting them)
    """
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

llm = init_chat_model() #qwen model3.5 used to test the single node model, url and api provided locally on the tutorial server

def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile()
display({"text/vnd.mermaid": graph.get_graph().draw_mermaid()}, raw=True)

def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)
stream_graph_updates('What are Large Language Models?')