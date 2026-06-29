"""
Day 1 toy graph: the minimum viable LangGraph.

Two nodes:
  1. greet        -> adds a greeting to state
  2. add_excited  -> reads state, adds enthusiasm

This has NO real logic - it exists purely to internalize the three
core concepts before building the real pipeline:

  - State:  a typed dict that flows through every node
  - Nodes:  plain functions, state in -> partial state out
  - Edges:  define what runs after what (here: linear, A -> B -> END)

"""
from typing import TypedDict

from langgraph.graph import StateGraph, END


class ToyState(TypedDict):
    name: str
    message: str


def greet(state: ToyState) -> dict:
    """Node 1: build a greeting from the input name."""
    return {"message": f"Hello, {state['name']}!"}


def add_excited(state: ToyState) -> dict:
    """Node 2: read the existing message, append to it."""
    return {"message": state["message"] + " Welcome to the pipeline!"}


def build_toy_graph():
    graph = StateGraph(ToyState)

    graph.add_node("greet", greet)
    graph.add_node("add_excited", add_excited)

    graph.set_entry_point("greet")
    graph.add_edge("greet", "add_excited")
    graph.add_edge("add_excited", END)

    return graph.compile()


_compiled_graph = build_toy_graph()


def run_toy_graph(name: str) -> dict:
    return _compiled_graph.invoke({"name": name, "message": ""})


if __name__ == "__main__":
    
    print(run_toy_graph("Jane Doe"))
