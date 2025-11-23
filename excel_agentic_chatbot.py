"""Agentic Excel analysis chatbot built around Gemini and LangGraph."""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, TypedDict

import pandas as pd
from langchain.schema import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph


@dataclass
class ExcelContext:
    """Container for the loaded workbook so agents share the same view."""

    df: pd.DataFrame
    file_path: Path


@dataclass
class AgentResponse:
    """Lightweight structure for agent answers."""

    name: str
    content: str


class ChatState(TypedDict):
    """Shared state flowing through the LangGraph pipeline."""

    question: str
    context: ExcelContext
    responses: List[AgentResponse]
    route: str | None
    final_answer: str | None


def _build_llm(api_key: str) -> ChatGoogleGenerativeAI:
    """Instantiate the Gemini model used across agents."""

    return ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0.3,
        google_api_key=api_key,
    )


def load_excel(file_path: str) -> ExcelContext:
    """Load an Excel workbook into a dataframe with basic cleaning."""

    path = Path(file_path)
    df = pd.read_excel(path)
    df.columns = [str(col).strip() for col in df.columns]
    return ExcelContext(df=df, file_path=path)


def route_question(state: ChatState) -> ChatState:
    """Choose which agent should answer based on the incoming question."""

    question = state["question"].lower()
    if any(keyword in question for keyword in ["trend", "forecast", "predict"]):
        route = "trend_agent"
    elif any(keyword in question for keyword in ["sum", "average", "mean", "median", "stats", "describe"]):
        route = "statistics_agent"
    else:
        route = "qa_agent"

    state["route"] = route
    return state


def _statistics_agent(state: ChatState, llm_builder: Callable[[], ChatGoogleGenerativeAI]) -> ChatState:
    df = state["context"].df
    summary = df.describe(include="all", datetime_is_numeric=True).transpose().reset_index()
    summary_text = summary.to_markdown(index=False)
    prompt = f"""
    You are the statistics agent. Provide a concise, human-friendly explanation of the key
    descriptive metrics for the uploaded spreadsheet. Keep the tone factual and cite any
    interesting outliers.

    Spreadsheet preview:
    {df.head().to_markdown(index=False)}

    Summary table:
    {summary_text}
    """
    completion = llm_builder().invoke([HumanMessage(content=prompt)])
    response = AgentResponse(name="statistics_agent", content=completion.content)
    state["responses"].append(response)
    return state


def _trend_agent(state: ChatState, llm_builder: Callable[[], ChatGoogleGenerativeAI]) -> ChatState:
    df = state["context"].df
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    trend_notes: list[str] = []
    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) < 2:
            continue
        diff = series.diff().dropna()
        direction = "increasing" if diff.mean() > 0 else "decreasing"
        trend_notes.append(f"Column '{col}' shows an overall {direction} pattern (Δ̄={diff.mean():.3f}).")
    if not trend_notes:
        trend_notes.append("No numeric columns with enough data to infer trends.")

    prompt = f"""
    You are the trend agent. Review the user's spreadsheet and provide a short summary of
    directional trends, patterns, or seasonality that a business user should know about.
    Use the provided heuristics when available.

    Spreadsheet preview:
    {df.head().to_markdown(index=False)}

    Heuristic findings:
    {' '.join(trend_notes)}
    """
    completion = llm_builder().invoke([HumanMessage(content=prompt)])
    response = AgentResponse(name="trend_agent", content=completion.content)
    state["responses"].append(response)
    return state


def _qa_agent(state: ChatState, llm_builder: Callable[[], ChatGoogleGenerativeAI]) -> ChatState:
    df = state["context"].df
    prompt = f"""
    You are the QA agent. Answer the user's question using only the spreadsheet content.
    Quote concrete numbers when possible. If the question requires filtering, groupby,
    or joins, explain your reasoning before answering.

    User question: {state['question']}
    Spreadsheet preview:
    {df.head().to_markdown(index=False)}
    """
    completion = llm_builder().invoke([HumanMessage(content=prompt)])
    response = AgentResponse(name="qa_agent", content=completion.content)
    state["responses"].append(response)
    return state


def aggregate_responses(state: ChatState, llm_builder: Callable[[], ChatGoogleGenerativeAI]) -> ChatState:
    responses = state["responses"]
    bullet_list = "\n".join([f"- {resp.name}: {resp.content}" for resp in responses])
    prompt = f"""
    You are the final drafting agent. Combine the upstream agent answers into a single
    concise reply tailored to the user's original question. Remove redundancy, keep the
    most reliable details, and present the result as a short narrative followed by 2-3
    actionable bullet points.

    User question: {state['question']}

    Agent findings:
    {bullet_list}
    """
    completion = llm_builder().invoke([HumanMessage(content=prompt)])
    state["final_answer"] = completion.content
    return state


def build_workflow(api_key: str) -> StateGraph:
    """Create a LangGraph workflow with routing and aggregation."""

    llm_builder = lambda: _build_llm(api_key)

    graph = StateGraph(ChatState)
    graph.add_node("router", route_question)
    graph.add_node("statistics_agent", lambda state: _statistics_agent(state, llm_builder))
    graph.add_node("trend_agent", lambda state: _trend_agent(state, llm_builder))
    graph.add_node("qa_agent", lambda state: _qa_agent(state, llm_builder))
    graph.add_node("aggregator", lambda state: aggregate_responses(state, llm_builder))

    graph.set_entry_point("router")
    graph.add_conditional_edges(
        "router",
        lambda state: state["route"],
        {
            "statistics_agent": "statistics_agent",
            "trend_agent": "trend_agent",
            "qa_agent": "qa_agent",
        },
    )
    graph.add_edge("statistics_agent", "aggregator")
    graph.add_edge("trend_agent", "aggregator")
    graph.add_edge("qa_agent", "aggregator")
    graph.add_edge("aggregator", END)
    return graph


def run_chat(file_path: str, question: str, api_key: str) -> str:
    """Convenience helper to run the full agent workflow."""

    context = load_excel(file_path)
    workflow = build_workflow(api_key).compile()
    initial_state: ChatState = {
        "question": question,
        "context": context,
        "responses": [],
        "route": None,
        "final_answer": None,
    }
    final_state = workflow.invoke(initial_state)
    return final_state["final_answer"] or "No answer generated."
