"""Agentic Excel analysis chatbot built around Gemini and LangGraph.

The pipeline keeps all numeric work in Python (pandas, scikit-learn, matplotlib)
and uses Gemini only for the final user-facing synthesis step.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, TypedDict

import matplotlib.pyplot as plt
import pandas as pd
from langchain.schema import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph
from sklearn.linear_model import LinearRegression


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
    routes: List[str]
    plots: List[str]
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
    """Choose which agents should answer based on the incoming question."""

    question = state["question"].lower()
    routes: list[str] = ["qa_agent"]

    stats_keywords = ["sum", "average", "mean", "median", "stats", "describe", "분석", "통계"]
    trend_keywords = ["trend", "forecast", "predict", "추세", "예측"]
    plot_keywords = ["plot", "chart", "graph", "시각화", "그래프"]
    ml_keywords = ["ml", "machine", "model", "회귀", "regression", "classification"]

    if any(keyword in question for keyword in stats_keywords):
        routes.append("statistics_agent")
    if any(keyword in question for keyword in trend_keywords):
        routes.append("trend_agent")
    if any(keyword in question for keyword in plot_keywords):
        routes.append("visualization_agent")
    if any(keyword in question for keyword in set(ml_keywords + trend_keywords)):
        routes.append("ml_agent")

    # Deduplicate while preserving order
    seen: set[str] = set()
    deduped = []
    for route in routes:
        if route not in seen:
            deduped.append(route)
            seen.add(route)

    state["routes"] = deduped
    return state


def _statistics_agent(state: ChatState) -> ChatState:
    """Pure-Python descriptive statistics for numeric and categorical columns."""

    df = state["context"].df
    summary = df.describe(include="all", datetime_is_numeric=True).transpose().reset_index()
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    top_head = df.head().to_markdown(index=False)
    summary_text = summary.to_markdown(index=False)
    notes: list[str] = []
    for col in numeric_cols:
        col_series = df[col].dropna()
        if col_series.empty:
            continue
        notes.append(
            f"{col}: mean={col_series.mean():.3f}, median={col_series.median():.3f}, std={col_series.std():.3f}, min={col_series.min():.3f}, max={col_series.max():.3f}"
        )

    content = (
        "Statistics preview\n" +
        f"Columns analyzed: {', '.join(numeric_cols) if numeric_cols else 'No numeric columns found.'}\n\n" +
        f"Data head:\n{top_head}\n\n" +
        f"Summary table:\n{summary_text}\n\n" +
        "Bullet insights:\n- " + "\n- ".join(notes if notes else ["No numeric insights computed."])
    )
    state["responses"].append(AgentResponse(name="statistics_agent", content=content))
    return state


def _trend_agent(state: ChatState) -> ChatState:
    """Python trend heuristics from deltas on numeric columns."""

    df = state["context"].df
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    trend_notes: list[str] = []
    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) < 3:
            continue
        diff = series.diff().dropna()
        direction = "increasing" if diff.mean() > 0 else "decreasing"
        trend_notes.append(
            f"{col}: overall {direction} pattern (avg Δ={diff.mean():.3f}, latest Δ={diff.iloc[-1]:.3f})."
        )
    if not trend_notes:
        trend_notes.append("No numeric columns with enough data to infer trends.")

    content = "Trend heuristics:\n- " + "\n- ".join(trend_notes)
    state["responses"].append(AgentResponse(name="trend_agent", content=content))
    return state


def _qa_agent(state: ChatState) -> ChatState:
    """Provide a schema-aware preview to help the final LLM ground its answer."""

    df = state["context"].df
    info_lines = [f"Rows: {len(df)}", f"Columns: {len(df.columns)}"]
    info_lines.extend([f"- {col}: {df[col].dtype}" for col in df.columns])
    preview = df.head().to_markdown(index=False)
    content = (
        "Dataset overview\n" +
        "\n".join(info_lines) +
        "\n\nPreview:\n" + preview
    )
    state["responses"].append(AgentResponse(name="qa_agent", content=content))
    return state


def _ml_agent(state: ChatState) -> ChatState:
    """Run a simple linear regression over index to give a forecast for numeric columns."""

    df = state["context"].df
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    forecasts: list[str] = []
    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) < 3:
            continue
        model = LinearRegression()
        X = pd.DataFrame({"index": range(len(series))})
        y = series.reset_index(drop=True)
        model.fit(X, y)
        next_index = [[len(series)]]
        predicted = model.predict(next_index)[0]
        forecasts.append(
            f"{col}: slope={model.coef_[0]:.4f}, intercept={model.intercept_:.4f}, next_value≈{predicted:.3f}"
        )

    content = "ML regression forecasts:\n- " + "\n- ".join(forecasts if forecasts else ["Not enough numeric data for regression."])
    state["responses"].append(AgentResponse(name="ml_agent", content=content))
    return state


def _visualization_agent(state: ChatState) -> ChatState:
    """Generate a quick Matplotlib plot based on user intent and available columns."""

    df = state["context"].df
    question = state["question"].lower()
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]

    selected_col = None
    for col in numeric_cols:
        if col.lower() in question:
            selected_col = col
            break
    if selected_col is None and numeric_cols:
        selected_col = numeric_cols[0]

    if selected_col:
        plt.figure(figsize=(6, 4))
        df[selected_col].plot(kind="line", title=f"{selected_col} over rows")
        plt.xlabel("Row index")
        plt.ylabel(selected_col)
        plot_path = state["context"].file_path.with_suffix("")
        plot_file = f"{plot_path.name}_{selected_col}_plot.png"
        plt.tight_layout()
        plt.savefig(plot_file)
        plt.close()
        state["plots"].append(plot_file)
        content = f"Generated line chart for {selected_col}: {plot_file}"
    else:
        content = "No numeric columns available for visualization."

    state["responses"].append(AgentResponse(name="visualization_agent", content=content))
    return state


def aggregate_responses(state: ChatState, llm_builder: Callable[[], ChatGoogleGenerativeAI]) -> ChatState:
    responses = state["responses"]
    bullet_list = "\n".join([f"- {resp.name}: {resp.content}" for resp in responses])
    plot_notes = "\n".join([f"- Plot saved at: {p}" for p in state.get("plots", [])])
    prompt = f"""
    You are the final drafting agent. Combine the upstream agent answers into a single
    concise reply tailored to the user's original question. Remove redundancy, keep the
    most reliable details, and present the result as a short narrative followed by 2-3
    actionable bullet points. Mention any generated plot paths so the user can view them.

    User question: {state['question']}

    Agent findings:
    {bullet_list}

    Plot artifacts:
    {plot_notes if plot_notes else 'No plots generated.'}
    """
    completion = llm_builder().invoke([HumanMessage(content=prompt)])
    state["final_answer"] = completion.content
    return state


def _run_agents(state: ChatState) -> ChatState:
    """Dispatch to each selected agent to gather Python-grounded findings."""

    agent_map: dict[str, Callable[[ChatState], ChatState]] = {
        "qa_agent": _qa_agent,
        "statistics_agent": _statistics_agent,
        "trend_agent": _trend_agent,
        "ml_agent": _ml_agent,
        "visualization_agent": _visualization_agent,
    }
    for route in state.get("routes", []):
        handler = agent_map.get(route)
        if handler:
            state = handler(state)
    return state


def build_workflow(api_key: str) -> StateGraph:
    """Create a LangGraph workflow with routing, multi-agent execution, and aggregation."""

    llm_builder = lambda: _build_llm(api_key)

    graph = StateGraph(ChatState)
    graph.add_node("router", route_question)
    graph.add_node("agent_driver", _run_agents)
    graph.add_node("aggregator", lambda state: aggregate_responses(state, llm_builder))

    graph.set_entry_point("router")
    graph.add_edge("router", "agent_driver")
    graph.add_edge("agent_driver", "aggregator")
    graph.add_edge("aggregator", END)
    return graph


def run_chat(file_path: str, question: str, api_key: str) -> str:
    """Convenience helper that returns only the final Gemini-composed answer."""

    final_state = run_chat_with_details(file_path=file_path, question=question, api_key=api_key)
    return final_state["final_answer"] or "No answer generated."


def run_chat_with_details(file_path: str, question: str, api_key: str) -> ChatState:
    """Execute the workflow and return the full state, including agent traces and plots.

    This is handy for interactive UIs (e.g., Streamlit) that want to surface
    intermediate agent insights and any generated chart artifacts alongside the
    final Gemini response.
    """

    context = load_excel(file_path)
    workflow = build_workflow(api_key).compile()
    initial_state: ChatState = {
        "question": question,
        "context": context,
        "responses": [],
        "routes": [],
        "plots": [],
        "final_answer": None,
    }
    return workflow.invoke(initial_state)
