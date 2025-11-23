"""Streamlit UI for the Gemini-powered Excel agentic chatbot."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

from excel_agentic_chatbot import run_chat_with_details


st.set_page_config(
    page_title="Excel Agentic Copilot (Gemini)",
    page_icon="📊",
    layout="wide",
)


def _inject_modern_css() -> None:
    """Add a light gradient, glass cards, and chip styling."""

    st.markdown(
        """
        <style>
            body {background: radial-gradient(circle at 10% 20%, #eef2ff 0, #ffffff 50%, #f8fafc 100%);} 
            .glass-card {
                background: rgba(255,255,255,0.7);
                border-radius: 18px;
                border: 1px solid rgba(99,102,241,0.12);
                box-shadow: 0 10px 40px rgba(99,102,241,0.08);
                padding: 1rem 1.25rem;
                backdrop-filter: blur(8px);
            }
            .chip {
                display: inline-flex;
                align-items: center;
                padding: 6px 12px;
                margin: 4px 6px 0 0;
                border-radius: 999px;
                background: #eef2ff;
                color: #312e81;
                font-weight: 600;
                border: 1px solid #c7d2fe;
                cursor: pointer;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _save_upload(upload) -> Optional[Path]:
    """Persist the uploaded Excel file to disk for downstream agents."""

    if upload is None:
        return None
    suffix = Path(upload.name).suffix or ".xlsx"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(upload.read())
        return Path(tmp.name)


def _render_agent_timeline(state) -> None:
    st.subheader("Agent Timeline ✨")
    if not state.get("responses"):
        st.info("아직 에이전트 결과가 없습니다. 질문을 입력해 보세요.")
        return
    for resp in state["responses"]:
        with st.container():
            st.markdown(
                f"<div class='glass-card'><h4>🤖 {resp.name}</h4><pre>{resp.content}</pre></div>",
                unsafe_allow_html=True,
            )


def _render_plots(state) -> None:
    plots = state.get("plots", [])
    if not plots:
        st.caption("생성된 그래프가 없습니다.")
        return
    cols = st.columns(min(3, len(plots)))
    for idx, plot_path in enumerate(plots):
        path_obj = Path(plot_path)
        if path_obj.exists():
            with cols[idx % len(cols)]:
                st.image(str(path_obj), caption=path_obj.name, use_column_width=True)
        else:
            st.warning(f"이미지 파일을 찾을 수 없습니다: {plot_path}")


def _render_data_snapshot(state) -> None:
    ctx = state.get("context")
    if not ctx:
        return
    df: pd.DataFrame = ctx.df
    st.subheader("Data Snapshot")
    st.caption(f"행 {len(df)}개 · 열 {len(df.columns)}개")
    st.dataframe(df.head(20))


_inject_modern_css()

st.title("📈 Excel Agentic Copilot")
st.write("Gemini + LangGraph 기반으로 라우팅·다중 에이전트·시각화를 결합한 혁신 UI")

if "question_input" not in st.session_state:
    st.session_state["question_input"] = "매출 추세를 요약하고 그래프를 보여줘"

with st.sidebar:
    st.header("⚙️ Control Center")
    api_key = st.text_input("Gemini API Key", value=os.getenv("GOOGLE_API_KEY", ""), type="password")
    uploaded = st.file_uploader("엑셀 파일 업로드 (.xlsx)", type=["xlsx", "xlsm", "xlsb", "xls"])
    st.text(" ")
    st.caption("질문 프롬프트")
    question = st.text_area("무엇을 분석할까요?", key="question_input", height=120)
    sample_cols = st.columns(2)
    with sample_cols[0]:
        if st.button("📊 통계 요약"):
            st.session_state["question_input"] = "각 열의 평균과 표준편차를 알려줘"
            st.experimental_rerun()
    with sample_cols[1]:
        if st.button("🤖 ML 예측 + 그래프"):
            st.session_state["question_input"] = "다음 분기 매출을 회귀로 예측하고 추세 그래프를 그려줘"
            st.experimental_rerun()
    run_clicked = st.button("Analyze 🚀", type="primary")

if "last_state" not in st.session_state:
    st.session_state["last_state"] = {}

if run_clicked:
    if not api_key:
        st.error("Gemini API Key를 입력해주세요.")
    elif uploaded is None:
        st.error("엑셀 파일을 업로드해주세요.")
    elif not question.strip():
        st.error("질문을 입력해주세요.")
    else:
        saved_path = _save_upload(uploaded)
        if saved_path is None:
            st.error("파일 저장에 실패했습니다. 다시 시도해주세요.")
        else:
            with st.spinner("에이전트가 데이터를 분석하고 있습니다..."):
                state = run_chat_with_details(
                    file_path=str(saved_path),
                    question=question,
                    api_key=api_key,
                )
                st.session_state["last_state"] = state
                st.success("완료! 아래 결과를 확인하세요.")

state = st.session_state.get("last_state", {})

if state.get("final_answer"):
    st.markdown(
        f"""
        <div class='glass-card'>
            <h3>최종 Gemini 답변</h3>
            <div>{state['final_answer']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

cols = st.columns([1, 1])
with cols[0]:
    _render_agent_timeline(state)
with cols[1]:
    st.subheader("Plots & Visuals")
    _render_plots(state)

with st.expander("데이터 미리보기"):
    _render_data_snapshot(state)

st.caption("모든 계산은 파이썬 내에서 처리되고 Gemini는 최종 답변만 담당합니다.")
