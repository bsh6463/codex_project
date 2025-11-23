# Agentic Excel Chatbot (Gemini + LangGraph)

This repo demonstrates a minimal agentic chatbot for Excel analysis using Gemini's API, LangChain, and LangGraph. The flow is:

1. **Router** classifies the user's question (statistics, trends/forecasting, ML, visualization, or general QA).
2. Multiple specialized **agents** run in Python: descriptive stats, trend heuristics, regression-based ML forecasts, and optional Matplotlib chart generation based on the prompt.
3. The **final aggregator** LLM composes a concise response that references all agent findings and any generated plot paths.

## Features
- Gemini 1.5 Pro via `langchain-google-genai` for the synthesis step.
- `pandas` ingestion and light cleaning of uploaded Excel files.
- Pure-Python stats/trend/ML calculations with `pandas` + `scikit-learn`.
- Matplotlib charting that saves a PNG when the question requests a plot/그래프.
- LangGraph workflow with routing, multi-agent execution, and response aggregation.

## Quickstart
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Export your Gemini API key:
   ```bash
   export GOOGLE_API_KEY="your-key"
   ```
3. Run the workflow:
   ```bash
   python - <<'PY'
   from excel_agentic_chatbot import run_chat

   answer = run_chat(
       file_path="/path/to/workbook.xlsx",
       question="지난 분기 매출 추세를 알려줘. 그래프도 보여줘",
       api_key="${GOOGLE_API_KEY}",
   )
   print(answer)
   PY
   ```

Generated plot files (if requested) will be saved next to the workbook path with a `_plot.png` suffix and referenced in the final answer.

## Streamlit cockpit (혁신 UI)
Launch the interactive front-end with Gemini wiring, agent traces, and inline plots:

```bash
export GOOGLE_API_KEY="your-key"
streamlit run streamlit_app.py
```

Features of the UI:
- Upload Excel files and ask questions from the sidebar "Control Center".
- One-click prompt chips for common tasks (통계 요약, ML 예측 + 그래프).
- Live agent timeline showing each agent's raw findings and any generated charts inline.
- Data preview table so you can validate the dataset before trusting the answer.

## Architecture notes
- The router uses keyword heuristics to pick the best agent set for each question.
- Each agent performs Python-only calculations (pandas, scikit-learn, matplotlib) and records findings or plot paths.
- The aggregator merges agent outputs with Gemini and returns a concise narrative plus bullet points.

You can extend this template by adding new agents (e.g., anomaly detection, cohort analysis) and wiring them into the LangGraph edges.

## 초보자를 위한 동작 설명
이 프로젝트는 "엑셀 파일을 주고, 질문을 하면 AI가 대신 분석해 답변하는" 파이썬 프로그램입니다. 주요 아이디어를 단계별로 설명하면 다음과 같습니다.

1. **엑셀 불러오기**: `pandas`가 엑셀 파일을 읽어 공통 데이터프레임(`ExcelContext`)으로 만듭니다. 모든 에이전트가 이 데이터를 공유합니다.
2. **질문 분류(라우터)**: 사용자의 질문에 들어 있는 키워드로 어떤 에이전트가 필요할지 결정합니다. 예를 들어 "평균"이 있으면 통계 에이전트, "그래프"가 있으면 시각화 에이전트가 선택됩니다.
3. **여러 에이전트 실행** (모두 파이썬 계산):
   - `qa_agent`: 데이터 크기와 열 타입, 앞부분 미리보기 제공.
   - `statistics_agent`: `pandas.describe`와 요약 통계를 계산.
   - `trend_agent`: 숫자 열의 증가/감소 추세를 간단히 파악.
   - `ml_agent`: 행 인덱스를 기준으로 선형회귀를 돌려 다음 값을 예측.
   - `visualization_agent`: 숫자 열로 Matplotlib 그래프를 그려 PNG 파일을 저장.
4. **최종 답변 작성**: 모든 에이전트의 결과(그래프 경로 포함)를 모아 Gemini LLM이 한글/영문 요약을 만듭니다. 핵심 요약과 함께 그래프 파일 경로를 알려주므로 이미지를 직접 확인할 수 있습니다.

### 실제 사용 흐름 예시
1. `requirements.txt`로 필요한 패키지를 설치합니다.
2. `GOOGLE_API_KEY` 환경변수에 Gemini API 키를 넣습니다.
3. `run_chat(엑셀경로, 질문, api_key)` 함수를 호출하면, 위 단계가 자동으로 실행되어 최종 답변 문자열을 돌려줍니다.

이 구조를 이해하면, 새로운 에이전트를 추가하거나 라우터 키워드를 바꾸는 식으로 손쉽게 기능을 확장할 수 있습니다.
