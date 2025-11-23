# Agentic Excel Chatbot (Gemini + LangGraph)

This repo demonstrates a minimal agentic chatbot for Excel analysis using Gemini's API, LangChain, and LangGraph. The flow is:

1. **Router** classifies the user's question (statistics, trends/forecasting, or general QA).
2. A specialized **agent** (statistics, trend, or QA) generates an answer grounded in the spreadsheet.
3. The **final aggregator** LLM composes a concise response tailored to the question.

## Features
- Gemini 1.5 Pro via `langchain-google-genai` for all LLM calls.
- `pandas` ingestion and light cleaning of uploaded Excel files.
- LangGraph workflow with conditional routing and response aggregation.

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
       question="지난 분기 매출 추세를 알려줘",
       api_key="${GOOGLE_API_KEY}",
   )
   print(answer)
   PY
   ```

## Architecture notes
- The router uses keyword heuristics to pick the best agent for each question.
- Each agent builds a context-rich prompt with dataframe previews and heuristic findings to reduce hallucinations.
- The aggregator merges agent outputs and returns a concise narrative plus bullet points.

You can extend this template by adding new agents (e.g., anomaly detection, cohort analysis) and wiring them into the LangGraph edges.
