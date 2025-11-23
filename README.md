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

## Architecture notes
- The router uses keyword heuristics to pick the best agent set for each question.
- Each agent performs Python-only calculations (pandas, scikit-learn, matplotlib) and records findings or plot paths.
- The aggregator merges agent outputs with Gemini and returns a concise narrative plus bullet points.

You can extend this template by adding new agents (e.g., anomaly detection, cohort analysis) and wiring them into the LangGraph edges.
