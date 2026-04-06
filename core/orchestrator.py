from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd

from agents.analyst import Analyst, AnalystResult
from agents.engineer import DataEngineer, EngineerResult
from agents.visualizer import Visualizer, VisualOutput
from core.llm_client import OllamaClient, extract_json


QueryType = Literal["data", "visual", "diagnostic", "descriptive"]


def classify_query(query: str) -> QueryType:
    q = query.lower().strip()

    cleaning_terms = ["clean", "cleanup", "remove null", "fix data", "ingest", "upload", "parse file"]
    visual_terms = ["chart", "graph", "plot", "visualize", "visualisation", "heatmap", "bar chart", "line chart"]
    diagnostic_terms = ["why", "reason", "cause", "diagnose", "root cause", "explain drop", "what caused"]
    data_terms = ["rows", "columns", "missing", "duplicates", "clean", "import", "load"]

    if any(term in q for term in cleaning_terms):
        return "data"
    if any(term in q for term in visual_terms):
        return "visual"
    if any(term in q for term in diagnostic_terms):
        return "diagnostic"
    if any(term in q for term in data_terms):
        return "descriptive"
    return "descriptive"


def should_use_agentic_loop(query_type: QueryType) -> bool:
    return query_type in {"diagnostic", "descriptive", "visual"}


@dataclass
class OrchestratorState:
    dataframe: pd.DataFrame | None = None
    engineer_report: dict[str, Any] | None = None
    visual_output: VisualOutput | None = None
    last_answer: str | None = None
    last_query: str | None = None
    last_route: str | None = None


class Orchestrator:
    def __init__(self, llm: OllamaClient):
        self.llm = llm
        self.engineer = DataEngineer(llm)
        self.visualizer = Visualizer()
        self.analyst = Analyst(llm)

    def plan(self, query: str, query_type: QueryType) -> dict[str, Any]:
        prompt = f"""
Create a tiny plan for the query.

Return valid JSON only:
{{
  "route": "engineer|visualizer|analyst",
  "steps": ["..."],
  "output_style": "..."
}}

Query: {query}
Query type: {query_type}
""".strip()

        try:
            raw = self.llm.generate(
                prompt=prompt,
                system="Return valid JSON only.",
                temperature=0.0,
                num_predict=120,
            )
            parsed = extract_json(raw)
            if parsed and isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        route = {
            "data": "engineer",
            "visual": "visualizer",
            "diagnostic": "analyst",
            "descriptive": "analyst",
        }.get(query_type, "analyst")

        return {
            "route": route,
            "steps": [f"Use {route} on the current dataset."],
            "output_style": "concise",
        }

    def reflect(self, query: str, draft_answer: str, evidence: str) -> str:
        prompt = f"""
Convert the raw data result into a concise, professional answer to the user's query.
Do not mention Python, code, or the extraction method.

Query: {query}
Raw Result: {draft_answer}

Final Answer:
""".strip()

        try:
            raw = self.llm.generate(
                prompt=prompt,
                system="You are a clear, direct data analyst.",
                temperature=0.2,
                num_predict=200,
            ).strip()
            return raw if raw else draft_answer
        except Exception:
            return draft_answer

    def process_upload(self, file_path: str) -> EngineerResult:
        raw_df = self.engineer.ingest(file_path)
        result = self.engineer.clean(raw_df)
        return result

    def handle_query(
        self,
        df: pd.DataFrame,
        query: str,
        chat_history: list[dict[str, str]] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        query_type = classify_query(query)
        plan = self.plan(query, query_type)

        route = plan.get("route", "analyst")
        evidence_blob: dict[str, Any] = {"plan": plan, "query_type": query_type}

        if route == "visualizer" or query_type == "visual":
            visual = self.visualizer.build(df)
            evidence_blob["visual"] = {
                "kpis": visual.kpis,
                "insights": visual.insights,
                "chart_count": len(visual.charts),
            }
            draft = "\n".join(visual.insights) or "No chart could be generated from the available columns."
            final = self.reflect(query, draft, str(evidence_blob))
            return final, {"route": "visualizer", **evidence_blob, "visual_output": visual}

        analyst_result: AnalystResult = self.analyst.answer(df, query, chat_history)
        evidence_blob["analyst"] = analyst_result.evidence
        final = self.reflect(query, analyst_result.answer, str(evidence_blob))
        return final, {"route": "analyst", **evidence_blob}