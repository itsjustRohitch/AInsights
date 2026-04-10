from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from core.llm_client import OllamaClient, extract_code_block


@dataclass
class AnalystResult:
    answer: str
    evidence: dict[str, Any]
    grounded: bool


class Analyst:
    def __init__(self, llm: OllamaClient):
        self.llm = llm

    def build_memory(self, df: pd.DataFrame) -> None:
        pass

    def answer(
        self,
        df: pd.DataFrame,
        query: str,
        chat_history: list[dict[str, str]] | None = None,
    ) -> AnalystResult:
        
        prompt = f"""
        Write pandas code to answer the query.
        Dataframe variable: df
        Columns: {list(df.columns)}
        Query: {query}
        
        Rule: Assign the exact final output (string or number) to a variable named 'final_answer'.
        Output ONLY a python code block. Do not explain.
        """.strip()

        try:
            raw_response = self.llm.generate(
                prompt=prompt,
                system="You output only valid Python code blocks.",
                temperature=0.0,
                num_predict=150,
            )
            code = extract_code_block(raw_response)
            
            local_env = {"df": df.copy()}
            exec(code, {"pd": pd}, local_env)
            
            result_val = str(local_env.get("final_answer", "Error: 'final_answer' variable missing in execution state."))
            return AnalystResult(answer=result_val, evidence={"code": code}, grounded=True)
            
        except Exception as e:
            return AnalystResult(
                answer=f"Computation error: {str(e)}", 
                evidence={"attempted_code": code if 'code' in locals() else "None"}, 
                grounded=False
            )