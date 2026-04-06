"""
AInsights — Agent B: The Visualizer v3
Fixes: _CUSTOM_CHART_PROMPT defined at module level (was causing
AttributeError when accessed as self._CUSTOM_CHART_PROMPT), exec
sandbox cleaned up, re imported at module level.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

log = logging.getLogger("ainsights.agent_b")

MIN_ROWS_SCATTER    = 10
MIN_ROWS_BOX        = 15
MIN_ROWS_TIMESERIES = 5
MAX_CATEGORIES_BAR  = 25
MAX_CHARTS          = 7

# ── Custom chart prompt (module-level — accessible without self) ───────────
_CUSTOM_CHART_PROMPT = """\
You are a data visualisation expert using Python and Plotly.
Write Python code to create a single chart that satisfies the user's request.

STRICT RULES:
1. Output ONLY raw Python code. No markdown, no triple backticks, no explanation.
2. Available variables: df (pandas DataFrame), px (plotly.express), go (plotly.graph_objects), pd (pandas), np (numpy).
3. Assign your final figure to a variable named exactly: fig
4. Do NOT call fig.show() or fig.write_html() or fig.write_image().
5. Apply this theme call BEFORE the last line:
   fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(255,255,255,0.02)',font=dict(color='#cbd5e1',family='Inter,sans-serif',size=12),margin=dict(t=52,b=36,l=48,r=16),hoverlabel=dict(bgcolor='#1e293b',bordercolor='#334155',font_color='#f1f5f9'))

DataFrame columns (name · dtype · sample values):
{schema}

User request: {request}

Python code:"""


class VisualizerAgent:

    def run(self, csv_path: Path) -> list[dict]:
        log.info("Agent B: loading %s …", csv_path.name)
        df = pd.read_csv(csv_path)

        if df.empty:
            return []

        num_cols      = df.select_dtypes(include="number").columns.tolist()
        cat_cols      = [
            c for c in df.select_dtypes(include=["object", "category"]).columns
            if df[c].nunique() <= MAX_CATEGORIES_BAR
        ]
        datetime_cols = self._detect_datetime_cols(df)

        log.info("numeric=%d cat=%d datetime=%d", len(num_cols), len(cat_cols), len(datetime_cols))

        n       = len(df)
        figures: list[dict] = []

        if len(num_cols) >= 3:
            figures.append(self._correlation_heatmap(df, num_cols, n))

        if datetime_cols and num_cols and n >= MIN_ROWS_TIMESERIES:
            figures.extend(self._time_series(df, datetime_cols, num_cols, n))

        for col in num_cols[:3]:
            figures.append(self._histogram(df, col, n))

        if cat_cols and num_cols:
            figures.append(self._bar_chart(df, cat_cols[0], num_cols[0], n))

        if len(num_cols) >= 2 and n >= MIN_ROWS_SCATTER:
            figures.append(self._scatter(df, num_cols[0], num_cols[1], cat_cols, n))

        if cat_cols and num_cols and n >= MIN_ROWS_BOX:
            figures.append(self._box_plot(df, cat_cols[0], num_cols[0], n))

        figures.append(self._summary_table(df, num_cols))

        # Deduplicate by title
        seen:   set[str]  = set()
        unique: list[dict] = []
        for f in figures:
            title = f.get("layout", {}).get("title", {}).get("text", "")
            if title not in seen:
                seen.add(title)
                unique.append(f)

        log.info("Agent B: %d charts generated.", len(unique))
        return unique[:MAX_CHARTS]

    # ── Theme ─────────────────────────────────────────────────────────────────
    def _theme(self, fig: go.Figure, title: str) -> go.Figure:
        fig.update_layout(
            title={
                "text":    title,
                "font":    {"size": 14, "color": "#e2e8f0"},
                "x":       0.0,
                "xanchor": "left",
                "pad":     {"l": 4},
            },
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor ="rgba(255,255,255,0.02)",
            font={"color": "#cbd5e1", "family": "Inter, sans-serif", "size": 12},
            margin={"t": 48, "b": 36, "l": 48, "r": 16},
            hoverlabel={
                "bgcolor":     "#1e293b",
                "bordercolor": "#334155",
                "font_color":  "#f1f5f9",
                "font_size":   12,
            },
            xaxis={"gridcolor": "#1e293b", "linecolor": "#334155", "tickcolor": "#475569"},
            yaxis={"gridcolor": "#1e293b", "linecolor": "#334155", "tickcolor": "#475569"},
            colorway=[
                "#818cf8", "#34d399", "#fb923c", "#f472b6",
                "#38bdf8", "#a78bfa", "#facc15", "#f87171",
            ],
            legend={
                "bgcolor":     "rgba(15,23,42,0.7)",
                "bordercolor": "#334155",
                "borderwidth": 1,
            },
        )
        return fig

    def _to_json(self, fig: go.Figure) -> dict:
        return json.loads(fig.to_json())

    # ── Datetime detection ────────────────────────────────────────────────────
    def _detect_datetime_cols(self, df: pd.DataFrame) -> list[str]:
        dt_cols = df.select_dtypes(include="datetime").columns.tolist()
        for col in df.select_dtypes(include="object").columns:
            try:
                parsed = pd.to_datetime(df[col], errors="coerce",
                                        infer_datetime_format=True)
                if parsed.notna().mean() > 0.7:
                    df[col] = parsed
                    dt_cols.append(col)
            except Exception:
                pass
        return dt_cols

    # ── Auto chart builders ───────────────────────────────────────────────────
    def _correlation_heatmap(
        self, df: pd.DataFrame, num_cols: list[str], n: int
    ) -> dict:
        corr = df[num_cols].corr().round(2)
        fig  = go.Figure(
            go.Heatmap(
                z=corr.values,
                x=corr.columns.tolist(),
                y=corr.columns.tolist(),
                colorscale="RdBu",
                zmid=0,
                text=corr.values.round(2),
                texttemplate="%{text}",
                hovertemplate="<b>%{x}</b> × <b>%{y}</b><br>r = %{z:.3f}<extra></extra>",
            )
        )
        return self._to_json(
            self._theme(fig, f"Correlation Matrix — {len(num_cols)} numeric columns ({n:,} rows)")
        )

    def _time_series(
        self,
        df: pd.DataFrame,
        datetime_cols: list[str],
        num_cols: list[str],
        n: int,
    ) -> list[dict]:
        figs     = []
        date_col = datetime_cols[0]
        df_s     = df.sort_values(date_col)
        for num_col in num_cols[:2]:
            fig = px.line(df_s, x=date_col, y=num_col, markers=True)
            fig.update_traces(line_color="#818cf8",
                              marker={"color": "#818cf8", "size": 5})
            figs.append(self._to_json(
                self._theme(fig, f"{num_col} Over Time — {n:,} data points by {date_col}")
            ))
        return figs

    def _histogram(self, df: pd.DataFrame, col: str, n: int) -> dict:
        non_null = df[col].dropna()
        mean_val = non_null.mean()
        std_val  = non_null.std()
        fig      = px.histogram(df, x=col, nbins=40, opacity=0.82)
        fig.update_traces(marker_color="#818cf8")
        fig.add_vline(
            x=mean_val, line_dash="dash", line_color="#34d399",
            annotation_text=f"mean={mean_val:.2f}",
            annotation_font_color="#34d399",
            annotation_font_size=11,
        )
        return self._to_json(
            self._theme(
                fig,
                f"Distribution of {col} — {n:,} values · mean={mean_val:.2f} · σ={std_val:.2f}",
            )
        )

    def _bar_chart(
        self, df: pd.DataFrame, cat_col: str, num_col: str, n: int
    ) -> dict:
        agg = (
            df.groupby(cat_col)[num_col]
            .agg(["mean", "count"])
            .sort_values("mean", ascending=False)
            .reset_index()
        )
        agg.columns = [cat_col, f"avg_{num_col}", "count"]
        fig = px.bar(
            agg, x=cat_col, y=f"avg_{num_col}",
            labels={cat_col: cat_col, f"avg_{num_col}": f"Average {num_col}"},
            hover_data={"count": True},
        )
        fig.update_traces(marker_color="#34d399", marker_line_width=0)
        return self._to_json(
            self._theme(
                fig,
                f"Average {num_col} by {cat_col} — "
                f"{agg[cat_col].nunique()} categories · {n:,} rows",
            )
        )

    def _scatter(
        self,
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        cat_cols: list[str],
        n: int,
    ) -> dict:
        sample_n  = min(2000, n)
        df_sample = df.sample(sample_n, random_state=42) if n > sample_n else df
        color_col = cat_cols[0] if cat_cols else None
        fig       = px.scatter(
            df_sample, x=x_col, y=y_col, color=color_col,
            opacity=0.68, trendline="ols",
            trendline_color_override="#f472b6",
        )
        corr       = df[[x_col, y_col]].dropna().corr().iloc[0, 1]
        color_note = f" · coloured by {color_col}" if color_col else ""
        return self._to_json(
            self._theme(
                fig,
                f"{x_col} vs {y_col} — r={corr:.2f} · {sample_n:,} points{color_note}",
            )
        )

    def _box_plot(
        self, df: pd.DataFrame, cat_col: str, num_col: str, n: int
    ) -> dict:
        n_cats = df[cat_col].nunique()
        fig    = px.box(
            df, x=cat_col, y=num_col, color=cat_col,
            points="outliers",
        )
        return self._to_json(
            self._theme(
                fig,
                f"{num_col} Distribution by {cat_col} — "
                f"quartiles across {n_cats} categories · {n:,} rows",
            )
        )

    def _summary_table(self, df: pd.DataFrame, num_cols: list[str]) -> dict:
        if num_cols:
            stats = df[num_cols].describe().T.round(3).reset_index()
            stats.rename(columns={"index": "Column"}, inplace=True)
        else:
            stats = df.describe(include="all").T.reset_index()
            stats.rename(columns={"index": "Column"}, inplace=True)

        fig = go.Figure(
            go.Table(
                header={
                    "values":     [f"<b>{c}</b>" for c in stats.columns],
                    "fill_color": "#1e293b",
                    "font":       {"color": "#94a3b8", "size": 12},
                    "line_color": "#334155",
                    "align":      "left",
                    "height":     32,
                },
                cells={
                    "values":     [stats[c].astype(str).tolist() for c in stats.columns],
                    "fill_color": [["#0f172a", "#111827"] * (len(stats) + 1)],
                    "font":       {"color": "#e2e8f0", "size": 11},
                    "line_color": "#1e293b",
                    "align":      "left",
                    "height":     28,
                },
            )
        )
        return self._to_json(self._theme(fig, "Descriptive Statistics"))

    # ── Custom chart generation ───────────────────────────────────────────────
    def generate_custom_chart(self, csv_path: Path, request: str) -> dict | None:
        """
        Ask the LLM to write Plotly code for a user-specified chart,
        execute it in a restricted sandbox, and return Plotly JSON.
        Returns None on failure.
        """
        from langchain_ollama import OllamaLLM

        df = pd.read_csv(csv_path)

        # Build compact schema string
        schema_lines = []
        for col in df.columns[:30]:
            dtype  = str(df[col].dtype)
            sample = df[col].dropna().head(3).tolist()
            schema_lines.append(f"  {col} · {dtype} · {sample}")
        if len(df.columns) > 30:
            schema_lines.append(f"  … and {len(df.columns) - 30} more columns")
        schema_str = "\n".join(schema_lines)

        llm = OllamaLLM(
            model=os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=0.0,
            num_predict=512,
            request_timeout=180,
        )

        # Use the module-level prompt constant (not self._CUSTOM_CHART_PROMPT)
        prompt = _CUSTOM_CHART_PROMPT.format(
            schema=schema_str,
            request=request,
        )

        for attempt in range(2):
            try:
                raw = llm.invoke(prompt).strip()
                # Strip markdown fences if model adds them
                raw = re.sub(r"```(?:python)?|```", "", raw).strip()

                if not raw:
                    log.warning("Custom chart attempt %d: empty response.", attempt + 1)
                    continue

                log.info("Custom chart attempt %d: executing code …", attempt + 1)
                fig_json = self._exec_custom_chart(raw, df)

                if fig_json:
                    log.info("Custom chart generated on attempt %d.", attempt + 1)
                    return fig_json

            except Exception as exc:
                log.warning("Custom chart attempt %d failed: %s", attempt + 1, exc)

        log.error("Custom chart generation failed after 2 attempts.")
        return None

    def _exec_custom_chart(self, code: str, df: pd.DataFrame) -> dict | None:
        """
        Execute LLM-generated Plotly chart code in a restricted namespace.
        The code must assign its figure to a variable named `fig`.
        Returns Plotly JSON dict or None.
        """
        safe_globals: dict = {
            "__builtins__": {
                "len": len, "range": range, "list": list, "dict": dict,
                "str": str, "int": int, "float": float, "bool": bool,
                "round": round, "abs": abs, "min": min, "max": max,
                "sum": sum, "sorted": sorted, "print": print,
                "isinstance": isinstance, "enumerate": enumerate,
                "zip": zip, "any": any, "all": all,
            },
            "pd": pd,
            "np": np,
            "px": px,
            "go": go,
            "df": df.copy(),
        }
        local_ns: dict = {}

        exec(code, safe_globals, local_ns)  # noqa: S102

        # Look for `fig` in local namespace first, then globals
        fig = local_ns.get("fig") or safe_globals.get("fig")

        if fig is None:
            raise ValueError(
                "Generated code did not produce a variable named 'fig'. "
                f"Variables found: {list(local_ns.keys())}"
            )

        if not isinstance(fig, go.Figure):
            raise ValueError(
                f"'fig' is type {type(fig).__name__}, expected plotly.graph_objects.Figure."
            )

        return json.loads(fig.to_json())