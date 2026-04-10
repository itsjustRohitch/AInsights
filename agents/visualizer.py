from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import altair as alt
import pandas as pd


def detect_column_types(df: pd.DataFrame) -> dict[str, list[str]]:
    numeric_cols = []
    categorical_cols = []
    date_cols = []

    for col in df.columns:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            numeric_cols.append(col)
            continue

        if pd.api.types.is_datetime64_any_dtype(s):
            date_cols.append(col)
            continue

        # Try lightweight date detection on object columns
        if s.dtype == "object":
            sample = s.dropna().astype(str).head(20)
            parsed = pd.to_datetime(sample, errors="coerce", infer_datetime_format=True)
            if parsed.notna().mean() >= 0.7:
                date_cols.append(col)
            else:
                nunique = s.nunique(dropna=True)
                if 1 <= nunique <= max(20, int(len(df) * 0.2)):
                    categorical_cols.append(col)
        else:
            categorical_cols.append(col)

    return {
        "numeric": numeric_cols,
        "categorical": categorical_cols,
        "date": date_cols,
    }


def _safe_top_categories(df: pd.DataFrame, col: str, limit: int = 10) -> pd.Series:
    return df[col].astype(str).value_counts().head(limit)


@dataclass
class VisualOutput:
    charts: list[alt.Chart]
    kpis: dict[str, Any]
    insights: list[str]


class Visualizer:
    def build(self, df: pd.DataFrame) -> VisualOutput:
        types = detect_column_types(df)
        charts: list[alt.Chart] = []
        insights: list[str] = []
        kpis: dict[str, Any] = {}

        numeric_cols = types["numeric"]
        categorical_cols = types["categorical"]
        date_cols = types["date"]

        if numeric_cols:
            for col in numeric_cols[:3]:
                kpis[f"mean_{col}"] = float(pd.to_numeric(df[col], errors="coerce").mean())
                kpis[f"sum_{col}"] = float(pd.to_numeric(df[col], errors="coerce").sum())

        if date_cols and numeric_cols:
            date_col = date_cols[0]
            num_col = numeric_cols[0]
            temp = df[[date_col, num_col]].copy()
            temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce")
            temp = temp.dropna(subset=[date_col, num_col]).sort_values(date_col)

            if not temp.empty:
                chart = (
                    alt.Chart(temp)
                    .mark_line()
                    .encode(
                        x=alt.X(f"{date_col}:T", title=date_col),
                        y=alt.Y(f"{num_col}:Q", title=num_col),
                        tooltip=[date_col, num_col],
                    )
                    .properties(title=f"{num_col} over {date_col}")
                )
                charts.append(chart)
                insights.append(f"Built line chart using {date_col} and {num_col}.")

        elif categorical_cols and numeric_cols:
            cat_col = categorical_cols[0]
            num_col = numeric_cols[0]
            temp = df[[cat_col, num_col]].copy()
            temp[cat_col] = temp[cat_col].astype(str)
            temp = temp.dropna()

            if not temp.empty:
                grouped = temp.groupby(cat_col, as_index=False)[num_col].mean()
                grouped = grouped.sort_values(num_col, ascending=False).head(15)

                chart = (
                    alt.Chart(grouped)
                    .mark_bar()
                    .encode(
                        x=alt.X(f"{cat_col}:N", sort="-y", title=cat_col),
                        y=alt.Y(f"{num_col}:Q", title=f"Mean {num_col}"),
                        tooltip=[cat_col, num_col],
                    )
                    .properties(title=f"Mean {num_col} by {cat_col}")
                )
                charts.append(chart)
                insights.append(f"Built bar chart using {cat_col} and {num_col}.")

        if len(categorical_cols) >= 2 and numeric_cols:
            c1, c2 = categorical_cols[:2]
            num_col = numeric_cols[0]
            temp = df[[c1, c2, num_col]].copy()
            temp[c1] = temp[c1].astype(str)
            temp[c2] = temp[c2].astype(str)
            temp = temp.dropna()

            if not temp.empty:
                agg = (
                    temp.groupby([c1, c2], as_index=False)[num_col]
                    .mean()
                    .sort_values(num_col, ascending=False)
                    .head(100)
                )

                chart = (
                    alt.Chart(agg)
                    .mark_rect()
                    .encode(
                        x=alt.X(f"{c1}:N", title=c1),
                        y=alt.Y(f"{c2}:N", title=c2),
                        color=alt.Color(f"{num_col}:Q", title=f"Mean {num_col}"),
                        tooltip=[c1, c2, num_col],
                    )
                    .properties(title=f"Heatmap of {num_col}")
                )
                charts.append(chart)
                insights.append(f"Built heatmap using {c1}, {c2}, and {num_col}.")

        if categorical_cols:
            top_cat = categorical_cols[0]
            top_values = _safe_top_categories(df, top_cat)
            insights.append(
                f"Top values in {top_cat}: "
                + ", ".join([f"{idx} ({val})" for idx, val in top_values.items()])
            )

        return VisualOutput(charts=charts, kpis=kpis, insights=insights)