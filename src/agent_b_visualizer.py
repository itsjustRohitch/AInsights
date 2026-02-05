import streamlit as st
import altair as alt
import pandas as pd
import numpy as np

class AgentB_Visualizer:
    """
    Agent B: The Adaptive Visualizer.
    
    Role:
    1. Scan the dataframe structure.
    2. Dynamically select the best charts based on available data types.
    3. Render 'Grounded' visuals (only using what exists).
    """

    def __init__(self, df):
        self.df = df
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        self.date_cols = df.select_dtypes(include=['datetime', 'datetimetz']).columns.tolist()
        
        # Heuristic: Try to find columns that look like dates but aren't typed yet
        # (In case Agent A missed one, though Agent A is pretty good now)
        if not self.date_cols:
            for col in self.categorical_cols:
                if 'date' in col.lower() or 'time' in col.lower():
                    try:
                        self.df[col] = pd.to_datetime(self.df[col])
                        self.date_cols.append(col)
                        self.categorical_cols.remove(col)
                    except:
                        pass

    def render_overview(self):
        """Main render loop."""
        if self.df is None or self.df.empty:
            st.warning("Agent B: DataFrame is empty. Nothing to visualize.")
            return

        # 1. STRATEGY REPORT (Agent Explains itself)
        with st.expander("👁️ Agent B: Visualization Strategy", expanded=False):
            st.write(f"**Detected Structure:**")
            st.write(f"- 🔢 **Metrics (Numeric):** {self.numeric_cols}")
            st.write(f"- 🔠 **Dimensions (Categorical):** {self.categorical_cols}")
            st.write(f"- 📅 **Timeline (Dates):** {self.date_cols}")
            
            strategy = []
            if self.numeric_cols: strategy.append("✅ Generating KPI Cards for top metrics.")
            if self.date_cols and self.numeric_cols: strategy.append("✅ Found Date column -> Generating Time Series Analysis.")
            if self.categorical_cols and self.numeric_cols: strategy.append("✅ Found Categories -> Generating Breakdowns & Heatmaps.")
            st.write("**Plan:**")
            for step in strategy:
                st.write(step)

        # 2. KPI SECTION
        if self.numeric_cols:
            st.subheader("📊 Key Metrics")
            # Create up to 4 columns for KPIs
            cols = st.columns(min(len(self.numeric_cols), 4))
            for i, col_name in enumerate(self.numeric_cols[:4]):
                total = self.df[col_name].sum()
                avg = self.df[col_name].mean()
                
                # Smart formatting
                if total > 1_000_000: fmt_total = f"{total/1_000_000:.1f}M"
                elif total > 1_000: fmt_total = f"{total/1_000:.1f}K"
                else: fmt_total = f"{total:,.0f}"
                
                cols[i].metric(label=col_name, value=fmt_total, delta=f"Avg: {avg:,.0f}")
            st.markdown("---")

        # 3. DYNAMIC CHARTING
        # We split the layout: Left (Trends/Big Charts), Right (Breakdowns)
        c1, c2 = st.columns([2, 1])

        # --- LEFT COLUMN: TRENDS ---
        with c1:
            if self.date_cols and self.numeric_cols:
                date_col = self.date_cols[0] # Use primary date
                metric_col = self.numeric_cols[0] # Use primary metric
                
                st.subheader(f"📈 {metric_col} Over Time")
                
                # Aggregation for cleaner charts
                chart_df = self.df.groupby(date_col)[metric_col].sum().reset_index()
                
                line = alt.Chart(chart_df).mark_line(point=True).encode(
                    x=alt.X(date_col, title="Date"),
                    y=alt.Y(metric_col, title=metric_col),
                    tooltip=[date_col, metric_col]
                ).properties(height=350).interactive()
                
                st.altair_chart(line, use_container_width=True)
            
            elif self.categorical_cols and self.numeric_cols:
                # If no date, show a big Bar Chart instead
                cat_col = self.categorical_cols[0]
                metric_col = self.numeric_cols[0]
                
                st.subheader(f"📊 {metric_col} by {cat_col}")
                bar = alt.Chart(self.df).mark_bar().encode(
                    x=alt.X(cat_col, sort='-y'),
                    y=alt.Y(metric_col),
                    color=cat_col,
                    tooltip=[cat_col, metric_col]
                ).properties(height=350).interactive()
                st.altair_chart(bar, use_container_width=True)

        # --- RIGHT COLUMN: BREAKDOWNS ---
        with c2:
            if len(self.categorical_cols) > 0 and self.numeric_cols:
                # Pick the second categorical col if available, else the first
                cat_col = self.categorical_cols[1] if len(self.categorical_cols) > 1 else self.categorical_cols[0]
                metric_col = self.numeric_cols[0]

                st.subheader(f"🍩 {cat_col} Distribution")
                pie = alt.Chart(self.df).mark_arc(innerRadius=50).encode(
                    theta=alt.Theta(field=metric_col, type="quantitative"),
                    color=alt.Color(field=cat_col, type="nominal"),
                    tooltip=[cat_col, metric_col]
                ).properties(height=350)
                st.altair_chart(pie, use_container_width=True)

        # 4. ADVANCED: HEATMAPS (If we have enough dimensions)
        if len(self.categorical_cols) >= 2 and self.numeric_cols:
            st.markdown("---")
            st.subheader("🔥 Interaction Heatmap")
            
            cat_x = self.categorical_cols[0]
            cat_y = self.categorical_cols[1]
            metric = self.numeric_cols[0]
            
            heatmap = alt.Chart(self.df).mark_rect().encode(
                x=cat_x,
                y=cat_y,
                color=alt.Color(metric, scale=alt.Scale(scheme='viridis')),
                tooltip=[cat_x, cat_y, metric]
            ).properties(height=400).interactive()
            
            st.altair_chart(heatmap, use_container_width=True)

        elif len(self.numeric_cols) >= 2:
             # Correlation Scatter Plot if we lack categories for heatmap
            st.markdown("---")
            st.subheader("🔗 Correlation Analysis")
            
            x_metric = self.numeric_cols[0]
            y_metric = self.numeric_cols[1]
            
            scatter = alt.Chart(self.df).mark_circle(size=60).encode(
                x=x_metric,
                y=y_metric,
                tooltip=self.numeric_cols
            ).properties(height=400).interactive()
            
            st.altair_chart(scatter, use_container_width=True)