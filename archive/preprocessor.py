import pandas as pd
import re
import traceback
import io
import numpy as np

class SmartPreprocessor:
    def __init__(self, df, llm_engine):
        self.df = df
        self.llm = llm_engine
        self.log = []

    def run_pipeline(self):
        """
        Robust 'AI Analyst' with Auto-Fallback.
        If AI code fails, it switches to manual rules to prevent app crashes.
        """
        # 1. OBSERVATION
        # Send a dictionary of the first row (easier for AI to understand)
        try:
            first_row = self.df.iloc[0].to_dict()
            data_snapshot = str(first_row)
        except:
            data_snapshot = "Empty DataFrame"

        # 2. REASONING PHASE
        prompt = f"""
        You are a Python Data Expert. 
        I have a dataset with these columns and first row values:
        {data_snapshot}
        
        YOUR GOAL: 
        Rename columns to match this standard format:
        ['Date', 'Region', 'Product', 'Sales', 'Profit']
        
        INSTRUCTIONS:
        1. Identify the matching columns from the input.
           - 'Order Date', 'Ship Date' -> 'Date'
           - 'Category', 'Sub-Category' -> 'Product'
           - 'Total', 'Revenue' -> 'Sales'
           - 'Net', 'Profit (USD)' -> 'Profit'
           
        2. Write Python code to:
           - Rename the columns to standard names.
           - Ensure 'Sales' and 'Profit' are numeric (coerce errors).
           - Convert 'Date' to datetime.
           
        3. OUTPUT:
        - Return ONLY valid Python code block wrapped in ```python ... ```.
        - Use 'df' variable directly.
        """

        try:
            self.log.append("🧐 AI Analyst is mapping columns...")
            response = self.llm.invoke(prompt)
            
            # Handle text extraction
            if hasattr(response, 'content'): 
                response_text = response.content
            else: 
                response_text = str(response)

            # Extract Code
            code_match = re.search(r"```python(.*?)```", response_text, re.DOTALL)
            
            if code_match:
                generated_code = code_match.group(1).strip()
                self.log.append("💡 AI generated cleaning strategy.")
                
                # 4. EXECUTION PHASE (With Crash Protection)
                local_vars = {'df': self.df.copy(), 'pd': pd, 'np': np}
                
                try:
                    exec(generated_code, {}, local_vars)
                    self.df = local_vars['df'] # Only update if exec worked
                    self.log.append("✅ Transformation successful.")
                except Exception as exec_error:
                    self.log.append(f"⚠️ AI Code Failed: {exec_error}. Switching to Manual Mode.")
                    self._manual_fallback() # <--- SAFETY NET
                        
            else:
                self.log.append("⚠️ AI was confused. Switching to Manual Mode.")
                self._manual_fallback()
            
            # 5. FINAL GUARANTEE (The "KeyError" Killer)
            # This ensures 'Profit' exists even if everything else failed
            self._ensure_columns_exist()
            
            return self.df, self.log

        except Exception as e:
            self.log.append(f"❌ Pipeline Error: {str(e)}")
            self._manual_fallback()
            self._ensure_columns_exist()
            return self.df, self.log

    def _manual_fallback(self):
        """Hardcoded backup rules for common datasets"""
        # 1. Normalize Names (remove spaces/case)
        self.df.columns = self.df.columns.str.strip().str.title()
        
        # 2. Common Mappings
        rename_map = {
            'Order Date': 'Date', 'Ship Date': 'Date', 'Fecha': 'Date',
            'Category': 'Product', 'Sub-Category': 'Product', 'Item': 'Product',
            'Total Revenue': 'Sales', 'Total': 'Sales', 'Amount': 'Sales',
            'Net Profit': 'Profit', 'Net': 'Profit'
        }
        
        # Apply renaming only if column exists
        final_map = {k: v for k, v in rename_map.items() if k in self.df.columns}
        self.df.rename(columns=final_map, inplace=True)
        self.log.append("⚙️ Applied manual backup renaming rules.")

    def _ensure_columns_exist(self):
        """Creates missing columns with 0 defaults to prevent KeyErrors"""
        required_defaults = {
            'Region': 'Unknown', 
            'Product': 'Generic', 
            'Sales': 0.0, 
            'Profit': 0.0,
            'Date': pd.Timestamp.now()
        }
        
        for col, default_val in required_defaults.items():
            if col not in self.df.columns:
                self.df[col] = default_val
                self.log.append(f"⚠️ Missing '{col}'. Created default column to prevent crash.")
                
        # Force numeric types
        for col in ['Sales', 'Profit']:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce').fillna(0)