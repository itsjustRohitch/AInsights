import pandas as pd
import numpy as np
import re
import os
import traceback
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
import json
from datetime import datetime

class AgentA_Engineer:
    """
    Agent A: Universal Data Ingestion & Engineering Agent
    
    Accepts: CSV, Excel, JSON, TXT, PDF, HTML, XML
    Outputs: Clean CSV with preserved column names
    Features: PDFPlumber integration, Hybrid LLM/Rule cleaning, Strict Schema Preservation
    """
    
    SUPPORTED_FORMATS = {
        '.csv': 'read_csv',
        '.xlsx': 'read_excel',
        '.xls': 'read_excel',
        '.json': 'read_json',
        '.txt': 'read_txt',
        '.pdf': 'read_pdf',
        '.html': 'read_html',
        '.xml': 'read_xml',
        '.tsv': 'read_tsv',
        '.parquet': 'read_parquet'
    }
    
    def __init__(self, llm_engine=None, output_dir: str = "data"):
        self.llm = llm_engine
        self.log = []
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def run(self, uploaded_file) -> Tuple[Optional[pd.DataFrame], List[str]]:
        self.log = []
        try:
            df = self._ingest_file(uploaded_file)
            if df is None:
                return None, self.log
            
            if not self._validate_dataframe(df):
                return None, self.log
            
            profile = self._analyze_data(df)
            
            if self.llm:
                df_clean = self._llm_clean(df, profile)
            else:
                self.log.append("⚙️ No LLM provided - using rule-based cleaning")
                df_clean = self._rule_based_clean(df, profile)
            
            df_clean = self._post_process(df_clean)
            
            output_path = self._export_csv(df_clean)
            
            self.log.append(f"✅ SUCCESS: Cleaned data saved to {output_path}")
            self.log.append(f"📊 Final Shape: {df_clean.shape[0]} rows × {df_clean.shape[1]} columns")
            
            return df_clean, self.log
            
        except Exception as e:
            self.log.append(f"❌ CRITICAL ERROR: {str(e)}")
            traceback.print_exc()
            return None, self.log

    def _ingest_file(self, uploaded_file) -> Optional[pd.DataFrame]:
        self.log.append("📂 STAGE 1: File Ingestion")
        try:
            if hasattr(uploaded_file, 'name'):
                filename = uploaded_file.name
                file_ext = Path(filename).suffix.lower()
            else:
                filename = str(uploaded_file)
                file_ext = Path(filename).suffix.lower()
            
            self.log.append(f"   📄 File: {filename}")
            
            if file_ext not in self.SUPPORTED_FORMATS:
                self.log.append(f"❌ Unsupported format: {file_ext}")
                return None
            
            reader_method = getattr(self, f'_{self.SUPPORTED_FORMATS[file_ext]}')
            df = reader_method(uploaded_file)
            
            if df is not None:
                self.log.append(f"   ✅ Loaded {len(df)} rows × {len(df.columns)} columns")
            
            return df
        except Exception as e:
            self.log.append(f"❌ Ingestion failed: {str(e)}")
            return None

    def _read_csv(self, file) -> pd.DataFrame:
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
        for encoding in encodings:
            try:
                if hasattr(file, 'seek'): file.seek(0)
                return pd.read_csv(file, encoding=encoding, low_memory=False)
            except:
                continue
        self.log.append("   ❌ All CSV encoding attempts failed")
        return None

    def _read_excel(self, file) -> pd.DataFrame:
        try:
            return pd.read_excel(file)
        except Exception as e:
            self.log.append(f"   ❌ Excel read error: {str(e)}")
            return None

    def _read_pdf(self, file) -> pd.DataFrame:
        try:
            import pdfplumber
            tables = []
            with pdfplumber.open(file) as pdf:
                for i, page in enumerate(pdf.pages):
                    extracted = page.extract_tables()
                    for table in extracted:
                        if len(table) > 1:
                            df_table = pd.DataFrame(table[1:], columns=table[0])
                            tables.append(df_table)
            
            if not tables:
                self.log.append("   ⚠️ No tables found in PDF")
                return None
            
            self.log.append(f"   ✓ Extracted {len(tables)} table(s)")
            
            if len(tables) > 1:
                df = pd.concat(tables, ignore_index=True)
                self.log.append("   ✓ Concatenated multiple tables")
            else:
                df = tables[0]
            
            return df
        except ImportError:
            self.log.append("   ❌ PDF support requires 'pdfplumber'")
            return None
        except Exception as e:
            self.log.append(f"   ❌ PDF read error: {str(e)}")
            return None

    def _read_json(self, file) -> pd.DataFrame:
        try:
            return pd.read_json(file)
        except:
            try:
                if hasattr(file, 'seek'): file.seek(0)
                data = json.load(file)
                return pd.json_normalize(data)
            except Exception as e:
                self.log.append(f"   ❌ JSON error: {str(e)}")
                return None

    def _read_txt(self, file) -> pd.DataFrame:
        try:
            if hasattr(file, 'seek'): file.seek(0)
            return pd.read_csv(file, sep=None, engine='python')
        except Exception as e:
            self.log.append(f"   ❌ TXT error: {str(e)}")
            return None

    def _read_html(self, file) -> pd.DataFrame:
        try:
            dfs = pd.read_html(file)
            return max(dfs, key=len) if dfs else None
        except Exception as e:
            self.log.append(f"   ❌ HTML error: {str(e)}")
            return None
            
    def _read_xml(self, file) -> pd.DataFrame:
        try:
            return pd.read_xml(file)
        except Exception as e:
            self.log.append(f"   ❌ XML error: {str(e)}")
            return None
            
    def _read_tsv(self, file) -> pd.DataFrame:
        return pd.read_csv(file, sep='\t')
        
    def _read_parquet(self, file) -> pd.DataFrame:
        return pd.read_parquet(file)

    def _validate_dataframe(self, df: pd.DataFrame) -> bool:
        self.log.append("🔍 STAGE 2: Data Validation")
        if df is None or df.empty:
            self.log.append("   ❌ DataFrame is empty")
            return False
        return True

    def _analyze_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        self.log.append("🧠 STAGE 3: Data Analysis")
        profile = {
            'shape': df.shape,
            'columns': list(df.columns),
            'currency_cols': [],
            'date_cols': [],
            'numeric_cols': [],
            'sample_data': {}
        }
        for col in df.columns:
            sample = df[col].dropna().head(5).tolist()
            profile['sample_data'][col] = sample
            if pd.api.types.is_numeric_dtype(df[col]):
                profile['numeric_cols'].append(col)
            elif pd.api.types.is_object_dtype(df[col]):
                sample_str = [str(x) for x in sample]
                if any(c in ''.join(sample_str) for c in ['$', '€', '£']):
                    profile['currency_cols'].append(col)
                elif any(self._looks_like_date(x) for x in sample_str):
                    profile['date_cols'].append(col)
        return profile

    def _looks_like_date(self, value: str) -> bool:
        patterns = [
            r'\d{4}-\d{2}-\d{2}', r'\d{2}/\d{2}/\d{4}', 
            r'\d{2}-\d{2}-\d{4}', r'\d{1,2}\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)'
        ]
        return any(re.search(p, str(value), re.IGNORECASE) for p in patterns)

    def _llm_clean(self, df: pd.DataFrame, profile: Dict) -> pd.DataFrame:
        self.log.append("🤖 STAGE 4: LLM-Powered Cleaning")
        summary = f"""
        Columns: {profile['columns']}
        Currency Candidates: {profile['currency_cols']}
        Date Candidates: {profile['date_cols']}
        Sample Data: {json.dumps({k: v[0] if v else None for k, v in profile['sample_data'].items()}, indent=2)}
        """
        prompt = f"""
        You are an expert Data Engineer. 
        DATA PROFILE: {summary}
        TASK: Write Python code to clean DataFrame 'df'.
        STRICT RULES:
        1. **NEVER RENAME COLUMNS**: Column names must stay exactly as they are.
        2. **Remove Duplicates**: Use df.drop_duplicates().
        3. **Fix Data Types**:
           - Convert Currency columns (remove '$', ',') to numeric.
           - Convert Date columns to datetime.
           - Ensure numeric columns are actually numbers (coerce errors).
        4. **Fill Missing**:
           - Numeric -> 0
           - Text -> "Unknown"
        OUTPUT: Only valid Python code wrapped in ```python ... ```.
        """
        try:
            response = self.llm.invoke(prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            code_match = re.search(r"```python(.*?)```", response_text, re.DOTALL)
            if code_match:
                clean_code = code_match.group(1).strip()
                self.log.append("   ✅ LLM generated cleaning strategy")
                local_vars = {'df': df.copy(), 'pd': pd, 'np': np, 're': re, 'datetime': datetime}
                exec(clean_code, {}, local_vars)
                df_clean = local_vars['df']
                if set(df_clean.columns) != set(df.columns):
                    self.log.append("   ⚠️ LLM renamed columns (Violation). Reverting to Rule-Based.")
                    return self._rule_based_clean(df, profile)
                return df_clean
            else:
                self.log.append("   ⚠️ No code generated. Using Rule-Based.")
                return self._rule_based_clean(df, profile)
        except Exception as e:
            self.log.append(f"   ⚠️ LLM Error: {e}. Using Rule-Based.")
            return self._rule_based_clean(df, profile)

    def _rule_based_clean(self, df: pd.DataFrame, profile: Dict) -> pd.DataFrame:
        self.log.append("⚙️ STAGE 4: Rule-Based Cleaning")
        df_clean = df.copy()
        df_clean.drop_duplicates(inplace=True)
        for col in profile['currency_cols']:
            df_clean[col] = df_clean[col].astype(str).str.replace(r'[$,]', '', regex=True)
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        for col in profile['date_cols']:
            df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
        num_cols = df_clean.select_dtypes(include=[np.number]).columns
        df_clean[num_cols] = df_clean[num_cols].fillna(0)
        obj_cols = df_clean.select_dtypes(include=['object']).columns
        df_clean[obj_cols] = df_clean[obj_cols].fillna("Unknown")
        return df_clean

    def _post_process(self, df: pd.DataFrame) -> pd.DataFrame:
        self.log.append("🔧 STAGE 5: Post-Processing")
        return df.dropna(axis=1, how='all').reset_index(drop=True)

    def _export_csv(self, df: pd.DataFrame) -> Path:
        self.log.append("💾 STAGE 6: Export")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / f"cleaned_data_{timestamp}.csv"
        df.to_csv(output_path, index=False)
        return output_path