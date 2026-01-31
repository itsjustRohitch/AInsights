import pandas as pd
import numpy as np
import os

# Ensure folder exists
os.makedirs('data', exist_ok=True)

# Create dummy sales data
dates = pd.date_range(start="2025-01-01", periods=100)
data = {
    'Date': dates,
    'Region': np.random.choice(['North', 'South', 'East', 'West'], size=100),
    'Product': np.random.choice(['Laptop', 'Mouse', 'Keyboard', 'Monitor'], size=100),
    'Sales': np.random.randint(100, 5000, size=100),
    'Profit': np.random.randint(10, 500, size=100)
}

df = pd.DataFrame(data)
df.to_csv('data/sales_data.csv', index=False)
print("✅ Success: 'data/sales_data.csv' created!")