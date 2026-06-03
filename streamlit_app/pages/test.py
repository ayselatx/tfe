import pandas as pd

df = pd.read_pickle("/home/aysel/tfe/streamlit_app/data/unimodal/metrics/global_unimodal_metrics.pkl")
print(df.columns)
print(df.head())
