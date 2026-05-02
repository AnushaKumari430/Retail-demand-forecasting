import pandas as pd
from prophet import Prophet
import pickle

# Load dataset (update path if needed)
df = pd.read_excel("data/raw_data.xlsx")

# Prepare data
df['Date'] = pd.to_datetime(df['Date'])
df_daily = df.groupby('Date')['Units Sold'].sum().reset_index()
df_daily.columns = ['ds', 'y']

# Train model
model = Prophet()
model.fit(df_daily)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved as model.pkl")