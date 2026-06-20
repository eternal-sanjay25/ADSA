"""Generate sample churn dataset with realistic distributions,
intentional missing values, outliers, and duplicate rows."""

import pandas as pd
import numpy as np
import os

np.random.seed(42)
os.makedirs('sample_data', exist_ok=True)
N = 5000

age            = np.random.normal(42, 12, N).clip(18, 80).astype(int)
gender         = np.random.choice(['Male', 'Female'], N)
location       = np.random.choice(
    ['Mumbai', 'Delhi', 'Chennai', 'Bangalore', 'Hyderabad'],
    N, p=[0.25, 0.25, 0.20, 0.20, 0.10],
)
tenure_months  = np.random.exponential(scale=24, size=N).clip(1, 72).astype(int)
contract_type  = np.random.choice(
    ['Month-to-Month', 'One Year', 'Two Year'],
    N, p=[0.55, 0.25, 0.20],
)
payment_method = np.random.choice(
    ['Credit Card', 'Bank Transfer', 'Digital Wallet', 'Cash'], N,
)
monthly_charge = np.random.normal(65, 30, N).clip(20, 150).round(2)
total_charges  = (monthly_charge * tenure_months * np.random.uniform(0.85, 1.15, N)).round(2)
data_usage_gb  = np.random.exponential(scale=8, size=N).clip(0, 50).round(1)
call_minutes   = np.random.normal(300, 120, N).clip(0, 800).astype(int)
support_tickets = np.random.poisson(lam=1.2, size=N).clip(0, 10)
has_internet   = np.random.choice(['Yes', 'No'], N, p=[0.80, 0.20])
has_phone      = np.random.choice(['Yes', 'No'], N, p=[0.90, 0.10])
tech_support   = np.random.choice(['Yes', 'No'], N, p=[0.40, 0.60])
streaming      = np.random.choice(['Yes', 'No'], N, p=[0.50, 0.50])

# Churn probability based on realistic factors
churn_prob = (
    0.05
    + 0.25 * (contract_type == 'Month-to-Month')
    + 0.15 * (monthly_charge > 90)
    + 0.20 * (support_tickets >= 3)
    - 0.15 * (tenure_months > 36)
    - 0.10 * (tech_support == 'Yes')
    + 0.08 * (payment_method == 'Cash')
    + np.random.normal(0, 0.05, N)
).clip(0, 1)
churn = (np.random.uniform(0, 1, N) < churn_prob).astype(int)

# Add missing values
age = age.astype(float)
age[np.random.choice(N, int(N * 0.04), replace=False)] = np.nan
monthly_charge = monthly_charge.astype(float)
monthly_charge[np.random.choice(N, int(N * 0.03), replace=False)] = np.nan
data_usage_gb = data_usage_gb.astype(float)
data_usage_gb[np.random.choice(N, int(N * 0.06), replace=False)] = np.nan

# Add outliers
monthly_charge[np.random.choice(N, 15, replace=False)] = np.random.uniform(300, 500, 15)

join_date = pd.date_range('2020-01-01', periods=N, freq='h')

df = pd.DataFrame({
    'CustomerID':     [f'CUST{str(i).zfill(5)}' for i in range(N)],
    'Age':            age,
    'Gender':         gender,
    'Location':       location,
    'JoinDate':       join_date,
    'TenureMonths':   tenure_months,
    'ContractType':   contract_type,
    'PaymentMethod':  payment_method,
    'MonthlyCharge':  monthly_charge,
    'TotalCharges':   total_charges,
    'DataUsageGB':    data_usage_gb,
    'CallMinutes':    call_minutes,
    'SupportTickets': support_tickets,
    'HasInternet':    has_internet,
    'HasPhone':       has_phone,
    'TechSupport':    tech_support,
    'Streaming':      streaming,
    'Churn':          churn,
})

# Add 12 duplicate rows
dup_idx = np.random.choice(N, 12, replace=False)
df = pd.concat([df, df.iloc[dup_idx]], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv('sample_data/churn.csv', index=False)

print(f'Generated: {len(df)} rows x {len(df.columns)} columns')
print(f'Churn rate: {df["Churn"].mean():.1%}')
print(f'Missing: Age={df["Age"].isnull().sum()}, MonthlyCharge={df["MonthlyCharge"].isnull().sum()}, DataUsageGB={df["DataUsageGB"].isnull().sum()}')
print(f'Saved: sample_data/churn.csv')
