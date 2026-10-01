import pandas as pd
import numpy as np

np.random.seed(42)
n = 1000

plans      = ['Basic', 'Standard', 'Premium 5G', 'Business Pro']
plan_price = {'Basic': 299, 'Standard': 549, 'Premium 5G': 899, 'Business Pro': 1299}
regions    = ['North', 'South', 'East', 'West', 'Central']
internets  = ['DSL', 'Fiber Optic', 'No Internet']
contracts  = ['Month-to-Month', 'One Year', 'Two Year']
genders    = ['Male', 'Female']
pay_methods= ['Auto Bank Transfer', 'Credit Card', 'Manual Bank Transfer', 'Mailed Check']

rows = []
for i in range(1, n + 1):
    gender   = np.random.choice(genders)
    age      = int(np.random.normal(42, 14))
    age      = max(18, min(75, age))
    region   = np.random.choice(regions)
    tenure   = int(np.random.exponential(24)) + 1
    tenure   = min(tenure, 72)
    plan     = np.random.choice(plans, p=[0.30, 0.33, 0.25, 0.12])
    base     = plan_price[plan]
    monthly  = round(base + np.random.uniform(-30, 80), 2)
    total    = round(monthly * tenure * np.random.uniform(0.95, 1.05), 2)
    internet = np.random.choice(internets, p=[0.38, 0.47, 0.15])
    contract = np.random.choice(contracts, p=[0.52, 0.30, 0.18])
    payment  = np.random.choice(pay_methods)
    num_lines= np.random.choice([1,2,3,4], p=[0.50,0.28,0.14,0.08])
    intl     = np.random.choice([0,1], p=[0.75,0.25])
    tech_sup = np.random.choice([0,1], p=[0.60,0.40])
    complaints = max(0, int(np.random.poisson(0.8)))
    calls    = max(0, int(np.random.normal(180, 80)))

    # ── Realistic churn probability ──
    p = 0.08
    if contract == 'Month-to-Month':  p += 0.22
    elif contract == 'One Year':      p += 0.06
    if tenure < 6:                    p += 0.18
    elif tenure < 12:                 p += 0.10
    elif tenure > 48:                 p -= 0.08
    if complaints >= 3:               p += 0.25
    elif complaints == 2:             p += 0.12
    if internet == 'Fiber Optic' and monthly > 800: p += 0.07
    if tech_sup == 0:                 p += 0.06
    if payment == 'Mailed Check':     p += 0.08
    if intl == 0 and plan == 'Basic': p += 0.04
    p = max(0.03, min(0.92, p))
    churn = 'Yes' if np.random.random() < p else 'No'

    rows.append({
        'CustomerID': f'CU-{i:04d}',
        'Gender': gender, 'Age': age, 'Region': region,
        'Tenure': tenure, 'Plan': plan,
        'MonthlyCharges': monthly, 'TotalCharges': total,
        'InternetService': internet, 'Contract': contract,
        'PaymentMethod': payment, 'NumPhoneLines': num_lines,
        'InternationalPlan': intl, 'TechSupport': tech_sup,
        'MonthlyCallMinutes': calls, 'Complaints': complaints,
        'Churn': churn
    })

df = pd.DataFrame(rows)
df.to_csv('/home/claude/teleinsight_v2/sample_data.csv', index=False)
yes = (df['Churn'] == 'Yes').sum()
print(f"Generated {len(df)} rows | Churn: {yes} ({yes/len(df):.1%}) | Non-Churn: {len(df)-yes}")
print(df.dtypes)
