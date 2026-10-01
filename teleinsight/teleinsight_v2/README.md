# TeleInsight v2 — Telecom Customer Analytics System
### DTI Final Project · Design Thinking & Innovation · 2026

---

## 🚀 What's New in v2

| Feature | v1 | v2 |
|---------|----|----|
| ML Model | Single Random Forest | **Ensemble: GBM + RF + ExtraTrees** |
| Accuracy | ~72% | **90.0%** |
| AUC-ROC | ~57% | **97.4%** |
| F1 Score | ~39% | **89.5%** |
| Cross-Validation | No | **5-Fold Stratified** |
| Feature Engineering | Basic | **+5 engineered features** |
| Class Balancing | No | **Oversampling** |
| Dataset Size | 500 rows | **1,000 rows** |
| UI Design | Standard | **Premium glassmorphism** |
| Font | DM Sans | **Outfit + JetBrains Mono** |
| Animated Counters | No | **Yes** |
| Confusion Matrix | No | **Yes** |
| Feature Importance | No | **Yes (visual bars)** |
| Risk Histogram | No | **Yes** |

---

## ⚙️ Setup — 3 Steps

### Step 1 — Install Dependencies
Open VS Code terminal in this folder:
```
pip install -r requirements.txt
```

### Step 2 — Run the App
```
python app.py
```

### Step 3 — Open Browser
```
http://localhost:5000
```

---

## 📁 Project Structure

```
teleinsight_v2/
├── app.py                    ← Flask backend (9 routes)
├── requirements.txt          ← pip dependencies
├── sample_data.csv           ← 1,000-row telecom dataset
├── generate_data.py          ← Regenerate sample data
├── models/
│   └── ml_models.py         ← Ensemble ML (GBM + RF + ET)
└── templates/
    ├── base.html             ← Premium sidebar layout
    ├── dashboard.html        ← KPIs + 5 animated charts
    ├── upload.html           ← Drag-drop file upload
    ├── segmentation.html     ← K-Means + 5 charts
    ├── churn.html            ← ML results + confusion matrix
    └── reports.html          ← Report center + tech stack
```

---

## 🤖 ML Models & Accuracy

### Churn Prediction — Voting Ensemble (Soft)
| Metric | Score |
|--------|-------|
| **Accuracy** | **90.0%** |
| **AUC-ROC** | **97.4%** |
| **Precision** | 93.4% |
| **Recall** | 85.9% |
| **F1 Score** | 89.5% |
| **CV Mean (5-fold)** | 88.1% ± 1.6% |

**Ensemble Components:**
- Gradient Boosting (300 estimators, lr=0.05, weight=2x)
- Random Forest (300 trees, balanced class weight)
- Extra Trees (200 trees, balanced class weight)

**Feature Engineering (+5 features):**
- `ChargesPerMonth` — MonthlyCharges / (Tenure+1)
- `TenureGroup` — Binned tenure categories
- `HighValueCustomer` — High charge + long tenure flag
- `AvgMonthlyVsTotal` — Consistency ratio
- `HighComplaint` — Binary complaint flag (≥3)

**Class Balancing:** Oversampling minority class to match majority

### Customer Segmentation — K-Means
- 5 clusters with RobustScaler
- Segments: Premium Loyals, Engaged Mid-Tier, New Customers, At-Risk Churners, Dormant Users
- Re-ranked by average monthly charges

---

## 🎨 UI Design

- **Font:** Outfit (headings) + JetBrains Mono (numbers/code)
- **Sidebar:** Dark glassmorphism (#080f1e) with glow accents
- **Cards:** White with colored top borders + hover lift
- **Animated:** Number counters, staggered page reveals
- **Charts:** Chart.js 4 — Line, Bar, Doughnut, Histogram
- **Color system:** Blue primary, green success, red danger, purple accent

---

## 📊 Dataset Columns

| Column | Type | Description |
|--------|------|-------------|
| CustomerID | String | Unique customer ID |
| Gender | String | Male / Female |
| Age | Integer | Customer age |
| Region | String | Geographic region |
| Tenure | Integer | Months with company |
| Plan | String | Basic / Standard / Premium 5G / Business Pro |
| MonthlyCharges | Float | Monthly billing amount |
| TotalCharges | Float | Total charged to date |
| InternetService | String | DSL / Fiber Optic / No Internet |
| Contract | String | Month-to-Month / One Year / Two Year |
| PaymentMethod | String | Payment type |
| NumPhoneLines | Integer | Number of lines |
| InternationalPlan | Integer | 0 or 1 |
| TechSupport | Integer | 0 or 1 |
| MonthlyCallMinutes | Integer | Monthly call usage |
| Complaints | Integer | Number of complaints |
| Churn | String | Yes / No |

---

*TeleInsight v2 — Built with Python + Flask + scikit-learn*
*DTI Project Final Submission 2026*
