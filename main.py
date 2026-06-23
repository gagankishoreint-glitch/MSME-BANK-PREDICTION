"""
IDBI MSME Risk Intelligence — FastAPI Backend
===============================================
Real XGBoost model trained on Give Me Some Credit (150k borrowers)
with MSME behavioral feature overlays.
"""

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
import pandas as pd
import numpy as np
import joblib
import os
import json

app = FastAPI(title="MSME Risk Intelligence API")
app.mount("/dashboard", StaticFiles(directory="static", html=True), name="static")

# ─── Global State ────────────────────────────────────────────
xgb_model   = None
explainer   = None
imputer     = None
feature_df  = None  # Synthetic MSME borrower portfolio
FEATURES    = None

# ─── Sector LGD (Loss Given Default) ────────────────────────
LGD_MAP = {
    'Retail':        0.35,
    'Manufacturing': 0.45,
    'Construction':  0.55,
    'Technology':    0.30,
    'Agriculture':   0.40,
    'Textile':       0.42,
}

# ─── Action Escalation Ladder ────────────────────────────────
ACTION_LADDER = {
    'Low': [
        {"priority": "routine", "action": "Continue Standard Monitoring",
         "detail": "No immediate action required. Review at next scheduled cycle."}
    ],
    'Medium': [
        {"priority": "medium", "action": "Schedule Relationship Manager Call",
         "detail": "Discuss business health with borrower within 7 days."},
        {"priority": "medium", "action": "Request Updated Financial Statements",
         "detail": "Obtain last 3 months bank statements and GST returns."}
    ],
    'High': [
        {"priority": "high", "action": "Initiate Document Review",
         "detail": "Pull all collateral documents and verify current valuation."},
        {"priority": "high", "action": "Cashflow Assessment",
         "detail": "Conduct detailed analysis of inflows vs outflows for past 6 months."},
        {"priority": "high", "action": "Collateral Verification",
         "detail": "Dispatch field officer to verify primary collateral on record."}
    ],
    'Critical': [
        {"priority": "critical", "action": "Initiate Restructuring Discussion",
         "detail": "Escalate to Senior RM for restructuring evaluation immediately."},
        {"priority": "critical", "action": "Evaluate Tenure Extension",
         "detail": "Assess feasibility of extending loan tenure to reduce immediate burden."},
        {"priority": "critical", "action": "Assign to Recovery Monitoring",
         "detail": "Flag in Recovery system and assign dedicated recovery officer."},
        {"priority": "critical", "action": "Prepare Legal Documentation",
         "detail": "Engage legal team to prepare NPA declaration and recovery notices."}
    ]
}

# ─── SHAP → Plain English Narratives ─────────────────────────
FEATURE_NARRATIVE = {
    'revolving_utilization':  lambda v: f"Working capital line utilized at {v*100:.0f}% — {'dangerously high' if v > 0.7 else 'elevated' if v > 0.4 else 'healthy'}.",
    'debt_ratio':             lambda v: f"Debt-to-income ratio is {v:.2f} — {'critical' if v > 1.0 else 'high' if v > 0.5 else 'manageable'}.",
    'late_30_59':             lambda v: f"Borrower had {int(v)} payment(s) 30-59 days late in the observation window.",
    'late_60_89':             lambda v: f"Borrower had {int(v)} payment(s) 60-89 days overdue — significant stress indicator.",
    'late_90_days':           lambda v: f"Borrower had {int(v)} serious delinquency event(s) (90+ days late) — NPA risk.",
    'open_credit_lines':      lambda v: f"Borrower holds {int(v)} open credit lines — {'over-leveraged' if v > 15 else 'normal'}.",
    'real_estate_loans':      lambda v: f"Borrower has {int(v)} real estate loan(s) as collateral exposure.",
    'num_dependents':         lambda v: f"Borrower supports {int(v)} dependent(s) — impacts disposable income.",
    'income_stability':       lambda v: f"Income stability index: {v:.2f} — {'strong' if v > 0.6 else 'moderate' if v > 0.3 else 'weak'}.",
    'gst_compliance_score':   lambda v: f"GST compliance score: {v:.2f}/1.0 — {'regular filer' if v > 0.8 else 'irregular filings detected' if v > 0.5 else 'non-compliant, high risk'}.",
    'emi_delay_count':        lambda v: f"EMI delays recorded: {int(v)} — {'no delays' if v == 0 else 'payment stress visible'}.",
    'cashflow_stress_ratio':  lambda v: f"Cashflow stress index: {v:.2f} — {'severe' if v > 2.0 else 'moderate' if v > 1.0 else 'low'}.",
    'working_capital_usage':  lambda v: f"Working capital drawn: {v*100:.0f}% of sanctioned limit.",
    'revenue_trend_index':    lambda v: f"Revenue trend index: {v:.2f} — {'growing' if v > 1.1 else 'stable' if v > 0.9 else 'declining'}.",
    'payment_history_score':  lambda v: f"Payment history score: {v:.2f}/1.0 — {'excellent' if v > 0.9 else 'fair' if v > 0.7 else 'poor track record'}.",
    'supplier_payment_risk':  lambda v: f"Supplier payment risk flag: {'Active — delays to creditors detected' if v > 0 else 'Clear — no creditor delays'}.",
}

def get_risk_band(pd_value: float) -> str:
    if pd_value < 0.20: return "Low"
    if pd_value < 0.50: return "Medium"
    if pd_value < 0.75: return "High"
    return "Critical"

def build_portfolio() -> pd.DataFrame:
    """Build a realistic MSME borrower portfolio using the real model's feature space."""
    np.random.seed(2024)
    N = 200
    sectors = ['Retail', 'Manufacturing', 'Construction', 'Technology', 'Agriculture', 'Textile']
    loan_types = ['Term Loan', 'Working Capital', 'Trade Credit']

    rows = []
    for i in range(N):
        sector = np.random.choice(sectors)
        loan_type = np.random.choice(loan_types)
        # Generate a risk profile biased by sector
        sector_risk_bias = {'Retail': 0.1, 'Manufacturing': 0.15, 'Construction': 0.25,
                            'Technology': 0.08, 'Agriculture': 0.18, 'Textile': 0.20}
        bias = sector_risk_bias[sector]

        # Core features (realistic distributions)
        rev_util       = float(np.clip(np.random.beta(2, 5) + bias * 0.5, 0.01, 0.99))
        debt_ratio     = float(np.clip(np.random.exponential(0.35) + bias, 0.01, 3.0))
        late_3059      = int(np.random.poisson(bias * 3))
        late_6089      = int(np.random.poisson(bias * 1.5))
        late_90        = int(np.random.poisson(bias * 0.8))
        open_lines     = int(np.random.poisson(8) + 2)
        re_loans       = int(np.random.poisson(1))
        num_dep        = int(np.random.poisson(1.5))
        income         = float(np.clip(np.random.lognormal(11, 0.8), 20000, 500000))

        # MSME behavioral features
        gst_score      = float(np.clip(1 - late_3059 * 0.12 + np.random.normal(0, 0.05), 0.0, 1.0))
        emi_delays     = min(late_3059 + late_6089, 12)
        cf_stress      = float(np.clip(debt_ratio * np.random.uniform(0.8, 1.2), 0, 5))
        wc_usage       = float(np.clip(rev_util * 0.6 + np.random.normal(0.1, 0.05), 0.0, 1.0))
        rev_trend      = float(np.clip(1.2 - debt_ratio * 0.4 + np.random.normal(0, 0.1), 0.2, 2.0))
        pay_hist       = float(np.clip(1 - late_90 * 0.2 - late_3059 * 0.05, 0.0, 1.0))
        supp_risk      = float((late_3059 > 2) + (late_90 > 0))
        inc_stability  = float(np.clip(income / 100000, 0.0, 1.0))

        outstanding    = float(np.random.uniform(200000, 2500000))

        # Build monthly journey (12 months)
        journey_events = []
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        for m_idx, month in enumerate(months):
            # GST signal degradation
            gst_ok = np.random.random() > (bias * 0.6 + m_idx * 0.02)
            emi_ok = np.random.random() > (bias * 0.5 + m_idx * 0.03)
            if not gst_ok:
                journey_events.append({"date": f"2024-{m_idx+1:02d}-07", "type": "warning",
                                        "desc": f"GST filing delayed — {month} 2024"})
            if not emi_ok:
                journey_events.append({"date": f"2024-{m_idx+1:02d}-15", "type": "alert",
                                        "desc": f"EMI payment overdue — {month} 2024"})
            elif m_idx == 0:
                journey_events.append({"date": f"2024-{m_idx+1:02d}-15", "type": "ok",
                                        "desc": f"EMI paid on time — {month} 2024"})

        feat_row = {
            'company_id':          f"MSME-{i+1:04d}",
            'sector':              sector,
            'loan_type':           loan_type,
            'outstanding_loan':    outstanding,
            'revolving_utilization': rev_util,
            'debt_ratio':          debt_ratio,
            'late_30_59':          late_3059,
            'late_60_89':          late_6089,
            'late_90_days':        late_90,
            'open_credit_lines':   open_lines,
            'real_estate_loans':   re_loans,
            'num_dependents':      num_dep,
            'income_stability':    inc_stability,
            'gst_compliance_score': gst_score,
            'emi_delay_count':     emi_delays,
            'cashflow_stress_ratio': cf_stress,
            'working_capital_usage': wc_usage,
            'revenue_trend_index': rev_trend,
            'payment_history_score': pay_hist,
            'supplier_payment_risk': supp_risk,
            'journey_events':      json.dumps(journey_events),
        }
        rows.append(feat_row)

    return pd.DataFrame(rows)

@app.on_event("startup")
def load_assets():
    global xgb_model, explainer, imputer, feature_df, FEATURES
    try:
        xgb_model = joblib.load('models/xgb_model.joblib')
        explainer = joblib.load('models/shap_explainer.joblib')
        imputer   = joblib.load('models/imputer.joblib')
        FEATURES  = joblib.load('models/feature_list.joblib')
        feature_df = build_portfolio()
        print(f"Real model loaded. Portfolio: {len(feature_df)} borrowers.")
    except Exception as e:
        print(f"Failed to load models: {e}")
        raise

def predict_portfolio(df: pd.DataFrame) -> np.ndarray:
    X = df[FEATURES].copy()
    X_imp = imputer.transform(X)
    return xgb_model.predict_proba(X_imp)[:, 1]

# ─── API Routes ───────────────────────────────────────────────

@app.get("/model-performance")
def get_model_performance():
    """Return training performance metrics for dashboard display."""
    try:
        with open('models/performance_report.json') as f:
            return json.load(f)
    except:
        return {}

@app.get("/portfolio")
def get_portfolio_summary():
    if feature_df is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    probs = predict_portfolio(feature_df)
    df = feature_df.copy()
    df['pd']    = probs
    df['band']  = df['pd'].apply(get_risk_band)
    df['lgd']   = df['sector'].map(lambda s: LGD_MAP.get(s, 0.4))
    df['el']    = df['outstanding_loan'] * df['pd'] * df['lgd']

    sector_stats = []
    for sector, grp in df.groupby('sector'):
        sector_stats.append({
            "sector":         sector,
            "exposure":       float(grp['outstanding_loan'].sum()),
            "avg_pd":         float(grp['pd'].mean()),
            "expected_loss":  float(grp['el'].sum()),
            "borrower_count": int(len(grp))
        })

    dist = df['band'].value_counts().to_dict()
    for band in ["Low", "Medium", "High", "Critical"]:
        dist.setdefault(band, 0)

    return {
        "total_borrowers":    int(len(df)),
        "avg_pd":             float(probs.mean()),
        "total_exposure":     float(df['outstanding_loan'].sum()),
        "total_expected_loss": float(df['el'].sum()),
        "risk_distribution":  dist,
        "sector_analytics":   sorted(sector_stats, key=lambda x: x['expected_loss'], reverse=True)
    }

@app.get("/borrowers")
def get_borrowers(limit: int = 50):
    if feature_df is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    probs = predict_portfolio(feature_df)
    df = feature_df.copy()
    df['pd'] = probs
    top = df.sort_values('pd', ascending=False).head(limit)

    return [
        {
            "company_id":      row['company_id'],
            "sector":          row['sector'],
            "loan_type":       row['loan_type'],
            "pd":              float(row['pd']),
            "outstanding_loan": float(row['outstanding_loan']),
            "risk_band":       get_risk_band(float(row['pd']))
        }
        for _, row in top.iterrows()
    ]

@app.get("/borrowers/{company_id}")
def get_borrower_details(company_id: str):
    if feature_df is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    row = feature_df[feature_df['company_id'] == company_id]
    if row.empty:
        raise HTTPException(status_code=404, detail="Borrower not found")
    row = row.iloc[0]

    # Predict PD
    X = pd.DataFrame([row[FEATURES].to_dict()])
    X_imp = imputer.transform(X)
    current_pd = float(xgb_model.predict_proba(X_imp)[0, 1])

    # Build model-derived 12-month trajectory
    # Simulate degrading monthly features and run model at each step
    timeline = []
    months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    base_features = row[FEATURES].to_dict()
    bias = current_pd  # Use final PD as a proxy for underlying risk level

    for m_idx, month in enumerate(months):
        # Simulate monthly deterioration proportional to final PD
        scale = (m_idx / 11)  # 0 → 1
        monthly_feats = base_features.copy()
        monthly_feats['revolving_utilization'] = float(np.clip(
            base_features['revolving_utilization'] * (0.5 + 0.5 * scale), 0.01, 0.99))
        monthly_feats['gst_compliance_score'] = float(np.clip(
            base_features['gst_compliance_score'] * (1.1 - 0.2 * scale), 0, 1))
        monthly_feats['payment_history_score'] = float(np.clip(
            base_features['payment_history_score'] * (1.05 - 0.15 * scale), 0, 1))
        monthly_feats['cashflow_stress_ratio'] = float(
            base_features['cashflow_stress_ratio'] * (0.6 + 0.6 * scale))

        Xm = pd.DataFrame([monthly_feats])
        Xm_imp = imputer.transform(Xm)
        m_pd = float(xgb_model.predict_proba(Xm_imp)[0, 1])
        timeline.append({"month": month, "pd": m_pd})

    # Journey events
    journey_events = []
    try:
        raw = json.loads(row['journey_events'])
        seen = set()
        for ev in raw:
            key = ev['date'] + ev['desc']
            if key not in seen:
                seen.add(key)
                journey_events.append(ev)
    except:
        pass

    sector  = row['sector']
    lgd     = LGD_MAP.get(sector, 0.4)
    ead     = float(row['outstanding_loan'])
    el      = current_pd * ead * lgd

    return {
        "company_id":       company_id,
        "sector":           sector,
        "loan_type":        row.get('loan_type', 'Term Loan'),
        "outstanding_loan": ead,
        "lgd_pct":          lgd,
        "expected_loss":    el,
        "potential_recovery": ead - el,
        "current_pd":       current_pd,
        "risk_band":        get_risk_band(current_pd),
        "timeline":         timeline,
        "risk_migration": {
            "start_band": get_risk_band(timeline[0]['pd']),
            "end_band":   get_risk_band(current_pd)
        },
        "journey_events":  journey_events,
        "raw_features":    {k: float(row[k]) for k in FEATURES},
        "action_ladder":   ACTION_LADDER.get(get_risk_band(current_pd), [])
    }

@app.get("/borrowers/{company_id}/explain")
def get_shap_explanation(company_id: str):
    if feature_df is None or explainer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    row = feature_df[feature_df['company_id'] == company_id]
    if row.empty:
        raise HTTPException(status_code=404, detail="Borrower not found")

    X = row[FEATURES].copy()
    X_imp = imputer.transform(X)
    shap_vals = explainer.shap_values(X_imp)

    feat_vals = X.iloc[0].to_dict()
    drivers = sorted(
        [
            {
                "feature":   f,
                "value":     float(feat_vals[f]),
                "impact":    float(shap_vals[0][i]),
                "narrative": FEATURE_NARRATIVE.get(f, lambda x: f"Feature value: {x:.3f}")(float(feat_vals[f]))
            }
            for i, f in enumerate(FEATURES)
        ],
        key=lambda x: abs(x['impact']),
        reverse=True
    )
    return {"key_drivers": drivers[:8]}
