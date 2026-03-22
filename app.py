
import os
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from flask import Flask, request, Response, send_from_directory
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

app = Flask(__name__)

# -------------------------------
# Safe JSON conversion
# -------------------------------
def to_py(obj):
    if isinstance(obj, dict):
        return {k: to_py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_py(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def safe_json(data, status=200):
    return Response(
        json.dumps(to_py(data)),
        status=status,
        mimetype="application/json"
    )

# -------------------------------
# Load dataset
# -------------------------------
XLSX_FILE = "Global_Cybersecurity_Threats_2015-2024.xlsx"

if not os.path.exists(XLSX_FILE):
    raise FileNotFoundError(f"{XLSX_FILE} not found in project folder")

df_raw = pd.read_excel(XLSX_FILE)
attack_desc = pd.read_excel(XLSX_FILE, sheet_name="Attack_desc")
df_raw = df_raw.merge(attack_desc, on="Attack_ID", how="left")

# -------------------------------
# Preprocessing
# -------------------------------
df = df_raw.copy()

CAT_COLS = [
    "Country",
    "Target Industry",
    "Attack Source",
    "Security Vulnerability Type",
    "Defense Mechanism Used"
]

LE = {}
for col in CAT_COLS:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    LE[col] = le

le_id = LabelEncoder()
df["Attack_ID_enc"] = le_id.fit_transform(df["Attack_ID"].astype(str))
LE["Attack_ID"] = le_id

target_le = LabelEncoder()
df["Attack Type"] = target_le.fit_transform(df["Attack Type"].astype(str))
ATTACK_MAP = {int(i): str(name) for i, name in enumerate(target_le.classes_)}

COUNTRY_MAP = {str(c): int(i) for i, c in enumerate(LE["Country"].classes_)}
INDUSTRY_MAP = {str(c): int(i) for i, c in enumerate(LE["Target Industry"].classes_)}
SOURCE_MAP = {str(c): int(i) for i, c in enumerate(LE["Attack Source"].classes_)}
VULN_MAP = {str(c): int(i) for i, c in enumerate(LE["Security Vulnerability Type"].classes_)}
DEFENSE_MAP = {str(c): int(i) for i, c in enumerate(LE["Defense Mechanism Used"].classes_)}
ATTACK_ID_MAP = {str(c): int(i) for i, c in enumerate(LE["Attack_ID"].classes_)}

FEATURE_COLS = [
    "Country",
    "Year",
    "Target Industry",
    "Financial Loss in Million Doller",
    "Number of Affected Users",
    "Attack Source",
    "Security Vulnerability Type",
    "Defense Mechanism Used",
    "Incident Resolution Time (in Hours)",
    "Attack_ID_enc"
]

X = df[FEATURE_COLS].values.astype(float)
y = df["Attack Type"].values.astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# -------------------------------
# Train models
# -------------------------------
MODELS = {
    "Random Forest": RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42),
    "Extra Trees": ExtraTreesClassifier(n_estimators=200, n_jobs=-1, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=150, learning_rate=0.1, max_depth=5, random_state=42
    ),
}

TRAINED = {}
RESULTS = {}

for name, model in MODELS.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    RESULTS[name] = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)) * 100, 2),
        "precision": round(float(precision_score(y_test, y_pred, average="weighted", zero_division=0)) * 100, 2),
        "recall": round(float(recall_score(y_test, y_pred, average="weighted", zero_division=0)) * 100, 2),
        "f1_score": round(float(f1_score(y_test, y_pred, average="weighted", zero_division=0)) * 100, 2),
    }
    TRAINED[name] = model

BEST = max(RESULTS, key=lambda m: RESULTS[m]["accuracy"])

# -------------------------------
# Feature importance
# -------------------------------
fi_pairs = sorted(
    zip(FEATURE_COLS, TRAINED["Random Forest"].feature_importances_),
    key=lambda x: x[1],
    reverse=True
)
FEAT_IMP = {str(k): round(float(v), 4) for k, v in fi_pairs}

# -------------------------------
# Chart data
# -------------------------------
atk_cnt = df_raw["Attack Type"].value_counts()
ATTACK_CHART = {
    "labels": [str(x) for x in atk_cnt.index.tolist()],
    "values": [int(x) for x in atk_cnt.values.tolist()]
}

cty_cnt = df_raw["Country"].value_counts()
COUNTRY_CHART = {
    "labels": [str(x) for x in cty_cnt.index.tolist()],
    "values": [int(x) for x in cty_cnt.values.tolist()]
}

ind_cnt = df_raw["Target Industry"].value_counts()
INDUSTRY_CHART = {
    "labels": [str(x) for x in ind_cnt.index.tolist()],
    "values": [int(x) for x in ind_cnt.values.tolist()]
}

src_cnt = df_raw["Attack Source"].value_counts()
SOURCE_CHART = {
    "labels": [str(x) for x in src_cnt.index.tolist()],
    "values": [int(x) for x in src_cnt.values.tolist()]
}

vuln_cnt = df_raw["Security Vulnerability Type"].value_counts()
VULN_CHART = {
    "labels": [str(x) for x in vuln_cnt.index.tolist()],
    "values": [int(x) for x in vuln_cnt.values.tolist()]
}

def_cnt = df_raw["Defense Mechanism Used"].value_counts()
DEFENSE_CHART = {
    "labels": [str(x) for x in def_cnt.index.tolist()],
    "values": [int(x) for x in def_cnt.values.tolist()]
}

loss_by_atk = df_raw.groupby("Attack Type")["Financial Loss in Million Doller"].mean().sort_values(ascending=False)
LOSS_CHART = {
    "labels": [str(x) for x in loss_by_atk.index.tolist()],
    "values": [round(float(x), 2) for x in loss_by_atk.values.tolist()]
}

res_by_atk = df_raw.groupby("Attack Type")["Incident Resolution Time (in Hours)"].mean().sort_values(ascending=False)
RESTIME_CHART = {
    "labels": [str(x) for x in res_by_atk.index.tolist()],
    "values": [round(float(x), 1) for x in res_by_atk.values.tolist()]
}

yr_cnt = df_raw["Year"].value_counts().sort_index()
YEAR_CHART = {
    "labels": [int(x) for x in yr_cnt.index.tolist()],
    "values": [int(x) for x in yr_cnt.values.tolist()]
}

MARKET = {
    "total_incidents": int(len(df_raw)),
    "total_countries": int(df_raw["Country"].nunique()),
    "total_attack_types": int(df_raw["Attack Type"].nunique()),
    "avg_loss": round(float(df_raw["Financial Loss in Million Doller"].mean()), 2),
    "avg_users_affected": int(df_raw["Number of Affected Users"].mean()),
    "avg_resolution_hrs": round(float(df_raw["Incident Resolution Time (in Hours)"].mean()), 1),
    "year_min": int(df_raw["Year"].min()),
    "year_max": int(df_raw["Year"].max()),
}

# -------------------------------
# Routes
# -------------------------------
@app.route("/")
def home():
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), "script.html")

@app.route("/api/init")
def api_init():
    return safe_json({
        "results": RESULTS,
        "best_model": BEST,
        "feat_imp": FEAT_IMP,
        "market": MARKET,
        "attack_chart": ATTACK_CHART,
        "country_chart": COUNTRY_CHART,
        "industry_chart": INDUSTRY_CHART,
        "source_chart": SOURCE_CHART,
        "vuln_chart": VULN_CHART,
        "defense_chart": DEFENSE_CHART,
        "loss_chart": LOSS_CHART,
        "restime_chart": RESTIME_CHART,
        "year_chart": YEAR_CHART,
        "options": {
            "countries": sorted(COUNTRY_MAP.keys()),
            "industries": sorted(INDUSTRY_MAP.keys()),
            "sources": sorted(SOURCE_MAP.keys()),
            "vulns": sorted(VULN_MAP.keys()),
            "defenses": sorted(DEFENSE_MAP.keys()),
            "attack_ids": sorted(ATTACK_ID_MAP.keys()),
        }
    })

@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        data = request.get_json(force=True, silent=True) or {}

        algo = data.get("algorithm", "Random Forest")
        if algo not in TRAINED:
            return safe_json({"success": False, "error": "Invalid algorithm"}, 400)

        country_enc = COUNTRY_MAP.get(str(data.get("country", "USA")), 0)
        industry_enc = INDUSTRY_MAP.get(str(data.get("industry", "Banking")), 0)
        source_enc = SOURCE_MAP.get(str(data.get("attack_source", "Hacker Group")), 0)
        vuln_enc = VULN_MAP.get(str(data.get("vulnerability", "Unpatched Software")), 0)
        defense_enc = DEFENSE_MAP.get(str(data.get("defense", "Firewall")), 0)
        atk_id_enc = ATTACK_ID_MAP.get(str(data.get("attack_id", "PH01")), 0)

        x = np.array([[
            float(country_enc),
            float(data.get("year", 2020)),
            float(industry_enc),
            float(data.get("financial_loss", 50.0)),
            float(data.get("affected_users", 500)),
            float(source_enc),
            float(vuln_enc),
            float(defense_enc),
            float(data.get("resolution_time", 36)),
            float(atk_id_enc),
        ]])

        clf = TRAINED[algo]
        pred_enc = int(clf.predict(x)[0])
        predicted = ATTACK_MAP.get(pred_enc, "Unknown")

        if hasattr(clf, "predict_proba"):
            probs = clf.predict_proba(x)[0]
            top_idx = probs.argsort()[::-1][:5]
            top5 = [
                {
                    "attack": str(ATTACK_MAP.get(int(i), "?")),
                    "prob": round(float(probs[i]) * 100, 2)
                }
                for i in top_idx
            ]
        else:
            top5 = [{"attack": predicted, "prob": 100.0}]

        rows = df_raw[df_raw["Attack Type"] == predicted]
        a_info = {}

        if not rows.empty:
            a_info = {
                "count": int(len(rows)),
                "avg_loss": round(float(rows["Financial Loss in Million Doller"].mean()), 2),
                "avg_users": int(rows["Number of Affected Users"].mean()),
                "avg_resolution": round(float(rows["Incident Resolution Time (in Hours)"].mean()), 1),
                "top_industry": str(rows["Target Industry"].mode()[0]),
                "top_source": str(rows["Attack Source"].mode()[0]),
                "year_range": [int(rows["Year"].min()), int(rows["Year"].max())],
            }

        return safe_json({
            "success": True,
            "attack_type": predicted,
            "algorithm": algo,
            "top5": top5,
            "attack_info": a_info,
            "metrics": RESULTS,
            "best": BEST,
        })

    except Exception as e:
        return safe_json({"success": False, "error": str(e)}, 500)

if __name__ == "__main__":
    app.run(port=5000)
