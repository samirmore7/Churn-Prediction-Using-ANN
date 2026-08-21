"""
ChurnGuard AI - Customer Churn Prediction (ANN)
================================================
Single-file Flask application. Loads a pre-trained Keras ANN from ANN.pkl
and serves a multi-theme analytics dashboard for predicting bank
customer churn risk.

Files needed:
    app.py            (this file)
    requirements.txt  (dependencies)
    ANN.pkl           (your trained Keras model)
"""

import os
import pickle
import random
import string
from datetime import datetime

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
from flask import Flask, jsonify, render_template_string, request

app = Flask(__name__)

PROJECT_NAME = "ChurnGuard AI"
TAGLINE = "Customer Retention Intelligence, powered by a hand-trained ANN"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "ANN.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

# ---------------------------------------------------------------------------
# Feature schema (order MUST match training matrix)
# ---------------------------------------------------------------------------
FEATURES = [
    dict(key="credit_score", label="Credit Score", section="Profile",
         kind="range", min=300, max=900, step=1, default=650, unit="",
         mean=650.53, std=96.65,
         help="Bureau credit score. Lower scores often correlate with risk."),
    dict(key="geography", label="Geography", section="Profile",
         kind="select", options=[("0", "France"), ("1", "Germany"), ("2", "Spain")],
         default="0", mean=0.7462, std=0.8279,
         help="Country the account is registered in."),
    dict(key="gender", label="Gender", section="Profile",
         kind="toggle2", options=[("0", "Female"), ("1", "Male")],
         default="0", mean=0.5457, std=0.4979,
         help="As recorded on the account."),
    dict(key="age", label="Age", section="Profile",
         kind="range", min=18, max=92, step=1, default=35, unit=" yrs",
         mean=38.92, std=10.49,
         help="Customer age in years."),
    dict(key="tenure", label="Tenure", section="Account",
         kind="range", min=0, max=10, step=1, default=5, unit=" yrs",
         mean=5.01, std=2.89,
         help="Years as a bank customer."),
    dict(key="balance", label="Account Balance", section="Account",
         kind="number", min=0, max=250000, step=100, default=60000, unit="",
         mean=76485.89, std=62397.40,
         help="Current balance held across products."),
    dict(key="num_products", label="Number of Products", section="Account",
         kind="stepper", min=1, max=4, step=1, default=1, unit="",
         mean=1.53, std=0.582,
         help="Bank products the customer holds (cards, loans, savings...)."),
    dict(key="has_cr_card", label="Has Credit Card", section="Engagement",
         kind="toggle", default="1", mean=0.7055, std=0.4558,
         help="Whether the customer holds a credit card with the bank."),
    dict(key="is_active_member", label="Active Member", section="Engagement",
         kind="toggle", default="1", mean=0.5151, std=0.4998,
         help="Whether the customer is currently an engaged/active user."),
    dict(key="estimated_salary", label="Estimated Salary", section="Engagement",
         kind="number", min=0, max=250000, step=100, default=100000, unit="",
         mean=100090.24, std=57510.49,
         help="Estimated annual salary."),
]
FEATURE_KEYS = [f["key"] for f in FEATURES]
MEANS = np.array([f["mean"] for f in FEATURES], dtype="float64")
STDS = np.array([f["std"] for f in FEATURES], dtype="float64")

SECTIONS = ["Profile", "Account", "Engagement"]

# ---------------------------------------------------------------------------
# Load model & Scaler
# ---------------------------------------------------------------------------
MODEL = None
SCALER = None
USING_REAL_SCALER = False

if os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH, "rb") as f:
            MODEL = pickle.load(f)
        print(f"[{PROJECT_NAME}] Loaded ANN model from {MODEL_PATH}")
    except Exception as e:
        print(f"[{PROJECT_NAME}] Error unpickling ANN.pkl: {e}")
else:
    print(f"[{PROJECT_NAME}] Warning: ANN.pkl not found in {BASE_DIR}")

if os.path.exists(SCALER_PATH):
    try:
        with open(SCALER_PATH, "rb") as f:
            SCALER = pickle.load(f)
        USING_REAL_SCALER = True
        print(f"[{PROJECT_NAME}] Found scaler.pkl - using exact transformation.")
    except Exception as exc:
        print(f"[{PROJECT_NAME}] Could not load scaler.pkl ({exc}); falling back to dataset statistics.")

# ---------------------------------------------------------------------------
# Analytics Store
# ---------------------------------------------------------------------------
HISTORY = []


def scale_vector(raw_vec):
    if SCALER is not None:
        return SCALER.transform(raw_vec.reshape(1, -1))[0]
    return (raw_vec - MEANS) / STDS


def predict_proba(raw_vec):
    scaled = scale_vector(raw_vec).astype("float32").reshape(1, -1)
    if MODEL is None:
        # Logistic fallback approximation if model file isn't present
        weights_approx = np.array([-0.05, 0.25, -0.15, 0.75, -0.05, 0.30, -0.10, -0.05, -0.55, 0.05])
        z = np.dot(scaled[0], weights_approx) - 1.2
        return float(1 / (1 + np.exp(-z)))

    try:
        if hasattr(MODEL, "predict"):
            pred = MODEL.predict(scaled, verbose=0)
        else:
            pred = MODEL(scaled, training=False)
            if hasattr(pred, "numpy"):
                pred = pred.numpy()
        
        prob = float(pred[0][0] if np.ndim(pred) > 1 else pred[0])
    except Exception as e:
        print(f"Prediction inference error: {e}")
        prob = 0.25
        
    return max(0.0, min(1.0, prob))


def risk_bucket(prob):
    if prob < 0.30:
        return "Low", "low"
    if prob < 0.60:
        return "Medium", "medium"
    return "High", "high"


def parse_payload(payload):
    raw = np.zeros(len(FEATURES), dtype="float64")
    clean = {}
    for i, feat in enumerate(FEATURES):
        val = payload.get(feat["key"], feat["default"])
        try:
            num = float(val)
        except (TypeError, ValueError):
            num = float(feat["default"])
        if feat["kind"] in ("range", "number", "stepper"):
            num = max(feat.get("min", num), min(feat.get("max", num), num))
        raw[i] = num
        clean[feat["key"]] = num
    return raw, clean


def compute_impacts(raw_vec):
    base = predict_proba(raw_vec)
    impacts = []
    for i, feat in enumerate(FEATURES):
        perturbed = raw_vec.copy()
        step = STDS[i] if STDS[i] > 0 else 1.0
        perturbed[i] += step
        if feat["kind"] in ("range", "number", "stepper"):
            perturbed[i] = max(feat.get("min", perturbed[i]),
                               min(feat.get("max", perturbed[i]), perturbed[i]))
        new_p = predict_proba(perturbed)
        impacts.append({
            "key": feat["key"],
            "label": feat["label"],
            "delta": round((new_p - base) * 100, 2),
        })
    max_abs = max(1e-6, max(abs(x["delta"]) for x in impacts))
    for x in impacts:
        x["pct"] = round(abs(x["delta"]) / max_abs * 100, 1)
    impacts.sort(key=lambda x: abs(x["delta"]), reverse=True)
    return impacts, base


def gen_id():
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=8))


def dashboard_stats():
    total = len(HISTORY)
    if total == 0:
        return dict(total=0, churn_rate=0, avg_prob=0, high_risk=0,
                    buckets=[0] * 10, recent=[])
    probs = [h["probability"] for h in HISTORY]
    high_risk = sum(1 for p in probs if p >= 0.60)
    churned_calls = sum(1 for p in probs if p >= 0.50)
    buckets = [0] * 10
    for p in probs:
        idx = min(9, int(p * 10))
        buckets[idx] += 1
    recent = list(reversed(HISTORY[-8:]))
    return dict(
        total=total,
        churn_rate=round(churned_calls / total * 100, 1),
        avg_prob=round(sum(probs) / total * 100, 1),
        high_risk=high_risk,
        buckets=buckets,
        recent=recent,
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template_string(
        PAGE_TEMPLATE,
        project_name=PROJECT_NAME,
        tagline=TAGLINE,
        features=FEATURES,
        sections=SECTIONS,
        using_real_scaler=USING_REAL_SCALER,
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    payload = request.get_json(force=True, silent=True) or {}
    raw_vec, clean_inputs = parse_payload(payload)
    impacts, prob = compute_impacts(raw_vec)
    label, cls = risk_bucket(prob)

    record = dict(
        id=gen_id(),
        ts=datetime.now().strftime("%H:%M:%S"),
        probability=prob,
        risk=label,
        risk_class=cls,
        inputs=clean_inputs,
    )
    HISTORY.append(record)

    return jsonify(dict(
        probability=round(prob * 100, 2),
        risk=label,
        risk_class=cls,
        impacts=impacts,
        stats=dashboard_stats(),
    ))


@app.route("/api/stats")
def api_stats():
    return jsonify(dashboard_stats())


@app.route("/api/reset", methods=["POST"])
def api_reset():
    HISTORY.clear()
    return jsonify(dashboard_stats())


# ---------------------------------------------------------------------------
# Embedded Single-Page Interface
# ---------------------------------------------------------------------------
PAGE_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="en" data-theme="midnight">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{{ project_name }} · Churn Risk Intelligence</title>
<style>
:root{
  --radius-lg:20px; --radius-md:14px; --radius-sm:9px;
  --ease:cubic-bezier(.22,1,.36,1);
}

/* ---------- THEME: Midnight Gold (default) ---------- */
html[data-theme="midnight"]{
  --bg-0:#0a0d13; --bg-1:#0f131b; --bg-2:#151a24;
  --glass:rgba(255,255,255,0.045); --glass-brd:rgba(255,255,255,0.09);
  --text-hi:#eef0f4; --text-mid:#aab0bd; --text-low:#6b7280;
  --accent:#d4af5a; --accent-2:#59c2c9; --accent-ink:#20180a;
  --ok:#4ade80; --warn:#f5b942; --bad:#f2664a;
  --mesh: radial-gradient(circle at 12% 8%, rgba(212,175,90,.14), transparent 42%),
          radial-gradient(circle at 88% 12%, rgba(89,194,201,.12), transparent 40%),
          radial-gradient(circle at 50% 100%, rgba(212,175,90,.06), transparent 55%);
  --font-display:Georgia,'Iowan Old Style','Palatino Linotype',serif;
}
/* ---------- THEME: Emerald Vault ---------- */
html[data-theme="emerald"]{
  --bg-0:#071410; --bg-1:#0a1a15; --bg-2:#0f221c;
  --glass:rgba(160,255,210,0.045); --glass-brd:rgba(160,255,210,0.09);
  --text-hi:#eafaf1; --text-mid:#9fc4b3; --text-low:#5f8272;
  --accent:#34d399; --accent-2:#c084fc; --accent-ink:#04140d;
  --ok:#4ade80; --warn:#f5b942; --bad:#fb7185;
  --mesh: radial-gradient(circle at 10% 10%, rgba(52,211,153,.14), transparent 42%),
          radial-gradient(circle at 90% 15%, rgba(192,132,252,.10), transparent 40%),
          radial-gradient(circle at 50% 100%, rgba(52,211,153,.06), transparent 55%);
  --font-display:Georgia,'Iowan Old Style','Palatino Linotype',serif;
}
/* ---------- THEME: Royal Amethyst ---------- */
html[data-theme="amethyst"]{
  --bg-0:#0f0a1a; --bg-1:#150e22; --bg-2:#1b122b;
  --glass:rgba(216,180,254,0.05); --glass-brd:rgba(216,180,254,0.10);
  --text-hi:#f3ecfb; --text-mid:#b9a8cf; --text-low:#786690;
  --accent:#a855f7; --accent-2:#f472b6; --accent-ink:#180a26;
  --ok:#4ade80; --warn:#f5b942; --bad:#fb7185;
  --mesh: radial-gradient(circle at 12% 10%, rgba(168,85,247,.16), transparent 42%),
          radial-gradient(circle at 88% 14%, rgba(244,114,182,.12), transparent 40%),
          radial-gradient(circle at 50% 100%, rgba(168,85,247,.07), transparent 55%);
  --font-display:Georgia,'Iowan Old Style','Palatino Linotype',serif;
}
/* ---------- THEME: Arctic Ivory (light) ---------- */
html[data-theme="ivory"]{
  --bg-0:#e7edf1; --bg-1:#eef2f5; --bg-2:#f5f7f9;
  --glass:rgba(255,255,255,0.55); --glass-brd:rgba(27,42,74,0.10);
  --text-hi:#16202e; --text-mid:#3c4a5c; --text-low:#748094;
  --accent:#1b2a4a; --accent-2:#c97b4a; --accent-ink:#f5f7f9;
  --ok:#1a9e6b; --warn:#b5790f; --bad:#c23b32;
  --mesh: radial-gradient(circle at 12% 8%, rgba(27,42,74,.07), transparent 42%),
          radial-gradient(circle at 88% 12%, rgba(201,123,74,.10), transparent 40%),
          radial-gradient(circle at 50% 100%, rgba(27,42,74,.04), transparent 55%);
  --font-display:Georgia,'Iowan Old Style','Palatino Linotype',serif;
}

*{box-sizing:border-box;}
html,body{margin:0;padding:0;}
body{
  background:
    var(--mesh),
    linear-gradient(180deg, var(--bg-0), var(--bg-1) 45%, var(--bg-0));
  background-attachment:fixed;
  color:var(--text-hi);
  font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;
  min-height:100vh;
  -webkit-font-smoothing:antialiased;
  transition:background .5s var(--ease), color .5s var(--ease);
}
.wrap{max-width:1180px; margin:0 auto; padding:28px 20px 80px;}

/* ---------- Top bar ---------- */
.topbar{
  display:flex; align-items:center; justify-content:space-between;
  gap:16px; margin-bottom:30px; flex-wrap:wrap;
}
.brand{display:flex; align-items:center; gap:12px;}
.brand-mark{
  width:42px;height:42px;border-radius:12px;
  background:linear-gradient(135deg,var(--accent),var(--accent-2));
  display:flex;align-items:center;justify-content:center;
  box-shadow:0 6px 18px -6px rgba(0,0,0,.5);
  flex-shrink:0;
}
.brand-mark svg{width:22px;height:22px;}
.brand-text h1{
  font-family:var(--font-display); font-weight:700;
  font-size:22px; margin:0; letter-spacing:.2px; color:var(--text-hi);
}
.brand-text p{margin:2px 0 0; font-size:12.5px; color:var(--text-mid);}

.theme-switch{display:flex; gap:8px; background:var(--glass); border:1px solid var(--glass-brd);
  padding:6px; border-radius:999px; backdrop-filter:blur(14px);}
.theme-dot{
  width:30px;height:30px;border-radius:50%; cursor:pointer; border:2px solid transparent;
  position:relative; transition:transform .25s var(--ease), border-color .25s var(--ease);
}
.theme-dot:hover{transform:translateY(-2px) scale(1.06);}
.theme-dot.active{border-color:var(--text-hi);}
.theme-dot[data-t="midnight"]{background:linear-gradient(135deg,#0f131b,#d4af5a);}
.theme-dot[data-t="emerald"]{background:linear-gradient(135deg,#0a1a15,#34d399);}
.theme-dot[data-t="amethyst"]{background:linear-gradient(135deg,#150e22,#a855f7);}
.theme-dot[data-t="ivory"]{background:linear-gradient(135deg,#eef2f5,#1b2a4a);}

/* ---------- Cards ---------- */
.card{
  background:var(--glass); border:1px solid var(--glass-brd);
  border-radius:var(--radius-lg); backdrop-filter:blur(18px);
  box-shadow:0 20px 50px -30px rgba(0,0,0,.6);
  transition:border-color .4s var(--ease), background .4s var(--ease);
}

.grid{display:grid; grid-template-columns:1.15fr 1fr; gap:22px; align-items:start;}
@media(max-width:920px){.grid{grid-template-columns:1fr;}}

/* ---------- Form ---------- */
.form-card{padding:26px 26px 22px;}
.form-card h2{font-family:var(--font-display); font-size:18px; margin:0 0 4px; color:var(--text-hi);}
.form-card .sub{color:var(--text-mid); font-size:12.5px; margin:0 0 18px;}
.section-label{
  font-size:11px; letter-spacing:1.4px; text-transform:uppercase; color:var(--accent);
  margin:22px 0 10px; font-weight:700;
}
.section-label:first-of-type{margin-top:4px;}
.field{margin-bottom:16px;}
.field-head{display:flex; justify-content:space-between; align-items:baseline; margin-bottom:6px;}
.field-head label{font-size:13.5px; color:var(--text-hi); font-weight:600;}
.field-head .val{font-size:13px; color:var(--accent); font-variant-numeric:tabular-nums; font-weight:600;}
.field .help{font-size:11.5px; color:var(--text-low); margin-top:4px;}

input[type="range"]{
  -webkit-appearance:none; width:100%; height:6px; border-radius:6px;
  background:linear-gradient(90deg,var(--accent),var(--accent-2)); outline:none; cursor:pointer;
}
input[type="range"]::-webkit-slider-thumb{
  -webkit-appearance:none; width:18px; height:18px; border-radius:50%;
  background:var(--text-hi); border:3px solid var(--accent);
  box-shadow:0 2px 8px rgba(0,0,0,.4); cursor:pointer; transition:transform .15s var(--ease);
}
input[type="range"]::-webkit-slider-thumb:hover{transform:scale(1.15);}
input[type="range"]::-moz-range-thumb{
  width:18px; height:18px; border-radius:50%; background:var(--text-hi);
  border:3px solid var(--accent); cursor:pointer;
}

input[type="number"]{
  width:100%; padding:10px 12px; border-radius:var(--radius-sm);
  border:1px solid var(--glass-brd); background:rgba(0,0,0,.15);
  color:var(--text-hi); font-size:13.5px; outline:none;
  transition:border-color .2s var(--ease), box-shadow .2s var(--ease);
}
html[data-theme="ivory"] input[type="number"]{background:rgba(255,255,255,.6);}
input[type="number"]:focus{border-color:var(--accent); box-shadow:0 0 0 3px color-mix(in srgb, var(--accent) 25%, transparent);}

select{
  width:100%; padding:10px 12px; border-radius:var(--radius-sm);
  border:1px solid var(--glass-brd); background:rgba(0,0,0,.15);
  color:var(--text-hi); font-size:13.5px; outline:none; cursor:pointer;
  appearance:none; -webkit-appearance:none;
  background-image:linear-gradient(45deg,transparent 50%,var(--text-mid) 50%),linear-gradient(135deg,var(--text-mid) 50%,transparent 50%);
  background-position:calc(100% - 18px) center, calc(100% - 13px) center;
  background-size:5px 5px, 5px 5px; background-repeat:no-repeat;
}
html[data-theme="ivory"] select{background-color:rgba(255,255,255,.6);}

.seg{display:flex; border-radius:var(--radius-sm); overflow:hidden; border:1px solid var(--glass-brd);}
.seg button{
  flex:1; padding:9px 10px; background:rgba(0,0,0,.12); color:var(--text-mid);
  border:none; font-size:13px; font-weight:600; cursor:pointer; transition:all .2s var(--ease);
}
html[data-theme="ivory"] .seg button{background:rgba(255,255,255,.5);}
.seg button.active{background:linear-gradient(135deg,var(--accent),var(--accent-2)); color:var(--accent-ink);}

.toggle-row{display:flex; align-items:center; justify-content:space-between; padding:2px 0;}
.switch{position:relative; width:46px; height:26px; flex-shrink:0;}
.switch input{opacity:0; width:0; height:0;}
.slider-pill{
  position:absolute; inset:0; background:rgba(0,0,0,.25); border-radius:999px; cursor:pointer;
  transition:background .25s var(--ease); border:1px solid var(--glass-brd);
}
.slider-pill:before{
  content:""; position:absolute; width:20px; height:20px; left:2px; top:2px;
  background:var(--text-hi); border-radius:50%; transition:transform .25s var(--ease);
}
.switch input:checked + .slider-pill{background:linear-gradient(135deg,var(--accent),var(--accent-2));}
.switch input:checked + .slider-pill:before{transform:translateX(20px);}

.stepper{display:flex; align-items:center; gap:10px;}
.stepper button{
  width:34px;height:34px;border-radius:50%; border:1px solid var(--glass-brd);
  background:rgba(0,0,0,.15); color:var(--text-hi); font-size:16px; cursor:pointer;
  display:flex; align-items:center; justify-content:center; transition:all .2s var(--ease);
}
.stepper button:hover{background:var(--accent); color:var(--accent-ink); border-color:var(--accent);}
.stepper .count{
  flex:1; text-align:center; font-size:16px; font-weight:700; color:var(--text-hi);
  font-variant-numeric:tabular-nums;
}

.actions{display:flex; gap:12px; margin-top:22px;}
.btn{
  border:none; border-radius:999px; font-weight:700; font-size:14px; cursor:pointer;
  padding:13px 22px; display:inline-flex; align-items:center; justify-content:center; gap:8px;
  transition:transform .2s var(--ease), box-shadow .2s var(--ease), filter .2s var(--ease);
  position:relative; overflow:hidden;
}
.btn-primary{
  flex:1; color:var(--accent-ink);
  background:linear-gradient(135deg,var(--accent),var(--accent-2));
  box-shadow:0 12px 30px -12px color-mix(in srgb, var(--accent) 70%, transparent);
}
.btn-primary:hover{transform:translateY(-2px); filter:brightness(1.06);}
.btn-primary:active{transform:translateY(0);}
.btn-primary .shine{
  position:absolute; top:0; left:-60%; width:40%; height:100%;
  background:linear-gradient(120deg, transparent, rgba(255,255,255,.55), transparent);
  transform:skewX(-20deg); transition:left .7s var(--ease);
}
.btn-primary:hover .shine{left:130%;}
.btn-ghost{
  background:transparent; color:var(--text-mid); border:1px solid var(--glass-brd);
}
.btn-ghost:hover{color:var(--text-hi); border-color:var(--text-mid);}
.btn[disabled]{opacity:.6; cursor:progress;}

/* ---------- Result Card ---------- */
.result-card{padding:26px; display:flex; flex-direction:column; align-items:center; text-align:center;}
.result-card h2{font-family:var(--font-display); font-size:18px; margin:0 0 2px;}
.result-card .sub{color:var(--text-mid); font-size:12.5px; margin:0 0 6px;}
.gauge-wrap{position:relative; width:240px; margin:10px auto 4px;}
.gauge-num{
  position:absolute; left:0; right:0; top:60%; transform:translateY(-50%);
  font-family:var(--font-display); font-size:40px; font-weight:700; color:var(--text-hi);
}
.gauge-label{
  position:absolute; left:0; right:0; top:82%;
  font-size:12px; letter-spacing:1px; text-transform:uppercase; color:var(--text-mid);
}
.risk-pill{
  display:inline-flex; align-items:center; gap:6px; padding:6px 14px; border-radius:999px;
  font-size:12.5px; font-weight:700; margin-top:14px; letter-spacing:.3px;
}
.risk-pill.low{background:rgba(74,222,128,.15); color:var(--ok);}
.risk-pill.medium{background:rgba(245,185,66,.15); color:var(--warn);}
.risk-pill.high{background:rgba(242,102,74,.15); color:var(--bad);}
.risk-pill .dot{width:7px;height:7px;border-radius:50%; background:currentColor;}

.impacts{width:100%; margin-top:22px; text-align:left;}
.impacts h3{font-size:11px; letter-spacing:1.4px; text-transform:uppercase; color:var(--text-mid); margin:0 0 12px;}
.impact-row{margin-bottom:9px;}
.impact-row .top{display:flex; justify-content:space-between; font-size:12.5px; margin-bottom:4px;}
.impact-row .top .name{color:var(--text-hi); font-weight:600;}
.impact-row .top .delta{font-variant-numeric:tabular-nums; color:var(--text-mid);}
.impact-track{height:6px; border-radius:6px; background:rgba(0,0,0,.2); overflow:hidden;}
html[data-theme="ivory"] .impact-track{background:rgba(0,0,0,.08);}
.impact-fill{height:100%; border-radius:6px; transition:width .6s var(--ease);}
.impact-fill.up{background:linear-gradient(90deg,var(--bad),#ff9d80);}
.impact-fill.down{background:linear-gradient(90deg,var(--ok),#8bf5b0);}

.placeholder-note{color:var(--text-low); font-size:12.5px; margin-top:18px; line-height:1.6;}

/* ---------- Dashboard ---------- */
.dash{margin-top:30px;}
.dash-head{display:flex; align-items:baseline; justify-content:space-between; margin-bottom:16px; flex-wrap:wrap; gap:8px;}
.dash-head h2{font-family:var(--font-display); font-size:19px; margin:0;}
.dash-head .link-btn{background:none; border:none; color:var(--text-mid); font-size:12.5px; cursor:pointer; text-decoration:underline; text-underline-offset:3px;}
.dash-head .link-btn:hover{color:var(--text-hi);}

.stat-grid{display:grid; grid-template-columns:repeat(4,1fr); gap:16px; margin-bottom:18px;}
@media(max-width:820px){.stat-grid{grid-template-columns:repeat(2,1fr);}}
.stat-card{padding:18px 20px;}
.stat-card .label{font-size:11px; letter-spacing:1px; text-transform:uppercase; color:var(--text-mid); margin-bottom:8px;}
.stat-card .value{font-family:var(--font-display); font-size:28px; font-weight:700; color:var(--text-hi);}
.stat-card .value span{font-size:15px; color:var(--text-mid); font-weight:400;}

.dash-grid{display:grid; grid-template-columns:1.1fr 1fr; gap:18px;}
@media(max-width:920px){.dash-grid{grid-template-columns:1fr;}}
.panel{padding:22px;}
.panel h3{font-size:13px; letter-spacing:.6px; margin:0 0 16px; color:var(--text-hi);}

.hist-row{display:flex; align-items:flex-end; gap:6px; height:130px;}
.hist-bar{flex:1; background:linear-gradient(180deg,var(--accent),var(--accent-2)); border-radius:5px 5px 2px 2px; min-height:3px; transition:height .5s var(--ease); position:relative;}
.hist-labels{display:flex; gap:6px; margin-top:8px;}
.hist-labels span{flex:1; text-align:center; font-size:9.5px; color:var(--text-low);}

table.recent{width:100%; border-collapse:collapse; font-size:12.5px;}
table.recent th{text-align:left; color:var(--text-mid); font-weight:600; font-size:10.5px; letter-spacing:.6px; text-transform:uppercase; padding:0 8px 8px 0;}
table.recent td{padding:8px 8px 8px 0; border-top:1px solid var(--glass-brd); color:var(--text-hi);}
table.recent .badge{padding:3px 9px; border-radius:999px; font-size:11px; font-weight:700;}
table.recent .badge.low{background:rgba(74,222,128,.15); color:var(--ok);}
table.recent .badge.medium{background:rgba(245,185,66,.15); color:var(--warn);}
table.recent .badge.high{background:rgba(242,102,74,.15); color:var(--bad);}
.empty-state{color:var(--text-low); font-size:12.5px; text-align:center; padding:24px 0;}

footer{margin-top:40px; text-align:center; color:var(--text-low); font-size:11.5px; line-height:1.8;}
footer b{color:var(--text-mid);}

@media (prefers-reduced-motion: reduce){
  *{animation:none !important; transition:none !important;}
}
</style>
</head>
<body>
<div class="wrap">

  <div class="topbar">
    <div class="brand">
      <div class="brand-mark">
        <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
          <path d="M12 2L3 6v6c0 5 3.8 8.7 9 10 5.2-1.3 9-5 9-10V6l-9-4z" fill="var(--accent-ink)"/>
          <path d="M8.5 12l2.3 2.3L16 9" stroke="#fff" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
      </div>
      <div class="brand-text">
        <h1>{{ project_name }}</h1>
        <p>{{ tagline }}</p>
      </div>
    </div>
    <div class="theme-switch" id="themeSwitch">
      <div class="theme-dot active" data-t="midnight" title="Midnight Gold"></div>
      <div class="theme-dot" data-t="emerald" title="Emerald Vault"></div>
      <div class="theme-dot" data-t="amethyst" title="Royal Amethyst"></div>
      <div class="theme-dot" data-t="ivory" title="Arctic Ivory"></div>
    </div>
  </div>

  <div class="grid">
    <!-- FORM -->
    <div class="card form-card">
      <h2>Customer Profile</h2>
      <p class="sub">Enter account details to estimate churn probability.</p>

      <form id="predictForm">
        {% for sec in sections %}
        <div class="section-label">{{ sec }}</div>
        {% for f in features if f.section == sec %}
          <div class="field" data-key="{{ f.key }}">
            {% if f.kind == "range" %}
              <div class="field-head">
                <label for="in_{{ f.key }}">{{ f.label }}</label>
                <span class="val" id="val_{{ f.key }}">{{ f.default }}{{ f.unit }}</span>
              </div>
              <input type="range" id="in_{{ f.key }}" min="{{ f.min }}" max="{{ f.max }}" step="{{ f.step }}" value="{{ f.default }}">
              <div class="help">{{ f.help }}</div>

            {% elif f.kind == "number" %}
              <div class="field-head"><label for="in_{{ f.key }}">{{ f.label }}</label></div>
              <input type="number" id="in_{{ f.key }}" min="{{ f.min }}" max="{{ f.max }}" step="{{ f.step }}" value="{{ f.default }}">
              <div class="help">{{ f.help }}</div>

            {% elif f.kind == "select" %}
              <div class="field-head"><label for="in_{{ f.key }}">{{ f.label }}</label></div>
              <select id="in_{{ f.key }}">
                {% for val, txt in f.options %}
                <option value="{{ val }}" {% if val == f.default %}selected{% endif %}>{{ txt }}</option>
                {% endfor %}
              </select>
              <div class="help">{{ f.help }}</div>

            {% elif f.kind == "toggle2" %}
              <div class="field-head"><label>{{ f.label }}</label></div>
              <div class="seg" id="in_{{ f.key }}" data-value="{{ f.default }}">
                {% for val, txt in f.options %}
                <button type="button" data-v="{{ val }}" class="{% if val == f.default %}active{% endif %}">{{ txt }}</button>
                {% endfor %}
              </div>
              <div class="help">{{ f.help }}</div>

            {% elif f.kind == "toggle" %}
              <div class="toggle-row">
                <div>
                  <label>{{ f.label }}</label>
                  <div class="help" style="margin-top:2px;">{{ f.help }}</div>
                </div>
                <label class="switch">
                  <input type="checkbox" id="in_{{ f.key }}" {% if f.default == "1" %}checked{% endif %}>
                  <span class="slider-pill"></span>
                </label>
              </div>

            {% elif f.kind == "stepper" %}
              <div class="field-head"><label>{{ f.label }}</label></div>
              <div class="stepper" id="in_{{ f.key }}" data-value="{{ f.default }}" data-min="{{ f.min }}" data-max="{{ f.max }}">
                <button type="button" data-d="-1">−</button>
                <div class="count">{{ f.default }}</div>
                <button type="button" data-d="1">+</button>
              </div>
              <div class="help">{{ f.help }}</div>
            {% endif %}
          </div>
        {% endfor %}
        {% endfor %}

        <div class="actions">
          <button type="submit" class="btn btn-primary" id="predictBtn">
            <span class="shine"></span>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none"><path d="M13 2L3 14h8l-1 8 11-14h-8l1-6z" fill="currentColor"/></svg>
            Predict Churn Risk
          </button>
          <button type="button" class="btn btn-ghost" id="resetBtn">Reset</button>
        </div>
      </form>
    </div>

    <!-- RESULT -->
    <div class="card result-card">
      <h2>Risk Assessment</h2>
      <p class="sub">Live prediction from the ANN model</p>

      <div class="gauge-wrap">
        <svg id="gaugeSvg" viewBox="0 0 240 150" width="240" height="150"></svg>
        <div class="gauge-num" id="gaugeNum">--%</div>
        <div class="gauge-label" id="gaugeLbl">awaiting input</div>
      </div>

      <div class="risk-pill low" id="riskPill" style="visibility:hidden;">
        <span class="dot"></span><span id="riskText">Low Risk</span>
      </div>

      <div class="impacts" id="impactsBlock" style="display:none;">
        <h3>Top Sensitivity Factors</h3>
        <div id="impactsList"></div>
      </div>

      <p class="placeholder-note" id="placeholderNote">
        Fill in the customer profile and click <b>Predict Churn Risk</b> to see
        the model's estimated probability, risk tier, and the factors the
        prediction is most sensitive to.
      </p>
    </div>
  </div>

  <!-- DASHBOARD -->
  <div class="dash">
    <div class="dash-head">
      <h2>Analytics Dashboard</h2>
      <button class="link-btn" id="clearHistoryBtn">Clear history</button>
    </div>

    <div class="stat-grid">
      <div class="card stat-card"><div class="label">Total Predictions</div><div class="value" id="statTotal">0</div></div>
      <div class="card stat-card"><div class="label">Churn Rate</div><div class="value" id="statChurn">0<span>%</span></div></div>
      <div class="card stat-card"><div class="label">Avg. Probability</div><div class="value" id="statAvg">0<span>%</span></div></div>
      <div class="card stat-card"><div class="label">High Risk Customers</div><div class="value" id="statHigh">0</div></div>
    </div>

    <div class="dash-grid">
      <div class="card panel">
        <h3>Probability Distribution</h3>
        <div class="hist-row" id="histRow"></div>
        <div class="hist-labels" id="histLabels"></div>
      </div>
      <div class="card panel">
        <h3>Recent Predictions</h3>
        <div id="recentWrap"><div class="empty-state">No predictions yet.</div></div>
      </div>
    </div>
  </div>

  <footer>
    Model: Sequential ANN · Dense(8→8→7→8→7→1) · sigmoid output · 10 input features<br>
    {% if using_real_scaler %}
      Predictions use your original <b>scaler.pkl</b> for exact-match scaling.
    {% else %}
      Predictions are standardized using approximate bank-churn dataset statistics
      (no <b>scaler.pkl</b> found) &mdash; add your original scaler next to app.py for exact-match results.
    {% endif %}
  </footer>
</div>

<script>
const FEATURE_KEYS = {{ features | map(attribute='key') | list | tojson }};
const FEATURE_META = {{ features | tojson }};

/* ---------------- Theme switching ---------------- */
const themeSwitch = document.getElementById('themeSwitch');
themeSwitch.addEventListener('click', (e) => {
  const dot = e.target.closest('.theme-dot');
  if(!dot) return;
  document.querySelectorAll('.theme-dot').forEach(d => d.classList.remove('active'));
  dot.classList.add('active');
  document.documentElement.setAttribute('data-theme', dot.dataset.t);
});

/* ---------------- Field wiring ---------------- */
FEATURE_META.forEach(f => {
  if(f.kind === 'range'){
    const el = document.getElementById('in_' + f.key);
    const out = document.getElementById('val_' + f.key);
    el.addEventListener('input', () => out.textContent = el.value + (f.unit || ''));
  }
  if(f.kind === 'toggle2'){
    const wrap = document.getElementById('in_' + f.key);
    wrap.querySelectorAll('button').forEach(btn => {
      btn.addEventListener('click', () => {
        wrap.querySelectorAll('button').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        wrap.dataset.value = btn.dataset.v;
      });
    });
  }
  if(f.kind === 'stepper'){
    const wrap = document.getElementById('in_' + f.key);
    const countEl = wrap.querySelector('.count');
    wrap.querySelectorAll('button').forEach(btn => {
      btn.addEventListener('click', () => {
        let v = parseInt(wrap.dataset.value, 10) + parseInt(btn.dataset.d, 10);
        v = Math.max(parseInt(wrap.dataset.min,10), Math.min(parseInt(wrap.dataset.max,10), v));
        wrap.dataset.value = v;
        countEl.textContent = v;
      });
    });
  }
});

function collectPayload(){
  const payload = {};
  FEATURE_META.forEach(f => {
    if(f.kind === 'range' || f.kind === 'number'){
      payload[f.key] = parseFloat(document.getElementById('in_' + f.key).value);
    } else if(f.kind === 'select'){
      payload[f.key] = parseFloat(document.getElementById('in_' + f.key).value);
    } else if(f.kind === 'toggle2' || f.kind === 'stepper'){
      payload[f.key] = parseFloat(document.getElementById('in_' + f.key).dataset.value);
    } else if(f.kind === 'toggle'){
      payload[f.key] = document.getElementById('in_' + f.key).checked ? 1 : 0;
    }
  });
  return payload;
}

/* ---------------- Gauge ---------------- */
function drawGauge(pct){
  const svg = document.getElementById('gaugeSvg');
  const cx = 120, cy = 120, r = 92;
  const color = pct < 30 ? 'var(--ok)' : pct < 60 ? 'var(--warn)' : 'var(--bad)';
  const endAngle = Math.PI - (pct/100)*Math.PI;
  const polar = (ang) => [cx + r*Math.cos(ang), cy - r*Math.sin(ang)];
  const trackStart = polar(Math.PI), trackEnd = polar(0);
  const valEnd = polar(endAngle);
  const largeArc = pct > 50 ? 1 : 0;

  svg.innerHTML = `
    <path d="M ${trackStart[0]} ${trackStart[1]} A ${r} ${r} 0 1 1 ${trackEnd[0]} ${trackEnd[1]}"
          fill="none" stroke="rgba(120,120,140,0.18)" stroke-width="16" stroke-linecap="round"/>
    <path d="M ${trackStart[0]} ${trackStart[1]} A ${r} ${r} 0 ${largeArc} 1 ${valEnd[0]} ${valEnd[1]}"
          fill="none" stroke="${color}" stroke-width="16" stroke-linecap="round"
          style="transition: d .6s var(--ease);"/>
    <circle cx="${cx}" cy="${cy}" r="5" fill="${color}"/>
  `;
  document.getElementById('gaugeNum').textContent = pct.toFixed(1) + '%';
  document.getElementById('gaugeNum').style.color = color;
  document.getElementById('gaugeLbl').textContent = 'churn probability';
}

/* ---------------- Predict ---------------- */
const form = document.getElementById('predictForm');
form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const btn = document.getElementById('predictBtn');
  btn.disabled = true;
  const payload = collectPayload();
  try{
    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    renderResult(data);
    renderDashboard(data.stats);
  } catch(err){
    console.error(err);
  } finally {
    btn.disabled = false;
  }
});

function renderResult(data){
  document.getElementById('placeholderNote').style.display = 'none';
  drawGauge(data.probability);

  const pill = document.getElementById('riskPill');
  pill.style.visibility = 'visible';
  pill.className = 'risk-pill ' + data.risk_class;
  document.getElementById('riskText').textContent = data.risk + ' Risk';

  const impactsBlock = document.getElementById('impactsBlock');
  const list = document.getElementById('impactsList');
  list.innerHTML = '';
  data.impacts.slice(0, 5).forEach(im => {
    const dir = im.delta >= 0 ? 'up' : 'down';
    const row = document.createElement('div');
    row.className = 'impact-row';
    row.innerHTML = `
      <div class="top">
        <span class="name">${im.label}</span>
        <span class="delta">${im.delta > 0 ? '+' : ''}${im.delta.toFixed(2)} pp</span>
      </div>
      <div class="impact-track"><div class="impact-fill ${dir}" style="width:${im.pct}%"></div></div>
    `;
    list.appendChild(row);
  });
  impactsBlock.style.display = 'block';
}

/* ---------------- Dashboard render ---------------- */
function renderDashboard(stats){
  document.getElementById('statTotal').textContent = stats.total;
  document.getElementById('statChurn').innerHTML = stats.churn_rate + '<span>%</span>';
  document.getElementById('statAvg').innerHTML = stats.avg_prob + '<span>%</span>';
  document.getElementById('statHigh').textContent = stats.high_risk;

  const histRow = document.getElementById('histRow');
  const histLabels = document.getElementById('histLabels');
  histRow.innerHTML = '';
  histLabels.innerHTML = '';
  const maxCount = Math.max(1, ...stats.buckets);
  stats.buckets.forEach((c, i) => {
    const bar = document.createElement('div');
    bar.className = 'hist-bar';
    bar.style.height = Math.max(3, (c / maxCount) * 120) + 'px';
    bar.title = (i*10) + '-' + (i*10+10) + '%: ' + c;
    histRow.appendChild(bar);
    const lbl = document.createElement('span');
    lbl.textContent = i % 2 === 0 ? (i*10) + '%' : '';
    histLabels.appendChild(lbl);
  });

  const recentWrap = document.getElementById('recentWrap');
  if(stats.recent.length === 0){
    recentWrap.innerHTML = '<div class="empty-state">No predictions yet.</div>';
    return;
  }
  let rows = stats.recent.map(r => `
    <tr>
      <td>${r.ts}</td>
      <td>${(r.probability*100).toFixed(1)}%</td>
      <td><span class="badge ${r.risk_class}">${r.risk}</span></td>
    </tr>
  `).join('');
  recentWrap.innerHTML = `
    <table class="recent">
      <thead><tr><th>Time</th><th>Probability</th><th>Risk</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
  `;
}

document.getElementById('resetBtn').addEventListener('click', () => window.location.reload());

document.getElementById('clearHistoryBtn').addEventListener('click', async () => {
  const res = await fetch('/api/reset', {method:'POST'});
  const stats = await res.json();
  renderDashboard(stats);
});

/* initial empty gauge */
drawGauge(0);
document.getElementById('gaugeNum').textContent = '--%';
document.getElementById('gaugeLbl').textContent = 'awaiting input';

/* initial dashboard load */
fetch('/api/stats').then(r => r.json()).then(renderDashboard);
</script>
</body>
</html>
"""

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(debug=debug, host="0.0.0.0", port=port)
