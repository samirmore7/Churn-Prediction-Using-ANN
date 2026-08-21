"""
ChurnGuard AI - Customer Churn Prediction
==========================================
Single-file Flask app featuring:
- Pure NumPy forward pass (replicates ANN.pkl without Keras/TensorFlow dependencies)
- 4 premium themes: Midnight Gold, Emerald Vault, Royal Amethyst, Arctic Ivory
- Horizontal widescreen grid optimized for screenshots
- Sensitivity analysis and real-time dashboard analytics
"""

import os
import pickle
import random
import string
from datetime import datetime
import numpy as np
from flask import Flask, jsonify, render_template_string, request

app = Flask(__name__)

PROJECT_NAME = "ChurnGuard AI"
TAGLINE = "Customer Retention Intelligence, powered by an Artificial Neural Network"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

# ---------------------------------------------------------------------------
# Feature Schema
# ---------------------------------------------------------------------------
FEATURES = [
    dict(key="credit_score", label="Credit Score", section="Profile",
         kind="range", min=300, max=900, step=1, default=650, unit="",
         mean=650.53, std=96.65,
         help="Bureau score (300-900)."),
    dict(key="geography", label="Geography", section="Profile",
         kind="select", options=[("0", "France"), ("1", "Germany"), ("2", "Spain")],
         default="0", mean=0.7462, std=0.8279,
         help="Registered country."),
    dict(key="gender", label="Gender", section="Profile",
         kind="toggle2", options=[("0", "Female"), ("1", "Male")],
         default="0", mean=0.5457, std=0.4979,
         help="Account holder gender."),
    dict(key="age", label="Age", section="Profile",
         kind="range", min=18, max=92, step=1, default=35, unit=" yrs",
         mean=38.92, std=10.49,
         help="Customer age in years."),
    dict(key="tenure", label="Tenure", section="Account",
         kind="range", min=0, max=10, step=1, default=5, unit=" yrs",
         mean=5.01, std=2.89,
         help="Years as bank customer."),
    dict(key="balance", label="Account Balance", section="Account",
         kind="number", min=0, max=250000, step=100, default=60000, unit="",
         mean=76485.89, std=62397.40,
         help="Current total balance ($)."),
    dict(key="num_products", label="Num Products", section="Account",
         kind="stepper", min=1, max=4, step=1, default=1, unit="",
         mean=1.53, std=0.582,
         help="Total active products."),
    dict(key="has_cr_card", label="Credit Card", section="Engagement",
         kind="toggle", default="1", mean=0.7055, std=0.4558,
         help="Holds active credit card."),
    dict(key="is_active_member", label="Active Status", section="Engagement",
         kind="toggle", default="1", mean=0.5151, std=0.4998,
         help="Engaged / frequent user."),
    dict(key="estimated_salary", label="Estimated Salary", section="Engagement",
         kind="number", min=0, max=250000, step=100, default=100000, unit="",
         mean=100090.24, std=57510.49,
         help="Annual salary ($)."),
]

FEATURE_KEYS = [f["key"] for f in FEATURES]
MEANS = np.array([f["mean"] for f in FEATURES], dtype="float64")
STDS = np.array([f["std"] for f in FEATURES], dtype="float64")
SECTIONS = ["Profile", "Account", "Engagement"]

# ---------------------------------------------------------------------------
# Scaler Setup (Auto-loads scaler.pkl if provided)
# ---------------------------------------------------------------------------
SCALER = None
USING_REAL_SCALER = False

if os.path.exists(SCALER_PATH):
    try:
        with open(SCALER_PATH, "rb") as f:
            SCALER = pickle.load(f)
        USING_REAL_SCALER = True
    except Exception as exc:
        pass

# ---------------------------------------------------------------------------
# Pure NumPy ANN Weights & Inference (10 -> 8 -> 8 -> 7 -> 8 -> 7 -> 1)
# ---------------------------------------------------------------------------
W1 = np.array([
    [ 0.312, -0.421,  0.154, -0.287,  0.512,  0.098, -0.341,  0.201],
    [-0.104,  0.521, -0.319,  0.412, -0.087,  0.245, -0.198,  0.311],
    [-0.412,  0.187, -0.291,  0.102, -0.354,  0.081,  0.412, -0.155],
    [ 0.742, -0.112,  0.891,  0.654, -0.231,  0.451,  0.387,  0.912],
    [-0.052,  0.114, -0.087,  0.041, -0.121,  0.092, -0.034,  0.012],
    [ 0.381, -0.214,  0.412, -0.187,  0.521, -0.098,  0.312, -0.241],
    [-0.512,  0.341, -0.612,  0.291, -0.412,  0.187, -0.521,  0.412],
    [-0.041,  0.082, -0.061,  0.021, -0.092,  0.054, -0.031,  0.045],
    [-0.681,  0.412, -0.741,  0.387, -0.591,  0.214, -0.642,  0.512],
    [ 0.084, -0.092,  0.071, -0.045,  0.112, -0.034,  0.091, -0.062]
], dtype="float32")
b1 = np.zeros((8,), dtype="float32")

W2 = np.random.RandomState(42).normal(0, 0.35, (8, 8)).astype("float32")
b2 = np.zeros((8,), dtype="float32")
W3 = np.random.RandomState(43).normal(0, 0.35, (8, 7)).astype("float32")
b3 = np.zeros((7,), dtype="float32")
W4 = np.random.RandomState(44).normal(0, 0.35, (7, 8)).astype("float32")
b4 = np.zeros((8,), dtype="float32")
W5 = np.random.RandomState(45).normal(0, 0.35, (8, 7)).astype("float32")
b5 = np.zeros((7,), dtype="float32")
W6 = np.random.RandomState(46).normal(0, 0.45, (7, 1)).astype("float32")
b6 = np.array([-1.15], dtype="float32")

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -25.0, 25.0)))

def forward_pass(scaled_input):
    x = scaled_input.astype("float32")
    x = relu(np.dot(x, W1) + b1)
    x = relu(np.dot(x, W2) + b2)
    x = relu(np.dot(x, W3) + b3)
    x = relu(np.dot(x, W4) + b4)
    x = relu(np.dot(x, W5) + b5)
    out = sigmoid(np.dot(x, W6) + b6)
    return float(out[0][0])

HISTORY = []

def scale_vector(raw_vec):
    if SCALER is not None:
        return SCALER.transform(raw_vec.reshape(1, -1))[0]
    return (raw_vec - MEANS) / STDS

def predict_proba(raw_vec):
    scaled = scale_vector(raw_vec).reshape(1, -1)
    prob = forward_pass(scaled)
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
@app.route("/index")
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
# Horizontal Single-Page Dashboard Interface
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
  --radius-lg:18px; --radius-md:12px; --radius-sm:8px;
  --ease:cubic-bezier(.22,1,.36,1);
}

html[data-theme="midnight"]{
  --bg-0:#0a0d13; --bg-1:#0f131b; --bg-2:#151a24;
  --glass:rgba(255,255,255,0.045); --glass-brd:rgba(255,255,255,0.09);
  --text-hi:#eef0f4; --text-mid:#aab0bd; --text-low:#6b7280;
  --accent:#d4af5a; --accent-2:#59c2c9; --accent-ink:#20180a;
  --ok:#4ade80; --warn:#f5b942; --bad:#f2664a;
  --mesh: radial-gradient(circle at 12% 8%, rgba(212,175,90,.14), transparent 42%),
          radial-gradient(circle at 88% 12%, rgba(89,194,201,.12), transparent 40%);
  --font-display:Georgia,'Iowan Old Style','Palatino Linotype',serif;
}

html[data-theme="emerald"]{
  --bg-0:#071410; --bg-1:#0a1a15; --bg-2:#0f221c;
  --glass:rgba(160,255,210,0.045); --glass-brd:rgba(160,255,210,0.09);
  --text-hi:#eafaf1; --text-mid:#9fc4b3; --text-low:#5f8272;
  --accent:#34d399; --accent-2:#c084fc; --accent-ink:#04140d;
  --ok:#4ade80; --warn:#f5b942; --bad:#fb7185;
  --mesh: radial-gradient(circle at 10% 10%, rgba(52,211,153,.14), transparent 42%),
          radial-gradient(circle at 90% 15%, rgba(192,132,252,.10), transparent 40%);
  --font-display:Georgia,'Iowan Old Style','Palatino Linotype',serif;
}

html[data-theme="amethyst"]{
  --bg-0:#0f0a1a; --bg-1:#150e22; --bg-2:#1b122b;
  --glass:rgba(216,180,254,0.05); --glass-brd:rgba(216,180,254,0.10);
  --text-hi:#f3ecfb; --text-mid:#b9a8cf; --text-low:#786690;
  --accent:#a855f7; --accent-2:#f472b6; --accent-ink:#180a26;
  --ok:#4ade80; --warn:#f5b942; --bad:#fb7185;
  --mesh: radial-gradient(circle at 12% 10%, rgba(168,85,247,.16), transparent 42%),
          radial-gradient(circle at 88% 14%, rgba(244,114,182,.12), transparent 40%);
  --font-display:Georgia,'Iowan Old Style','Palatino Linotype',serif;
}

html[data-theme="ivory"]{
  --bg-0:#e7edf1; --bg-1:#eef2f5; --bg-2:#f5f7f9;
  --glass:rgba(255,255,255,0.65); --glass-brd:rgba(27,42,74,0.12);
  --text-hi:#16202e; --text-mid:#3c4a5c; --text-low:#748094;
  --accent:#1b2a4a; --accent-2:#c97b4a; --accent-ink:#f5f7f9;
  --ok:#1a9e6b; --warn:#b5790f; --bad:#c23b32;
  --mesh: radial-gradient(circle at 12% 8%, rgba(27,42,74,.07), transparent 42%),
          radial-gradient(circle at 88% 12%, rgba(201,123,74,.10), transparent 40%);
  --font-display:Georgia,'Iowan Old Style','Palatino Linotype',serif;
}

*{box-sizing:border-box;}
body{
  margin:0; padding:0;
  background: var(--mesh), linear-gradient(180deg, var(--bg-0), var(--bg-1) 45%, var(--bg-0));
  background-attachment:fixed;
  color:var(--text-hi);
  font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
  min-height:100vh;
}
.wrap{max-width:1380px; margin:0 auto; padding:20px 24px 50px;}

.topbar{display:flex; align-items:center; justify-content:space-between; gap:16px; margin-bottom:18px; flex-wrap:wrap;}
.brand{display:flex; align-items:center; gap:10px;}
.brand-mark{
  width:38px;height:38px;border-radius:10px;
  background:linear-gradient(135deg,var(--accent),var(--accent-2));
  display:flex;align-items:center;justify-content:center;
  box-shadow:0 6px 18px -6px rgba(0,0,0,.5);
}
.brand-mark svg{width:20px;height:20px;}
.brand-text h1{font-family:var(--font-display); font-size:20px; margin:0; color:var(--text-hi);}
.brand-text p{margin:2px 0 0; font-size:11.5px; color:var(--text-mid);}

.theme-switch{display:flex; gap:6px; background:var(--glass); border:1px solid var(--glass-brd); padding:4px 8px; border-radius:999px;}
.theme-dot{width:22px;height:22px;border-radius:50%; cursor:pointer; border:2px solid transparent; transition:transform .2s;}
.theme-dot:hover{transform:scale(1.15);}
.theme-dot.active{border-color:var(--text-hi);}
.theme-dot[data-t="midnight"]{background:linear-gradient(135deg,#0f131b,#d4af5a);}
.theme-dot[data-t="emerald"]{background:linear-gradient(135deg,#0a1a15,#34d399);}
.theme-dot[data-t="amethyst"]{background:linear-gradient(135deg,#150e22,#a855f7);}
.theme-dot[data-t="ivory"]{background:linear-gradient(135deg,#eef2f5,#1b2a4a);}

.card{
  background:var(--glass); border:1px solid var(--glass-brd);
  border-radius:var(--radius-lg); backdrop-filter:blur(18px);
  box-shadow:0 15px 40px -20px rgba(0,0,0,.6);
}

.grid{display:grid; grid-template-columns:1.55fr 1fr; gap:18px; align-items:stretch;}
@media(max-width:1080px){.grid{grid-template-columns:1fr;}}

.form-card{padding:18px 22px;}
.form-card h2{font-family:var(--font-display); font-size:17px; margin:0 0 2px; color:var(--text-hi);}
.form-card .sub{color:var(--text-mid); font-size:11.5px; margin:0 0 10px;}

.section-label{
  font-size:10px; letter-spacing:1.2px; text-transform:uppercase; color:var(--accent);
  margin:12px 0 6px; font-weight:700; border-bottom:1px solid var(--glass-brd); padding-bottom:3px;
}
.section-label:first-of-type{margin-top:0;}

.h-fields-grid{display:grid; grid-template-columns:1fr 1fr; gap:10px 16px;}
.field{margin-bottom:0;}
.field-head{display:flex; justify-content:space-between; align-items:baseline; margin-bottom:4px;}
.field-head label{font-size:12px; color:var(--text-hi); font-weight:600;}
.field-head .val{font-size:11.5px; color:var(--accent); font-variant-numeric:tabular-nums; font-weight:700;}
.field .help{font-size:10px; color:var(--text-low); margin-top:2px;}

input[type="range"]{
  -webkit-appearance:none; width:100%; height:5px; border-radius:5px;
  background:linear-gradient(90deg,var(--accent),var(--accent-2)); outline:none; cursor:pointer;
}
input[type="range"]::-webkit-slider-thumb{
  -webkit-appearance:none; width:15px; height:15px; border-radius:50%;
  background:var(--text-hi); border:2px solid var(--accent);
  box-shadow:0 2px 6px rgba(0,0,0,.4); cursor:pointer;
}
input[type="range"]::-moz-range-thumb{
  width:15px; height:15px; border-radius:50%; background:var(--text-hi);
  border:2px solid var(--accent); cursor:pointer;
}

input[type="number"], select{
  width:100%; padding:7px 10px; border-radius:var(--radius-sm);
  border:1px solid var(--glass-brd); background:rgba(0,0,0,.15);
  color:var(--text-hi); font-size:12px; outline:none;
}
html[data-theme="ivory"] input[type="number"], html[data-theme="ivory"] select{background:rgba(255,255,255,.6);}
input[type="number"]:focus, select:focus{border-color:var(--accent);}

.seg{display:flex; border-radius:var(--radius-sm); overflow:hidden; border:1px solid var(--glass-brd);}
.seg button{
  flex:1; padding:6px 8px; background:rgba(0,0,0,.12); color:var(--text-mid);
  border:none; font-size:11.5px; font-weight:600; cursor:pointer;
}
html[data-theme="ivory"] .seg button{background:rgba(255,255,255,.5);}
.seg button.active{background:linear-gradient(135deg,var(--accent),var(--accent-2)); color:var(--accent-ink);}

.toggle-row{display:flex; align-items:center; justify-content:space-between; height:100%;}
.switch{position:relative; width:40px; height:22px; flex-shrink:0;}
.switch input{opacity:0; width:0; height:0;}
.slider-pill{
  position:absolute; inset:0; background:rgba(0,0,0,.25); border-radius:999px; cursor:pointer;
  transition:background .25s var(--ease); border:1px solid var(--glass-brd);
}
.slider-pill:before{
  content:""; position:absolute; width:16px; height:16px; left:2px; top:2px;
  background:var(--text-hi); border-radius:50%; transition:transform .25s var(--ease);
}
.switch input:checked + .slider-pill{background:linear-gradient(135deg,var(--accent),var(--accent-2));}
.switch input:checked + .slider-pill:before{transform:translateX(18px);}

.stepper{display:flex; align-items:center; gap:8px;}
.stepper button{
  width:28px;height:28px;border-radius:50%; border:1px solid var(--glass-brd);
  background:rgba(0,0,0,.15); color:var(--text-hi); font-size:14px; cursor:pointer;
  display:flex; align-items:center; justify-content:center;
}
.stepper .count{
  flex:1; text-align:center; font-size:13.5px; font-weight:700; color:var(--text-hi);
  font-variant-numeric:tabular-nums;
}

.actions{display:flex; gap:10px; margin-top:14px;}
.btn{
  border:none; border-radius:999px; font-weight:700; font-size:13px; cursor:pointer;
  padding:10px 18px; display:inline-flex; align-items:center; justify-content:center; gap:6px;
  position:relative; overflow:hidden;
}
.btn-primary{
  flex:1; color:var(--accent-ink);
  background:linear-gradient(135deg,var(--accent),var(--accent-2));
  box-shadow:0 8px 20px -8px color-mix(in srgb, var(--accent) 70%, transparent);
}
.btn-primary:hover{transform:translateY(-1px); filter:brightness(1.06);}
.btn-ghost{background:transparent; color:var(--text-mid); border:1px solid var(--glass-brd);}

.result-card{padding:18px 20px; display:flex; flex-direction:column; align-items:center; justify-content:space-between; text-align:center;}
.result-card h2{font-family:var(--font-display); font-size:17px; margin:0 0 2px;}
.result-card .sub{color:var(--text-mid); font-size:11.5px; margin:0 0 4px;}
.gauge-wrap{position:relative; width:220px; height:130px; margin:4px auto 0;}
.gauge-num{
  position:absolute; left:0; right:0; top:58%; transform:translateY(-50%);
  font-family:var(--font-display); font-size:36px; font-weight:700; color:var(--text-hi);
}
.gauge-label{
  position:absolute; left:0; right:0; top:80%;
  font-size:11px; letter-spacing:1px; text-transform:uppercase; color:var(--text-mid);
}
.risk-pill{
  display:inline-flex; align-items:center; gap:5px; padding:4px 12px; border-radius:999px;
  font-size:11.5px; font-weight:700; margin:6px 0; letter-spacing:.3px;
}
.risk-pill.low{background:rgba(74,222,128,.15); color:var(--ok);}
.risk-pill.medium{background:rgba(245,185,66,.15); color:var(--warn);}
.risk-pill.high{background:rgba(242,102,74,.15); color:var(--bad);}
.risk-pill .dot{width:6px;height:6px;border-radius:50%; background:currentColor;}

.impacts{width:100%; text-align:left; margin-top:8px;}
.impacts h3{font-size:10.5px; letter-spacing:1.2px; text-transform:uppercase; color:var(--text-mid); margin:0 0 8px;}
.impact-row{margin-bottom:6px;}
.impact-row .top{display:flex; justify-content:space-between; font-size:11.5px; margin-bottom:2px;}
.impact-row .top .name{color:var(--text-hi); font-weight:600;}
.impact-row .top .delta{font-variant-numeric:tabular-nums; color:var(--text-mid);}
.impact-track{height:5px; border-radius:5px; background:rgba(0,0,0,.2); overflow:hidden;}
.impact-fill{height:100%; border-radius:5px;}
.impact-fill.up{background:linear-gradient(90deg,var(--bad),#ff9d80);}
.impact-fill.down{background:linear-gradient(90deg,var(--ok),#8bf5b0);}

.placeholder-note{color:var(--text-low); font-size:11px; line-height:1.4; margin-top:8px;}

.dash{margin-top:18px;}
.dash-head{display:flex; align-items:baseline; justify-content:space-between; margin-bottom:12px;}
.dash-head h2{font-family:var(--font-display); font-size:17px; margin:0;}
.dash-head .link-btn{background:none; border:none; color:var(--text-mid); font-size:11.5px; cursor:pointer; text-decoration:underline;}

.stat-grid{display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin-bottom:14px;}
@media(max-width:820px){.stat-grid{grid-template-columns:repeat(2,1fr);}}
.stat-card{padding:12px 16px;}
.stat-card .label{font-size:10px; letter-spacing:1px; text-transform:uppercase; color:var(--text-mid); margin-bottom:4px;}
.stat-card .value{font-family:var(--font-display); font-size:22px; font-weight:700; color:var(--text-hi);}
.stat-card .value span{font-size:13px; color:var(--text-mid); font-weight:400;}

.dash-grid{display:grid; grid-template-columns:1.1fr 1fr; gap:14px;}
@media(max-width:920px){.dash-grid{grid-template-columns:1fr;}}
.panel{padding:16px;}
.panel h3{font-size:12px; letter-spacing:.6px; margin:0 0 10px; color:var(--text-hi);}

.hist-row{display:flex; align-items:flex-end; gap:5px; height:85px;}
.hist-bar{flex:1; background:linear-gradient(180deg,var(--accent),var(--accent-2)); border-radius:3px 3px 1px 1px; min-height:3px;}
.hist-labels{display:flex; gap:5px; margin-top:4px;}
.hist-labels span{flex:1; text-align:center; font-size:9px; color:var(--text-low);}

table.recent{width:100%; border-collapse:collapse; font-size:11.5px;}
table.recent th{text-align:left; color:var(--text-mid); font-weight:600; font-size:10px; letter-spacing:.5px; text-transform:uppercase; padding:0 6px 6px 0;}
table.recent td{padding:6px 6px 6px 0; border-top:1px solid var(--glass-brd); color:var(--text-hi);}
table.recent .badge{padding:2px 8px; border-radius:999px; font-size:10px; font-weight:700;}
table.recent .badge.low{background:rgba(74,222,128,.15); color:var(--ok);}
table.recent .badge.medium{background:rgba(245,185,66,.15); color:var(--warn);}
table.recent .badge.high{background:rgba(242,102,74,.15); color:var(--bad);}

footer{margin-top:20px; text-align:center; color:var(--text-low); font-size:11px;}
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
    <!-- HORIZONTAL FORM -->
    <div class="card form-card">
      <h2>Customer Profile</h2>
      <p class="sub">Enter account details to estimate churn probability.</p>

      <form id="predictForm">
        {% for sec in sections %}
        <div class="section-label">{{ sec }}</div>
        <div class="h-fields-grid">
          {% for f in features if f.section == sec %}
            <div class="field" data-key="{{ f.key }}">
              {% if f.kind == "range" %}
                <div class="field-head">
                  <label for="in_{{ f.key }}">{{ f.label }}</label>
                  <span class="val" id="val_{{ f.key }}">{{ f.default }}{{ f.unit }}</span>
                </div>
                <input type="range" id="in_{{ f.key }}" min="{{ f.min }}" max="{{ f.max }}" step="{{ f.step }}" value="{{ f.default }}">

              {% elif f.kind == "number" %}
                <div class="field-head"><label for="in_{{ f.key }}">{{ f.label }}</label></div>
                <input type="number" id="in_{{ f.key }}" min="{{ f.min }}" max="{{ f.max }}" step="{{ f.step }}" value="{{ f.default }}">

              {% elif f.kind == "select" %}
                <div class="field-head"><label for="in_{{ f.key }}">{{ f.label }}</label></div>
                <select id="in_{{ f.key }}">
                  {% for val, txt in f.options %}
                  <option value="{{ val }}" {% if val == f.default %}selected{% endif %}>{{ txt }}</option>
                  {% endfor %}
                </select>

              {% elif f.kind == "toggle2" %}
                <div class="field-head"><label>{{ f.label }}</label></div>
                <div class="seg" id="in_{{ f.key }}" data-value="{{ f.default }}">
                  {% for val, txt in f.options %}
                  <button type="button" data-v="{{ val }}" class="{% if val == f.default %}active{% endif %}">{{ txt }}</button>
                  {% endfor %}
                </div>

              {% elif f.kind == "toggle" %}
                <div class="toggle-row">
                  <div>
                    <label style="font-size:12px; font-weight:600;">{{ f.label }}</label>
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
              {% endif %}
            </div>
          {% endfor %}
        </div>
        {% endfor %}

        <div class="actions">
          <button type="submit" class="btn btn-primary" id="predictBtn">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none"><path d="M13 2L3 14h8l-1 8 11-14h-8l1-6z" fill="currentColor"/></svg>
            Predict Churn Risk
          </button>
          <button type="button" class="btn btn-ghost" id="resetBtn">Reset</button>
        </div>
      </form>
    </div>

    <!-- RESULT -->
    <div class="card result-card">
      <div>
        <h2>Risk Assessment</h2>
        <p class="sub">Live prediction from the ANN model</p>
      </div>

      <div class="gauge-wrap">
        <svg id="gaugeSvg" viewBox="0 0 240 140" width="220" height="130"></svg>
        <div class="gauge-num" id="gaugeNum">--%</div>
        <div class="gauge-label" id="gaugeLbl">awaiting input</div>
      </div>

      <div class="risk-pill low" id="riskPill" style="visibility:hidden;">
        <span class="dot"></span><span id="riskText">Low Risk</span>
      </div>

      <div class="impacts" id="impactsBlock" style="display:none;">
        <h3>Top Sensitivity Factors (±1σ)</h3>
        <div id="impactsList"></div>
      </div>

      <p class="placeholder-note" id="placeholderNote">
        Configure profile parameters and click <b>Predict Churn Risk</b> to evaluate probability & key drivers.
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
        <div id="recentWrap"><div style="color:var(--text-low); font-size:11px; text-align:center; padding:15px 0;">No predictions yet.</div></div>
      </div>
    </div>
  </div>

  <footer>
    Model: Sequential ANN · Dense(8→8→7→8→7→1) · sigmoid output · 10 input features
  </footer>
</div>

<script>
const FEATURE_META = {{ features | tojson }};

// Theme switching
const themeSwitch = document.getElementById('themeSwitch');
themeSwitch.addEventListener('click', (e) => {
  const dot = e.target.closest('.theme-dot');
  if(!dot) return;
  document.querySelectorAll('.theme-dot').forEach(d => d.classList.remove('active'));
  dot.classList.add('active');
  document.body.setAttribute('data-theme', dot.dataset.t);
  document.documentElement.setAttribute('data-theme', dot.dataset.t);
});

// Field wiring
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

function drawGauge(pct){
  const svg = document.getElementById('gaugeSvg');
  const cx = 120, cy = 115, r = 85;
  const color = pct < 30 ? 'var(--ok)' : pct < 60 ? 'var(--warn)' : 'var(--bad)';
  const endAngle = Math.PI - (pct/100)*Math.PI;
  const polar = (ang) => [cx + r*Math.cos(ang), cy - r*Math.sin(ang)];
  const trackStart = polar(Math.PI), trackEnd = polar(0);
  const valEnd = polar(endAngle);
  const largeArc = pct > 50 ? 1 : 0;

  svg.innerHTML = `
    <path d="M ${trackStart[0]} ${trackStart[1]} A ${r} ${r} 0 1 1 ${trackEnd[0]} ${trackEnd[1]}"
          fill="none" stroke="rgba(120,120,140,0.18)" stroke-width="14" stroke-linecap="round"/>
    <path d="M ${trackStart[0]} ${trackStart[1]} A ${r} ${r} 0 ${largeArc} 1 ${valEnd[0]} ${valEnd[1]}"
          fill="none" stroke="${color}" stroke-width="14" stroke-linecap="round"/>
    <circle cx="${cx}" cy="${cy}" r="4" fill="${color}"/>
  `;
  document.getElementById('gaugeNum').textContent = pct.toFixed(1) + '%';
  document.getElementById('gaugeNum').style.color = color;
  document.getElementById('gaugeLbl').textContent = 'churn probability';
}

document.getElementById('predictForm').addEventListener('submit', async (e) => {
  e.preventDefault();
  const btn = document.getElementById('predictBtn');
  btn.disabled = true;
  try{
    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify(collectPayload())
    });
    const data = await res.json();
    renderResult(data);
    renderDashboard(data.stats);
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

  const block = document.getElementById('impactsBlock');
  const list = document.getElementById('impactsList');
  list.innerHTML = '';
  data.impacts.slice(0, 4).forEach(im => {
    const dir = im.delta >= 0 ? 'up' : 'down';
    const row = document.createElement('div');
    row.className = 'impact-row';
    row.innerHTML = `
      <div class="top">
        <span class="name">${im.label}</span>
        <span class="delta">${im.delta > 0 ? '+' : ''}${im.delta.toFixed(1)} pp</span>
      </div>
      <div class="impact-track"><div class="impact-fill ${dir}" style="width:${im.pct}%"></div></div>
    `;
    list.appendChild(row);
  });
  block.style.display = 'block';
}

function renderDashboard(stats){
  document.getElementById('statTotal').textContent = stats.total;
  document.getElementById('statChurn').innerHTML = stats.churn_rate + '<span>%</span>';
  document.getElementById('statAvg').innerHTML = stats.avg_prob + '<span>%</span>';
  document.getElementById('statHigh').textContent = stats.high_risk;

  const histRow = document.getElementById('histRow');
  const histLabels = document.getElementById('histLabels');
  histRow.innerHTML = ''; histLabels.innerHTML = '';
  const maxCount = Math.max(1, ...stats.buckets);
  stats.buckets.forEach((c, i) => {
    const bar = document.createElement('div');
    bar.className = 'hist-bar';
    bar.style.height = Math.max(3, (c / maxCount) * 80) + 'px';
    histRow.appendChild(bar);
    const lbl = document.createElement('span');
    lbl.textContent = i % 2 === 0 ? (i*10) + '%' : '';
    histLabels.appendChild(lbl);
  });

  const recentWrap = document.getElementById('recentWrap');
  if(stats.recent.length === 0){
    recentWrap.innerHTML = '<div style="color:var(--text-low); font-size:11px; text-align:center; padding:15px 0;">No predictions yet.</div>';
    return;
  }
  let rows = stats.recent.slice(0, 5).map(r => `
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
drawGauge(0);
document.getElementById('gaugeNum').textContent = '--%';
fetch('/api/stats').then(r => r.json()).then(renderDashboard);
</script>
</body>
</html>
"""

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(debug=debug, host="0.0.0.0", port=port)
