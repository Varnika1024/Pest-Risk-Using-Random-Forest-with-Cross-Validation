#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PestGuard 3D  -  Pest-Risk Prediction with Random Forest Regression
===================================================================
One file. Runs in Google Colab and in PyCharm / any local Python (3.9+).

  PyCharm / terminal :  python app.py            -> browser tab opens automatically
  Google Colab       :  upload app.py + the two CSVs, then in a cell:  %run app.py

ML lifecycle covered
  1. Data preparation    : cleaning, de-duplication, numeric coercion, median / mode imputation
  2. Feature processing  : one-hot encoding of crop label, StandardScaler (for the linear baseline)
  3. Exploratory         : PCA (3 components) + K-Means (k chosen by silhouette score)
  4. Training            : RandomForestRegressor (primary) vs Ridge linear regression (baseline)
  5. Evaluation          : 5-fold cross-validation + hold-out test, MAE / RMSE / R2
  6. Tuning              : RandomizedSearchCV over RF hyper-parameters
  7. Deployment          : Flask web panel + REST API  (/api/predict, /api/health, /api/monitor)
  8. Monitoring          : prediction log + data-drift check (Population Stability Index)
"""
import os, sys, re, json, math, time, sqlite3, secrets, subprocess, threading, webbrowser, warnings
from collections import deque

warnings.filterwarnings("ignore")


def _ensure(pip_name, import_name):
    try:
        __import__(import_name)
    except ImportError:
        print(f"[setup] installing {pip_name} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pip_name])


for _p, _i in [("flask", "flask"), ("scikit-learn", "sklearn"), ("pandas", "pandas"),
               ("numpy", "numpy"), ("joblib", "joblib")]:
    _ensure(_p, _i)

import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, session, Response
from werkzeug.security import generate_password_hash, check_password_hash
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, silhouette_score
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --------------------------------------------------------------------------- config
TARGET = "pest_pressure"
CAT = ["label"]
DEV_FILE = "model_development_data.csv"        # used to TRAIN the model
FIELD_FILE = "farmer_field_holdout_data.csv"   # shown first (hold-out, never trained on)
CHART_FEATURES = ["soil_moisture", "humidity", "fertilizer_usage"]
PORT = int(os.environ.get("PORT", 5000))
RANDOM_STATE = 42

try:
    HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:  # notebook / exec
    HERE = os.getcwd()


def in_colab():
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def find_file(name):
    search = [HERE, os.getcwd(), "/content", "/content/drive/MyDrive", "/mnt/user-data/uploads",
              os.path.expanduser("~/Downloads"), os.path.expanduser("~/Desktop")]
    for d in search:
        p = os.path.join(d, name)
        if os.path.isfile(p):
            return p
    if in_colab():
        print(f"[data] '{name}' not found - please upload it now ...")
        from google.colab import files
        up = files.upload()
        for k in up:
            if k.endswith(name) or name in k:
                return os.path.join(os.getcwd(), k)
    raise FileNotFoundError(
        f"Could not find '{name}'. Put it in the same folder as app.py (or /content in Colab).")


# --------------------------------------------------------------------------- data prep
def clean(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def sanitize(o):
    """Make numpy / NaN values JSON-safe."""
    if isinstance(o, dict):
        return {str(k): sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [sanitize(v) for v in o]
    if isinstance(o, np.ndarray):
        return sanitize(o.tolist())
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if (math.isnan(f) or math.isinf(f)) else round(f, 4)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def metrics(y, p):
    return {"MAE": float(mean_absolute_error(y, p)),
            "RMSE": float(math.sqrt(mean_squared_error(y, p))),
            "R2": float(r2_score(y, p))}


STATE = {}   # filled by train_all()


def make_pre(num_cols, cat_cols, scale):
    num_steps = [("imp", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("sc", StandardScaler()))
    transformers = [("num", Pipeline(num_steps), num_cols)]
    if cat_cols:
        transformers.append(("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols))
    return ColumnTransformer(transformers)


def prepare_input(df_raw, dedupe=True):
    """Align any uploaded dataset to the training schema."""
    df = clean(df_raw)
    if dedupe:
        df = df.drop_duplicates().reset_index(drop=True)
    num_cols, cat_cols = STATE["num_cols"], STATE["cat_cols"]
    missing = []
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            df[c] = np.nan
            missing.append(c)
    for c in cat_cols:
        if c not in df.columns:
            df[c] = "unknown"
            missing.append(c)
        df[c] = df[c].astype(str)
    return df, missing


# --------------------------------------------------------------------------- training
def train_all():
    dev_path = find_file(DEV_FILE)
    sig = f"{os.path.getsize(dev_path)}-{int(os.path.getmtime(dev_path))}-v4"
    cache = os.path.join(HERE, "pest_model_bundle.joblib")
    if os.path.isfile(cache):
        try:
            b = joblib.load(cache)
            if b.get("sig") == sig:
                STATE.update(b)
                print("[model] loaded cached model bundle")
                return
        except Exception:
            pass

    print("[data] loading", dev_path)
    dev = clean(pd.read_csv(dev_path))
    n_raw = len(dev)
    dev = dev.drop_duplicates()
    dev[TARGET] = pd.to_numeric(dev[TARGET], errors="coerce")
    dev = dev.dropna(subset=[TARGET]).reset_index(drop=True)
    cat_cols = [c for c in CAT if c in dev.columns]
    num_cols = [c for c in dev.columns if c != TARGET and c not in cat_cols
                and pd.api.types.is_numeric_dtype(dev[c])]
    for c in cat_cols:
        dev[c] = dev[c].astype(str)
    X, y = dev[num_cols + cat_cols], dev[TARGET]
    print(f"[data] {n_raw} rows -> {len(dev)} after cleaning | {len(num_cols)} numeric + {len(cat_cols)} categorical features")

    # ---- random forest + hyper-parameter search
    rf_pipe = Pipeline([("pre", make_pre(num_cols, cat_cols, scale=False)),
                        ("rf", RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1))])
    space = {"rf__n_estimators": [150, 250, 400],
             "rf__max_depth": [None, 6, 10, 16],
             "rf__min_samples_leaf": [1, 3, 6, 12],
             "rf__min_samples_split": [2, 6, 12],
             "rf__max_features": ["sqrt", 0.5, 1.0]}
    print("[train] hyper-parameter search (RandomizedSearchCV, 12 candidates x 3 folds) ...")
    t0 = time.time()
    search = RandomizedSearchCV(rf_pipe, space, n_iter=12, cv=3, scoring="neg_mean_absolute_error",
                                random_state=RANDOM_STATE, n_jobs=1, refit=True)
    search.fit(X, y)
    rf = search.best_estimator_
    best_params = {k.replace("rf__", ""): v for k, v in search.best_params_.items()}
    print(f"[train] best params {best_params}  ({time.time() - t0:.0f}s)")

    # ---- baselines
    lin = Pipeline([("pre", make_pre(num_cols, cat_cols, scale=True)), ("lr", Ridge(alpha=1.0))]).fit(X, y)
    dummy = Pipeline([("pre", make_pre(num_cols, cat_cols, scale=False)),
                      ("d", DummyRegressor(strategy="mean"))]).fit(X, y)

    # ---- 5-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {"MAE": "neg_mean_absolute_error", "RMSE": "neg_root_mean_squared_error", "R2": "r2"}

    def cv_of(est):
        r = cross_validate(est, X, y, cv=kf, scoring=scoring, n_jobs=1)
        return {"MAE": float(-r["test_MAE"].mean()), "MAE_std": float(r["test_MAE"].std()),
                "RMSE": float(-r["test_RMSE"].mean()), "RMSE_std": float(r["test_RMSE"].std()),
                "R2": float(r["test_R2"].mean()), "R2_std": float(r["test_R2"].std())}

    print("[eval] 5-fold cross-validation ...")
    cv = {"Random Forest (tuned)": cv_of(rf),
          "Linear baseline (Ridge)": cv_of(lin),
          "Mean predictor": cv_of(dummy)}

    # ---- hold-out evaluation on the farmer field data (never used for training)
    hold = {}
    try:
        fpath = find_file(FIELD_FILE)
        fd = clean(pd.read_csv(fpath))
        STATE.update(num_cols=num_cols, cat_cols=cat_cols)
        fd, _ = prepare_input(fd, dedupe=False)
        fd[TARGET] = pd.to_numeric(fd[TARGET], errors="coerce")
        fd = fd.dropna(subset=[TARGET])
        Xh, yh = fd[num_cols + cat_cols], fd[TARGET]
        hold = {"Random Forest (tuned)": metrics(yh, np.clip(rf.predict(Xh), 0, 100)),
                "Linear baseline (Ridge)": metrics(yh, np.clip(lin.predict(Xh), 0, 100)),
                "Mean predictor": metrics(yh, dummy.predict(Xh)),
                "n": int(len(fd))}
        print("[eval] hold-out:", {k: (round(v['R2'], 3) if isinstance(v, dict) else v) for k, v in hold.items()})
    except Exception as e:
        print("[eval] hold-out skipped:", e)

    # ---- feature importance (collapse one-hot columns back to their source column)
    names = rf.named_steps["pre"].get_feature_names_out()
    imp = rf.named_steps["rf"].feature_importances_
    agg = {}
    for nme, v in zip(names, imp):
        col = next((c for c in cat_cols if nme.startswith("cat__" + c)), nme.replace("num__", "", 1))
        agg[col] = agg.get(col, 0.0) + float(v)
    importance = dict(sorted(agg.items(), key=lambda kv: -kv[1]))

    # ---- exploratory analysis: PCA + K-Means
    expl = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    Z = expl.fit_transform(dev[num_cols])
    pca = PCA(n_components=3, random_state=RANDOM_STATE).fit(Z)
    P = pca.transform(Z)
    best_k, best_s = 2, -1
    for k in range(2, 7):
        km_ = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(Z)
        s = silhouette_score(Z, km_.labels_, sample_size=1000, random_state=RANDOM_STATE)
        if s > best_s:
            best_k, best_s = k, s
    km = KMeans(n_clusters=best_k, n_init=10, random_state=RANDOM_STATE).fit(Z)
    clusters = []
    for k in range(best_k):
        m = km.labels_ == k
        clusters.append({"cluster": k, "size": int(m.sum()), "mean_pest_pressure": float(y[m].mean())})
    rs = np.random.RandomState(RANDOM_STATE)
    sel = rs.choice(len(dev), size=min(500, len(dev)), replace=False)
    explore = {"k": best_k, "silhouette": float(best_s),
               "explained_variance": pca.explained_variance_ratio_.tolist(),
               "clusters": clusters,
               "points": {"x": P[sel, 0], "y": P[sel, 1], "z": P[sel, 2],
                          "cluster": km.labels_[sel], "risk": y.to_numpy()[sel]}}

    # ---- reference statistics for drift monitoring and suggestions
    quant = {}
    for c in num_cols:
        e = np.unique(np.nanquantile(dev[c], np.linspace(0, 1, 11)))
        if len(e) >= 3:
            e = e.copy(); e[0], e[-1] = -np.inf, np.inf
            h = np.histogram(dev[c].dropna(), bins=e)[0].astype(float)
            quant[c] = {"edges": e.tolist(), "exp": (h / h.sum()).tolist()}
        else:
            quant[c] = {"edges": e.tolist(), "exp": [1.0]}
    stats = {c: {"p25": float(dev[c].quantile(.25)), "p50": float(dev[c].quantile(.5)),
                 "p75": float(dev[c].quantile(.75)), "mean": float(dev[c].mean())} for c in num_cols}
    train_pred = np.clip(rf.predict(X), 0, 100)
    bundle = dict(sig=sig, model=rf, linear=lin, num_cols=num_cols, cat_cols=cat_cols, best_params=best_params,
                  cv=cv, holdout=hold, importance=importance, explore=explore, expl=expl, pca=pca, km=km,
                  quant=quant, stats=stats, risk_cuts=[float(y.quantile(1 / 3)), float(y.quantile(2 / 3))],
                  train_pred_mean=float(train_pred.mean()), n_train=int(len(dev)),
                  target_range=[float(y.min()), float(y.max())])
    STATE.update(bundle)
    try:
        joblib.dump(bundle, cache)
    except Exception:
        pass
    print("[model] ready.")


# --------------------------------------------------------------------------- monitoring
MONITOR = deque(maxlen=200)
MON_FILE = os.path.join(HERE, "monitor_log.jsonl")


def psi(ref, new_vals):
    """Population Stability Index of new data vs the training distribution (bins + proportions from training)."""
    new_vals = new_vals[~np.isnan(new_vals)]
    edges, exp = np.array(ref["edges"], dtype=float), np.array(ref["exp"], dtype=float)
    if len(new_vals) < 5 or len(exp) < 2:
        return None
    act = np.histogram(new_vals, bins=edges)[0].astype(float)
    act = (act + 0.5) / (act.sum() + 0.5 * len(act))      # Laplace smoothing: small uploads must not look like drift
    exp = np.clip(exp, 1e-4, None)
    return float(np.sum((act - exp) * np.log(act / exp)))


def drift_report(df):
    rows = []
    for c in STATE["num_cols"]:
        v = df[c].to_numpy(dtype=float)
        p = psi(STATE["quant"][c], v)
        if p is not None:
            rows.append({"feature": c, "psi": p})
    rows.sort(key=lambda r: -r["psi"])
    mx = rows[0]["psi"] if rows else 0
    status = "Stable" if mx < 0.1 else ("Watch" if mx < 0.25 else "Drift detected")
    for r in rows:
        r["flag"] = "ok" if r["psi"] < 0.1 else ("watch" if r["psi"] < 0.25 else "drift")
    return {"status": status, "max_psi": mx, "top": rows[:6]}


def log_event(kind, source, n, mean_pred, m, drift_status):
    e = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "kind": kind, "source": source, "rows": int(n),
         "mean_pred": round(float(mean_pred), 2), "MAE": None if not m else round(m["MAE"], 3),
         "R2": None if not m else round(m["R2"], 3), "drift": drift_status}
    MONITOR.append(e)
    try:
        with open(MON_FILE, "a") as f:
            f.write(json.dumps(e) + "\n")
    except Exception:
        pass


# --------------------------------------------------------------------------- suggestions
def level_of(v):
    lo, hi = STATE["risk_cuts"]
    return "Low" if v < lo else ("Moderate" if v < hi else "High")


def build_suggestions(df, preds, corr):
    lo, hi = STATE["risk_cuts"]
    mean = float(np.mean(preds))
    level = level_of(mean)
    pct_high = float(np.mean(preds >= hi) * 100)
    st = STATE["stats"]
    out = []

    if level == "High":
        out.append({"severity": "high", "title": f"Overall pest risk is HIGH ({mean:.0f}/100)",
                    "text": "Start field scouting within 24-48 hours, beginning with the highest-risk blocks listed below. "
                            "Confirm pest species and counts before acting, then use targeted Integrated Pest Management "
                            "(spot treatment, biological agents, pheromone or sticky traps) instead of blanket spraying."})
    elif level == "Moderate":
        out.append({"severity": "moderate", "title": f"Overall pest risk is MODERATE ({mean:.0f}/100)",
                    "text": "Increase scouting to about twice a week, place monitoring traps, and keep biological-control options "
                            "ready. Review irrigation and fertilizer schedules now so conditions do not tip into high risk."})
    else:
        out.append({"severity": "low", "title": f"Overall pest risk is LOW ({mean:.0f}/100)",
                    "text": "Continue routine weekly scouting and good field hygiene (remove crop residue and weeds). "
                            "Re-run this analysis whenever new field readings arrive."})

    if pct_high >= 15 and level != "High":
        out.append({"severity": "moderate", "title": f"Pockets of high risk: {pct_high:.0f}% of samples",
                    "text": "The average looks acceptable, but a share of samples are predicted high risk. Inspect those areas individually."})
    elif level == "High":
        out.append({"severity": "high", "title": f"{pct_high:.0f}% of samples are predicted high risk",
                    "text": "Prioritise treatment and monitoring by sample rather than treating the whole field uniformly."})

    def has(c):
        return c in df.columns and df[c].notna().any()

    if has("humidity"):
        m, p75 = float(df["humidity"].mean()), st["humidity"]["p75"]
        if m > p75:
            out.append({"severity": "moderate", "title": f"Humidity is above typical ({m:.0f}% vs {st['humidity']['p50']:.0f}% median)",
                        "text": "Improve airflow and plant spacing, avoid late-day overhead irrigation, and watch for humidity-loving pests and fungal disease."})
    if has("fertilizer_usage"):
        m, p75 = float(df["fertilizer_usage"].mean()), st["fertilizer_usage"]["p75"]
        if m > p75:
            out.append({"severity": "moderate", "title": f"Fertilizer usage is high ({m:.0f} vs {st['fertilizer_usage']['p50']:.0f} median)",
                        "text": "Heavy nitrogen can produce soft, lush growth that many pests prefer. Consider a soil test and split applications."})
    if has("soil_moisture"):
        m = float(df["soil_moisture"].mean())
        if m > st["soil_moisture"]["p75"]:
            out.append({"severity": "info", "title": f"Soil moisture is on the wet side ({m:.1f})",
                        "text": "Check drainage and avoid waterlogging, which stresses roots and favours soil-borne pests."})
        elif m < st["soil_moisture"]["p25"]:
            out.append({"severity": "info", "title": f"Soil moisture is on the dry side ({m:.1f})",
                        "text": "Water-stressed crops are more vulnerable. Adjust irrigation frequency."})

    top = list(STATE["importance"].items())[:3]
    out.append({"severity": "info", "title": "What the model relies on most",
                "text": "Strongest drivers in the trained Random Forest: " + ", ".join(f"{k} ({v * 100:.0f}%)" for k, v in top) +
                        ". Prioritise accurate measurement of these inputs."})

    strong = [f"{k} ({v:+.2f})" for k, v in corr.items() if v is not None and abs(v) >= 0.3]
    if strong:
        out.append({"severity": "info", "title": "Relationships in your data",
                    "text": "Features moving together with predicted risk in this dataset: " + ", ".join(strong) + "."})

    worst = np.argsort(-preds)[:5]
    out.append({"severity": "info", "title": "Inspect these samples first",
                "text": "Highest predicted risk: " + ", ".join(f"#{int(i)} ({preds[i]:.0f})" for i in worst) + "."})
    out.append({"severity": "note", "title": "Decision support only",
                "text": "These suggestions come from a statistical model and general IPM practice. Confirm with field scouting and your local agricultural extension officer before applying any pesticide."})
    return level, mean, pct_high, out


# --------------------------------------------------------------------------- analysis
def analyze(df_raw, source_name, n_show):
    df, missing = prepare_input(df_raw)
    if len(df) < 2:
        raise ValueError("The dataset needs at least 2 rows.")
    total_feats = len(STATE["num_cols"]) + len(STATE["cat_cols"])
    if len(missing) > total_feats * 0.6:
        raise ValueError("Too few matching columns. Expected columns like: " +
                         ", ".join(STATE["num_cols"][:10]) + " ...")
    X = df[STATE["num_cols"] + STATE["cat_cols"]]
    preds = np.clip(STATE["model"].predict(X), 0, 100)
    n_show = int(max(10, min(n_show, 300, len(df))))

    actual = None
    m = None
    if TARGET in df.columns:
        a = pd.to_numeric(df[TARGET], errors="coerce").to_numpy()
        ok = ~np.isnan(a)
        if ok.sum() >= 2:
            actual = a
            m = metrics(a[ok], preds[ok])

    def norm(col):
        if col not in df.columns or df[col].isna().all():
            return None
        v = pd.to_numeric(df[col], errors="coerce").to_numpy(float)
        lo, hi = np.nanmin(v), np.nanmax(v)
        return (v - lo) / (hi - lo) * 100 if hi > lo else np.full_like(v, 50.0)

    series = {c: (None if norm(c) is None else norm(c)[:n_show]) for c in CHART_FEATURES}
    chart = {"n": n_show, "series": series, "pest_predicted": preds[:n_show],
             "pest_actual": None if actual is None else actual[:n_show]}

    corr = {}
    for c in CHART_FEATURES:
        if c in df.columns and df[c].notna().sum() > 3 and np.std(preds) > 1e-9:
            corr[c] = float(np.corrcoef(df[c].fillna(df[c].mean()), preds)[0, 1])
        else:
            corr[c] = None

    level, mean, pct_high, suggestions = build_suggestions(df, preds, corr)
    lo, hi = STATE["risk_cuts"]
    counts = {"Low": int((preds < lo).sum()), "Moderate": int(((preds >= lo) & (preds < hi)).sum()),
              "High": int((preds >= hi).sum())}

    Z = STATE["expl"].transform(df[STATE["num_cols"]])
    P = STATE["pca"].transform(Z)
    cl = STATE["km"].predict(Z)
    cap = np.arange(len(df))[:400]
    proj = {"x": P[cap, 0], "y": P[cap, 1], "z": P[cap, 2], "cluster": cl[cap], "risk": preds[cap]}

    scatter = None
    if actual is not None:
        ok = np.where(~np.isnan(actual))[0][:400]
        scatter = {"actual": actual[ok], "pred": preds[ok]}

    drift = drift_report(df)
    log_event("dashboard", source_name, len(df), preds.mean(), m, drift["status"])

    ho = STATE["holdout"].get("Random Forest (tuned)", {}).get("R2")
    cvr2 = STATE["cv"]["Random Forest (tuned)"]["R2"]
    skill = {"cv_r2": cvr2, "holdout_r2": ho, "weak": bool(cvr2 < 0.1 and (ho is None or ho < 0.1))}
    return {"source": source_name, "rows": int(len(df)), "missing_columns": missing, "skill": skill,
            "mean_risk": mean, "level": level, "pct_high": pct_high, "counts": counts, "cuts": [lo, hi],
            "metrics": m, "chart": chart, "terrain": preds[:100], "projection": proj, "scatter": scatter,
            "suggestions": suggestions, "drift": drift,
            "model": {"best_params": STATE["best_params"], "cv": STATE["cv"], "holdout": STATE["holdout"],
                      "importance": dict(list(STATE["importance"].items())[:10]), "n_train": STATE["n_train"],
                      "explore": {k: v for k, v in STATE["explore"].items() if k != "points"},
                      "points": STATE["explore"]["points"]}}


# --------------------------------------------------------------------------- web app
app = Flask(__name__)
app.secret_key = secrets.token_hex(32)
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024
DB = os.path.join(HERE, "users.db")


def db():
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row
    return c


def init_db():
    with db() as c:
        c.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL,"
                  " phone TEXT UNIQUE NOT NULL, pw TEXT NOT NULL, created TEXT)")


def jres(obj, code=200):
    return Response(json.dumps(sanitize(obj)), status=code, mimetype="application/json")


def login_required(fn):
    from functools import wraps

    @wraps(fn)
    def w(*a, **k):
        if "uid" not in session:
            return jres({"error": "Please log in first."}, 401)
        return fn(*a, **k)
    return w


@app.get("/")
def index():
    return Response(HTML, mimetype="text/html")


@app.post("/api/signup")
def signup():
    d = request.get_json(force=True, silent=True) or {}
    name, phone, pw = str(d.get("name", "")).strip(), re.sub(r"[\s\-]", "", str(d.get("phone", ""))), str(d.get("password", ""))
    if len(name) < 2:
        return jres({"error": "Please enter your name."}, 400)
    if not re.fullmatch(r"\+?\d{10,13}", phone):
        return jres({"error": "Enter a valid phone number (10-13 digits)."}, 400)
    if len(pw) < 6:
        return jres({"error": "Password must be at least 6 characters."}, 400)
    try:
        with db() as c:
            cur = c.execute("INSERT INTO users(name,phone,pw,created) VALUES(?,?,?,?)",
                            (name, phone, generate_password_hash(pw), time.strftime("%Y-%m-%d %H:%M:%S")))
            session["uid"], session["name"] = cur.lastrowid, name
    except sqlite3.IntegrityError:
        return jres({"error": "This phone number is already registered. Please log in."}, 409)
    return jres({"ok": True, "name": name})


@app.post("/api/login")
def login():
    d = request.get_json(force=True, silent=True) or {}
    phone, pw = re.sub(r"[\s\-]", "", str(d.get("phone", ""))), str(d.get("password", ""))
    with db() as c:
        u = c.execute("SELECT * FROM users WHERE phone=?", (phone,)).fetchone()
    if not u or not check_password_hash(u["pw"], pw):
        return jres({"error": "Wrong phone number or password."}, 401)
    session["uid"], session["name"] = u["id"], u["name"]
    return jres({"ok": True, "name": u["name"]})


@app.post("/api/logout")
def logout():
    session.clear()
    return jres({"ok": True})


@app.get("/api/me")
def me():
    return jres({"logged_in": "uid" in session, "name": session.get("name")})


@app.get("/api/health")
def health():
    return jres({"status": "ok", "model": "RandomForestRegressor", "target": TARGET,
                 "n_train": STATE.get("n_train"), "features": STATE.get("num_cols", []) + STATE.get("cat_cols", [])})


@app.post("/api/analyze")
@login_required
def api_analyze():
    try:
        n = int(request.form.get("n", 50))
    except ValueError:
        n = 50
    source = request.form.get("source", "farmer")
    try:
        if source == "upload":
            f = request.files.get("file")
            if not f or not f.filename:
                return jres({"error": "Choose a CSV file to upload."}, 400)
            df = pd.read_csv(f)
            name = os.path.basename(f.filename)
        else:
            df = pd.read_csv(find_file(FIELD_FILE))
            name = FIELD_FILE
        return jres(analyze(df, name, n))
    except pd.errors.ParserError:
        return jres({"error": "That file could not be read as CSV."}, 400)
    except Exception as e:
        return jres({"error": str(e)}, 400)


@app.post("/api/predict")
def api_predict():
    """REST prediction: POST {"records":[{...features...}]} or a single feature object."""
    d = request.get_json(force=True, silent=True)
    if d is None:
        return jres({"error": "Send JSON."}, 400)
    recs = d.get("records") if isinstance(d, dict) and "records" in d else (d if isinstance(d, list) else [d])
    try:
        df, missing = prepare_input(pd.DataFrame(recs), dedupe=False)
        p = np.clip(STATE["model"].predict(df[STATE["num_cols"] + STATE["cat_cols"]]), 0, 100)
    except Exception as e:
        return jres({"error": str(e)}, 400)
    log_event("api", "REST", len(df), p.mean(), None, "n/a")
    return jres({"predictions": [{"pest_risk": float(v), "level": level_of(v)} for v in p],
                 "missing_features_filled": missing})


@app.get("/api/monitor")
@login_required
def api_monitor():
    items = list(MONITOR)[-20:][::-1]
    return jres({"events": items, "total": len(MONITOR)})


# --------------------------------------------------------------------------- front end
HTML = r"""<!DOCTYPE html>
<html lang="en" data-theme="light">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PestGuard 3D</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.27.0/plotly.min.js"></script>
<style>
:root[data-theme="light"]{--bg:#eef3ea;--card:rgba(255,255,255,.82);--text:#16210f;--muted:#5b6b52;--line:#d3ddcb;--accent:#2f7d32;--soft:#e3ecdc}
:root[data-theme="dark"]{--bg:#0a110e;--card:rgba(18,28,23,.82);--text:#e8f1e4;--muted:#93a58b;--line:#25352d;--accent:#7ddf64;--soft:#16231d}
*{box-sizing:border-box}
html,body{margin:0}
body{background:var(--bg);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;min-height:100vh;transition:background .4s,color .4s}
#bg{position:fixed;inset:0;width:100%;height:100%;z-index:0;pointer-events:none}
.wrap{position:relative;z-index:1;max-width:1240px;margin:0 auto;padding:18px 18px 60px}
header{display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap;margin-bottom:18px}
.brand{font-weight:800;font-size:1.5rem;letter-spacing:-.02em}
.brand span{color:var(--accent)}
.hright{display:flex;align-items:center;gap:18px;flex-wrap:wrap}
.who{color:var(--muted);font-size:.95rem}
.themebox{display:flex;align-items:center;gap:12px;font-size:.85rem;color:var(--muted)}
.card{background:var(--card);border:1px solid var(--line);border-radius:18px;padding:18px;backdrop-filter:blur(10px);-webkit-backdrop-filter:blur(10px);margin-bottom:16px}
h2{margin:0 0 4px;font-size:1.15rem}
.sub{color:var(--muted);font-size:.9rem;margin:0 0 12px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media(max-width:900px){.grid2{grid-template-columns:1fr}}
.hidden{display:none!important}

/* ---- button template (Uiverse by felipesntr) ---- */
button.btn{
  font-size: 1rem;
  padding: 0.8em 2em;
  background-color: #000;
  border: 3px solid yellow;
  border-radius: 1em;
  color: #fff;
  font-weight: bolder;
  transition: cubic-bezier(0.68, -0.55, 0.265, 1.55) 0.4s;
  box-shadow: -5px 5px 0px 0px yellow;
  cursor:pointer;
}
button.btn:hover{transform: translate(5px, -5px);}
button.btn.sm{font-size:.85rem;padding:.55em 1.2em}
button.btn.ghost{opacity:.85}
button.btn:disabled{opacity:.5;cursor:wait}

/* ---- toggle switch template (Uiverse by EddyBel) ---- */
.switch {
  font-size: 17px;
  position: relative;
  display: inline-block;
  width: 3.5em;
  height: 2em;
  transform-style: preserve-3d;
  perspective: 500px;
  animation: toggle__animation 3s infinite;
}
.switch::before {
  content: "";
  position: absolute;
  width: 100%;
  height: 100%;
  left: 0;
  top: 0;
  filter: blur(20px);
  z-index: -1;
  border-radius: 50px;
  background-color: #d8ff99;
  background-image: radial-gradient(at 21% 46%, hsla(183,65%,60%,1) 0px, transparent 50%),
radial-gradient(at 23% 25%, hsla(359,74%,70%,1) 0px, transparent 50%),
radial-gradient(at 20% 1%, hsla(267,83%,75%,1) 0px, transparent 50%),
radial-gradient(at 86% 87%, hsla(204,69%,68%,1) 0px, transparent 50%),
radial-gradient(at 99% 41%, hsla(171,72%,77%,1) 0px, transparent 50%),
radial-gradient(at 55% 24%, hsla(138,60%,62%,1) 0px, transparent 50%);
}
.switch input {opacity: 0;width: 0;height: 0;}
.slider {
  position: absolute;
  cursor: pointer;
  top: 0;left: 0;right: 0;bottom: 0;
  background-color: #fdfefedc;
  transition: .4s;
  border-radius: 30px;
}
.slider:before {
  position: absolute;
  content: "";
  height: 1.4em;
  width: 1.4em;
  left: 0.3em;
  bottom: 0.35em;
  transition: .4s;
  border-radius: 50%;
  box-shadow: rgba(0, 0, 0, 0.17) 0px -10px 10px 0px inset,
     rgba(0, 0, 0, 0.09) 0px -1px 15px -8px;
  background-color: #ff99fd;
  background-image: radial-gradient(at 81% 39%, hsla(327,79%,79%,1) 0px, transparent 50%),
  radial-gradient(at 11% 72%, hsla(264,64%,79%,1) 0px, transparent 50%),
  radial-gradient(at 23% 20%, hsla(75,98%,71%,1) 0px, transparent 50%);
}
.input__check:checked + .slider {background-color: #17202A;}
.input__check:checked + .slider:before {transform: translateX(1.5em);}
@keyframes toggle__animation {
  0%, 100% {transform: translateY(-10px) rotateX(15deg) rotateY(-20deg);}
  50% {transform: translateY(0px) rotateX(15deg) rotateY(-20deg);}
}

/* ---- input template (Uiverse by satyamchaudharydev) ---- */
.form button {border: none;background: none;color: #8b8ba7;cursor:pointer}
.form {
  --timing: 0.3s;
  --width-of-input: 200px;
  --height-of-input: 40px;
  --border-height: 2px;
  --input-bg: #fff;
  --border-color: #2f2ee9;
  --border-radius: 30px;
  --after-border-radius: 1px;
  position: relative;
  width: var(--width-of-input);
  height: var(--height-of-input);
  display: flex;
  align-items: center;
  padding-inline: 0.8em;
  border-radius: var(--border-radius);
  transition: border-radius 0.5s ease;
  background: var(--input-bg,#fff);
}
.input {
  font-size: 0.9rem;
  background-color: transparent;
  width: 100%;
  height: 100%;
  padding-inline: 0.5em;
  padding-block: 0.7em;
  border: none;
  color:#111;
}
.form:before {
  content: "";
  position: absolute;
  background: var(--border-color);
  transform: scaleX(0);
  transform-origin: center;
  width: 100%;
  height: var(--border-height);
  left: 0;
  bottom: 0;
  border-radius: 1px;
  transition: transform var(--timing) ease;
}
.form:focus-within {border-radius: var(--after-border-radius);}
input:focus {outline: none;}
.form:focus-within:before {transform: scale(1);}
.reset {border: none;background: none;opacity: 0;visibility: hidden;}
input:not(:placeholder-shown) ~ .reset {opacity: 1;visibility: visible;}
.form svg {width: 17px;margin-top: 3px;}
.form{border:1px solid var(--line)}

/* ---- auth ---- */
.authwrap{display:flex;justify-content:center;padding-top:6vh}
.auth{width:100%;max-width:430px}
.auth .form{--width-of-input:100%;margin-bottom:14px}
.tabs{display:flex;gap:8px;margin-bottom:18px}
.tab{flex:1;text-align:center;padding:.6em;border-radius:12px;cursor:pointer;border:1px solid var(--line);color:var(--muted);font-weight:600}
.tab.on{background:var(--accent);color:#fff;border-color:var(--accent)}
.msg{min-height:1.3em;margin:6px 0 12px;font-size:.9rem;color:#d33}
.hero h1{font-size:2.1rem;margin:0 0 6px;letter-spacing:-.03em;text-align:center}
.hero p{color:var(--muted);text-align:center;margin:0 0 18px}

/* ---- dashboard ---- */
.controls{display:flex;gap:14px;align-items:center;flex-wrap:wrap}
.controls .form{--width-of-input:150px}
.fname{color:var(--muted);font-size:.9rem}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-bottom:16px}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:14px;backdrop-filter:blur(10px)}
.kpi .l{color:var(--muted);font-size:.78rem;text-transform:uppercase;letter-spacing:.05em}
.kpi .v{font-size:1.7rem;font-weight:800;margin-top:4px}
.pill{display:inline-block;padding:.15em .7em;border-radius:99px;font-weight:700;font-size:1rem}
.Low{background:#2e7d3233;color:#2e9c3a}.Moderate{background:#f9a82533;color:#d98a00}.High{background:#e5393533;color:#e53935}
.plot{width:100%;height:420px}
.plot.sm{height:340px}
#terrain{width:100%;height:360px;border-radius:14px;overflow:hidden;position:relative;background:linear-gradient(180deg,var(--soft),transparent)}
#tip{position:absolute;pointer-events:none;background:#000c;color:#fff;font-size:.8rem;padding:4px 8px;border-radius:8px;display:none}
table{width:100%;border-collapse:collapse;font-size:.88rem}
th,td{text-align:left;padding:7px 8px;border-bottom:1px solid var(--line)}
th{color:var(--muted);font-weight:600}
.sug{display:flex;flex-direction:column;gap:10px}
.s{border-left:5px solid var(--line);background:var(--soft);padding:10px 14px;border-radius:10px}
.s b{display:block;margin-bottom:2px}
.s.high{border-color:#e53935}.s.moderate{border-color:#f9a825}.s.low{border-color:#2e9c3a}.s.info{border-color:#3b82f6}.s.note{border-color:var(--muted);opacity:.85}
.warn{background:#f9a82522;border:1px solid #f9a82566;padding:10px 14px;border-radius:12px;margin-bottom:16px;font-size:.9rem}
.spin{display:inline-block;width:14px;height:14px;border:3px solid #fff5;border-top-color:#fff;border-radius:50%;animation:sp 1s linear infinite;vertical-align:-2px;margin-right:8px}
@keyframes sp{to{transform:rotate(360deg)}}
.tag{font-size:.75rem;padding:.1em .6em;border-radius:99px;border:1px solid var(--line);color:var(--muted);margin-left:6px}
.ok{color:#2e9c3a}.watch{color:#d98a00}.drift{color:#e53935}
</style>
</head>
<body>
<canvas id="bg"></canvas>
<div class="wrap">
  <header>
    <div class="brand">Pest<span>Guard</span> 3D</div>
    <div class="hright">
      <span class="who" id="who"></span>
      <div class="themebox">
        <span>Light</span>
        <label class="switch">
          <input type="checkbox" class="input__check" id="themeToggle">
          <span class="slider"></span>
        </label>
        <span>Dark</span>
      </div>
      <button class="btn sm hidden" id="logoutBtn">Log out</button>
    </div>
  </header>

  <!-- ===== AUTH ===== -->
  <section id="authView" class="hidden">
    <div class="authwrap"><div class="card auth">
      <div class="hero"><h1>Pest-risk intelligence</h1><p>Random Forest regression for smarter crop protection</p></div>
      <div class="tabs"><div class="tab on" id="tabLogin">Log in</div><div class="tab" id="tabSignup">Sign up</div></div>

      <div id="nameRow" class="hidden">
        <form class="form" onsubmit="return false">
          <button type="button" tabindex="-1"><svg width="17" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="8" r="4"/><path d="M4 21c0-4 4-6 8-6s8 2 8 6"/></svg></button>
          <input class="input" id="inName" placeholder="Full name" type="text" autocomplete="name">
          <button class="reset" type="reset"><svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M6 18L18 6M6 6l12 12"></path></svg></button>
        </form>
      </div>
      <form class="form" onsubmit="return false">
        <button type="button" tabindex="-1"><svg width="17" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 16.9v3a2 2 0 0 1-2.2 2 19.8 19.8 0 0 1-8.6-3.1 19.5 19.5 0 0 1-6-6A19.8 19.8 0 0 1 2.1 4.2 2 2 0 0 1 4.1 2h3a2 2 0 0 1 2 1.7c.1 1 .4 1.9.7 2.8a2 2 0 0 1-.5 2.1L8.1 9.9a16 16 0 0 0 6 6l1.3-1.3a2 2 0 0 1 2.1-.4c.9.3 1.8.6 2.8.7a2 2 0 0 1 1.7 2z"/></svg></button>
        <input class="input" id="inPhone" placeholder="Phone number" type="tel" autocomplete="tel">
        <button class="reset" type="reset"><svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M6 18L18 6M6 6l12 12"></path></svg></button>
      </form>
      <form class="form" onsubmit="return false">
        <button type="button" tabindex="-1"><svg width="17" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="11" width="16" height="10" rx="2"/><path d="M8 11V7a4 4 0 0 1 8 0v4"/></svg></button>
        <input class="input" id="inPass" placeholder="Password (min 6 characters)" type="password" autocomplete="current-password">
        <button class="reset" type="reset"><svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M6 18L18 6M6 6l12 12"></path></svg></button>
      </form>
      <div class="msg" id="authMsg"></div>
      <button class="btn" id="authBtn" style="width:calc(100% - 5px)">Log in</button>
    </div></div>
  </section>

  <!-- ===== DASHBOARD ===== -->
  <section id="dashView" class="hidden">
    <div class="card">
      <h2>Dataset</h2>
      <p class="sub">The model is trained on <i>model_development_data.csv</i>. Start with the farmer field hold-out data, or upload your own CSV with the same kind of columns.</p>
      <div class="controls">
        <button class="btn" id="btnField">Farmer field data</button>
        <button class="btn" id="btnUpload">Upload dataset</button>
        <input type="file" id="fileIn" accept=".csv,text/csv" class="hidden">
        <form class="form" onsubmit="return false" title="Samples shown in chart">
          <button type="button" tabindex="-1"><svg width="17" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M4 20V10M10 20V4M16 20v-8M22 20H2"/></svg></button>
          <input class="input" id="nShow" placeholder="Samples" type="number" min="10" max="300" value="50">
          <button class="reset" type="reset"><svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M6 18L18 6M6 6l12 12"></path></svg></button>
        </form>
        <span class="fname" id="fname"></span>
      </div>
      <div class="msg" id="dashMsg" style="margin-bottom:0"></div>
    </div>

    <div id="results" class="hidden">
      <div id="skillWarn" class="warn hidden"></div>
      <div id="missingWarn" class="warn hidden"></div>
      <div class="kpis" id="kpis"></div>

      <div class="card">
        <h2>Field features comparison across samples</h2>
        <p class="sub">Soil moisture, humidity and fertilizer usage are the dataset values (scaled 0-100). The red line is the recorded pest pressure; if the file has no pest_pressure column it shows the model's prediction. The dotted line is always the model's prediction.</p>
        <div id="chMain" class="plot"></div>
      </div>

      <div class="grid2">
        <div class="card">
          <h2>3D risk terrain</h2>
          <p class="sub">Each column is a sample; height and colour show predicted risk. Drag to rotate, scroll to zoom.</p>
          <div id="terrain"><div id="tip"></div></div>
        </div>
        <div class="card">
          <h2>3D PCA &amp; K-Means map</h2>
          <p class="sub">Training data coloured by cluster (small dots); your samples as diamonds coloured by predicted risk.</p>
          <div id="chPca" class="plot sm"></div>
        </div>
      </div>

      <div class="card">
        <h2>Suggestions <span class="tag" id="lvlTag"></span></h2>
        <p class="sub">Recommended actions based on the predicted risk.</p>
        <div class="sug" id="sugs"></div>
      </div>

      <div class="grid2">
        <div class="card"><h2>Feature importance</h2><p class="sub">What drives the Random Forest's predictions.</p><div id="chImp" class="plot sm"></div></div>
        <div class="card" id="scCard"><h2>Predicted vs actual</h2><p class="sub">Closer to the diagonal is better.</p><div id="chSc" class="plot sm"></div></div>
      </div>

      <div class="card">
        <h2>Model report</h2>
        <p class="sub" id="modelSub"></p>
        <div style="overflow-x:auto"><table id="modelTbl"></table></div>
        <p class="sub" style="margin-top:12px" id="explSub"></p>
      </div>

      <div class="card">
        <h2>Monitoring &amp; data drift <span class="tag" id="driftTag"></span></h2>
        <p class="sub">Population Stability Index (PSI) against the training data: below 0.1 stable, 0.1-0.25 watch, above 0.25 drift.</p>
        <div style="overflow-x:auto"><table id="driftTbl"></table></div>
        <h2 style="margin-top:16px">Recent prediction log</h2>
        <div style="overflow-x:auto"><table id="logTbl"></table></div>
      </div>
    </div>
  </section>
</div>

<script>
const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let LAST = null, mode = 'login';

/* ---------- theme ---------- */
function setTheme(t){
  document.documentElement.setAttribute('data-theme', t);
  $('themeToggle').checked = (t === 'dark');
  try{ localStorage.setItem('pg-theme', t); }catch(e){}
  updateBg();
  if(LAST) renderAll(LAST);
}
$('themeToggle').addEventListener('change', e => setTheme(e.target.checked ? 'dark' : 'light'));
(function(){
  let t = null; try{ t = localStorage.getItem('pg-theme'); }catch(e){}
  if(!t) t = window.matchMedia && matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  document.documentElement.setAttribute('data-theme', t); $('themeToggle').checked = (t === 'dark');
})();
const dark = () => document.documentElement.getAttribute('data-theme') === 'dark';
const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();

/* ---------- animated 3D background ---------- */
let bgScene, bgCam, bgRen, bgPts, bgGrid;
function initBg(){
  const cv = $('bg');
  bgRen = new THREE.WebGLRenderer({canvas: cv, alpha: true, antialias: true});
  bgRen.setPixelRatio(Math.min(devicePixelRatio, 2));
  bgScene = new THREE.Scene();
  bgCam = new THREE.PerspectiveCamera(55, 1, .1, 200); bgCam.position.set(0, 6, 18);
  const N = 700, pos = new Float32Array(N * 3);
  for(let i = 0; i < N; i++){ pos[i*3] = (Math.random()-.5)*60; pos[i*3+1] = Math.random()*24-4; pos[i*3+2] = (Math.random()-.5)*50 - 5; }
  const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  bgPts = new THREE.Points(g, new THREE.PointsMaterial({size: .14, transparent: true, opacity: .8}));
  bgScene.add(bgPts);
  bgGrid = new THREE.GridHelper(80, 40); bgGrid.position.y = -5; bgGrid.material.transparent = true; bgGrid.material.opacity = .25;
  bgScene.add(bgGrid);
  updateBg(); resizeBg(); addEventListener('resize', resizeBg);
  let t = 0;
  (function loop(){ t += .0025; bgPts.rotation.y = t; bgGrid.rotation.y = Math.sin(t) * .15;
    bgCam.position.x = Math.sin(t * 2) * 2; bgCam.lookAt(0, 2, 0); bgRen.render(bgScene, bgCam); requestAnimationFrame(loop); })();
}
function updateBg(){ if(!bgPts) return; const c = new THREE.Color(css('--accent')); bgPts.material.color = c; bgGrid.material.color = c; }
function resizeBg(){ bgRen.setSize(innerWidth, innerHeight, false); bgCam.aspect = innerWidth / innerHeight; bgCam.updateProjectionMatrix(); }

/* ---------- views / auth ---------- */
function show(view){ $('authView').classList.toggle('hidden', view !== 'auth'); $('dashView').classList.toggle('hidden', view !== 'dash');
  $('logoutBtn').classList.toggle('hidden', view !== 'dash'); }
function setMode(m){
  mode = m; $('tabLogin').classList.toggle('on', m === 'login'); $('tabSignup').classList.toggle('on', m === 'signup');
  $('nameRow').classList.toggle('hidden', m === 'login'); $('authBtn').textContent = m === 'login' ? 'Log in' : 'Create account';
  $('inPass').autocomplete = m === 'login' ? 'current-password' : 'new-password'; $('authMsg').textContent = '';
}
$('tabLogin').onclick = () => setMode('login'); $('tabSignup').onclick = () => setMode('signup');
async function api(url, opts){
  const r = await fetch(url, opts); let j = {}; try{ j = await r.json(); }catch(e){}
  if(!r.ok) throw new Error(j.error || ('Request failed (' + r.status + ')'));
  return j;
}
async function submitAuth(){
  $('authMsg').textContent = ''; $('authBtn').disabled = true;
  try{
    const body = {phone: $('inPhone').value, password: $('inPass').value};
    if(mode === 'signup') body.name = $('inName').value;
    const j = await api(mode === 'login' ? '/api/login' : '/api/signup', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
    enterDash(j.name);
  }catch(e){ $('authMsg').textContent = e.message; }
  $('authBtn').disabled = false;
}
$('authBtn').onclick = submitAuth;
['inName','inPhone','inPass'].forEach(i => $(i).addEventListener('keydown', e => { if(e.key === 'Enter') submitAuth(); }));
$('logoutBtn').onclick = async () => { await api('/api/logout', {method: 'POST'}); LAST = null; $('results').classList.add('hidden'); $('who').textContent = ''; show('auth'); };

function enterDash(name){ $('who').textContent = 'Hello, ' + name; show('dash'); runAnalysis('farmer'); }

/* ---------- analysis ---------- */
$('btnField').onclick = () => runAnalysis('farmer');
$('btnUpload').onclick = () => $('fileIn').click();
$('fileIn').onchange = () => { if($('fileIn').files.length) runAnalysis('upload'); };
async function runAnalysis(source){
  const fd = new FormData(); fd.append('source', source); fd.append('n', $('nShow').value || 50);
  if(source === 'upload') fd.append('file', $('fileIn').files[0]);
  const bf = $('btnField'), bu = $('btnUpload'); const t0 = bf.textContent, t1 = bu.textContent;
  bf.disabled = bu.disabled = true; (source === 'farmer' ? bf : bu).innerHTML = '<span class="spin"></span>Predicting';
  $('dashMsg').textContent = '';
  try{
    const j = await api('/api/analyze', {method: 'POST', body: fd});
    j.log = (await api('/api/monitor')).events;
    LAST = j; $('fname').textContent = 'Showing: ' + j.source + ' (' + j.rows + ' rows)';
    $('results').classList.remove('hidden'); renderAll(j);
  }catch(e){
    if(/log in/i.test(e.message)){ show('auth'); } else { $('dashMsg').textContent = e.message; }
  }
  bf.disabled = bu.disabled = false; bf.textContent = t0; bu.textContent = t1;
  if(source === 'upload') $('fileIn').value = '';
}

/* ---------- rendering ---------- */
function plotLayout(extra){
  return Object.assign({paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
    font: {color: css('--text'), family: 'system-ui, sans-serif'}, margin: {l: 55, r: 15, t: 10, b: 45},
    xaxis: {gridcolor: css('--line'), zerolinecolor: css('--line')}, yaxis: {gridcolor: css('--line'), zerolinecolor: css('--line')}}, extra || {});
}
const CFG = {responsive: true, displaylogo: false};

function renderAll(j){
  renderKpis(j); renderMain(j); renderTerrain(j.terrain); renderPca(j); renderSug(j); renderImp(j); renderScatter(j); renderModel(j); renderDrift(j);
  const sw = $('skillWarn');
  if(j.skill && j.skill.weak){ sw.classList.remove('hidden');
    sw.innerHTML = '<b>Low model skill on this dataset.</b> Cross-validated R\u00b2 is ' + j.skill.cv_r2.toFixed(2) + (j.skill.holdout_r2 != null ? ' and hold-out R\u00b2 is ' + j.skill.holdout_r2.toFixed(2) : '') +
      ', about the same as always predicting the average. In these files pest pressure does not vary with the other columns, so predictions stay near the mean and the risk level and suggestions below are indicative only. A dataset where pest pressure genuinely depends on field conditions will give a much stronger model.'; }
  else sw.classList.add('hidden');
  const w = $('missingWarn');
  if(j.missing_columns && j.missing_columns.length){ w.classList.remove('hidden');
    w.innerHTML = 'Your file has no values for: <b>' + j.missing_columns.map(esc).join(', ') + '</b>. These were filled with typical training values, so predictions may be less reliable.'; }
  else w.classList.add('hidden');
}
function renderKpis(j){
  const m = j.metrics, f = x => x == null ? '-' : x.toFixed(2);
  const k = [['Average predicted risk', j.mean_risk.toFixed(1) + ' / 100'],
    ['Risk level', '<span class="pill ' + j.level + '">' + j.level + '</span>'],
    ['High-risk samples', j.pct_high.toFixed(0) + '%'],
    ['Low / Mod / High', j.counts.Low + ' / ' + j.counts.Moderate + ' / ' + j.counts.High]];
  if(m){ k.push(['MAE on this data', f(m.MAE)], ['RMSE', f(m.RMSE)], ['R²', f(m.R2)]); }
  $('kpis').innerHTML = k.map(([l, v]) => '<div class="kpi"><div class="l">' + l + '</div><div class="v">' + v + '</div></div>').join('');
}
function renderMain(j){
  const c = j.chart, x = [...Array(c.n).keys()], S = c.series, tr = [];
  const add = (name, y, color, w, dash) => { if(y) tr.push({x, y, name, mode: 'lines', line: {color, width: w, dash: dash || 'solid'}, opacity: w > 2 ? 1 : .8}); };
  add('Soil Moisture', S.soil_moisture, '#4caf50', 2);
  add('Humidity', S.humidity, '#1f77b4', 2);
  add('Fertilizer Usage', S.fertilizer_usage, '#ff9933', 2);
  if(c.pest_actual){
    add('Predicted Pest Risk (model)', c.pest_predicted, dark() ? '#ffffff' : '#222', 2.5, 'dot');
    add('Pest Pressure (Dominant Factor)', c.pest_actual, '#d62728', 4.5);
  } else add('Pest Pressure (Dominant Factor)', c.pest_predicted, '#d62728', 4.5);
  Plotly.react('chMain', tr, plotLayout({xaxis: {title: 'Sample Index', gridcolor: css('--line')}, yaxis: {title: 'Normalized Value (0 - 100%)', range: [-5, 105], gridcolor: css('--line')},
    legend: {x: 1, xanchor: 'right', y: 1, bgcolor: dark() ? 'rgba(20,30,25,.8)' : 'rgba(255,255,255,.8)', bordercolor: css('--line'), borderwidth: 1}}), CFG);
}
function renderPca(j){
  const P = j.model.points, Q = j.projection, pal = ['#4c78a8','#f58518','#54a24b','#b279a2','#e45756','#9d755d'];
  const t1 = {type: 'scatter3d', mode: 'markers', name: 'Training clusters', x: P.x, y: P.y, z: P.z,
    marker: {size: 2.5, opacity: .45, color: P.cluster.map(c => pal[c % pal.length])}, hoverinfo: 'skip'};
  const t2 = {type: 'scatter3d', mode: 'markers', name: 'Your samples', x: Q.x, y: Q.y, z: Q.z,
    marker: {size: 5, symbol: 'diamond', color: Q.risk, cmin: 0, cmax: 100, colorscale: 'RdYlGn', reversescale: true, line: {width: .5, color: '#000'},
      colorbar: {title: 'Risk', thickness: 10, len: .6}}, text: Q.risk.map((r, i) => 'Sample #' + i + ' risk ' + r.toFixed(0)), hoverinfo: 'text'};
  const ax = t => ({title: t, gridcolor: css('--line'), zerolinecolor: css('--line'), backgroundcolor: 'rgba(0,0,0,0)', showbackground: false});
  Plotly.react('chPca', [t1, t2], plotLayout({margin: {l: 0, r: 0, t: 0, b: 0}, legend: {orientation: 'h', y: 0},
    scene: {xaxis: ax('PC1'), yaxis: ax('PC2'), zaxis: ax('PC3'), bgcolor: 'rgba(0,0,0,0)'}}), CFG);
}
function renderImp(j){
  const e = Object.entries(j.model.importance).reverse();
  Plotly.react('chImp', [{type: 'bar', orientation: 'h', x: e.map(a => a[1] * 100), y: e.map(a => a[0]), marker: {color: css('--accent')}}],
    plotLayout({margin: {l: 140, r: 15, t: 10, b: 40}, xaxis: {title: 'Importance (%)', gridcolor: css('--line')}}), CFG);
}
function renderScatter(j){
  const card = $('scCard'); if(!j.scatter){ card.classList.add('hidden'); return; } card.classList.remove('hidden');
  const s = j.scatter;
  Plotly.react('chSc', [{type: 'scatter', mode: 'markers', x: s.actual, y: s.pred, marker: {color: '#d62728', size: 6, opacity: .6}, name: 'Samples'},
    {type: 'scatter', mode: 'lines', x: [0, 100], y: [0, 100], line: {color: css('--muted'), dash: 'dash'}, name: 'Perfect'}],
    plotLayout({xaxis: {title: 'Actual pest pressure', gridcolor: css('--line')}, yaxis: {title: 'Predicted', gridcolor: css('--line')}, showlegend: false}), CFG);
}
function renderSug(j){
  $('lvlTag').textContent = j.level + ' risk'; $('lvlTag').className = 'tag pill ' + j.level;
  $('sugs').innerHTML = j.suggestions.map(s => '<div class="s ' + s.severity + '"><b>' + esc(s.title) + '</b>' + esc(s.text) + '</div>').join('');
}
function renderModel(j){
  const M = j.model, f = (v, d) => v == null ? '-' : v.toFixed(d || 2), pm = x => f(x.m) + ' ± ' + f(x.s);
  $('modelSub').textContent = 'Trained on ' + M.n_train + ' cleaned rows. Best hyper-parameters (RandomizedSearchCV): ' +
    Object.entries(M.best_params).map(([k, v]) => k + '=' + v).join(', ') + '.';
  const names = Object.keys(M.cv);
  let h = '<tr><th>Model</th><th>CV MAE</th><th>CV RMSE</th><th>CV R²</th><th>Hold-out MAE</th><th>Hold-out RMSE</th><th>Hold-out R²</th></tr>';
  names.forEach(n => { const c = M.cv[n], o = M.holdout[n];
    h += '<tr><td><b>' + esc(n) + '</b></td><td>' + pm({m: c.MAE, s: c.MAE_std}) + '</td><td>' + pm({m: c.RMSE, s: c.RMSE_std}) + '</td><td>' + pm({m: c.R2, s: c.R2_std}) + '</td>' +
      (o ? '<td>' + f(o.MAE) + '</td><td>' + f(o.RMSE) + '</td><td>' + f(o.R2) + '</td>' : '<td>-</td><td>-</td><td>-</td>') + '</tr>'; });
  $('modelTbl').innerHTML = h;
  const E = M.explore;
  $('explSub').textContent = 'Exploratory analysis: PCA keeps ' + (E.explained_variance.reduce((a, b) => a + b, 0) * 100).toFixed(0) + '% of variance in 3 components; K-Means chose k=' + E.k +
    ' (silhouette ' + E.silhouette.toFixed(2) + '). ' + E.clusters.map(c => 'Cluster ' + c.cluster + ': ' + c.size + ' rows, mean pest pressure ' + c.mean_pest_pressure.toFixed(1)).join(' | ') + '.';
}
function renderDrift(j){
  const d = j.drift; $('driftTag').textContent = d.status; $('driftTag').className = 'tag ' + (d.status === 'Stable' ? 'ok' : d.status === 'Watch' ? 'watch' : 'drift');
  $('driftTbl').innerHTML = '<tr><th>Feature</th><th>PSI</th><th>Status</th></tr>' + d.top.map(r => '<tr><td>' + esc(r.feature) + '</td><td>' + r.psi.toFixed(3) + '</td><td class="' + r.flag + '">' + r.flag + '</td></tr>').join('');
  $('logTbl').innerHTML = '<tr><th>Time</th><th>Type</th><th>Source</th><th>Rows</th><th>Mean pred.</th><th>MAE</th><th>Drift</th></tr>' +
    (j.log || []).slice(0, 8).map(e => '<tr><td>' + esc(e.time) + '</td><td>' + esc(e.kind) + '</td><td>' + esc(e.source) + '</td><td>' + e.rows + '</td><td>' + e.mean_pred + '</td><td>' + (e.MAE == null ? '-' : e.MAE) + '</td><td>' + esc(e.drift) + '</td></tr>').join('');
}

/* ---------- 3D risk terrain (Three.js, drag to rotate) ---------- */
let T = null;
function initTerrain(){
  const el = $('terrain'); const ren = new THREE.WebGLRenderer({antialias: true, alpha: true}); ren.setPixelRatio(Math.min(devicePixelRatio, 2));
  el.insertBefore(ren.domElement, el.firstChild);
  const scene = new THREE.Scene(), cam = new THREE.PerspectiveCamera(45, 1, .1, 200);
  scene.add(new THREE.AmbientLight(0xffffff, .65)); const dl = new THREE.DirectionalLight(0xffffff, .9); dl.position.set(8, 14, 10); scene.add(dl);
  const grp = new THREE.Group(); scene.add(grp);
  T = {el, ren, scene, cam, grp, rotY: .6, rotX: .75, dist: 20, auto: true, bars: [], ray: new THREE.Raycaster(), mouse: new THREE.Vector2()};
  function size(){ const w = el.clientWidth, h = el.clientHeight; ren.setSize(w, h); cam.aspect = w / h; cam.updateProjectionMatrix(); }
  new ResizeObserver(size).observe(el); size();
  let drag = false, lx = 0, ly = 0;
  ren.domElement.addEventListener('pointerdown', e => { drag = true; T.auto = false; lx = e.clientX; ly = e.clientY; ren.domElement.setPointerCapture(e.pointerId); });
  ren.domElement.addEventListener('pointerup', () => drag = false);
  ren.domElement.addEventListener('pointermove', e => {
    if(drag){ T.rotY += (e.clientX - lx) * .008; T.rotX = Math.max(.2, Math.min(1.5, T.rotX + (e.clientY - ly) * .006)); lx = e.clientX; ly = e.clientY; }
    const r = ren.domElement.getBoundingClientRect(); T.mouse.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
    T.ray.setFromCamera(T.mouse, cam); const hit = T.ray.intersectObjects(T.bars)[0], tip = $('tip');
    if(hit){ tip.style.display = 'block'; tip.style.left = (e.clientX - r.left + 12) + 'px'; tip.style.top = (e.clientY - r.top + 12) + 'px'; tip.textContent = 'Sample #' + hit.object.userData.i + ' - risk ' + hit.object.userData.v.toFixed(0); }
    else tip.style.display = 'none';
  });
  ren.domElement.addEventListener('wheel', e => { e.preventDefault(); T.dist = Math.max(8, Math.min(40, T.dist + e.deltaY * .02)); }, {passive: false});
  (function loop(){ if(T.auto) T.rotY += .004; grp.rotation.y = T.rotY;
    cam.position.set(0, Math.sin(T.rotX) * T.dist, Math.cos(T.rotX) * T.dist); cam.lookAt(0, 1.5, 0); ren.render(scene, cam); requestAnimationFrame(loop); })();
}
function renderTerrain(vals){
  if(!T) initTerrain();
  T.bars.forEach(b => { T.grp.remove(b); b.geometry.dispose(); b.material.dispose(); }); T.bars = [];
  if(T.base){ T.grp.remove(T.base); T.base.geometry.dispose(); T.base.material.dispose(); T.base = null; }
  const side = Math.ceil(Math.sqrt(vals.length)), gap = 1.15, off = (side - 1) * gap / 2;
  vals.forEach((v, i) => {
    const h = .15 + v / 100 * 6, col = new THREE.Color().setHSL((1 - v / 100) * .33, .8, dark() ? .55 : .48);
    const m = new THREE.Mesh(new THREE.BoxGeometry(.9, h, .9), new THREE.MeshStandardMaterial({color: col, roughness: .45, metalness: .1}));
    m.position.set((i % side) * gap - off, h / 2, Math.floor(i / side) * gap - off); m.userData = {i, v}; T.grp.add(m); T.bars.push(m);
  });
  const base = new THREE.Mesh(new THREE.BoxGeometry(side * gap + .6, .1, side * gap + .6), new THREE.MeshStandardMaterial({color: dark() ? 0x1b2a23 : 0xcfd9c6}));
  base.position.y = -.05; T.grp.add(base); T.base = base;
  T.dist = 10 + side * 1.1;
}

/* ---------- start ---------- */
initBg();
(async () => {
  try{ const m = await api('/api/me'); if(m.logged_in) enterDash(m.name); else show('auth'); }catch(e){ show('auth'); }
})();
</script>
</body>
</html>
"""


# --------------------------------------------------------------------------- launcher
def main():
    init_db()
    train_all()
    url = f"http://127.0.0.1:{PORT}"
    print(f"\n[server] PestGuard 3D running at {url}\n")
    if in_colab():
        threading.Thread(target=lambda: app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False, threaded=True),
                         daemon=True).start()
        time.sleep(1.5)
        from google.colab import output
        output.serve_kernel_port_as_window(PORT)      # opens the web panel in a new tab
        print("If no tab opened, click the link above. Keep this cell running / do not restart the runtime.")
    else:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
        app.run(host="127.0.0.1", port=PORT, debug=False, use_reloader=False, threaded=True)


if __name__ == "__main__":
    main()
