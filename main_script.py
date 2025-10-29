from fastapi import FastAPI, HTTPException, Body, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from pathlib import Path
import io
import os

import pandas as pd
import numpy as np
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pyodbc
from datetime import date as Date

app = FastAPI(title="Ele ML API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Utilities ----------------

def _pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df

def _load_lookup_tables() -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    ana_articoli = None
    ana_macchine = None
    try:
        p_art = Path("esportazioneARTICOLI.xlsx")
        p_mac1 = Path("esportazioneMACCH REPARTI.xlsx")   # con spazio
        p_mac2 = Path("esportazioneMACCH_REPARTI.xlsx")   # con underscore
        if p_art.exists():
            ana_articoli = _normalize_cols(pd.read_excel(p_art))
        if p_mac1.exists():
            ana_macchine = _normalize_cols(pd.read_excel(p_mac1))
        elif p_mac2.exists():
            ana_macchine = _normalize_cols(pd.read_excel(p_mac2))
    except Exception as e:
        print(f"Warn: lookup non caricati: {e}")
    return ana_articoli, ana_macchine

def _clean_log(log: pd.DataFrame,
               ana_articoli: Optional[pd.DataFrame],
               ana_macchine: Optional[pd.DataFrame]) -> pd.DataFrame:
    log = _normalize_cols(log)

    if "idmacchina" not in log.columns:
        alt = _pick_col(log, ["id_macchina", "macchina_id", "id"])
        if alt:
            log["idmacchina"] = log[alt]

    if "durata" in log.columns:
        log["durata"] = pd.to_numeric(log["durata"], errors="coerce").fillna(0)
    else:
        log["durata"] = 0

    for col in ["causalefermo", "gruppo_macchina", "tipoarticolo", "classearticolo"]:
        if col in log.columns:
            log[col] = log[col].astype(str).str.strip()

    if "causalefermo" in log.columns:
        cf = log["causalefermo"].astype(str).str.strip().str.upper()
        map_cf = {
            "MANC PROG": "MANC. PROG.",
            "MANC.PROG": "MANC. PROG.",
            "MANC. PROG": "MANC. PROG.",
            "MANCANZA PROGRAMMAZIONE": "MANC. PROG.",
            "IN PRODUZIONE": "IN FUNZIONE",
            "FUNZIONE": "IN FUNZIONE",
        }
        log["causalefermo"] = cf.replace(map_cf)

    if ana_macchine is not None:
        idm_col = _pick_col(ana_macchine, ["idmacchina", "id_macchina", "macchina_id", "id"])
        grp_col = _pick_col(ana_macchine, ["gruppo_macchina", "gruppomacchina", "gruppo", "reparto", "reparto_descr"])
        if idm_col:
            am = ana_macchine.rename(columns={idm_col: "idmacchina"})
            cols_keep = ["idmacchina"]
            if grp_col:
                am = am.rename(columns={grp_col: "gruppo_macchina"})
                cols_keep.append("gruppo_macchina")
            am = am[cols_keep].drop_duplicates()
            log = log.merge(am, on="idmacchina", how="left", suffixes=("", "_ana"))

    def _norm_tipo(s: pd.Series) -> pd.Series:
        s_norm = s.astype(str).str.strip().str.upper()
        mask = s_norm.isin(["RIVETTI FORATI", "RIV.FOR.", "RIV. FOR."])
        s_out = s_norm.copy()
        s_out[mask] = "RIVETTO FORATO"
        return s_out

    if "tipoarticolo" in log.columns:
        log["tipoarticolo"] = _norm_tipo(log["tipoarticolo"])

    if "datainizio" in log.columns:
        log["datainizio"] = pd.to_datetime(log["datainizio"], errors="coerce")
        if "orainizio" in log.columns:
            ts = pd.to_datetime(log["datainizio"].astype(str) + " " + log["orainizio"].astype(str), errors="coerce")
        else:
            ts = log["datainizio"]
        log["ts_inizio"] = ts

    if "turno" in log.columns:
        log = log[log["turno"] != "T0"].reset_index(drop=True)

    dedup_cols = [c for c in ["datainizio", "idmacchina", "commessa", "causalefermo", "orainizio", "durata"] if c in log.columns]
    if dedup_cols:
        log = log.drop_duplicates(subset=dedup_cols, keep="first").reset_index(drop=True)

    if "datainizio" in log.columns:
        log = log[log["datainizio"] >= pd.Timestamp("2024-10-01")]

    return log

# --------------- DB Loader ----------------

def load_log_from_db() -> pd.DataFrame:
    conn_str = os.getenv("DB_CONNECTION_STRING")
    if not conn_str:
        raise HTTPException(status_code=500, detail="Stringa di connessione al database non configurata.")
    try:
        with pyodbc.connect(conn_str) as cnxn:
            raw = pd.read_sql("SELECT * FROM LOG_STATI", cnxn)
        ana_articoli, ana_macchine = _load_lookup_tables()
        return _clean_log(raw, ana_articoli, ana_macchine)
    except pyodbc.Error as ex:
        sqlstate = ex.args[0] if ex.args else "unknown"
        raise HTTPException(status_code=500, detail=f"Errore di connessione al database: {sqlstate}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore durante il caricamento/cleaning dei dati: {str(e)}")

def load_log() -> pd.DataFrame:
    return load_log_from_db()

# --------------- Analisi ABC --------------

def esegui_analisi_abc() -> pd.DataFrame:
    log = load_log_from_db()
    needed = {"causalefermo", "durata", "idmacchina"}
    if not needed.issubset(log.columns):
        raise HTTPException(status_code=500, detail=f"Colonne mancanti. Servono: {needed}")
    mask_prod = ~log["causalefermo"].str.upper().isin(["IN FUNZIONE", "MANC. PROG."])
    dfp = log[mask_prod].copy()
    abc = (
        dfp.groupby("causalefermo")
           .agg(
               ore_totali=("durata", lambda s: s.sum() / 3600.0),
               frequenza=("durata", "count"),
               macchine_coinvolte=("idmacchina", "nunique"),
           )
           .reset_index()
           .rename(columns={"causalefermo": "causale"})
    )
    return abc

def add_abc_classes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "ore_totali" not in df.columns:
        return df
    df = df.sort_values("ore_totali", ascending=False)
    perc = (df["ore_totali"] / df["ore_totali"].sum() * 100).fillna(0)
    df["perc"] = perc
    df["cumsum_perc"] = perc.cumsum()
    def lab(x):
        if x <= 80: return "A"
        if x <= 95: return "B"
        return "C"
    df["classe_abc"] = df["cumsum_perc"].apply(lab)
    return df

# --------------- Survival -----------------

def build_runs_for_km(log: pd.DataFrame) -> pd.DataFrame:
    req = {"idmacchina", "causalefermo", "durata"}
    if not req.issubset(log.columns):
        raise HTTPException(status_code=500, detail=f"Colonne mancanti per survival: {req}")
    if "datainizio" in log.columns and "orainizio" in log.columns:
        ts = pd.to_datetime(log["datainizio"].astype(str) + " " + log["orainizio"].astype(str), errors="coerce")
    elif "datainizio" in log.columns:
        ts = pd.to_datetime(log["datainizio"], errors="coerce")
    else:
        ts = pd.Series(pd.NaT, index=log.index)
    sa = log.assign(ts=ts).sort_values(["idmacchina", "ts"])
    sa["next_causale"] = sa.groupby("idmacchina")["causalefermo"].shift(-1)
    runs = sa[sa["causalefermo"].str.upper() == "IN FUNZIONE"].copy()
    runs["duration_hours"] = pd.to_numeric(runs["durata"], errors="coerce").fillna(0) / 3600.0
    runs["event"] = (
        runs["next_causale"].notna() &
        ~runs["next_causale"].str.upper().isin(["IN FUNZIONE", "MANC. PROG."])
    ).astype(int)
    runs = runs[runs["duration_hours"] > 0]
    return runs

def km_curve(durations: np.ndarray, events: np.ndarray) -> Dict[str, List[float]]:
    order = np.argsort(durations)
    t = durations[order]
    e = events[order]
    unique_times = np.unique(t)
    n = len(t)
    s = 1.0
    surv_times = [0.0]
    surv_vals = [1.0]
    at_risk = n
    for ut in unique_times:
        d_i = int(e[t == ut].sum())
        c_i = int((1 - e[t == ut]).sum())
        if at_risk > 0:
            s = s * (1.0 - (d_i / at_risk))
        surv_times.append(float(ut))
        surv_vals.append(float(s))
        at_risk -= (d_i + c_i)
    median = None
    for tt, ss in zip(surv_times, surv_vals):
        if ss <= 0.5:
            median = tt
            break
    return {"time": surv_times, "survival": surv_vals, "median": None if median is None else float(median)}

def plot_survival(series: Dict[str, Any]) -> bytes:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.step(series["time"], series["survival"], where="post", label="KM")
    ax.set_xlabel("Hours in function")
    ax.set_ylabel("Survival probability")
    ax.set_title("Survival curve")
    ax.grid(True, alpha=0.3)
    if series.get("median") is not None:
        ax.axvline(series["median"], color="red", linestyle="--", alpha=0.8)
        ax.axhline(0.5, color="gray", linestyle=":", alpha=0.6)
    ax.legend()
    buf = io.BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format="png", dpi=140)
    plt.close(fig)
    buf.seek(0)
    return buf.read()

# --------------- Modello scrap ------------

DATA_DIR = Path(".")
P_MODEL = DATA_DIR / "scrap_prediction_model_v1.pkl"
model_pkg: Optional[dict] = None

@app.on_event("startup")
async def load_model_on_startup():
    global model_pkg
    if P_MODEL.exists():
        try:
            model_pkg = joblib.load(P_MODEL)
            print("Modello caricato.")
        except Exception as e:
            model_pkg = None
            print(f"Attenzione: impossibile caricare il modello ({e}). "
                  "Installa scikit-learn e verifica la compatibilità di versione.")
    else:
        model_pkg = None
        print("Attenzione: modello non trovato.")

def safe_le_transform(le, val: str) -> int:
    try:
        classes = set(map(str, getattr(le, "classes_", [])))
        return int(le.transform([val])[0]) if val in classes else -1
    except Exception:
        return -1

def preprocess_scrap(order: Dict[str, Any], pkg: dict) -> pd.DataFrame:
    feats = pkg["features"]
    enc = pkg["label_encoders"]

    ora_inizio = order.get("ora_inizio")
    giorno_inizio = order.get("giorno_inizio")
    mese_inizio = order.get("mese_inizio")
    if order.get("data_inizio_ordine"):
        d = pd.to_datetime(order["data_inizio_ordine"], errors="coerce")
        if pd.notna(d):
            # usa solo la data (niente ora)
            ora_inizio = -1
            giorno_inizio = int(d.dayofweek + 1)
            mese_inizio = int(d.month)
            order["data_inizio_ordine"] = d.date().isoformat()

    macchina_encoded = safe_le_transform(enc["macchina"], str(order.get("macchina", "")).strip().upper())
    gruppo_macchina_encoded = safe_le_transform(enc["gruppo_macchina"], str(order.get("gruppo_macchina", "")).strip().upper())
    tipoarticolo_encoded = safe_le_transform(enc["tipoarticolo"], str(order.get("tipoarticolo", "")).strip().upper())
    classearticolo_encoded = safe_le_transform(enc["classearticolo"], str(order.get("classearticolo", "")).strip().upper())

    durata_per_qta = float(order.get("durata_per_qta", 0) or 0)
    durata_totale_ordine = float(order.get("durata_totale_ordine", 0) or 0)
    durata_minuti_setup = float(order.get("durata_minuti_setup", 0) or 0)
    kg_prodotti = float(order.get("kg_prodotti", 0) or 0)

    ora_inizio = int(ora_inizio if ora_inizio is not None else -1)
    giorno_inizio = int(giorno_inizio if giorno_inizio is not None else -1)
    mese_inizio = int(mese_inizio if mese_inizio is not None else -1)

    setup_per_kg = (durata_minuti_setup / kg_prodotti) if kg_prodotti not in (0, None) else 0.0
    efficienza = (kg_prodotti / durata_totale_ordine) if durata_totale_ordine not in (0, None) else 0.0
    macchina_x_setup = macchina_encoded * durata_minuti_setup
    tipo_x_classe = tipoarticolo_encoded * classearticolo_encoded
    is_monday = 1 if giorno_inizio == 1 else 0
    is_weekend_start = 1 if giorno_inizio in (5, 6) else 0
    is_month_start = 1 if order.get("data_inizio_ordine") and pd.to_datetime(order["data_inizio_ordine"], errors="coerce").day <= 7 else 0

    row = {
        "macchina_encoded": macchina_encoded,
        "gruppo_macchina_encoded": gruppo_macchina_encoded,
        "tipoarticolo_encoded": tipoarticolo_encoded,
        "classearticolo_encoded": classearticolo_encoded,
        "durata_per_qta": durata_per_qta,
        "durata_minuti_setup": durata_minuti_setup,
        "kg_prodotti": kg_prodotti,
        "ora_inizio": ora_inizio,
        "giorno_inizio": giorno_inizio,
        "mese_inizio": mese_inizio,
        "setup_per_kg": setup_per_kg,
        "efficienza": efficienza,
        "macchina_x_setup": macchina_x_setup,
        "tipo_x_classe": tipo_x_classe,
        "is_monday": is_monday,
        "is_weekend_start": is_weekend_start,
        "is_month_start": is_month_start,
    }
    X = pd.DataFrame([{k: row.get(k, 0) for k in feats}]).fillna(0)
    return X

# --------------- Schemas ------------------

class ScrapOrder(BaseModel):
    macchina: Optional[str] = None
    gruppo_macchina: Optional[str] = None
    tipoarticolo: Optional[str] = None
    classearticolo: Optional[str] = None
    durata_per_qta: Optional[float] = 0
    durata_minuti_setup: Optional[float] = 0
    kg_prodotti: Optional[float] = 0
    data_inizio_ordine: Optional[str] = Field(None, description="YYYY-MM-DD")

# --------------- Endpoints ----------------

@app.get("/health")
def health():
    db_status = "ok"
    db_error_details = None
    try:
        conn_str = os.getenv("DB_CONNECTION_STRING")
        if not conn_str:
            raise ValueError("DB_CONNECTION_STRING non impostata.")
        with pyodbc.connect(conn_str, timeout=10) as cnxn:
            cur = cnxn.cursor()
            cur.execute("SELECT 1")
            r = cur.fetchone()
            if not (r and r[0] == 1):
                raise ConnectionError("Query test DB inattesa.")
    except Exception as e:
        db_status = "error"
        db_error_details = str(e)
    try:
        odbc_drivers = pyodbc.drivers()
    except Exception as e:
        odbc_drivers = [f"error: {e}"]
    return {
        "api_status": "ok",
        "model_loaded": bool(model_pkg is not None),
        "database_connection": {"status": db_status, "details": db_error_details},
        "odbc_drivers": odbc_drivers,
    }


@app.post("/scrap/predict")
def scrap_predict(payload: Dict[str, Any] = Body(...)):
    if model_pkg is None:
        raise HTTPException(status_code=500, detail=f"Modello non trovato: {P_MODEL.name}")
    model = model_pkg["model"]
    thr = float(model_pkg.get("threshold", 0.5))

    if isinstance(payload, dict) and "orders" in payload and isinstance(payload["orders"], list):
        orders = payload["orders"]
    elif isinstance(payload, dict) and "order" in payload and isinstance(payload["order"], dict):
        orders = [payload["order"]]
    else:
        orders = [payload]

    for o in orders:
        if isinstance(o, dict) and o.get("data_inizio_ordine"):
            d = pd.to_datetime(o["data_inizio_ordine"], errors="coerce")
            if pd.notna(d):
                o["data_inizio_ordine"] = d.date().isoformat()

    X = pd.concat([preprocess_scrap(o, model_pkg) for o in orders], ignore_index=True)
    proba = model.predict_proba(X)[:, 1]
    out = []
    for i, p in enumerate(proba):
        out.append({
            "index": i,
            "risk_probability": float(p),
            "risk_class": "ALTO" if float(p) >= thr else "BASSO",
            "threshold": thr
        })
    return {"results": out, "count": len(out)}

@app.get("/scrap/predict")
def scrap_predict_get(
    macchina: Optional[str] = Query(None),
    gruppo_macchina: Optional[str] = Query(None),
    tipoarticolo: Optional[str] = Query(None),
    classearticolo: Optional[str] = Query(None),
    durata_per_qta: Optional[float] = Query(0),
    durata_minuti_setup: Optional[float] = Query(0),
    kg_prodotti: Optional[float] = Query(0),
    durata_totale_ordine: Optional[float] = Query(0),
    data_inizio_ordine: Optional[Date] = Query(None, description="YYYY-MM-DD")
):
    if model_pkg is None:
        raise HTTPException(status_code=500, detail=f"Modello non trovato: {P_MODEL.name}")
    model = model_pkg["model"]
    thr = float(model_pkg.get("threshold", 0.5))
    order = {
        "macchina": macchina,
        "gruppo_macchina": gruppo_macchina,
        "tipoarticolo": tipoarticolo,
        "classearticolo": classearticolo,
        "durata_per_qta": durata_per_qta,
        "durata_minuti_setup": durata_minuti_setup,
        "kg_prodotti": kg_prodotti,
        "durata_totale_ordine": durata_totale_ordine,
        "data_inizio_ordine": data_inizio_ordine.isoformat() if data_inizio_ordine else None,
    }
    X = preprocess_scrap(order, model_pkg)
    p = float(model.predict_proba(X)[:, 1][0])
    return {"results": [{
        "index": 0,
        "risk_probability": p,
        "risk_class": "ALTO" if p >= thr else "BASSO",
        "threshold": thr
    }], "count": 1}

@app.get("/fermi/abc")
def fermi_abc(
    limit: Optional[int] = Query(None, ge=1),
    order_by: str = Query("ore_totali"),
    descending: bool = Query(True)
):
    df = esegui_analisi_abc()
    if order_by not in df.columns:
        raise HTTPException(status_code=400, detail=f"order_by non valido. Scegli tra: {df.columns.tolist()}")
    df = add_abc_classes(df)
    df_sorted = df.sort_values(order_by, ascending=not descending)
    if limit:
        df_sorted = df_sorted.head(int(limit))
    return df_sorted.to_dict(orient="records")

@app.get("/survival/machine/{machine_id}")
def survival_machine(machine_id: str, as_image: bool = Query(False)):
    log = load_log()
    runs = build_runs_for_km(log[log["idmacchina"].astype(str).str.upper() == str(machine_id).upper()])
    if runs.empty:
        raise HTTPException(status_code=404, detail=f"Nessun episodio 'IN FUNZIONE' per macchina {machine_id}")
    series = km_curve(runs["duration_hours"].to_numpy(), runs["event"].to_numpy())
    if as_image:
        png = plot_survival(series)
        return Response(content=png, media_type="image/png")
    return {"machine": machine_id, **series}

@app.get("/survival/group/{group_name}")
def survival_group(group_name: str, as_image: bool = Query(False)):
    log = load_log()
    if "gruppo_macchina" not in log.columns:
        raise HTTPException(status_code=400, detail="Colonna 'gruppo_macchina' assente")
    grp = str(group_name).strip().upper()
    sel = log[log["gruppo_macchina"].astype(str).str.upper() == grp]
    if sel.empty:
        raise HTTPException(status_code=404, detail=f"Nessun dato per gruppo {group_name}")
    runs = build_runs_for_km(sel)
    if runs.empty:
        raise HTTPException(status_code=404, detail=f"Nessun episodio 'IN FUNZIONE' nel gruppo {group_name}")
    series = km_curve(runs["duration_hours"].to_numpy(), runs["event"].to_numpy())
    if as_image:
        png = plot_survival(series)
        return Response(content=png, media_type="image/png")
    return {"group": group_name, **series}

@app.get("/survival/groups")
def survival_groups(top: int = Query(3, ge=1, le=20)):
    log = load_log()
    if "gruppo_macchina" not in log.columns:
        raise HTTPException(status_code=400, detail="Colonna 'gruppo_macchina' assente")
    counts = log.groupby("gruppo_macchina")["durata"].count().sort_values(ascending=False).head(top)
    out = []
    for g, _ in counts.items():
        runs = build_runs_for_km(log[log["gruppo_macchina"] == g])
        if runs.empty:
            continue
        series = km_curve(runs["duration_hours"].to_numpy(), runs["event"].to_numpy())
        out.append({"group": g, **series})
    return {"series": out}

