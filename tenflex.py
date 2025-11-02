# streamlit_app.py
# Tenis AI+ — Momios sintéticos (api-tennis.com) — versión Streamlit
# - Individual y Lote por múltiples match_key
# - Exportación a Excel (descarga)
# - Resultados oficiales (ganador/marcador)
# - Integra cuotas Bet365 (Home/Away)
# - Columna "Acerto pronostico" en Excel
# Reqs: streamlit, requests, pandas, openpyxl, unidecode, urllib3

import os
import io
import json
import math
from datetime import datetime, timedelta, date

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from unidecode import unidecode

import pandas as pd
import streamlit as st

# ===================== CONFIG =====================
st.set_page_config(page_title="Tenis AI+ (Streamlit)", layout="wide")
BASE_URL = "https://api.api-tennis.com/tennis/"

RANK_BUCKETS = {
    "GS": 1.30,      # Grand Slam
    "ATP/WTA": 1.15,
    "Challenger": 1.00,
    "ITF": 0.85
}
RANK_BUCKETS.setdefault("Other", 0.95)

# ===================== UTILIDADES =====================
def normalize(s: str) -> str:
    return unidecode(s or "").strip().lower()

def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

def safe_int(x, default=None):
    try:
        return int(x)
    except Exception:
        return default

def logistic(x):
    return 1.0 / (1.0 + math.exp(-x))

def clamp(v, a, b):
    return max(a, min(b, v))

def make_session():
    s = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=0.6,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=10, pool_maxsize=10)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s

SESSION = make_session()
HTTP_TIMEOUT = 25  # seg por request

# ===================== API WRAPPER =====================
def call_api(method: str, params: dict):
    """Llama a la API y maneja casos de éxito sin 'result' (retorna {})."""
    params = {k: v for k, v in params.items() if v is not None}
    url = BASE_URL
    q = {"method": method, **params}
    r = SESSION.get(url, params=q, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    data = r.json()

    if str(data.get("success")) == "1":
        return data.get("result", {})

    if str(data.get("error")) == "1":
        try:
            detail = (data.get("result") or [{}])[0]
            cod = detail.get("cod")
            msg = detail.get("msg") or "API error"
        except Exception:
            cod, msg = None, "API error"
        raise RuntimeError(f"{method} → {msg} (cod={cod})")

    raise RuntimeError(f"{method} → Respuesta no esperada: {data}")

def try_get_players(api_key: str, player_name_like: str):
    try:
        res = call_api("get_players", {"APIkey": api_key, "player": player_name_like})
        return res or []
    except Exception:
        return []

# ===================== ODDS HELPERS (Bet365) =====================
def get_bet365_odds_for_match(api_key: str, match_key: int):
    try:
        res = call_api("get_odds", {"APIkey": api_key, "match_key": match_key}) or {}
        m = res.get(str(match_key)) or res.get(int(match_key))
        if not isinstance(m, dict):
            return (None, None)

        ha = m.get("Home/Away") or {}
        home = (ha.get("Home") or {})
        away = (ha.get("Away") or {})

        def pick_b365(d):
            if not isinstance(d, dict):
                return None
            for k in d.keys():
                if str(k).strip().lower() == "bet365":
                    return d[k]
            return None

        def to_float(x):
            try:
                return float(x)
            except Exception:
                return None

        home_b365 = to_float(pick_b365(home))
        away_b365 = to_float(pick_b365(away))
        return (home_b365, away_b365)
    except Exception:
        return (None, None)

# ===================== FIXTURE HELPERS =====================
def list_fixtures(api_key: str, date_start: str, date_stop: str, tz: str, player_key=None):
    params = {
        "APIkey": api_key,
        "date_start": date_start,
        "date_stop": date_stop,
        "timezone": tz
    }
    if player_key:
        params["player_key"] = player_key
    res = call_api("get_fixtures", params) or []
    return res

def get_fixture_by_key(api_key: str, match_key: int, tz: str = "Europe/Berlin", center_date: str | None = None, progress_cb=None):
    # 1) Intento directo
    try:
        res = call_api("get_events", {"APIkey": api_key, "event_key": match_key}) or []
        if isinstance(res, list):
            for m in res:
                if safe_int(m.get("event_key")) == int(match_key):
                    return m
        elif isinstance(res, dict) and safe_int(res.get("event_key")) == int(match_key):
            return res
    except Exception:
        pass

    # 2) Fallback con ventanas
    if center_date:
        try:
            base = datetime.strptime(center_date, "%Y-%m-%d").date()
        except Exception:
            base = datetime.utcnow().date()
    else:
        base = datetime.utcnow().date()

    CHUNK_SIZES = [7, 3, 1]
    RINGS = [14, 28, 56, 112, 200]

    steps = 0
    total_steps = len(RINGS) * 60
    for ring in RINGS:
        start_global = base - timedelta(days=ring)
        stop_global  = base + timedelta(days=10)
        cur_start = start_global
        while cur_start <= stop_global:
            hit_this_window = False
            for chunk in CHUNK_SIZES:
                cur_stop = min(cur_start + timedelta(days=chunk - 1), stop_global)
                try:
                    fixtures = list_fixtures(
                        api_key,
                        cur_start.strftime("%Y-%m-%d"),
                        cur_stop.strftime("%Y-%m-%d"),
                        tz
                    ) or []
                    for m in fixtures:
                        if safe_int(m.get("event_key")) == int(match_key):
                            return m
                    hit_this_window = True
                    break
                except requests.HTTPError as http_err:
                    if http_err.response is not None and http_err.response.status_code == 500:
                        continue
                    else:
                        raise
                except Exception:
                    continue
            steps += 1
            if progress_cb:
                progress_cb(min(1.0, steps/total_steps))
            step = max(CHUNK_SIZES) if hit_this_window else 1
            cur_start = cur_start + timedelta(days=step)

    raise ValueError(f"No se encontró el match_key={match_key} alrededor de {base}.")

# ===================== FEATURE ENGINEERING =====================
def get_player_matches(api_key: str, player_key: int, days_back=365):
    stop = datetime.utcnow().date()
    start = (stop - timedelta(days=days_back)).strftime("%Y-%m-%d")
    stop_str = stop.strftime("%Y-%m-%d")
    res = list_fixtures(api_key, start, stop_str, "Europe/Berlin", player_key=player_key) or []
    clean = []
    for m in res:
        status = (m.get("event_status") or "").lower()
        if "finished" in status or m.get("event_winner") in ("First Player", "Second Player"):
            clean.append(m)
    return clean

def is_win_for_name(match, player_name_norm: str):
    fp = normalize(match.get("event_first_player"))
    sp = normalize(match.get("event_second_player"))
    w = match.get("event_winner")
    if w == "First Player":
        return fp == player_name_norm
    if w == "Second Player":
        return sp == player_name_norm
    res = (match.get("event_final_result") or "").strip().lower()
    if fp == player_name_norm and (res.startswith("2 - 0") or res.startswith("2 - 1")):
        return True
    if sp == player_name_norm and (res.startswith("0 - 2") or res.startswith("1 - 2")):
        return True
    return False

def winrate_60d_and_lastN(matches, player_name_norm: str, N=10, days=60):
    now = datetime.utcnow()
    def days_ago(m):
        try:
            d = datetime.strptime(m["event_date"], "%Y-%m-%d")
            return (now - d).days
        except Exception:
            return 10**6

    recent = [m for m in matches if days_ago(m) <= days]
    wr60 = (sum(is_win_for_name(m, player_name_norm) for m in recent) / len(recent)) if recent else 0.5

    sorted_all = sorted(matches, key=lambda x: (x.get("event_date") or "", x.get("event_time") or "00:00"), reverse=True)
    lastN = sorted_all[:N]
    wrN = (sum(is_win_for_name(m, player_name_norm) for m in lastN) / len(lastN)) if lastN else 0.5

    last_date = sorted_all[0]["event_date"] if sorted_all else None
    return wr60, wrN, last_date, sorted_all

def compute_momentum(sorted_matches, player_name_norm: str):
    streak = 0
    for m in sorted_matches:
        w = is_win_for_name(m, player_name_norm)
        if w:
            streak = +1 if streak < 0 else streak + 1
        else:
            streak = -1 if streak > 0 else -1
        if streak >= 4:
            return +1
        if streak <= -3:
            return -1
    return 0

def rest_days(last_date_str):
    if not last_date_str:
        return None
    d = datetime.strptime(last_date_str, "%Y-%m-%d").date()
    return (datetime.utcnow().date() - d).days

def rest_score(days):
    if days is None:
        return 0.0
    return clamp(1.0 - abs(days - 7) / 21.0, 0.0, 1.0)

def league_bucket(league_name: str):
    s = (league_name or "").lower()
    if any(k in s for k in ["grand slam", "roland", "wimbledon", "us open", "australian open"]):
        return "GS"
    if any(k in s for k in ["atp", "wta"]):
        return "ATP/WTA"
    if "challenger" in s:
        return "Challenger"
    if "itf" in s:
        return "ITF"
    return "Other"

def surface_winrate(matches, player_name_norm: str, surface: str):
    if not surface:
        return 0.5
    sur = surface.lower()
    hist = [m for m in matches if (m.get("event_tournament_surface") or "").lower() == sur]
    if not hist:
        return 0.5
    return sum(is_win_for_name(m, player_name_norm) for m in hist) / len(hist)

def travel_penalty(last_match_country, current_country, days_since):
    if not last_match_country or not current_country or days_since is None:
        return 0.0
    if last_match_country.strip().lower() == current_country.strip().lower():
        return 0.0
    if days_since <= 3:
        return 0.15
    if days_since <= 5:
        return 0.07
    return 0.0

def elo_synth_from_opposition(matches, player_name_norm: str):
    if not matches:
        return 0.0
    score = 0.0
    for m in matches[:20]:
        bucket = league_bucket(m.get("league_name", ""))
        weight = RANK_BUCKETS.get(bucket, 1.0)
        w = is_win_for_name(m, player_name_norm)
        score += (1.0 if w else -1.0) * weight
    score = score / (20.0 * 1.30)
    return clamp(score, -1.0, 1.0)

def compute_h2h(api_key, player_key_a, player_key_b, years_back=5):
    stop = datetime.utcnow().date()
    start = (stop - timedelta(days=365*years_back)).strftime("%Y-%m-%d")
    stop_str = stop.strftime("%Y-%m-%d")

    res_a = list_fixtures(api_key, start, stop_str, "Europe/Berlin", player_key=player_key_a) or []
    res_b = list_fixtures(api_key, start, stop_str, "Europe/Berlin", player_key=player_key_b) or []

    def key_of(m):
        return (normalize(m.get("event_first_player")),
                normalize(m.get("event_second_player")),
                m.get("event_date"))

    idx_b = {key_of(m): m for m in res_b}
    wins_a = wins_b = 0

    for ma in res_a:
        k = key_of(ma)
        mb = idx_b.get(k)
        if not mb:
            continue
        w = ma.get("event_winner")
        if w == "First Player":
            wins_a += 1
        elif w == "Second Player":
            wins_b += 1

    total = wins_a + wins_b
    pct_a = wins_a / total if total else 0.5
    return wins_a, wins_b, pct_a

# ===================== MODELO =====================
def calibrate_probability(diff, weights, gamma=3.0, bias=0.0, bonus=0.0, malus=0.0):
    wsum = sum(weights.values()) or 1.0
    w = {k: v/wsum for k, v in weights.items()}
    z = (w.get("wr60",0)*diff.get("wr60",0) +
         w.get("wr10",0)*diff.get("wr10",0) +
         w.get("h2h",0)*diff.get("h2h",0) +
         w.get("rest",0)*diff.get("rest",0) +
         w.get("surface",0)*diff.get("surface",0) +
         w.get("elo",0)*diff.get("elo",0) +
         w.get("momentum",0)*diff.get("momentum",0) -
         w.get("travel",0)*diff.get("travel",0) +
         bias)
    p = logistic(gamma * z + bonus - malus)
    return clamp(p, 0.05, 0.95)

def invert_bo3_set_prob(pm):
    lo, hi = 0.05, 0.95
    for _ in range(40):
        mid = 0.5*(lo+hi)
        pm_mid = mid*mid*(3 - 2*mid)
        if pm_mid < pm: lo = mid
        else: hi = mid
    return 0.5*(lo+hi)

def bo3_distribution(p_set):
    s = p_set; q = 1 - s
    p20 = s*s
    p21 = 2*s*s*q
    p12 = 2*q*q*s
    p02 = q*q
    tot = p20 + p21 + p12 + p02
    return {"2:0": p20/tot, "2:1": p21/tot, "1:2": p12/tot, "0:2": p02/tot}

def to_decimal(p):
    p = clamp(p, 0.01, 0.99)
    return round(1.0/p, 3)

# ===================== CÁLCULO CORE =====================
def compute_from_fixture(api_key: str, meta: dict, surface_hint: str,
                         weights: dict, gamma: float, bias: float):
    match_key = safe_int(meta.get("event_key"))
    tz = meta.get("timezone") or "Europe/Berlin"
    date_str = meta.get("event_date") or datetime.utcnow().strftime("%Y-%m-%d")

    api_p1 = meta.get("event_first_player")
    api_p2 = meta.get("event_second_player")
    api_p1n = normalize(api_p1)
    api_p2n = normalize(api_p2)

    p1k = safe_int(meta.get("first_player_key"))
    p2k = safe_int(meta.get("second_player_key"))

    surface_api = (meta.get("event_tournament_surface") or "").strip() or None
    surface_final = (surface_hint or "").strip().lower() or (surface_api.lower() if surface_api else None)

    # --- Últimos partidos / features
    lastA = get_player_matches(api_key, p1k, days_back=365) if p1k else []
    lastB = get_player_matches(api_key, p2k, days_back=365) if p2k else []

    wr60_A, wr10_A, lastA_date, sortedA = winrate_60d_and_lastN(lastA, api_p1n, N=10, days=60)
    wr60_B, wr10_B, lastB_date, sortedB = winrate_60d_and_lastN(lastB, api_p2n, N=10, days=60)

    momA = compute_momentum(sortedA, api_p1n)
    momB = compute_momentum(sortedB, api_p2n)

    rA_days = rest_days(lastA_date)
    rB_days = rest_days(lastB_date)
    rA = rest_score(rA_days)
    rB = rest_score(rB_days)

    surf_wrA = surface_winrate(lastA, api_p1n, surface_final)
    surf_wrB = surface_winrate(lastB, api_p2n, surface_final)

    lastA_country = lastA and (lastA[0].get("country") or lastA[0].get("event_tournament_country"))
    lastB_country = lastB and (lastB[0].get("country") or lastB[0].get("event_tournament_country"))
    tourn_country = meta.get("country") or meta.get("event_tournament_country")
    travA = travel_penalty(lastA_country, tourn_country, rA_days or 999)
    travB = travel_penalty(lastB_country, tourn_country, rB_days or 999)

    if p1k and p2k:
        _, _, h2h_pct_a = compute_h2h(api_key, p1k, p2k, years_back=5)
    else:
        h2h_pct_a = 0.5
    h2h_pct_b = 1.0 - h2h_pct_a

    eloA = elo_synth_from_opposition(sortedA, api_p1n)
    eloB = elo_synth_from_opposition(sortedB, api_p2n)

    total_obs = len(sortedA) + len(sortedB)
    reg_alpha = 0.0
    if total_obs < 6: reg_alpha = 0.6
    elif total_obs < 12: reg_alpha = 0.35
    elif total_obs < 20: reg_alpha = 0.2

    wr60_A = (1-reg_alpha)*wr60_A + reg_alpha*0.5
    wr60_B = (1-reg_alpha)*wr60_B + reg_alpha*0.5
    wr10_A = (1-reg_alpha)*wr10_A + reg_alpha*0.5
    wr10_B = (1-reg_alpha)*wr10_B + reg_alpha*0.5
    surf_wrA = (1-reg_alpha)*surf_wrA + reg_alpha*0.5
    surf_wrB = (1-reg_alpha)*surf_wrB + reg_alpha*0.5
    h2h_pct_a = (1-reg_alpha)*h2h_pct_a + reg_alpha*0.5
    h2h_pct_b = 1 - h2h_pct_a
    eloA = (1-reg_alpha)*eloA
    eloB = (1-reg_alpha)*eloB

    diff = {
        "wr60": wr60_A - wr60_B,
        "wr10": wr10_A - wr10_B,
        "h2h":  h2h_pct_a - h2h_pct_b,
        "rest": rA - rB,
        "surface": surf_wrA - surf_wrB,
        "elo": eloA - eloB,
        "momentum": (0.03 if momA > 0 else (-0.03 if momA < 0 else 0.0)) -
                    (0.03 if momB > 0 else (-0.03 if momB < 0 else 0.0)),
        "travel": travA - travB,
    }

    pA = calibrate_probability(diff=diff, weights=weights, gamma=gamma, bias=bias)
    pB = 1 - pA

    p_set_A = invert_bo3_set_prob(pA)
    dist = bo3_distribution(p_set_A)

    # ========= Resultado oficial =========
    event_status = (meta.get("event_status") or "").strip()
    event_winner_side = meta.get("event_winner")
    if event_winner_side == "First Player":
        winner_name = api_p1
    elif event_winner_side == "Second Player":
        winner_name = api_p2
    else:
        winner_name = None
    final_sets_str = (meta.get("event_final_result") or "").strip() or None

    # ========= Bet365 odds (Home/Away) =========
    b365_home, b365_away = get_bet365_odds_for_match(api_key, match_key) if match_key else (None, None)
    bet365_p1 = b365_home
    bet365_p2 = b365_away

    out = {
        "match_key": int(match_key) if match_key is not None else None,
        "inputs": {
            "date": date_str,
            "player1": api_p1,
            "player2": api_p2,
            "timezone": tz,
            "surface_used": surface_final or "(no especificada)",
        },
        "notes": [
            "Momios sintéticos (decimales) = 1 / prob. No incluyen margen de casa.",
            "Factores: forma (60d/10), H2H, descanso, superficie, ELO sintético, momentum, viaje, regularización.",
            "Los pesos se normalizan para sumar 1."
        ],
        "features": {
            "player1": {
                "wr60": round(wr60_A,3),
                "wr10": round(wr10_A,3),
                "h2h": round(h2h_pct_a,3),
                "rest_days": rA_days,
                "rest_score": round(rA,3),
                "surface_wr": round(surf_wrA,3),
                "elo_synth": round(eloA,3),
                "momentum": momA,
                "travel_penalty": round(travA,3),
            },
            "player2": {
                "wr60": round(wr60_B,3),
                "wr10": round(wr10_B,3),
                "h2h": round(h2h_pct_b,3),
                "rest_days": rB_days,
                "rest_score": round(rB,3),
                "surface_wr": round(surf_wrB,3),
                "elo_synth": round(eloB,3),
                "momentum": momB,
                "travel_penalty": round(travB,3),
            },
            "diff_A_minus_B": {k: round(v,4) for k,v in diff.items()},
        },
        "weights_used": {k: round(v,3) for k,v in weights.items()},
        "gamma": gamma,
        "bias": bias,
        "regularization_alpha": reg_alpha,
        "probabilities": {
            "match": {"player1": round(pA,4), "player2": round(pB,4)},
            "final_sets": {k: round(v,4) for k,v in dist.items()}
        },
        "synthetic_odds_decimal": {
            "player1": to_decimal(pA),
            "player2": to_decimal(pB),
            "2:0": to_decimal(dist["2:0"]),
            "2:1": to_decimal(dist["2:1"]),
            "1:2": to_decimal(dist["1:2"]),
            "0:2": to_decimal(dist["0:2"])
        },
        "bet365_odds_decimal": {
            "player1": bet365_p1,
            "player2": bet365_p2
        },
        "official_result": {
            "status": event_status,
            "winner_side": event_winner_side,
            "winner_name": winner_name,
            "final_sets": final_sets_str
        }
    }
    return out

# ===================== HELPERS UI =====================
def find_match_by_names(api_key, date_str, p1, p2, tz):
    p1n, p2n = normalize(p1), normalize(p2)
    base = datetime.strptime(date_str, "%Y-%m-%d").date()

    def scan_day(d):
        fixtures = list_fixtures(api_key, d, d, tz)
        cand = []
        for m in fixtures:
            fp = normalize(m.get("event_first_player"))
            sp = normalize(m.get("event_second_player"))
            if (p1n in fp and p2n in sp) or (p1n in sp and p2n in fp):
                cand.append(m)
        if not cand:
            for m in fixtures:
                fp = normalize(m.get("event_first_player"))
                sp = normalize(m.get("event_second_player"))
                if any(x in fp for x in p1n.split()) and any(x in sp for x in p2n.split()):
                    cand.append(m)
                elif any(x in sp for x in p1n.split()) and any(x in fp for x in p2n.split()):
                    cand.append(m)
        return cand[0] if cand else None

    m = scan_day(date_str)
    if not m:
        for k in [1]:
            for dd in [base - timedelta(days=k), base + timedelta(days=k)]:
                hit = scan_day(dd.strftime("%Y-%m-%d"))
                if hit:
                    m = hit
                    break
            if m:
                break

    if not m:
        raise ValueError(f"No se encontró el partido '{p1}' vs '{p2}' cerca de {date_str} (tz {tz}).")
    return m

def parse_batch_keys(raw: str):
    parts = [p.strip() for p in raw.replace(",", " ").replace("\n", " ").split(" ") if p.strip()]
    keys = []
    for p in parts:
        if p.isdigit():
            keys.append(int(p))
    seen = set()
    dedup = []
    for k in keys:
        if k not in seen:
            seen.add(k)
            dedup.append(k)
    return dedup

def build_excel_bytes(batch_results: list):
    rows = []
    for r in batch_results:
        mk = r.get("match_key")
        inp = r.get("inputs", {})
        probs = r.get("probabilities", {}).get("match", {})
        odds = r.get("synthetic_odds_decimal", {})
        feats = r.get("features", {})
        off = r.get("official_result", {})
        b365 = r.get("bet365_odds_decimal", {}) or {}
        f1 = feats.get("player1", {})
        f2 = feats.get("player2", {})
        diff = feats.get("diff_A_minus_B", {})

        odds_p1 = odds.get("player1")
        odds_p2 = odds.get("player2")
        winner_side = off.get("winner_side")

        favored_side = None
        try:
            if odds_p1 is not None and odds_p2 is not None:
                if float(odds_p1) < float(odds_p2):
                    favored_side = "First Player"
                elif float(odds_p2) < float(odds_p1):
                    favored_side = "Second Player"
        except Exception:
            favored_side = None

        if favored_side and winner_side in ("First Player", "Second Player"):
            acerto = "Si" if favored_side == winner_side else "No"
        else:
            acerto = ""

        row = {
            "match_key": mk,
            "date": inp.get("date"),
            "player1": inp.get("player1"),
            "player2": inp.get("player2"),
            "surface_used": inp.get("surface_used"),
            "p_player1": probs.get("player1"),
            "p_player2": probs.get("player2"),
            "odds_player1": odds_p1,
            "odds_player2": odds_p2,
            "odds_2_0": odds.get("2:0"),
            "odds_2_1": odds.get("2:1"),
            "odds_1_2": odds.get("1:2"),
            "odds_0_2": odds.get("0:2"),
            "bet365_player1": b365.get("player1"),
            "bet365_player2": b365.get("player2"),
            "p1_wr60": f1.get("wr60"),
            "p1_wr10": f1.get("wr10"),
            "p1_h2h": f1.get("h2h"),
            "p1_rest_days": f1.get("rest_days"),
            "p1_surface_wr": f1.get("surface_wr"),
            "p1_elo": f1.get("elo_synth"),
            "p1_momentum": f1.get("momentum"),
            "p1_travel": f1.get("travel_penalty"),
            "p2_wr60": f2.get("wr60"),
            "p2_wr10": f2.get("wr10"),
            "p2_h2h": f2.get("h2h"),
            "p2_rest_days": f2.get("rest_days"),
            "p2_surface_wr": f2.get("surface_wr"),
            "p2_elo": f2.get("elo_synth"),
            "p2_momentum": f2.get("momentum"),
            "p2_travel": f2.get("travel_penalty"),
            "diff_wr60": diff.get("wr60"),
            "diff_wr10": diff.get("wr10"),
            "diff_h2h": diff.get("h2h"),
            "diff_rest": diff.get("rest"),
            "diff_surface": diff.get("surface"),
            "diff_elo": diff.get("elo"),
            "diff_momentum": diff.get("momentum"),
            "diff_travel": diff.get("travel"),
            "status": off.get("status"),
            "winner_name": off.get("winner_name"),
            "final_sets": off.get("final_sets"),
            "Acerto pronostico": acerto,
        }
        rows.append(row)

    df = pd.DataFrame(rows).sort_values(by=["date","match_key"], ascending=True, na_position="last")

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="resumen")
        jrows = [{"match_key": r.get("match_key"), "json": json.dumps(r, ensure_ascii=False)} for r in batch_results]
        pd.DataFrame(jrows).to_excel(writer, index=False, sheet_name="json")
    output.seek(0)
    return output, df

# ===================== UI =====================
st.title("🎾 Tenis AI+ — Momios sintéticos (api-tennis.com)")

with st.sidebar:
    st.header("🔑 Credenciales")
    api_key_env = os.getenv("API_TENNIS_KEY", "")
    api_key_secret = st.secrets.get("API_TENNIS_KEY") if hasattr(st, "secrets") else None
    default_api = api_key_secret or api_key_env
    API_KEY = st.text_input("API Key", value=default_api, type="password", help="Puedes definir API_TENNIS_KEY en Secrets o variable de entorno.")

    st.header("🧮 Pesos del modelo (se normalizan)")
    w_wr60 = st.slider("wr60 (forma 60 días)", 0.0, 1.0, 0.30, 0.01)
    w_wr10 = st.slider("wr10 (últimos 10)",   0.0, 1.0, 0.20, 0.01)
    w_h2h  = st.slider("h2h",                 0.0, 1.0, 0.15, 0.01)
    w_rest = st.slider("rest (descanso)",     0.0, 1.0, 0.05, 0.01)
    w_surf = st.slider("surface",             0.0, 1.0, 0.15, 0.01)
    w_elo  = st.slider("elo sintético",       0.0, 1.0, 0.10, 0.01)
    w_mom  = st.slider("momentum",            0.0, 1.0, 0.05, 0.01)
    w_trav = st.slider("travel (malus)",      0.0, 1.0, 0.00, 0.01)
    gamma  = st.slider("gamma (agresividad)", 0.5, 5.0, 3.0, 0.1)
    bias   = st.slider("bias (sesgo)",       -0.5, 0.5, 0.0, 0.01)

weights = {
    "wr60": w_wr60, "wr10": w_wr10, "h2h": w_h2h, "rest": w_rest,
    "surface": w_surf, "elo": w_elo, "momentum": w_mom, "travel": w_trav
}

tab1, tab2, tab3 = st.tabs(["🧍 Cálculo individual", "📦 Lote por match_key", "📊 Resultados oficiales"])

# -------- Tab 1: Individual --------
with tab1:
    colA, colB, colC = st.columns([1,1,1])
    with colA:
        # Fecha con calendario
        _date_obj = st.date_input("Fecha", value=datetime.utcnow().date())
        date_str = _date_obj.strftime("%Y-%m-%d")

        tz = st.text_input("Timezone (IANA)", value="America/Mexico_City")
        surface = st.text_input("Superficie (opcional: hard/clay/grass/indoor)", value="")
    with colB:
        player1 = st.text_input("Jugador 1 (Home)", value="Okamura")
        player2 = st.text_input("Jugador 2 (Away)", value="Morvayova")
    with colC:
        manual_mk = st.text_input("Match Key (opcional)", value="")
        use_center_date = st.checkbox("Usar fecha estimada para buscar por match_key (opcional)")
        center_date = None
        if use_center_date:
            _center_dt = st.date_input("Fecha estimada (YYYY-MM-DD)", value=datetime.utcnow().date(), key="center_date_ind")
            center_date = _center_dt.strftime("%Y-%m-%d")

    run_individual = st.button("Calcular (individual)")
    if run_individual:
        if not API_KEY:
            st.error("Falta API Key.")
        else:
            try:
                with st.spinner("Buscando meta del partido…"):
                    if manual_mk.strip().isdigit():
                        prog = st.progress(0)
                        meta = get_fixture_by_key(API_KEY, int(manual_mk.strip()), tz=tz,
                                                  center_date=center_date,
                                                  progress_cb=lambda x: prog.progress(int(x*100)))
                    else:
                        meta = find_match_by_names(API_KEY, date_str, player1, player2, tz)

                with st.spinner("Calculando momios sintéticos…"):
                    result = compute_from_fixture(API_KEY, meta, surface, weights, gamma, bias)

                st.success("Listo (individual).")
                st.subheader("Resultado (JSON)")
                st.json(result, expanded=False)

                probs = result.get("probabilities", {}).get("match", {})
                odds = result.get("synthetic_odds_decimal", {})
                b365 = result.get("bet365_odds_decimal", {})
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("P(Player1)", probs.get("player1"))
                    st.metric("Momio sintético P1", odds.get("player1"))
                    st.metric("Bet365 P1", b365.get("player1"))
                with col2:
                    st.metric("P(Player2)", probs.get("player2"))
                    st.metric("Momio sintético P2", odds.get("player2"))
                    st.metric("Bet365 P2", b365.get("player2"))
                with col3:
                    st.write("Final sets (sintético):")
                    st.write(result.get("probabilities", {}).get("final_sets", {}))
                    st.write("Oficial:")
                    st.write(result.get("official_result", {}))

                st.session_state["last_result_single"] = result
            except Exception as e:
                st.error(str(e))

# -------- Tab 2: Lote --------
with tab2:
    st.write("Pega múltiples *match_key* (uno por línea, separados por coma o espacio).")
    raw_keys = st.text_area("Match Keys", height=150, placeholder="12345678\n98765432, 11122233 44455566")

    # Fecha estimada opcional con switch + calendario
    use_center_batch = st.checkbox("Usar fecha estimada para acelerar búsqueda (opcional)")
    center_date_batch = None
    if use_center_batch:
        _center_dt_b = st.date_input("Fecha estimada (YYYY-MM-DD)", value=datetime.utcnow().date(), key="center_date_batch")
        center_date_batch = _center_dt_b.strftime("%Y-%m-%d")

    run_batch = st.button("Calcular Lote")
    export_placeholder = st.empty()

    if run_batch:
        if not API_KEY:
            st.error("Falta API Key.")
        else:
            keys = parse_batch_keys(raw_keys)
            if not keys:
                st.warning("No se detectaron claves válidas.")
            else:
                progress = st.progress(0)
                log = st.container()
                results = []
                errors = []
                total = len(keys)
                for idx, mk in enumerate(keys, start=1):
                    try:
                        log.write(f"[{idx}/{total}] Buscando match_key {mk}…")
                        meta = get_fixture_by_key(API_KEY, mk, tz="America/Mexico_City",
                                                  center_date=center_date_batch)
                        out = compute_from_fixture(API_KEY, meta, surface, weights, gamma, bias)
                        results.append(out)
                        p1 = out['inputs']['player1']; p2 = out['inputs']['player2']; d = out['inputs']['date']
                        log.write(f"   OK: {p1} vs {p2} (date: {d})")
                    except Exception as e:
                        errors.append((mk, str(e)))
                        log.write(f"   ERROR {mk}: {e}")
                    progress.progress(int(100*idx/total))

                st.session_state["last_results_batch"] = results
                st.success(f"Lote finalizado. Éxitos: {len(results)} — Errores: {len(errors)}")

                if results:
                    xls_bytes, df = build_excel_bytes(results)
                    st.dataframe(df, use_container_width=True)
                    export_placeholder.download_button(
                        "⬇️ Descargar Excel (lote)",
                        data=xls_bytes.getvalue(),
                        file_name="momios_sinteticos_batch.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )

                if errors:
                    with st.expander("Ver errores"):
                        st.write(errors)

# -------- Tab 3: Resultados oficiales --------
with tab3:
    st.write("Pega *match_key* para consultar **solo** resultados oficiales (ganador/marcador).")
    raw_keys_R = st.text_area("Match Keys (Resultados)", height=150, placeholder="12345678\n98765432 11122233")

    use_center_R = st.checkbox("Usar fecha estimada (opcional)")
    center_date_R = None
    if use_center_R:
        _center_dt_R = st.date_input("Fecha estimada (YYYY-MM-DD)", value=datetime.utcnow().date(), key="center_date_R")
        center_date_R = _center_dt_R.strftime("%Y-%m-%d")

    run_res = st.button("Consultar Resultados")

    if run_res:
        if not API_KEY:
            st.error("Falta API Key.")
        else:
            keys = parse_batch_keys(raw_keys_R)
            if not keys:
                st.warning("No se detectaron claves válidas.")
            else:
                progress = st.progress(0)
                log = st.container()
                results = []
                errors = []
                total = len(keys)
                for idx, mk in enumerate(keys, start=1):
                    try:
                        log.write(f"[{idx}/{total}] Resultado de match_key {mk}…")
                        meta = get_fixture_by_key(API_KEY, mk, tz="America/Mexico_City",
                                                  center_date=center_date_R)
                        item = {
                            "match_key": safe_int(meta.get("event_key")),
                            "date": meta.get("event_date"),
                            "time": meta.get("event_time"),
                            "league": meta.get("league_name"),
                            "tournament": meta.get("event_tournament_name"),
                            "player1": meta.get("event_first_player"),
                            "player2": meta.get("event_second_player"),
                            "status": meta.get("event_status"),
                            "winner_side": meta.get("event_winner"),
                            "winner_name": (
                                meta.get("event_first_player") if meta.get("event_winner") == "First Player"
                                else (meta.get("event_second_player") if meta.get("event_winner") == "Second Player" else None)
                            ),
                            "final_sets": (meta.get("event_final_result") or "").strip() or None
                        }
                        results.append(item)
                    except Exception as e:
                        errors.append((mk, str(e)))
                        log.write(f"   ERROR {mk}: {e}")
                    progress.progress(int(100*idx/total))

                st.success("Resultados listos." if not errors else "Resultados con errores (ver abajo).")
                if results:
                    st.dataframe(pd.DataFrame(results).sort_values(by=["date","match_key"], na_position="last"),
                                 use_container_width=True)
                if errors:
                    with st.expander("Ver errores"):
                        st.write(errors)

# ===================== NOTAS =====================
st.caption(
    "Tip: en Streamlit Cloud, añade un archivo 'requirements.txt' con: "
    "streamlit, requests, pandas, openpyxl, unidecode, urllib3. "
    "Para usar Secrets, ve a Settings → Secrets y agrega API_TENNIS_KEY."
)
