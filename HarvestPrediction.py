#!/usr/bin/env python3
"""
dssat_orchestrator.py

Unified pipeline with verbose terminal prints so user sees stage-by-stage progress.
Usage examples:
  python dssat_orchestrator.py --pin 302031 --planting 2025-05-14 --crop wheat
  python dssat_orchestrator.py --latlon 26.90 75.90 --planting 2025-05-14 --crop wheat
"""

import os
import sys
import argparse
import time
import traceback
from datetime import datetime, timedelta
import re
import json

import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# DSSATTools binding (must be on PYTHONPATH)
try:
    from DSSATTools import DSSAT, Crop, SoilProfile, Weather, Management
except Exception as e:
    print("ERROR: Could not import DSSATTools. Ensure DSSATTools is installed and available on PYTHONPATH.")
    print("Exception:", e)
    sys.exit(1)

# optional geocoding
try:
    from geopy.geocoders import Nominatim
except Exception:
    Nominatim = None

# ---------- CONFIG ----------
OUT_BASE = "Final_Yield_outputs_1.0"
SOL_DIR = os.path.join(OUT_BASE, "sol_files")
WTH_DIR = os.path.join(OUT_BASE, "wth_files")
PLOT_DIR = os.path.join(OUT_BASE, "plots")
CSV_DIR = os.path.join(OUT_BASE, "csv")
LOG_DIR = os.path.join(OUT_BASE, "logs")
for d in (OUT_BASE, SOL_DIR, WTH_DIR, PLOT_DIR, CSV_DIR, LOG_DIR):
    os.makedirs(d, exist_ok=True)

WARMUP_DAYS = 30
DEFAULT_GROWTH_DAYS = 160
SOILGRIDS_DEPTHS = ["0-5cm", "5-15cm", "15-30cm"]

OPENMETEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
OPENMETEO_FORECAST = "https://api.open-meteo.com/v1/forecast"
SOILGRIDS_BASE = "https://rest.isric.org/soilgrids/v2.0/properties/query"

# ---------- Helpers (soil) ----------
def geocode_pin(pin):
    """Return (lat, lon) for given Indian PIN using Nominatim (if available)."""
    if Nominatim is None:
        raise RuntimeError("geopy not installed; cannot geocode. Install with `pip install geopy` or pass lat/lon manually.")
    geolocator = Nominatim(user_agent="dssat_pin_geocoder")
    q = f"{pin}, India"
    print(f"[GEOCODE] Looking up PIN '{pin}' via Nominatim...")
    loc = geolocator.geocode(q, timeout=15)
    if loc is None:
        raise RuntimeError(f"Geocoding failed for PIN {pin}.")
    print(f"[GEOCODE] Found location: {loc.address} (lat={loc.latitude:.6f}, lon={loc.longitude:.6f})")
    return float(loc.latitude), float(loc.longitude)

def _parse_depth_label_to_bottom_mm(label):
    m = re.match(r"^\s*\d+\s*-\s*(\d+)\s*cm\s*$", label)
    if m:
        bottom_cm = int(m.group(1))
        return bottom_cm * 10  # cm -> mm
    return None

def fetch_soilgrids_properties(lat, lon, properties=("sand","silt","clay","soc","bdod"), depths=SOILGRIDS_DEPTHS):
    """Query SoilGrids; returns list of dicts per depth (properties as floats)."""
    print(f"[SOIL] Fetching SoilGrids properties for lat={lat:.4f}, lon={lon:.4f} ...")
    params = {
        "lon": lon,
        "lat": lat,
        "depths": ",".join(depths),
        "properties": ",".join(properties)
    }
    r = requests.get(SOILGRIDS_BASE, params=params, timeout=60)
    r.raise_for_status()
    j = r.json()
    props_obj = j.get("properties", {}) or {}

    prop_entries = {}
    if isinstance(props_obj, dict):
        for p in properties:
            if p in props_obj:
                prop_entries[p] = props_obj[p]
    if not prop_entries and isinstance(props_obj.get("layers"), list):
        for entry in props_obj["layers"]:
            name = entry.get("name")
            if name:
                prop_entries[name] = entry

    layers_out = []
    for depth_label in depths:
        row = {p: np.nan for p in properties}
        for p in properties:
            info = prop_entries.get(p)
            if not info:
                continue
            d_factor = None
            try:
                d_factor = info.get("unit_measure", {}).get("d_factor")
            except Exception:
                d_factor = None
            depth_list = info.get("depths") or info.get("layers") or []
            if isinstance(depth_list, list):
                for entry in depth_list:
                    label = entry.get("label")
                    if not label and entry.get("range"):
                        rng = entry["range"]
                        top = int(rng.get("top_depth", 0))
                        bottom = int(rng.get("bottom_depth", 0))
                        label = f"{top}-{bottom}cm"
                    if label == depth_label:
                        vals = entry.get("values", {}) or {}
                        meanv = None
                        if isinstance(vals, dict):
                            meanv = vals.get("mean") or vals.get("m") or None
                        elif isinstance(vals, (int, float, str)):
                            meanv = vals
                        if meanv is not None:
                            try:
                                val = float(meanv)
                                if d_factor:
                                    val = val / float(d_factor)
                                row[p] = val
                                break
                            except Exception:
                                pass
        for k in row:
            try:
                row[k] = float(row[k])
            except Exception:
                row[k] = np.nan
        layers_out.append(row)

    # heuristic: scale down weird large numbers
    for layer in layers_out:
        for k in list(layer.keys()):
            v = layer[k]
            if not np.isnan(v) and v > 100:
                layer[k] = v / 10.0

    print(f"[SOIL] Parsed SoilGrids layers: {layers_out}")
    return layers_out

def texture_to_dssat_class(sand, silt, clay):
    if np.isnan(sand) or np.isnan(silt) or np.isnan(clay):
        return "SIL"
    if clay >= 35:
        return "CL"
    if sand >= 70:
        return "SL"
    if silt >= 50:
        return "SIL"
    if sand > clay and sand > silt:
        return "SL"
    return "L"

def write_simple_sol(layers, lat, lon, out_path, depth_labels=SOILGRIDS_DEPTHS):
    print(f"[SOL] Writing .SOL file to: {out_path} ...")
    def compute_mm_depths(labels, n_layers):
        mm = []
        for i, lab in enumerate(labels[:n_layers]):
            bottom_mm = _parse_depth_label_to_bottom_mm(lab)
            if bottom_mm is None:
                seq = [50, 150, 300, 600, 1200]
                bottom_mm = seq[i] if i < len(seq) else seq[-1]
            mm.append(bottom_mm)
        while len(mm) < n_layers:
            mm.append(mm[-1] * 2)
        return mm

    with open(out_path, "w") as f:
        f.write("*SOILS: Auto-generated from SoilGrids (approximate)\n")
        f.write("@SITE        COUNTRY  LAT     LONG    SCS FAMILY\n")
        f.write(f"AUTO_{int(time.time())}    INDIA    {lat:.4f}  {lon:.4f}  Generated\n\n")
        f.write("@SLB  SLLL  SDUL  SSAT  SBDM  SLOC  SAND  SILT  CLAY\n")
        n_layers = len(layers)
        mm_depths = compute_mm_depths(depth_labels, n_layers)
        for i, layer in enumerate(layers):
            sand = layer.get("sand", np.nan)
            silt = layer.get("silt", np.nan)
            clay = layer.get("clay", np.nan)
            bdod = layer.get("bdod", np.nan)
            soc = layer.get("soc", np.nan)

            if not np.isnan(soc):
                soc_pct = soc / 10.0 if soc > 10 else soc
            else:
                soc_pct = 0.5

            ssat = 0.45
            if not np.isnan(bdod) and bdod > 0:
                ssat = max(0.25, min(0.60, 1.0 - bdod / 2.65))

            if np.isnan(clay):
                slll = 0.10
            else:
                slll = 0.08 + 0.001 * float(clay)
                slll = max(0.05, min(0.30, slll))

            sdul = min(ssat - 0.02, slll + 0.15)
            bd = bdod if not np.isnan(bdod) and bdod > 0 else 1.30

            f.write(f"{mm_depths[i]:4d}  {slll:.3f}  {sdul:.3f}  {ssat:.3f}  {bd:.3f}  {soc_pct:.3f}  "
                    f"{(sand if not np.isnan(sand) else 0):5.1f}  {(silt if not np.isnan(silt) else 0):5.1f}  {(clay if not np.isnan(clay) else 0):5.1f}\n")
    print(f"[SOL] .SOL file written: {out_path}")
    return out_path

# ---------- Helpers (weather) ----------
def sanitize_rhum(s):
    s2 = s.copy()
    s2 = s2.ffill().bfill().fillna(60.0)
    s2 = s2.clip(0.0, 100.0)
    return s2

def fetch_open_meteo_range(lat, lon, start_date, end_date, timezone="Asia/Kolkata"):
    sd = pd.to_datetime(start_date).strftime("%Y-%m-%d")
    ed = pd.to_datetime(end_date).strftime("%Y-%m-%d")
    print(f"[WEATHER] Fetching Open-Meteo archive: {sd} -> {ed}")
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": sd,
        "end_date": ed,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,shortwave_radiation_sum",
        "hourly": "relativehumidity_2m",
        "timezone": timezone
    }
    r = requests.get(OPENMETEO_ARCHIVE, params=params, timeout=30)
    r.raise_for_status()
    j = r.json()
    daily = j.get("daily", {})
    if not daily:
        print("[WEATHER] Open-Meteo returned no daily data for requested range.")
        return pd.DataFrame(columns=["TMIN","TMAX","RAIN","SRAD","RHUM"])
    dates = pd.to_datetime(daily["time"])
    df = pd.DataFrame(index=dates)
    df["TMAX"] = np.array(daily.get("temperature_2m_max", [np.nan]*len(dates)))
    df["TMIN"] = np.array(daily.get("temperature_2m_min", [np.nan]*len(dates)))
    df["RAIN"] = np.array(daily.get("precipitation_sum", [0.0]*len(dates)))
    df["SRAD"] = np.array(daily.get("shortwave_radiation_sum", [np.nan]*len(dates)))
    rh_hourly = j.get("hourly", {}).get("relativehumidity_2m")
    hour_times = j.get("hourly", {}).get("time")
    if rh_hourly and hour_times:
        try:
            rh_ser = pd.Series(rh_hourly, index=pd.to_datetime(hour_times))
            rh_daily = rh_ser.resample("D").mean()
            df["RHUM"] = rh_daily.reindex(df.index).values
        except Exception:
            df["RHUM"] = np.nan
    else:
        df["RHUM"] = np.nan
    df["RHUM"] = sanitize_rhum(df["RHUM"])
    print(f"[WEATHER] Retrieved {len(df)} days of archive weather.")
    return df

def fetch_open_meteo_forecast(lat, lon, days=7, timezone="Asia/Kolkata"):
    print(f"[WEATHER] Fetching Open-Meteo forecast ({days} days)...")
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,shortwave_radiation_sum",
        "hourly": "relativehumidity_2m",
        "timezone": timezone,
    }
    r = requests.get(OPENMETEO_FORECAST, params=params, timeout=20)
    r.raise_for_status()
    j = r.json()
    daily = j.get("daily", {})
    if not daily:
        print("[WEATHER] Open-Meteo forecast returned no daily data.")
        return pd.DataFrame(columns=["TMIN","TMAX","RAIN","SRAD","RHUM"])
    dates = pd.to_datetime(daily["time"])
    df = pd.DataFrame(index=dates)
    df["TMAX"] = np.array(daily.get("temperature_2m_max", [np.nan]*len(dates)))
    df["TMIN"] = np.array(daily.get("temperature_2m_min", [np.nan]*len(dates)))
    df["RAIN"] = np.array(daily.get("precipitation_sum", [0.0]*len(dates)))
    df["SRAD"] = np.array(daily.get("shortwave_radiation_sum", [np.nan]*len(dates)))
    rh_hourly = j.get("hourly", {}).get("relativehumidity_2m")
    hour_times = j.get("hourly", {}).get("time")
    if rh_hourly and hour_times:
        try:
            rh_ser = pd.Series(rh_hourly, index=pd.to_datetime(hour_times))
            rh_daily = rh_ser.resample("D").mean()
            df["RHUM"] = rh_daily.reindex(df.index).values
        except Exception:
            df["RHUM"] = np.nan
    else:
        df["RHUM"] = np.nan
    df["RHUM"] = sanitize_rhum(df["RHUM"])
    if len(df) > days:
        df = df.iloc[:days]
    print(f"[WEATHER] Retrieved {len(df)} days of forecast.")
    return df

def build_tmy_from_past_years(lat, lon, start_date, end_date, years=5):
    print(f"[WEATHER] Building TMY from last {years} years for {start_date.date()} -> {end_date.date()} ...")
    frames = []
    today = datetime.now().date()
    for y in range(today.year - years, today.year):
        try:
            def safe_replace(d, ynew):
                try:
                    return d.replace(year=ynew)
                except ValueError:
                    return d.replace(year=ynew, day=28)
            start_y = safe_replace(start_date, y)
            end_y = safe_replace(end_date, y)
            df = fetch_open_meteo_range(lat, lon, start_y, end_y)
            if not df.empty:
                frames.append(df)
        except Exception:
            continue
    if not frames:
        print("[WEATHER] No historical frames found; using synthetic default weather for TMY.")
        days = (end_date - start_date).days + 1
        dates = pd.date_range(start_date, periods=days, freq="D")
        df = pd.DataFrame({
            "TMIN": np.repeat(18.0, len(dates)),
            "TMAX": np.repeat(30.0, len(dates)),
            "RAIN": np.repeat(0.1, len(dates)),
            "SRAD": np.repeat(18.0, len(dates)),
            "RHUM": np.repeat(60.0, len(dates)),
        }, index=dates)
        return df
    target_index = pd.date_range(start_date, end_date, freq="D")
    stacked = []
    for f in frames:
        vals = {}
        for col in ["TMIN","TMAX","RAIN","SRAD","RHUM"]:
            arr = f[col].to_numpy()
            tlen = len(target_index)
            alen = len(arr)
            if alen >= tlen:
                arr2 = arr[:tlen]
            else:
                arr2 = np.concatenate([arr, np.repeat(arr[-1], tlen - alen)])
            vals[col] = arr2
        df2 = pd.DataFrame(vals, index=target_index)
        stacked.append(df2)
    avg = pd.concat(stacked).groupby(level=0).mean()
    avg["RHUM"] = sanitize_rhum(avg["RHUM"])
    print(f"[WEATHER] Built TMY with {len(avg)} days.")
    return avg

def stitch_weather(past_df, forecast_df, tmy_df):
    print("[WEATHER] Stitching past + forecast + TMY into a continuous daily series...")
    parts = []
    if past_df is not None and not past_df.empty:
        parts.append(past_df)
    if forecast_df is not None and not forecast_df.empty:
        parts.append(forecast_df)
    if tmy_df is not None and not tmy_df.empty:
        parts.append(tmy_df)
    if not parts:
        raise RuntimeError("No weather data available to stitch.")
    df = pd.concat(parts)
    df = df[~df.index.duplicated(keep="first")]
    df = df.sort_index()
    df = df.asfreq("D")
    df = df.ffill().bfill()
    df["RHUM"] = sanitize_rhum(df["RHUM"])
    print(f"[WEATHER] Stitched weather has {len(df)} days from {df.index.min().date()} -> {df.index.max().date()}.")
    return df

# ---------- Management & maturity helpers ----------
def random_management(planting_date):
    print("[MGMT] Creating default/randomized Management object.")
    m = Management(planting_date=planting_date)
    try:
        from numpy.random import default_rng
        rng = default_rng()
        rs = int(rng.integers(20, 46))
        if hasattr(m, "planting_details"):
            try:
                m.planting_details.update({"PLRS": rs})
            except Exception:
                try:
                    m.planting_details["PLRS"] = rs
                except Exception:
                    pass
    except Exception:
        pass
    print(f"[MGMT] Management prepared (PLRS randomised if supported).")
    return m

def parse_mdat_from_summary(summary_df, planting_date):
    try:
        if summary_df is None or summary_df.empty:
            return None
        if "MDAT" not in summary_df.columns:
            return None
        raw = summary_df["MDAT"].iloc[0]
        if pd.isna(raw):
            return None
        m = int(raw)
        if 1 <= m <= 366:
            return datetime(planting_date.year, 1, 1) + timedelta(days=m-1)
        s = str(m)
        if len(s) == 8:
            try:
                return datetime.strptime(s, "%Y%m%d")
            except Exception:
                pass
    except Exception:
        pass
    return None

def detect_maturity_from_plantgro(plantgro_df, planting_date, crop_name):
    df = plantgro_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        if "DATE" in df.columns:
            df.index = pd.to_datetime(df["DATE"])
        else:
            df.index = pd.date_range(planting_date - timedelta(days=WARMUP_DAYS), periods=len(df), freq="D")
    if "GSTD" in df.columns:
        gst = df["GSTD"].dropna()
        if not gst.empty:
            max_stage = gst.max()
            candidates = df[df["GSTD"] == max_stage]
            if not candidates.empty:
                return (candidates.index[-1], candidates.iloc[-1])
    target_col = "HWAD" if crop_name.lower() == "wheat" else "GWAD"
    if target_col in df.columns:
        s = df[target_col].dropna()
        if not s.empty:
            final_val = float(s.iloc[-1])
            if final_val > 0:
                threshold = final_val * 0.995
                idxs = s[s >= threshold].index
                if len(idxs) > 0:
                    idx = idxs[0]
                    return (idx, df.loc[idx])
                else:
                    return (s.index[-1], df.loc[s.index[-1]])
    if not df.empty:
        idx = df.dropna(how="all").index[-1]
        return (idx, df.loc[idx])
    return (None, None)

# ---------- Orchestrator ----------
def run_for_farmer(pin=None, latlon=None, planting_date_str=None, crop_name="wheat", out_base=OUT_BASE):
    """Top-level function to run the whole pipeline once. Returns dict summary."""
    os.makedirs(out_base, exist_ok=True)
    csv_path = None
    wth_out = None
    sol_path = None

    try:
        print("[RUN] Starting pipeline...")
        # 1) geocode or use latlon
        if pin:
            try:
                lat, lon = geocode_pin(pin)
            except Exception as e:
                print("[GEOCODE] Geocoding failed; falling back to approximate default coordinates for the PIN. Error:", e)
                lat, lon = 26.90, 75.90
        elif latlon:
            lat, lon = float(latlon[0]), float(latlon[1])
            print(f"[INPUT] Using provided lat/lon {lat:.4f}, {lon:.4f}")
        else:
            raise ValueError("Either pin or latlon must be provided.")

        # 2) soil (SoilGrids)
        try:
            layers = fetch_soilgrids_properties(lat, lon, properties=("sand","silt","clay","soc","bdod"), depths=SOILGRIDS_DEPTHS)
            topvals = layers[0] if layers else None
            if topvals is None or all(np.isnan(v) for v in topvals.values()):
                raise RuntimeError("SoilGrids returned empty for top layer.")
            print("[SOIL] SoilGrids fetch successful.")
        except Exception as e:
            print("[SOIL] SoilGrid fetch failed; using fallback generic soil. Error:", e)
            layers = [{"sand":30.0,"silt":40.0,"clay":30.0,"bdod":1.35,"soc":0.6} for _ in range(3)]

        sol_name = f"soil_{int(time.time())}.SOL"
        sol_dir = os.path.join(out_base, "sol_files")
        os.makedirs(sol_dir, exist_ok=True)
        sol_path = os.path.join(sol_dir, sol_name)
        try:
            write_simple_sol(layers, lat, lon, sol_path)
        except Exception as e:
            print("[SOL] Warning: failed to write SOL file:", e)

        # 3) weather: past (planting-warmup -> yesterday), forecast (7d), then TMY to cover growth
        planting_date = datetime.strptime(planting_date_str, "%Y-%m-%d")
        today = datetime.now().date()
        yesterday = today - timedelta(days=1)
        past_start = (planting_date.date() - timedelta(days=WARMUP_DAYS))

        past_df = pd.DataFrame()
        if past_start <= yesterday:
            try:
                past_df = fetch_open_meteo_range(lat, lon, past_start, yesterday)
                print(f"[WEATHER] Fetched past weather: {past_df.index.min().date()} -> {past_df.index.max().date()} ({len(past_df)} days)")
            except Exception as e:
                print("[WEATHER] Warning: failed to fetch past weather:", e)
                past_df = pd.DataFrame()
        else:
            print("[WEATHER] No past weather fetch required (planting in the future + warmup).")

        try:
            forecast_df = fetch_open_meteo_forecast(lat, lon, days=7)
            if not forecast_df.empty:
                print(f"[WEATHER] Forecast fetched: {forecast_df.index.min().date()} -> {forecast_df.index.max().date()}")
            else:
                print("[WEATHER] Forecast empty.")
        except Exception as e:
            print("[WEATHER] Warning: failed to fetch forecast:", e)
            forecast_df = pd.DataFrame()

        forecast_end = forecast_df.index.max().date() if not forecast_df.empty else (datetime.now().date() + timedelta(days=6))
        tmy_start = forecast_end + timedelta(days=1)
        tmy_end = planting_date.date() + timedelta(days=DEFAULT_GROWTH_DAYS)
        tmy_df = pd.DataFrame()
        if tmy_start <= tmy_end:
            try:
                tmy_df = build_tmy_from_past_years(lat, lon, datetime.combine(tmy_start, datetime.min.time()), datetime.combine(tmy_end, datetime.min.time()))
                print("[WEATHER] TMY built successfully.")
            except Exception as e:
                print("[WEATHER] Warning: failed to build TMY:", e)
                tmy_df = pd.DataFrame()
        else:
            print("[WEATHER] No TMY needed (forecast covers the required period).")

        # stitch weather
        try:
            stitched = stitch_weather(past_df, forecast_df, tmy_df)
            # Ensure coverage at least planting -> planting+DEFAULT_GROWTH_DAYS
            needed_end = planting_date.date() + timedelta(days=DEFAULT_GROWTH_DAYS)
            if stitched.index.max().date() < needed_end:
                add_days = (needed_end - stitched.index.max().date()).days
                extra_idx = pd.date_range(stitched.index.max() + timedelta(days=1), periods=add_days, freq="D")
                last_row = stitched.iloc[-1]
                extra = pd.DataFrame([last_row.values]*add_days, index=extra_idx, columns=stitched.columns)
                stitched = pd.concat([stitched, extra])
                print(f"[WEATHER] Extended stitched weather by {add_days} days to cover simulation horizon.")
            # save stitched CSV
            wth_out = os.path.join(out_base, "wth_files", f"weather_{lat:.4f}_{lon:.4f}_{planting_date_str}.csv")
            stitched.to_csv(wth_out, index=True)
            print(f"[WEATHER] Saved stitched weather CSV: {wth_out}")
        except Exception as e:
            print("[WEATHER] ERROR: could not stitch weather:", e)
            # fallback to a generated random weather for simulation continuity:
            print("[WEATHER] Falling back to generated random weather.")
            total_days = WARMUP_DAYS + DEFAULT_GROWTH_DAYS
            start_date = planting_date - timedelta(days=WARMUP_DAYS)
            dates = pd.date_range(start_date, periods=total_days, freq="D")
            rng = np.random.default_rng()
            stitched = pd.DataFrame({
                "TMIN": np.round(rng.uniform(12,20,size=total_days),2),
                "TMAX": np.round(rng.uniform(25,35,size=total_days),2),
                "SRAD": np.round(rng.uniform(12,28,size=total_days),2),
                "RAIN": np.round(np.where(rng.random(size=total_days) < 0.25, rng.exponential(scale=6.0,size=total_days), 0.0),3),
                "RHUM": np.round(rng.uniform(40,95,size=total_days),1)
            }, index=dates)
            wth_out = os.path.join(out_base, "wth_files", f"weather_random_{int(time.time())}.csv")
            stitched.to_csv(wth_out, index=True)
            print(f"[WEATHER] Saved fallback random weather CSV: {wth_out}")

        # 4) prepare DSSAT objects
        mapping = {"TMIN":"TMIN","TMAX":"TMAX","RAIN":"RAIN","SRAD":"SRAD","RHUM":"RHUM"}
        print("[DSSAT] Building Weather object for DSSAT...")
        weather_obj = Weather(stitched, mapping, lat, lon, elev=0)
        print("[DSSAT] Weather object built.")

        # SoilProfile: create from topsoil class (safe default)
        try:
            top = layers[0] if layers else {"sand":30,"silt":40,"clay":30}
            dssat_soil_class = texture_to_dssat_class(top.get("sand",np.nan), top.get("silt",np.nan), top.get("clay",np.nan))
            soil = SoilProfile(default_class=dssat_soil_class)
            print(f"[DSSAT] SoilProfile created with default_class='{dssat_soil_class}'.")
        except Exception:
            soil = SoilProfile(default_class="SIL")
            print("[DSSAT] SoilProfile creation failed; used default_class='SIL'.")

        management = random_management(planting_date)

        # 5) run DSSAT once
        dssat = DSSAT()
        try:
            print(f"[DSSAT] Running DSSAT for crop='{crop_name}', planting_date='{planting_date_str}' ...")
            dssat.run(soil, weather_obj, Crop(crop_name), management)
            print("[DSSAT] DSSAT run completed successfully.")
        except Exception:
            print("[DSSAT] DSSAT run failed; traceback:")
            traceback.print_exc()
            try:
                dssat.close()
            except Exception:
                pass
            return {"ok": False, "error": "DSSAT run failed. See logs."}

        # 6) parse outputs
        print("[DSSAT] Parsing outputs...")
        summary_df = dssat.output.get("Summary")
        plantgro = dssat.output.get("PlantGro")
        maturity_date = None
        maturity_row = None
        try:
            mdate_from_summary = parse_mdat_from_summary(summary_df, planting_date)
        except Exception:
            mdate_from_summary = None

        if mdate_from_summary is not None:
            maturity_date = mdate_from_summary
            try:
                if plantgro is not None and not plantgro.empty:
                    doy = maturity_date.timetuple().tm_yday
                    matches = [idx for idx in plantgro.index if idx.timetuple().tm_yday == doy]
                    if matches:
                        maturity_row = plantgro.loc[matches[-1]]
                    else:
                        diffs = np.abs((plantgro.index - maturity_date).astype('timedelta64[D]'))
                        pos = int(diffs.argmin())
                        maturity_row = plantgro.iloc[pos]
            except Exception:
                maturity_row = None
        else:
            if plantgro is not None:
                maturity_date, maturity_row = detect_maturity_from_plantgro(plantgro, planting_date, crop_name)

        # yield extraction
        predicted_yield = None
        target_col = "HWAD" if crop_name.lower() == "wheat" else "GWAD"
        if plantgro is not None and target_col in plantgro.columns:
            try:
                s = plantgro[target_col].dropna()
                if not s.empty:
                    predicted_yield = float(s.iloc[-1])
            except Exception:
                predicted_yield = None

        if predicted_yield is None and summary_df is not None:
            for k in ("YIELD","YLD","GYLD","GWAD","HWAD"):
                if k in summary_df.columns:
                    try:
                        v = summary_df[k].iloc[0]
                        predicted_yield = float(v)
                        break
                    except Exception:
                        pass

        print("[DSSAT] Output parsing complete.")

        # save outputs: PlantGro CSV + plots
        if plantgro is not None and not plantgro.empty:
            # ensure datetime index
            if not pd.api.types.is_datetime64_any_dtype(plantgro.index):
                if "DATE" in plantgro.columns:
                    plantgro.index = pd.to_datetime(plantgro["DATE"])
                else:
                    plantgro.index = pd.date_range(planting_date - timedelta(days=WARMUP_DAYS), periods=len(plantgro), freq="D")
            csv_path = os.path.join(out_base, "csv", f"PlantGro_{crop_name}_{lat:.4f}_{lon:.4f}_{planting_date_str}.csv")
            plantgro.to_csv(csv_path, index=True)
            print(f"[OUTPUT] Saved PlantGro CSV: {csv_path}")
            preferred = [c for c in ("HWAD","GWAD") if c in plantgro.columns]
            if not preferred:
                numeric_cols = plantgro.select_dtypes(include=[np.number]).columns.tolist()
                preferred = numeric_cols[:3]
            for var in preferred:
                try:
                    plt.figure(figsize=(10,4))
                    ax = plantgro[var].plot(title=f"{crop_name} - {var} over time")
                    ax.set_xlabel("Date"); ax.set_ylabel(var)
                    png = os.path.join(out_base,"plots", f"{crop_name}_{var}_{int(time.time())}.png")
                    plt.tight_layout(); plt.savefig(png); plt.close()
                    print(f"[OUTPUT] Saved plot: {png}")
                except Exception as ex:
                    print("[OUTPUT] Plot failed for", var, ex)
        else:
            print("[OUTPUT] Warning: no PlantGro output to save/plot.")

        # close dssat
        try:
            dssat.close()
            print("[DSSAT] DSSAT closed.")
        except Exception:
            pass

        # Prepare summary
        summary = {
            "ok": True,
            "lat": lat,
            "lon": lon,
            "crop": crop_name,
            "planting_date": planting_date_str,
            "predicted_yield_kg_ha": predicted_yield,
            "predicted_maturity_date": maturity_date.strftime("%Y-%m-%d") if maturity_date is not None else None,
            "plantgro_csv": csv_path if csv_path else None,
            "weather_csv": wth_out if wth_out else None,
            "soil_sol": sol_path if sol_path and os.path.exists(sol_path) else None
        }
        # save summary json
        summary_path = os.path.join(out_base, f"summary_{int(time.time())}.json")
        with open(summary_path, "w") as jf:
            json.dump(summary, jf, indent=2)
        print(f"[OUTPUT] Saved pipeline summary JSON: {summary_path}")

        # print output lines for farmer
        print("\n===== RESULT =====")
        if summary["predicted_maturity_date"]:
            print("Predicted maturity date:", summary["predicted_maturity_date"])
        else:
            print("Predicted maturity date: (not available)")
        if summary["predicted_yield_kg_ha"] is not None:
            print("Predicted yield (kg/ha):", f"{summary['predicted_yield_kg_ha']:.2f}")
        else:
            print("Predicted yield: (not available)")

        print("[RUN] Pipeline finished successfully.")
        return summary

    except Exception as e:
        print("[RUN] Pipeline error:", e)
        traceback.print_exc()
        return {"ok": False, "error": str(e)}

# ---------- CLI ----------
def main():
    parser = argparse.ArgumentParser(description="Unified DSSAT orchestrator (verbose)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pin", help="Indian PIN code to geocode")
    group.add_argument("--latlon", nargs=2, type=float, metavar=("LAT","LON"), help="Latitude and longitude")
    parser.add_argument("--crop", required=True, help="Crop name recognizable by DSSATTools (e.g. wheat, rice)")
    parser.add_argument("--planting", required=True, help="Planting date YYYY-MM-DD")
    parser.add_argument("--out", default=OUT_BASE, help="Output base folder")
    args = parser.parse_args()

    if args.pin:
        summary = run_for_farmer(pin=args.pin, planting_date_str=args.planting, crop_name=args.crop, out_base=args.out)
    else:
        summary = run_for_farmer(latlon=args.latlon, planting_date_str=args.planting, crop_name=args.crop, out_base=args.out)

    if not summary.get("ok"):
        print("\nPipeline failed. See logs printed above.")
    else:
        print("\nPipeline succeeded. Outputs saved under:", args.out)

if __name__ == "__main__":
    main()