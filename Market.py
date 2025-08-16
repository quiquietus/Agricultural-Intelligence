#!/usr/bin/env python3
"""
Market.py

Requirements:
    pip install requests

Usage:
    - Set DATA_GOV_API_KEY environment variable with your data.gov.in API key,
      or paste it when prompted.
    - Run: python mandi_price_nearby.py
    - Or pass CLI args: 
        python mandi_price_nearby.py --pincode 110001 --crop "wheat" --yield_kg 1000 --radius_km 50

What it does:
    1. Geocodes the provided PIN code to get a lat/lon (via Nominatim).
    2. Fetches mandi price records from data.gov.in Agmarknet dataset.
    3. Geocodes market names (market + district + state) and caches results to disk.
    4. Computes distance and filters markets inside radius_km.
    5. Shows price per unit (modal price) and estimated total for the provided yield.
Notes:
    - Nominatim usage policy: this script sleeps 1s between geocode requests to be polite.
    - data.gov.in API may limit the number of records per request; this script requests a large limit and will page if needed.
"""

import requests
import time
import math
import os
import json
import argparse
from urllib.parse import quote_plus
from dotenv import load_dotenv
load_dotenv()

# ---------- Config ----------
DATA_GOV_RESOURCE_ID = "9ef84268-d588-465a-a308-a864a43d0070"
DATA_GOV_API_URL = f"https://api.data.gov.in/resource/{DATA_GOV_RESOURCE_ID}"
GEOCODE_CACHE_FILE = "geocode_cache.json"
USER_AGENT = "mandi-price-nearby-script/1.0 (contact: you@example.com)"  # replace contact if you want

# ---------- Utilities ----------
def load_cache():
    try:
        with open(GEOCODE_CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_cache(cache):
    with open(GEOCODE_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

def haversine_km(lat1, lon1, lat2, lon2):
    # calculate great-circle distance between two points in km
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    c = 2*math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

# ---------- Geocoding (Nominatim) ----------
def nominatim_geocode(query, cache, sleep_secs=1.0):
    """Geocode using nominatim; cache results using the cache dict; returns (lat, lon) or None."""
    key = query.strip().lower()
    if key in cache:
        return cache[key]
    # Respect Nominatim policy: include a user agent and throttle requests
    url = f"https://nominatim.openstreetmap.org/search?format=json&q={quote_plus(query)}&limit=1&addressdetails=0"
    headers = {"User-Agent": USER_AGENT}
    try:
        resp = requests.get(url, headers=headers, timeout=20)
        if resp.status_code == 200:
            data = resp.json()
            if data:
                lat = float(data[0]["lat"])
                lon = float(data[0]["lon"])
                cache[key] = [lat, lon]
                save_cache(cache)
                time.sleep(sleep_secs)
                return (lat, lon)
            else:
                cache[key] = None
                save_cache(cache)
                time.sleep(sleep_secs)
                return None
        else:
            # on error, do not cache; raise or return None
            print(f"Geocode request failed for '{query}', status: {resp.status_code}")
            return None
    except Exception as e:
        print(f"Exception during geocode for '{query}': {e}")
        return None

def geocode_pincode(pincode, cache):
    # Use Nominatim to geocode "pincode, India"
    q = f"{pincode}, India"
    return nominatim_geocode(q, cache)

# ---------- Data.gov.in Agmarknet fetching ----------
def fetch_agmarknet_records(api_key, commodity_filter=None, max_records=5000):
    """
    Fetch records from data.gov.in. Returns list of records (dictionaries).
    commodity_filter: if provided, we'll use it to filter locally (dataset's filtering param depends on API).
    Note: This function requests `limit=max_records`. If you need more data, increase max_records.
    """
    params = {
        "api-key": api_key,
        "format": "json",
        "limit": max_records,
    }
    # Some datasets accept filters in the URL; to keep this general we will filter locally after fetching
    resp = requests.get(DATA_GOV_API_URL, params=params, headers={"User-Agent": USER_AGENT}, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if "records" in data:
        records = data["records"]
    elif "data" in data:
        records = data["data"]
    else:
        # try top-level array
        records = data if isinstance(data, list) else []
    # optional local filter on commodity
    if commodity_filter:
        cf = commodity_filter.strip().lower()
        filtered = []
        for r in records:
            comet = r.get("commodity") or r.get("Commodity") or r.get("COMMODITY") or ""
            if cf in comet.lower():
                filtered.append(r)
        return filtered
    return records

# ---------- Main logic ----------
def find_markets_within_radius(pincode, crop, yield_kg, radius_km, api_key, max_records=5000):
    cache = load_cache()
    farmer_loc = geocode_pincode(pincode, cache)
    if not farmer_loc:
        raise ValueError(f"Could not geocode pincode {pincode}. Try a different PIN or check internet.")
    farmer_lat, farmer_lon = farmer_loc
    print(f"Farmer location (lat,lon): {farmer_lat:.6f}, {farmer_lon:.6f}")

    print("Fetching mandi price records from data.gov.in (may take a few seconds)...")
    records = fetch_agmarknet_records(api_key, commodity_filter=None, max_records=max_records)
    print(f"Fetched {len(records)} records; filtering for commodity '{crop}' and extracting markets...")

    # Build a small structure keyed by market+district+state with best modal_price found among records.
    market_entries = {}  # key -> {market, district, state, modal_prices: [..] }
    crop_lc = crop.strip().lower()
    for r in records:
        commodity = (r.get("commodity") or r.get("Commodity") or "").strip()
        if commodity == "":
            continue
        if crop_lc not in commodity.lower():
            continue
        market = (r.get("market") or r.get("Market") or r.get("market_name") or "").strip()
        district = (r.get("district") or r.get("District") or "").strip()
        state = (r.get("state") or r.get("State") or "").strip()
        modal_price_raw = r.get("modal_price") or r.get("modalPrice") or r.get("Modal Price") or r.get("price")
        if not market:
            continue
        # try parse modal price to float
        try:
            modal_price = None
            if modal_price_raw is None:
                modal_price = None
            else:
                # some values may contain commas
                modal_price = float(str(modal_price_raw).replace(",", "").strip())
        except Exception:
            modal_price = None
        key = f"{market}||{district}||{state}"
        if key not in market_entries:
            market_entries[key] = {
                "market": market,
                "district": district,
                "state": state,
                "prices": []
            }
        if modal_price is not None:
            market_entries[key]["prices"].append(modal_price)

    print(f"Unique candidate markets for '{crop}': {len(market_entries)}")
    # Geocode each market (market,district,state,India), compute distance
    results = []
    total_markets = len(market_entries)
    i = 0
    for key, entry in market_entries.items():
        i += 1
        market = entry["market"]
        district = entry["district"]
        state = entry["state"]
        # build geocode query
        components = ", ".join([p for p in [market, district, state, "India"] if p])
        geocode_q = components
        loc = nominatim_geocode(geocode_q, cache)
        if not loc:
            # try a simpler geocode (market + district)
            loc = nominatim_geocode(f"{market}, {district}, India", cache)
        if not loc:
            # try market + state
            loc = nominatim_geocode(f"{market}, {state}, India", cache)
        if not loc:
            # skip if still unknown
            # print(f"Could not geocode market: {components}")
            continue
        lat, lon = loc
        dist = haversine_km(farmer_lat, farmer_lon, lat, lon)
        if dist <= radius_km:
            # aggregate price: choose average or modal or max. We'll report min/avg/max to give sense.
            prices = entry["prices"]
            if not prices:
                continue
            p_min = min(prices)
            p_max = max(prices)
            p_avg = sum(prices)/len(prices)
            # modal price in dataset is typically per quintal (100 kg). We will return the prices as-is but indicate unit may vary.
            results.append({
                "market": market,
                "district": district,
                "state": state,
                "lat": lat,
                "lon": lon,
                "distance_km": round(dist, 2),
                "price_min": round(p_min, 2),
                "price_avg": round(p_avg, 2),
                "price_max": round(p_max, 2),
                "num_price_points": len(prices)
            })
    # sort by price_avg descending
    results.sort(key=lambda x: x["price_avg"], reverse=True)
    # compute estimated value for farmer based on price per quintal: convert yield_kg -> quintals (1 quintal = 100 kg)
    yield_quintals = yield_kg / 100.0
    for r in results:
        r["estimated_total_at_avg"] = round(r["price_avg"] * yield_quintals, 2)
        r["estimated_total_at_min"] = round(r["price_min"] * yield_quintals, 2)
        r["estimated_total_at_max"] = round(r["price_max"] * yield_quintals, 2)

    return results

# ---------- CLI / Interactive ----------
def main():
    parser = argparse.ArgumentParser(description="Find mandi prices near a farmer's PIN code.")
    parser.add_argument("--pincode", "-p", type=str, help="Farmer's PIN code (India)")
    parser.add_argument("--crop", "-c", type=str, help="Crop/commodity name (e.g., wheat)")
    parser.add_argument("--yield_kg", "-y", type=float, help="Yield in kilograms (kg)")
    parser.add_argument("--radius_km", "-r", type=float, default=50.0, help="Search radius in kilometers")
    parser.add_argument("--api_key", type=str, help="data.gov.in API key (optional; will read from DATA_GOV_API_KEY env var if not supplied)")
    parser.add_argument("--max_records", type=int, default=5000, help="Max records to fetch from data.gov.in (increase if needed)")
    args = parser.parse_args()

    pincode = args.pincode or input("Enter farmer PIN code: ").strip()
    crop = args.crop or input("Enter crop name (e.g., wheat): ").strip()
    try:
        yield_kg = args.yield_kg if args.yield_kg is not None else float(input("Enter yield (kg): ").strip())
    except Exception:
        print("Invalid yield value.")
        return
    radius_km = args.radius_km if args.radius_km is not None else float(input("Enter radius (km): ").strip())
    api_key = args.api_key or os.environ.get("DATA_GOV_API_KEY") or input("Enter data.gov.in API key (or set DATA_GOV_API_KEY env var): ").strip()

    if not api_key:
        print("data.gov.in API key is required. Get one for free at data.gov.in and set DATA_GOV_API_KEY env var.")
        return

    try:
        results = find_markets_within_radius(pincode, crop, yield_kg, radius_km, api_key, max_records=args.max_records)
    except Exception as e:
        print("Error:", e)
        return

    if not results:
        print(f"No markets found within {radius_km} km for crop '{crop}'. Try increasing radius or check the crop name spelling.")
        return

    print("\nMarkets within radius (sorted by average price descending):\n")
    for idx, r in enumerate(results, start=1):
        print(f"{idx}. {r['market']}, {r['district']}, {r['state']} — {r['distance_km']} km away")
        print(f"    Prices (per unit reported): min {r['price_min']}, avg {r['price_avg']}, max {r['price_max']}  (based on {r['num_price_points']} data points)")
        print(f"    Estimated total for {yield_kg} kg -> min ₹{r['estimated_total_at_min']}, avg ₹{r['estimated_total_at_avg']}, max ₹{r['estimated_total_at_max']}")
        print(f"    Coordinates: {r['lat']:.6f}, {r['lon']:.6f}")
        print()

if __name__ == "__main__":
    main()
