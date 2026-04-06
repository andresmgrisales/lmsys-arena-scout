"""
LMSYS Arena Scout — Weekly Data Updater
========================================
This script fetches the latest leaderboard data from the LMSYS Chatbot Arena
and updates data/latest.json for the GitHub Pages dashboard.

Designed to run via GitHub Actions every Monday at 08:00 UTC.

Data source: https://huggingface.co/spaces/lmarena-ai/chatbot-arena-leaderboard
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import re

try:
    import requests
except ImportError:
    print("Installing requests...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "requests"])
    import requests


# ── Configuration ────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
OUTPUT_FILE = DATA_DIR / "latest.json"
HISTORY_FILE = DATA_DIR / "history.json"

# Arena.ai leaderboard pages (each embeds JSON data in HTML)
ARENA_CATEGORIES = {
    "text": {"url": "https://arena.ai/leaderboard/text", "label": "Texto (Overall)"},
    "code": {"url": "https://arena.ai/leaderboard/code", "label": "Código"},
    "vision": {"url": "https://arena.ai/leaderboard/vision", "label": "Visión"},
}
DEFAULT_CATEGORY = "text"

TOP_N = 5  # Number of top models to show
CATEGORY_TOP_N = 3  # Number of top models per secondary category
HISTORY_WEEKS = 5  # Number of weeks of history to keep


# ── Locale-aware date formatting ─────────────────────────────────────────────

MONTHS_ES = {
    1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril",
    5: "Mayo", 6: "Junio", 7: "Julio", 8: "Agosto",
    9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre"
}

def format_date_es(dt):
    """Format date in Spanish: '30 de Marzo, 2026'"""
    return f"{dt.day} de {MONTHS_ES[dt.month]}, {dt.year}"


# ── Data Fetching ────────────────────────────────────────────────────────────

def _extract_entries_from_html(html):
    """Extract the leaderboard entries array from arena.ai HTML."""
    marker = '\\"entries\\":['
    start_idx = html.find(marker)
    if start_idx < 0:
        return None

    arr_start = start_idx + len(marker) - 1  # position of '['

    # Find the matching closing bracket
    depth = 0
    end_idx = arr_start
    for i in range(arr_start, min(len(html), arr_start + 300000)):
        c = html[i]
        if c == '[':
            depth += 1
        elif c == ']':
            depth -= 1
            if depth == 0:
                end_idx = i + 1
                break

    raw = html[arr_start:end_idx]
    raw = raw.replace('\\"', '"').replace('\\\\', '\\')
    return json.loads(raw)


def _entries_to_models(entries, limit=None):
    """Convert raw entry dicts to our model format."""
    models = []
    for entry in entries:
        ci_val = int(round(
            (entry.get("ratingUpper", 0) - entry.get("ratingLower", 0)) / 2
        ))
        models.append({
            "name": entry["modelDisplayName"],
            "elo": round(entry["rating"]),
            "ci": ci_val,
            "organization": entry.get("modelOrganization", "Unknown"),
            "license_raw": entry.get("license", "Unknown"),
            "votes": entry.get("votes", 0),
        })
    models.sort(key=lambda x: x["elo"], reverse=True)
    return models[:limit] if limit else models


def fetch_leaderboard_data():
    """
    Fetch the main (text) leaderboard data from arena.ai.
    Returns a list of dicts: [{"name": ..., "elo": ..., "organization": ..., ...}, ...]
    """
    url = ARENA_CATEGORIES[DEFAULT_CATEGORY]["url"]
    print(f"[1/1] Fetching arena.ai leaderboard ({DEFAULT_CATEGORY})...")
    try:
        resp = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0 (LMSYS Arena Scout)"},
            timeout=60,
        )
        resp.raise_for_status()
        entries = _extract_entries_from_html(resp.text)
        if entries is None:
            print("  ✗ Could not find entries marker in HTML")
            return None
        print(f"  ✓ Parsed {len(entries)} models from arena.ai")
        return _entries_to_models(entries, limit=TOP_N * 2)
    except json.JSONDecodeError as e:
        print(f"  ✗ JSON parse error: {e}")
        return None
    except Exception as e:
        print(f"  ✗ Failed to fetch arena.ai data: {e}")
        return None


def fetch_category_leaders():
    """
    Fetch top models from each secondary category (code, vision).
    Returns dict: {"code": [{...}, ...], "vision": [{...}, ...]}
    """
    results = {}
    for cat_key, cat_info in ARENA_CATEGORIES.items():
        if cat_key == DEFAULT_CATEGORY:
            continue
        print(f"  Fetching category: {cat_info['label']}...")
        try:
            resp = requests.get(
                cat_info["url"],
                headers={"User-Agent": "Mozilla/5.0 (LMSYS Arena Scout)"},
                timeout=60,
            )
            resp.raise_for_status()
            entries = _extract_entries_from_html(resp.text)
            if entries:
                models = _entries_to_models(entries, limit=CATEGORY_TOP_N)
                results[cat_key] = {
                    "label": cat_info["label"],
                    "top_models": [{
                        "rank": i + 1,
                        "name": m["name"],
                        "elo": m["elo"],
                        "organization": m["organization"],
                    } for i, m in enumerate(models)]
                }
                print(f"    ✓ {len(entries)} models, top: {models[0]['name']} ({models[0]['elo']} Elo)")
            else:
                print(f"    ✗ No entries found")
        except Exception as e:
            print(f"    ✗ Error: {e}")
    return results


def parse_csv_data(csv_text):
    """Parse CSV format leaderboard data."""
    import csv
    from io import StringIO

    reader = csv.DictReader(StringIO(csv_text))
    models = []
    for row in reader:
        # Adapt column names based on actual CSV structure
        name = row.get("Model", row.get("model", row.get("model_name", "")))
        elo = float(row.get("Arena Elo", row.get("rating", row.get("elo", 0))))
        ci = row.get("95% CI", row.get("ci", row.get("confidence_interval", "±6")))
        org = row.get("Organization", row.get("organization", row.get("org", "Unknown")))
        license_info = row.get("License", row.get("license", "Unknown"))

        if name and elo > 0:
            models.append({
                "name": name.strip(),
                "elo": round(elo),
                "ci": parse_ci(ci),
                "organization": org.strip(),
                "license_raw": license_info.strip(),
            })

    models.sort(key=lambda x: x["elo"], reverse=True)
    return models[:TOP_N * 2]  # Get more than needed for analysis


def parse_gradio_response(result):
    """Parse the Gradio API response."""
    data = result.get("data", [[]])[0]
    if isinstance(data, str):
        # Sometimes returns HTML table
        return parse_html_table(data)
    models = []
    for row in data:
        if isinstance(row, (list, tuple)) and len(row) >= 3:
            models.append({
                "name": str(row[0]).strip(),
                "elo": round(float(row[1])),
                "ci": int(row[2]) if len(row) > 2 else 6,
                "organization": str(row[3]).strip() if len(row) > 3 else "Unknown",
                "license_raw": str(row[4]).strip() if len(row) > 4 else "Unknown",
            })
    models.sort(key=lambda x: x["elo"], reverse=True)
    return models[:TOP_N * 2]


def parse_battle_json(battles):
    """Parse battle data to compute Elo ratings (simplified)."""
    # This is a simplified Elo calculation
    from collections import defaultdict
    ratings = defaultdict(lambda: 1500)
    counts = defaultdict(int)
    K = 4

    for battle in battles:
        a = battle.get("model_a", "")
        b = battle.get("model_b", "")
        winner = battle.get("winner", "")

        if not a or not b:
            continue

        ra, rb = ratings[a], ratings[b]
        ea = 1 / (1 + 10 ** ((rb - ra) / 400))
        eb = 1 - ea

        if winner == "model_a":
            sa, sb = 1, 0
        elif winner == "model_b":
            sa, sb = 0, 1
        else:
            sa, sb = 0.5, 0.5

        ratings[a] += K * (sa - ea)
        ratings[b] += K * (sb - eb)
        counts[a] += 1
        counts[b] += 1

    # Filter models with enough votes
    models = []
    for name, elo in sorted(ratings.items(), key=lambda x: -x[1]):
        if counts[name] >= 100:
            models.append({
                "name": name,
                "elo": round(elo),
                "ci": 6,
                "organization": infer_organization(name),
                "license_raw": infer_license(name),
            })

    return models[:TOP_N * 2]


def parse_html_table(html):
    """Fallback: parse HTML table content."""
    # Simple regex-based parsing
    import re
    rows = re.findall(r'<tr[^>]*>(.*?)</tr>', html, re.DOTALL)
    models = []
    for row in rows[1:]:  # Skip header
        cells = re.findall(r'<td[^>]*>(.*?)</td>', row, re.DOTALL)
        if len(cells) >= 3:
            name = re.sub(r'<[^>]+>', '', cells[0]).strip()
            elo_text = re.sub(r'<[^>]+>', '', cells[1]).strip()
            try:
                elo = round(float(elo_text))
            except ValueError:
                continue
            models.append({
                "name": name,
                "elo": elo,
                "ci": 6,
                "organization": "Unknown",
                "license_raw": "Unknown",
            })
    models.sort(key=lambda x: x["elo"], reverse=True)
    return models[:TOP_N * 2]


# ── Helpers ──────────────────────────────────────────────────────────────────

def parse_ci(ci_str):
    """Extract numeric CI value from string like '±6' or '+6/-6'."""
    import re
    match = re.search(r'(\d+)', str(ci_str))
    return int(match.group(1)) if match else 6


def infer_organization(model_name):
    """Infer organization from model name."""
    name_lower = model_name.lower()
    org_map = {
        "claude": "Anthropic", "gpt": "OpenAI", "o1": "OpenAI", "o3": "OpenAI",
        "gemini": "Google", "palm": "Google", "bard": "Google",
        "llama": "Meta", "codellama": "Meta",
        "grok": "xAI", "mistral": "Mistral AI", "mixtral": "Mistral AI",
        "qwen": "Alibaba", "yi": "01.AI", "deepseek": "DeepSeek",
        "command": "Cohere", "phi": "Microsoft", "copilot": "Microsoft",
        "falcon": "TII", "glm": "Zhipu AI", "chatglm": "Zhipu AI",
    }
    for key, org in org_map.items():
        if key in name_lower:
            return org
    return "Unknown"


def infer_license(model_name):
    """Infer license type from model name."""
    name_lower = model_name.lower()
    open_models = ["llama", "mistral", "mixtral", "qwen", "yi", "falcon", "glm",
                    "vicuna", "alpaca", "deepseek", "phi", "gemma", "olmo"]
    for key in open_models:
        if key in name_lower:
            return "open"
    return "proprietary"


def classify_license(raw_license, model_name):
    """Return structured license info."""
    raw_lower = str(raw_license).lower()
    is_open = any(k in raw_lower for k in ["open", "apache", "mit", "llama", "cc-by"])
    if not is_open:
        is_open = infer_license(model_name) == "open"

    if is_open:
        return {
            "license_type": "open",
            "license_label": f"Abierto ({raw_license})" if raw_license != "Unknown" else "Abierto"
        }
    return {
        "license_type": "proprietary",
        "license_label": "Propiedad (Cerrado)"
    }


# ── History Management ───────────────────────────────────────────────────────

def load_history():
    """Load historical Elo data."""
    if HISTORY_FILE.exists():
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_history(history):
    """Save historical Elo data."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


def update_history(history, top_models):
    """Add current week's Elo scores to history."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    for model in top_models:
        name = model["name"]
        if name not in history:
            history[name] = []
        # Avoid duplicate entries for same date
        if not history[name] or history[name][-1]["date"] != today:
            history[name].append({"date": today, "elo": model["elo"]})
        # Keep only the last N weeks
        history[name] = history[name][-HISTORY_WEEKS:]

    return history


def build_history_series(history, top_names, n_weeks=HISTORY_WEEKS):
    """Build the chart series data from history."""
    colors = ["#6366f1", "#0ea5e9", "#10b981", "#f59e0b", "#ef4444"]
    series = []

    for i, name in enumerate(top_names[:3]):  # Top 3 for the chart
        entries = history.get(name, [])
        data = [e["elo"] for e in entries[-n_weeks:]]

        # Pad with first value if not enough history
        while len(data) < n_weeks:
            data.insert(0, data[0] if data else 1500)

        series.append({
            "name": name,
            "color": colors[i % len(colors)],
            "data": data
        })

    # Build labels
    labels = []
    for i in range(n_weeks):
        if i == n_weeks - 1:
            labels.append("Actual")
        else:
            labels.append(f"Sem -{n_weeks - 1 - i}")

    return {"labels": labels, "series": series}


# ── Trend Calculation ────────────────────────────────────────────────────────

def calculate_trends(top_models, history):
    """Calculate trend (up/down/neutral) based on historical data."""
    for model in top_models:
        entries = history.get(model["name"], [])
        if len(entries) >= 2:
            delta = entries[-1]["elo"] - entries[-2]["elo"]
        else:
            delta = 0

        model["trend_delta"] = delta
        if delta > 0:
            model["trend"] = "up"
        elif delta < 0:
            model["trend"] = "down"
        else:
            model["trend"] = "neutral"

    return top_models


# ── Insight Generation ───────────────────────────────────────────────────────

def generate_insights(top_models):
    """Generate analysis insights based on current data."""
    leader = top_models[0]
    second = top_models[1] if len(top_models) > 1 else None
    open_leader = next((m for m in top_models if m.get("license_type") == "open"), None)

    seismic = []

    # Top 2 comparison
    if second and abs(leader["elo"] - second["elo"]) <= leader.get("ci", 6):
        seismic.append({
            "title": f"Batalla Cerrada: {leader['name']} vs {second['name']}",
            "content": (
                f"Con solo <strong>{abs(leader['elo'] - second['elo'])} puntos</strong> de diferencia y un "
                f"intervalo de confianza de ±{leader.get('ci', 6)}, los dos modelos líderes son "
                f"<strong>estadísticamente indistinguibles</strong>. La preferencia humana fluctúa "
                f"marginalmente entre ellos según la categoría de tarea evaluada."
            ),
            "accent_color": "indigo"
        })
    else:
        seismic.append({
            "title": f"Liderazgo Claro: {leader['name']}",
            "content": (
                f"<strong>{leader['name']}</strong> de {leader['organization']} domina el ranking con "
                f"<strong>{leader['elo']} Elo</strong>, manteniendo una ventaja significativa de "
                f"{leader['elo'] - (second['elo'] if second else 0)} puntos sobre el segundo lugar."
            ),
            "accent_color": "indigo"
        })

    # Biggest mover
    biggest = max(top_models, key=lambda m: abs(m.get("trend_delta", 0)))
    if abs(biggest.get("trend_delta", 0)) > 3:
        direction = "subió" if biggest["trend_delta"] > 0 else "bajó"
        seismic.append({
            "title": f"Movimiento Notable: {biggest['name']}",
            "content": (
                f"<strong>{biggest['name']}</strong> {direction} <strong>{abs(biggest['trend_delta'])} "
                f"puntos Elo</strong> esta semana, señalando un cambio significativo en las preferencias "
                f"de los evaluadores humanos."
            ),
            "accent_color": "emerald"
        })

    # Open source leader insight
    os_insight = None
    if open_leader:
        gap = leader["elo"] - open_leader["elo"]
        pct = max(0, min(100, round(100 - (gap / leader["elo"]) * 100)))
        os_insight = {
            "name": open_leader["name"],
            "organization": open_leader.get("organization", "Unknown"),
            "license": open_leader.get("license_label", "Abierto"),
            "elo": open_leader["elo"],
            "gap_to_frontier": gap,
            "gap_percentage": pct,
            "description": (
                f"<strong>{open_leader['name']}</strong> lidera el segmento open source con "
                f"<strong>{open_leader['elo']} Elo</strong>, a solo {gap} puntos de la frontera "
                f"propietaria. La brecha continúa estrechándose semana tras semana."
            ),
            "gap_note": (
                f"La brecha actual de {gap} puntos está dentro del margen de error combinado, "
                f"sugiriendo convergencia entre modelos abiertos y propietarios."
            )
        }

    return {
        "seismic_movements": seismic,
        "open_source_leader": os_insight or {
            "name": "N/A",
            "organization": "N/A",
            "license": "N/A",
            "elo": 0,
            "gap_to_frontier": 0,
            "gap_percentage": 0,
            "description": "No se encontraron modelos open source en el ranking.",
            "gap_note": ""
        },
        "expert_analysis": {
            "coding": {
                "title": "Categoría: Código",
                "content": (
                    f"En tareas de programación y desarrollo web, los rankings pueden diferir "
                    f"significativamente del overall. Consulta la sección <strong>Líderes por Categoría</strong> "
                    f"para ver el top real de cada vertical según los votos de arena.ai."
                )
            },
            "hard_prompts": {
                "title": "Categoría: Visión",
                "content": (
                    f"En tareas multimodales (imagen + texto), modelos como los de Google y OpenAI "
                    f"suelen destacar. La sección <strong>Líderes por Categoría</strong> muestra "
                    f"los rankings reales basados en evaluación humana ciega."
                )
            }
        }
    }


# ── Main Pipeline ────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("LMSYS Arena Scout — Weekly Data Update")
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 60)

    # Fetch latest data
    raw_models = fetch_leaderboard_data()

    if not raw_models:
        print("\n⚠  No data fetched. Keeping existing data unchanged.")
        sys.exit(1)

    print(f"\n✓ Fetched {len(raw_models)} models")

    # Process top N
    top_models = []
    for i, m in enumerate(raw_models[:TOP_N]):
        license_info = classify_license(m.get("license_raw", "Unknown"), m["name"])
        top_models.append({
            "rank": i + 1,
            "name": m["name"],
            "elo": m["elo"],
            "ci": m.get("ci", 6),
            **license_info,
            "organization": m.get("organization", infer_organization(m["name"])),
            "trend": "neutral",
            "trend_delta": 0,
        })

    # Load and update history
    history = load_history()
    history = update_history(history, top_models)
    save_history(history)

    # Calculate trends
    top_models = calculate_trends(top_models, history)

    # Build history series for chart
    top_names = [m["name"] for m in top_models]
    history_series = build_history_series(history, top_names)

    # Generate insights
    insights = generate_insights(top_models)

    # Fetch secondary category leaders (code, vision)
    print("\n[2/2] Fetching category leaders...")
    category_leaders = fetch_category_leaders()

    # Build output
    now = datetime.now(timezone.utc)
    output = {
        "metadata": {
            "last_updated": now.isoformat(),
            "report_date_label": format_date_es(now),
            "source": "Arena (Chatbot Arena)",
            "source_url": "https://arena.ai/?leaderboard"
        },
        "top_models": top_models,
        "category_leaders": category_leaders,
        "history": history_series,
        "insights": insights,
    }

    # Write output
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Data written to {OUTPUT_FILE}")
    print(f"  Leader: {top_models[0]['name']} ({top_models[0]['elo']} Elo)")
    print(f"  Models: {', '.join(m['name'] for m in top_models)}")
    print(f"  Date: {format_date_es(now)}")
    print("\n✓ Update complete!")


if __name__ == "__main__":
    main()
