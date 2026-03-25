"""
app.py — Folsom AQI Monitor · FLC Los Rios STEM Fair 2026
Live AQI forecast dashboard backed by FastAPI + LightGBM on Render.

Run locally:  streamlit run app.py
Deploy:       Push to GitHub → Streamlit Community Cloud
"""

import html
import os
import sys
import textwrap
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import plotly.graph_objects as go
import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# ── Page config — MUST be the very first Streamlit call ───────────────────────
st.set_page_config(
    page_title="Folsom AQI Monitor — FLC Los Rios STEM Fair 2026",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Constants ─────────────────────────────────────────────────────────────────

def _get_api_url() -> str:
    # Safely load the secrets inside a function to avoid crashing cold-starts
    url = os.getenv("API_URL", "")
    if not url:
        try:
            url = st.secrets.get("API_URL", "")
        except Exception:
            pass
    return url or "http://localhost:8000"

API_URL = _get_api_url()
TZ      = ZoneInfo("America/Los_Angeles")

COLORS = {
    "background":     "#0a0f1e",
    "card":           "#111827",
    "card_border":    "#1f2937",
    "text_primary":   "#f9fafb",
    "text_secondary": "#9ca3af",
    "text_muted":     "#6b7280",
}

AQI_COLORS = {
    "Good":                            "#00e400",
    "Moderate":                        "#ffff00",
    "Unhealthy for Sensitive Groups":  "#ff7e00",
    "Unhealthy":                       "#ff0000",
    "Very Unhealthy":                  "#8f3f97",
    "Hazardous":                       "#7e0023",
}

AQI_COLORS_RGBA = {
    "Good":                            "rgba(0,228,0,",
    "Moderate":                        "rgba(255,255,0,",
    "Unhealthy for Sensitive Groups":  "rgba(255,126,0,",
    "Unhealthy":                       "rgba(255,0,0,",
    "Very Unhealthy":                  "rgba(143,63,151,",
    "Hazardous":                       "rgba(126,0,35,",
}

ADVISORIES = {
    "Good":
        "Air quality is satisfactory. No health precautions needed.",
    "Moderate":
        "Unusually sensitive people should consider limiting prolonged outdoor exertion.",
    "Unhealthy for Sensitive Groups":
        "People with heart or lung disease, older adults, and children should reduce prolonged outdoor exertion.",
    "Unhealthy":
        "Everyone should reduce prolonged outdoor exertion. Sensitive groups should avoid outdoor exertion.",
    "Very Unhealthy":
        "Everyone should avoid prolonged outdoor exertion. Sensitive groups should remain indoors.",
    "Hazardous":
        "Health alert: Everyone should avoid all outdoor exertion. Remain indoors with windows closed.",
}

HORIZON_LABELS = {
    "6h":  "6 HOUR FORECAST",
    "12h": "12 HOUR FORECAST",
    "24h": "24 HOUR FORECAST",
    "48h": "48 HOUR FORECAST",
}

# ── Gemini AI config ──────────────────────────────────────────────────────────

from ai_layer import answer_question

def _get_gemini_key() -> str:
    """Retrieve GEMINI_API_KEY from Streamlit secrets or env to validate UI chatbox availability."""
    try:
        key = st.secrets.get("GEMINI_API_KEY", "")
        if key:
            return key
    except Exception:
        pass
    return os.environ.get("GEMINI_API_KEY", "")


# ── Global CSS ────────────────────────────────────────────────────────────────

def inject_css():
    st.html("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&display=swap');

    /* ── RESET: Strict Instrumentation Profile ── */
    html, body, [class*="css"] {
        font-family: 'JetBrains Mono', monospace !important;
        letter-spacing: -0.01em !important;
    }
    .stApp {
        background-color: #000000;
        background-image: none !important;
    }
    .block-container {
        padding: 0.5rem 1rem 1rem 1rem !important;
        max-width: 1100px !important;
        margin: 0 auto !important;
    }

    /* ── HIDE CHROME ── */
    #MainMenu, footer, header, .stDeployButton, [data-testid="stToolbar"] { visibility: hidden; display: none; }

    /* ── CARDS: Flat Matte Utility ── */
    .aqi-card {
        background: #000000;
        border: 1px solid #333333;
        border-radius: 2px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
    }

    .horizon-card {
        background: #000000;
        border-radius: 2px;
        padding: 0.5rem;
        text-align: left;
        border: 1px solid #222222;
        transition: border-color 0.1s ease;
    }
    .horizon-card:hover {
        border-color: #444444;
    }

    .uncertainty-box {
        position: relative;
        height: 6px;
        background: #111;
        border-radius: 1px;
        margin: 0.8rem 0 0.5rem 0;
        overflow: hidden;
    }
    .ci-range {
        position: absolute;
        height: 100%;
        background: #333;
        opacity: 0.6;
    }
    .point-prediction {
        position: absolute;
        height: 100%;
        width: 3px;
        background: #fff;
        box-shadow: 0 0 5px #fff;
        z-index: 2;
    }

    /* ── HEADER & TELEMETRY ── */
    .header-title {
        font-size: 14px;
        font-weight: 700;
        color: #fff;
        letter-spacing: 0.05em;
    }
    .header-sub {
        font-size: 10px;
        color: #444;
        font-weight: 400;
    }
    .telemetry-row {
        display: flex;
        gap: 1.5rem;
        padding: 0.5rem;
        background: #111;
        border: 1px solid #222;
        border-radius: 2px;
        margin-bottom: 1rem;
    }
    .tel-item {
        display: flex;
        gap: 0.4rem;
        align-items: baseline;
    }
    .tel-label {
        font-size: 9px;
        color: #555;
        font-weight: 700;
    }}
    .tel-value {{
        font-size: 9px;
        color: #AAA;
    }}

    /* ── AI SUMMARY ── */
    .ai-summary-card {{
        background: #0D0D0D;
        border: 1px solid #222;
        border-left: 2px solid #555;
        padding: 0.75rem 1rem;
        margin-bottom: 1.5rem;
        margin-bottom: 0.3rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .ai-summary-text {
        font-size: 13px;
        line-height: 1.6;
        color: #aaaaaa;
    }

    /* ── HEADER ── */
    .header-title {
        font-size: 20px;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: -0.02em;
        margin: 0;
    }
    .header-sub {
        font-size: 11px;
        color: #555555;
        margin-top: 0.1rem;
    }

    /* ── BANNER / STATUS ── */
    .banner-warn, .banner-error, .banner-stale {
        border-radius: 2px;
        padding: 0.5rem 1rem;
        font-size: 12px;
        margin-bottom: 0.75rem;
        border: 1px solid transparent;
    }
    .banner-stale { background: #1a0f00; border-color: #552200; color: #ff8800; }

    /* ── PLOTLY OVERRIDES ── */
    .js-plotly-plot .plotly { border-radius: 2px; border: 1px solid #222222; }

    /* ── STEM Fair Optimized Mobile ── */
    .forecast-grid {
        display: grid;
        grid-template_columns: repeat(4, 1fr);
        gap: 0.5rem;
        margin-bottom: 0.5rem;
    }
    @media (max-width: 640px) {
        .block-container { padding: 0.5rem !important; }
        .aqi-number { font-size: 28px !important; }
        .forecast-grid { grid-template-columns: repeat(2, 1fr); }
        .telemetry-row { flex-wrap: wrap; gap: 0.5rem 1rem; }
    }
    @media (max-width: 400px) {
        .forecast-grid { grid-template-columns: 1fr; }
    }
    </style>
    """)


# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_forecast(api_url: str) -> dict | None:
    """
    Fetch /forecast from FastAPI backend.
    Returns parsed dict on success, None on any failure.
    Never raises — all exceptions caught and logged to stderr.
    """
    try:
        # Render free tier goes to sleep after 15m of inactivity.
        # A cold start + loading 12 models takes ~20-40 seconds. We use a 60s timeout here
        # so Streamlit shows a spinner instead of instantly crashing. 
        resp = requests.get(f"{api_url}/forecast", timeout=60)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"[dashboard] API fetch failed: {e}", file=sys.stderr)
        return None


def fetch_with_retry(api_url: str, max_attempts: int = 3) -> dict | None:
    """Retry load_forecast up to max_attempts times."""
    for attempt in range(max_attempts):
        if attempt > 0:
            load_forecast.clear()
            # Removed 10-second sleep which blocked the Streamlit UI thread
            with st.spinner("Retrying connection to backend..."):
                time.sleep(1.5)
        result = load_forecast(api_url)
        if result is not None:
            return result
    return None


# ── AQI helpers ───────────────────────────────────────────────────────────────

def get_aqi_color(category: str) -> str:
    return AQI_COLORS.get(category, "#9ca3af")


def get_aqi_rgba(category: str, alpha: float) -> str:
    base = AQI_COLORS_RGBA.get(category, "rgba(156,163,175,")
    return f"{base}{alpha})"


def format_timestamp(ts_str: str) -> str:
    """Convert ISO timestamp to '5:00 PM PST · Mar 6'"""
    try:
        ts = datetime.fromisoformat(ts_str)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=TZ)
        else:
            ts = ts.astimezone(TZ)
        tz_name = "PDT" if ts.dst() else "PST"
        h = str(int(ts.strftime("%I")))
        d = str(int(ts.strftime("%d")))
        return ts.strftime(f"{h}:%M %p {tz_name} · %b {d}")
    except Exception:
        return ts_str


def format_valid_at(ts_str: str) -> str:
    """Convert valid_at to '5:00 PM today' or '5:00 PM tomorrow'"""
    try:
        ts  = datetime.fromisoformat(ts_str).astimezone(TZ)
        now = datetime.now(tz=TZ)
        h = str(int(ts.strftime("%I")))
        d = str(int(ts.strftime("%d")))
        if ts.date() == now.date():
            return ts.strftime(f"{h}:%M %p today")
        elif (ts.date() - now.date()).days == 1:
            return ts.strftime(f"{h}:%M %p tomorrow")
        else:
            return ts.strftime(f"{h}:%M %p %b {d}")
    except Exception:
        return ts_str


def time_until_refresh() -> str:
    """Compute countdown to next 5-minute cache refresh."""
    secs = int(300 - (time.time() % 300))
    m, s = divmod(secs, 60)
    return f"~{m}m {s:02d}s"


def data_age_minutes(generated_at: str) -> int:
    """Return how many minutes old the forecast data is."""
    try:
        ts  = datetime.fromisoformat(generated_at)
        now = datetime.now(tz=ts.tzinfo or TZ)
        return int((now - ts).total_seconds() / 60)
    except Exception:
        return 0


# ── Gauge figure ──────────────────────────────────────────────────────────────

def make_gauge_figure(aqi_value: float, category: str, color: str) -> go.Figure:
    """Build the Plotly gauge indicator figure."""
    aqi_clamped = max(1, min(500, aqi_value))

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=aqi_clamped,
        number={
            "font": {"size": 72, "color": color, "family": "Space Grotesk"},
            "valueformat": ".0f",
        },
        title={
            "text": (
                f"<b style='color:{color}'>{category}</b>"
                f"<br><span style='font-size:13px;color:#9ca3af;font-weight:400'>"
                f"Air Quality Index</span>"
            ),
            "font": {"size": 22, "color": color, "family": "Space Grotesk"},
        },
        gauge={
            "axis": {
                "range":     [0, 500],
                "tickvals":  [0, 50, 100, 150, 200, 300, 500],
                "ticktext":  ["0", "50", "100", "150", "200", "300", "500"],
                "tickcolor": "#374151",
                "tickfont":  {"size": 10, "color": "#6b7280"},
            },
            "bar": {
                "color":     color,
                "thickness": 0.22,
            },
            "bgcolor":     "#1f2937",
            "bordercolor": "#0a0f1e",
            "borderwidth": 2,
            "steps": [
                {"range": [0,   50],  "color": "rgba(0,228,0,0.10)"},
                {"range": [50,  100], "color": "rgba(255,255,0,0.10)"},
                {"range": [100, 150], "color": "rgba(255,126,0,0.10)"},
                {"range": [150, 200], "color": "rgba(255,0,0,0.10)"},
                {"range": [200, 300], "color": "rgba(143,63,151,0.10)"},
                {"range": [300, 500], "color": "rgba(126,0,35,0.10)"},
            ],
            "threshold": {
                "line":      {"color": color, "width": 5},
                "thickness": 0.80,
                "value":     aqi_clamped,
            },
        },
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=30, r=30, t=70, b=10),
        height=300,
        font={"color": "#f9fafb", "family": "Space Grotesk"},
    )
    return fig


# ── History chart ─────────────────────────────────────────────────────────────

def make_history_chart(history_72h: list, category: str) -> go.Figure:
    """Build the 72-hour actual vs. forecast Plotly line chart."""
    color      = get_aqi_color(category)
    color_band = get_aqi_rgba(category, 0.12)

    times, actuals, forecasts, ci_lo, ci_hi = [], [], [], [], []
    for h in history_72h:
        ts = h.get("timestamp")
        if not ts:
            continue
        times.append(ts)
        actuals.append(h.get("actual_aqi"))
        forecasts.append(h.get("forecast_aqi"))
        ci_lo.append(h.get("ci_lo"))
        ci_hi.append(h.get("ci_hi"))

    has_actuals   = any(v is not None for v in actuals)
    has_forecasts = any(v is not None for v in forecasts)

    y_vals = [v for v in actuals + forecasts + ci_hi if v is not None]
    y_max  = max(200, int(max(y_vals) * 1.15) + 10) if y_vals else 200

    fig = go.Figure()

    for lo, hi_b, rgba in [
        (0,   50,  "rgba(0,228,0,0.06)"),
        (50,  100, "rgba(255,255,0,0.06)"),
        (100, 150, "rgba(255,126,0,0.06)"),
        (150, 200, "rgba(255,0,0,0.06)"),
    ]:
        if lo < y_max:
            fig.add_hrect(y0=lo, y1=min(hi_b, y_max),
                          fillcolor=rgba, line_width=0, layer="below")

    if has_forecasts and any(v is not None for v in ci_hi):
        fig.add_trace(go.Scatter(
            x=times + times[::-1],
            y=ci_hi + ci_lo[::-1],
            fill="toself",
            fillcolor=color_band,
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
            name="90% CI",
        ))

    if has_forecasts:
        fig.add_trace(go.Scatter(
            x=times, y=forecasts,
            name="6h Forecast",
            line=dict(color=color, width=2, dash="dot"),
            mode="lines",
            connectgaps=False,
            hovertemplate="%{x|%b %-d %-I %p}<br>Forecast: %{y:.0f} AQI<extra></extra>",
        ))

    if has_actuals:
        fig.add_trace(go.Scatter(
            x=times, y=actuals,
            name="Actual AQI",
            line=dict(color="#f9fafb", width=2.5),
            mode="lines",
            connectgaps=False,
            hovertemplate="%{x|%b %-d %-I %p}<br>Actual: %{y:.0f} AQI<extra></extra>",
        ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#111827",
        margin=dict(l=10, r=10, t=20, b=20),
        height=260,
        font={"color": "#9ca3af", "family": "Space Grotesk", "size": 11},
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.01,
            xanchor="right",  x=1,
            bgcolor="rgba(17,24,39,0.9)",
            bordercolor="#1f2937",
            borderwidth=1,
            font=dict(color="#9ca3af", size=10),
        ),
        xaxis=dict(
            tickformat="%b %-d\n%-I%p",
            tickcolor="#374151",
            gridcolor="#1f2937",
            showgrid=True,
            zeroline=False,
            tickfont=dict(size=9, color="#6b7280"),
        ),
        yaxis=dict(
            range=[0, y_max],
            tickcolor="#374151",
            gridcolor="#1f2937",
            showgrid=True,
            zeroline=False,
            tickfont=dict(size=9, color="#6b7280"),
            title=dict(text="AQI", font=dict(size=10, color="#6b7280")),
        ),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="#1f2937",
            bordercolor="#374151",
            font=dict(color="#f9fafb", size=11),
        ),
    )
    return fig


# ── UI components ─────────────────────────────────────────────────────────────

def render_header(generated_at: str, current: dict):
    """Minimal instrumentation header with telemetry integrated."""
    source    = current.get("source", "—")
    src_label = "AIRNOW" if source == "AirNow" else "OPNMETEO"
    ts_str    = current.get("timestamp", generated_at)
    
    try:
        gen_ts  = datetime.fromisoformat(generated_at)
        now     = datetime.now(tz=gen_ts.tzinfo or TZ)
        elapsed = max(0, int((now - gen_ts).total_seconds() / 60))
        remain  = max(0, 60 - elapsed)
        refr_ttl = f"{remain}m"
    except Exception:
        refr_ttl = "—"

    st.html(f"""
        <div style="display:flex; justify-content:space-between; align-items:flex-end; margin-bottom:1rem; border-bottom:1px solid #222; padding-bottom:0.5rem;">
            <div>
                <div class="header-title">FOLSOM_AQI_MONITOR</div>
                <div class="header-sub">LAT: 38.6780°N | LON: 121.1761°W | ALT: 66M</div>
            </div>
            <div class="telemetry-row" style="margin-bottom:0; border:none; background:none; padding:0;">
                <div class="tel-item"><span class="tel-label">SYS.VER:</span><span class="tel-value">v2.0</span></div>
                <div class="tel-item"><span class="tel-label">DATA.SRC:</span><span class="tel-value">{src_label}</span></div>
                <div class="tel-item"><span class="tel-label">FRESH:</span><span class="tel-value">{elapsed}m</span></div>
                <div class="tel-item"><span class="tel-label">REFR.TTL:</span><span class="tel-value">{refr_ttl}</span></div>
            </div>
        </div>
    """)


def render_gauge(current_aqi: int, category: str, color: str):
    """Render the AQI gauge with smooth animation on value change."""
    if "prev_aqi" not in st.session_state:
        st.session_state.prev_aqi = current_aqi

    gauge_placeholder = st.empty()

    if abs(current_aqi - st.session_state.prev_aqi) > 2:
        start  = float(st.session_state.prev_aqi)
        end    = float(current_aqi)
        frames = 6
        for i in range(frames + 1):
            t     = i / frames
            ease  = t * t * (3 - 2 * t)
            val   = start + (end - start) * ease
            gauge_placeholder.plotly_chart(
                make_gauge_figure(val, category, color),
                use_container_width=True,
                config={"displayModeBar": False},
            )
            time.sleep(0.01)
        st.session_state.prev_aqi = current_aqi
    else:
        gauge_placeholder.plotly_chart(
            make_gauge_figure(current_aqi, category, color),
            use_container_width=True,
            config={"displayModeBar": False},
        )


def render_advisory(category: str, color: str):
    """Strict linear advisory banner."""
    advisory = ADVISORIES.get(category, "No advisory available.")
    st.markdown(
        f"""
        <div class="advisory-banner" style="border-left:3px solid {color}; background:#080808;">
            <div style="font-family:'JetBrains Mono', monospace; font-size:11px;">
                <span style="color:{color}; font-weight:700;">[{category.upper()}]</span>
                <span style="color:#888; margin-left:0.5rem;">{advisory}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_ai_summary(data: dict):
    """
    Render the AI-generated plain-English summary card.
    Reads from data['ai_summary'] — generated hourly by inference.py.
    If the field is empty (key missing or blank), the card is hidden entirely.
    """
    summary = data.get("ai_summary", "").strip()
    if not summary:
        return   # Backend hasn't generated it yet (no GEMINI_API_KEY set on server)

    st.html(f"""
        <div class="ai-summary-card">
            <div class="ai-summary-label">✦ AI Summary</div>
            <div class="ai-summary-text">{summary}</div>
        </div>
    """)


def render_forecast_cards(forecasts: dict):
    """4 horizon cards with scientific bullet graphs for uncertainty (Fix #9)."""
    grid_html = '<div class="forecast-grid">'
    
    # Standardize scale to 200 for the visual bars (or max of forecasts if higher)
    all_vals = [f.get("ci_hi", 100) for f in forecasts.values()]
    max_scale = max(200, max(all_vals) if all_vals else 200)

    for h_key in ["6h", "12h", "24h", "48h"]:
        fc = forecasts.get(h_key, {})
        if not fc:
            continue
        aqi      = fc.get("aqi", 0)
        ci_lo    = fc.get("ci_lo", 0)
        ci_hi    = fc.get("ci_hi", 0)
        cat      = fc.get("category", "Good")
        valid_at = fc.get("valid_at", "")
        color    = get_aqi_color(cat)
        label    = HORIZON_LABELS.get(h_key, h_key.upper())
        valid_str = format_valid_at(valid_at)

        # Calc percentages for CSS positioning
        ci_lo_pct = (ci_lo / max_scale) * 100
        ci_hi_pct = (ci_hi / max_scale) * 100
        aqi_pct   = (aqi / max_scale) * 100
        ci_width  = ci_hi_pct - ci_lo_pct

        grid_html += textwrap.dedent(f"""
            <div class="horizon-card">
                <div style="font-size:10px; font-weight:700; color:#444; margin-bottom:0.4rem;">
                    T+{h_key.upper()} 
                    <span style="color:#222; margin-left:1rem; font-weight:400;">{valid_str}</span>
                </div>
                <div style="display:flex; align-items:baseline; gap:0.5rem;">
                    <div style="font-size:24px; font-weight:700; color:#fff;">{aqi}</div>
                    <div style="font-size:10px; color:{color}; font-weight:700;">{cat.upper()}</div>
                </div>
                
                <!-- Uncertainty Viz (Bullet Graph) -->
                <div class="uncertainty-box">
                    <div class="ci-range" style="left:{ci_lo_pct}%; width:{ci_width}%;"></div>
                    <div class="point-prediction" style="left:{aqi_pct}%;"></div>
                </div>
                <div style="display:flex; justify-content:space-between; font-size:9px; color:#444; margin-top:0.3rem; font-family:monospace;">
                    <span>{ci_lo}</span>
                    <span>{ci_hi}</span>
                </div>
            </div>
        """)
    
    grid_html += '</div>'
    st.html(grid_html)


def render_telemetry(current: dict, generated_at: str):
    """Simplified telemetry footer for details not in the header."""
    ts_str    = current.get("timestamp", generated_at)
    pollutant = current.get("primary_pollutant", "PM2.5")
    
    st.html(f"""
        <div class="telemetry-row">
            <div class="tel-item"><span class="tel-label">TSTAMP:</span><span class="tel-value">{format_timestamp(ts_str)}</span></div>
            <div class="tel-item"><span class="tel-label">PARAM:</span><span class="tel-value">{pollutant}</span></div>
            <div class="tel-item"><span class="tel-label">LOC:</span><span class="tel-value">FOLSOM_CA_STN_1</span></div>
        </div>
    """)


def render_history_chart(history_72h: list, category: str):
    """72h forecast vs. actual chart inside an expander."""
    with st.expander("📈  Past 72 Hours — Forecast vs Actual", expanded=False):
        has_data = any(
            h.get("actual_aqi") is not None or h.get("forecast_aqi") is not None
            for h in history_72h
        )
        if not has_data:
            st.markdown(
                """
                <div style="color:#6b7280;font-size:13px;padding:1rem 0;text-align:center;">
                    Forecast history is building up — check back in a few hours.<br>
                    <span style="font-size:11px;">The system logs predictions as it runs;
                    the chart fills in over the first 72 hours of operation.</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.plotly_chart(
                make_history_chart(history_72h, category),
                use_container_width=True,
                config={"displayModeBar": False},
            )


def render_ai_chat(data: dict):
    """
    Single-turn AI chatbox grounded in the current forecast data.
    Users type a question, press Enter, and see the AI's answer.
    The last Q&A pair is stored in session state so it survives reruns.
    """
    st.markdown(
        '<div class="chat-section-header">🤖 &nbsp; Ask the AI about air quality</div>',
        unsafe_allow_html=True,
    )

    # Show last Q&A pair if it exists
    last_q = st.session_state.get("chat_last_q", "")
    last_a = st.session_state.get("chat_last_a", "")

    if last_q and last_a:
        safe_q = html.escape(last_q)
        safe_a = html.escape(last_a)
        st.markdown(
            f'<div class="chat-q">{safe_q}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="chat-a">'
            f'<div class="chat-a-label">✦ AI</div>'
            f'{safe_a}'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Chat input — Streamlit re-runs app on submit
    question = st.chat_input(
        "Ask anything — e.g. 'Is it safe to run outside today?' or 'How does the model work?'",
        key="ai_chat_input",
    )

    if question:
        # Check key availability first
        api_key = _get_gemini_key()
        if not api_key:
            st.session_state["chat_last_q"] = question
            st.session_state["chat_last_a"] = (
                "⚠️ The AI assistant isn't configured yet. "
                "Add GEMINI_API_KEY to your Streamlit secrets to enable this feature."
            )
            st.rerun()

        with st.spinner("Thinking..."):
            answer = answer_question(question, data)

        st.session_state["chat_last_q"] = question
        st.session_state["chat_last_a"] = answer
        st.rerun()


def render_about():
    """About / methodology expander."""
    with st.expander("ℹ️  About This Forecast", expanded=False):
        st.markdown(
            """
            <div style="color:#9ca3af;font-size:13px;line-height:1.75;">
            This dashboard uses a machine learning model trained on 2+ years of hourly
            air quality and meteorological data for Folsom, CA.<br><br>
            The model (<strong style="color:#f9fafb;">LightGBM</strong>) produces separate
            forecasts for <strong style="color:#f9fafb;">6, 12, 24, and 48-hour horizons</strong>.
            Confidence intervals represent the 1st–99th percentile range of expected outcomes
            based on quantile regression.<br><br>
            <strong style="color:#f9fafb;">Data sources:</strong><br>
            &bull; Current readings: AirNow (U.S. EPA sensor network)<br>
            &bull; Weather inputs: Open-Meteo historical and forecast API<br>
            &bull; Training data: 2022–present, Folsom monitoring station<br><br>
            <strong style="color:#f9fafb;">AI layer:</strong> Plain-English summaries and
            the Q&amp;A chatbox are powered by Google Gemini 2.0 Flash.<br><br>
            <em>Built by Arsan, FLC Los Rios STEM Fair 2026</em>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_footer():
    st.markdown(
        '<div class="page-footer">'
        'Built with LightGBM + Streamlit &nbsp;·&nbsp; Folsom, CA '
        '&nbsp;·&nbsp; FLC Los Rios STEM Fair 2026'
        '</div>',
        unsafe_allow_html=True,
    )


def render_error(message: str, kind: str = "error"):
    css_class = f"banner-{kind}"
    st.markdown(
        f'<div class="{css_class}">{message}</div>',
        unsafe_allow_html=True,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    inject_css()

    # ── Auto-refresh — avoid wiping input while user is chatting (Fix #8) ───
    # If the user has just interacted with the input (ai_chat_input is in state), 
    # we double the interval to 10 minutes to reduce the chance of a mid-type wipe.
    interval_ms = 300_000 if not st.session_state.get("ai_chat_input") else 600_000
    st_autorefresh(interval=interval_ms, limit=None, key="aqi_autorefresh")

    # ── Load data ─────────────────────────────────────────────────────────
    data = None
    try:
        data = fetch_with_retry(API_URL, max_attempts=3)
    except Exception as e:
        print(f"[dashboard] Unhandled error in fetch: {e}", file=sys.stderr)

    # ── Total failure state ────────────────────────────────────────────────
    if data is None:
        st.markdown(
            '<p class="header-title">🌬️ Folsom Air Quality Monitor</p>',
            unsafe_allow_html=True,
        )
        render_error(
            "⚠️ Unable to load forecast. "
            "Please check your connection or try refreshing the page.",
            kind="error",
        )
        if st.button("🔄 Try Again"):
            load_forecast.clear()
            st.rerun()
        render_footer()
        return

    # ── Extract fields ─────────────────────────────────────────────────────
    try:
        generated_at = data.get("generated_at", "")
        current      = data.get("current", {})
        forecasts    = data.get("forecasts", {})
        history_72h  = data.get("history_72h", [])

        raw_aqi  = current.get("aqi", 1)
        aqi      = max(1, int(raw_aqi)) if raw_aqi else 1
        category = current.get("category", "Good")
        color    = get_aqi_color(category)

        sensor_missing = (raw_aqi == 0 or raw_aqi is None)

        age = data_age_minutes(generated_at)
        if age > 120:
            render_error(
                f"⚠️ Data may be outdated — last updated {format_timestamp(generated_at)}",
                kind="stale",
            )

    except Exception as e:
        print(f"[dashboard] Error parsing API response: {e}", file=sys.stderr)
        render_error("Something went wrong. Please refresh the page.")
        render_footer()
        return

    # ── Render ────────────────────────────────────────────────────────────
    try:
        render_header(generated_at, current)

        if sensor_missing:
            render_error(
                'SYSTEM_MSG: LOCAL_SENSOR_OFFLINE // FALLBACK_TO_OPNMETEO_MODEL',
                kind="warn",
            )

        render_gauge(aqi, category, color)
        render_advisory(category, color)

        render_ai_summary(data)

        st.markdown("<div style='height:0.25rem'></div>", unsafe_allow_html=True)
        render_forecast_cards(forecasts)
        render_telemetry(current, generated_at)

        st.markdown("<div style='height:1rem; border-bottom:1px solid #111; margin-bottom:1rem;'></div>", unsafe_allow_html=True)

        render_history_chart(history_72h, category)
        render_ai_chat(data)

        st.markdown("<hr style='border:none; border-top:1px solid #111; margin:2rem 0;'>", unsafe_allow_html=True)

        render_about()
        render_footer()

    except Exception as e:
        print(f"[dashboard] Unhandled render error: {e}", file=sys.stderr)
        render_error("Something went wrong. Please refresh the page.")


if __name__ == "__main__":
    main()
