# Folsom AQI Monitor — Dashboard
**FLC Los Rios STEM Fair 2026**

Live AQI forecast dashboard for Folsom, CA.  
Backed by a LightGBM ML model running on your backend server.

---

## What this repo contains

| File | Purpose |
|------|---------|
| `app.py` | The Streamlit dashboard — the only file visitors see |
| `qr_generator.py` | Run once to create the printable QR code |
| `requirements.txt` | Python packages for Streamlit Cloud |
| `.streamlit/config.toml` | Dark navy theme |
| `assets/booth_qr.png` | Pre-generated QR code (commit this after running qr_generator.py) |

---

## Deployment guide (20 minutes, no experience needed)

### Step 1 — Create a GitHub account and repository (5 min)

1. Go to **github.com** → click **Sign up** (or **Sign in** if you have an account)
2. After signing in, click the **+** icon (top-right) → **New repository**
3. Repository name: `folsom-aqi-dashboard`
4. Set visibility to **Public** (required for Streamlit free tier)
5. Click **Create repository**
6. On the next page, click **Add file → Upload files**
7. Drag in all files from this folder — including the hidden `.streamlit/` folder
8. Click **Commit changes**

> **Tip for the `.streamlit/` folder:** GitHub's file uploader may not show hidden folders.
> If it doesn't appear, use the GitHub Desktop app or drag the folder directly.
> Alternatively, create `.streamlit/config.toml` via **Add file → Create new file**
> and paste the contents manually.

---

### Step 2 — Deploy to Streamlit Community Cloud (5 min)

1. Go to **share.streamlit.io**
2. Click **Sign in with GitHub** — authorise the connection
3. Click **New app**
4. Fill in:
   - **Repository:** `your-github-username/folsom-aqi-dashboard`
   - **Branch:** `main`
   - **Main file path:** `app.py`
5. Click **Deploy!**
6. Wait ~2 minutes while Streamlit installs packages
7. Your app will open automatically at a URL like:
   `https://folsom-aqi-dashboard-xxxx.streamlit.app`

---

### Step 3 — Add your backend API URL as a secret (2 min)

The dashboard needs to know where your backend is running.

1. In the Streamlit Cloud dashboard, find your app
2. Click the **⋮** (three dots) next to the app name → **Settings**
3. Click **Secrets** in the left sidebar
4. Paste this (replace the URL with your actual backend address):

```
API_URL = "http://folsom-aqi.onrender.com"
```

5. Click **Save**
6. The app will automatically restart — wait ~30 seconds

> **To verify it worked:** Open the app URL. The gauge should show a real AQI number
> (not an error banner). If you see "API unavailable", double-check the URL in Secrets.

---

### Step 4 — Generate and print the QR code (3 min)

After the app is deployed and working:

1. Copy your Streamlit app URL (e.g. `https://folsom-aqi-dashboard.streamlit.app`)
2. On your Windows machine, open PowerShell in the project folder:

```powershell
.\venv\Scripts\Activate.ps1
pip install qrcode[pil] Pillow
python qr_generator.py https://your-actual-app-url.streamlit.app
```

3. This creates `assets/booth_qr.png`
4. Commit it to GitHub:
   - Drag `assets/booth_qr.png` to your GitHub repo via **Add file → Upload files**
5. Print at home or a copy shop:
   - **Home:** Print on any printer at 8×8 inches (no scaling / fit to page OFF)
   - **Copy shop:** Request "print at 100%, 8×8 inches" on matte paper

> The QR code has 30% error correction — it scans even with minor damage or glare.
> Make sure there is at least 5mm of white border around the QR pattern when mounting.

---

### Step 5 — Test before the fair (5 min)

1. Open the app URL on your phone
2. Check that the AQI number matches a real AQI app (AirNow, IQAir, etc.)
3. Scan the printed QR code with your phone camera — it should open the dashboard instantly
4. **Wake the app:** Streamlit free tier goes to sleep after ~7 days of inactivity.
   Open the URL on your laptop **10 minutes before the fair starts** to wake it up.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| App shows "Module not found" | Check `requirements.txt` has all packages, redeploy |
| Gauge doesn't appear | Try Chrome or Firefox; some old iOS Safari versions have Plotly issues |
| "API unavailable" banner | Verify `API_URL` secret is set correctly in Streamlit Cloud Settings |
| API URL correct but still failing | Confirm your backend server is running: open `http://YOUR_IP/health` |
| QR code doesn't scan | Ensure minimum 5mm white border around QR when printed; increase print size |
| App is asleep / loading spinner | Visit the URL 10 minutes before the fair; it wakes up in ~30 seconds |
| Auto-refresh not working | Disable any browser extensions that block JavaScript timers |
| Forecast shows old data (orange banner) | Check that your backend `refresh.py` cron job is running hourly |

---

## How auto-refresh works

- The page automatically reloads every **5 minutes** (the countdown shows in the header)
- The backend itself refreshes the forecast every **60 minutes** via cron job
- The `ttl=300` cache in `app.py` means the dashboard re-fetches from the backend every 5 minutes
- You can manually force a refresh by clicking the **🔄 Refresh** button in the header

---

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `API_URL` | `http://localhost:8000` | Your FastAPI backend URL |

Set via Streamlit Cloud Secrets (TOML format: `API_URL = "https://..."`)  
Or locally via a `.env` file in this directory.

---

## Local development

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
# Create .env with: API_URL=http://localhost:8000
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

*FLC Los Rios · STEM Fair 2026 · Built with Streamlit + LightGBM*
