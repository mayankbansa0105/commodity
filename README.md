# ⛽ Natural Gas Commodity Analytics Dashboard

A real-time natural gas market intelligence dashboard built with Python & Streamlit.

---

## 📊 Features

- **Live NG Futures prices** — Henry Hub (NG=F) via Yahoo Finance
- **Last 7 days electricity demand** — real EIA-930 US48 data (with key) or temperature model
- **14-day weather forecast** — OpenMeteo API (free, no key required)
- **Storage & injection analysis** — real EIA weekly storage; current week, last week, last year, 5-year average
- **🛢️ NG Fundamentals & Price Drivers** — real EIA dry production, total consumption, power burn (Bcf/day) with YoY, sector demand split, and prod−demand balance
- **Deviation analysis** — WoW, YoY, and vs 5yr average (absolute + %)
- **Daily injection breakdown** — bar chart + cumulative curve + table
- **Supply & demand factors** — data-driven bullish/bearish/watch signals
- **5-factor price recommendation** — momentum + storage + weather + production + demand → BULLISH/BEARISH/NEUTRAL with confidence and price targets
- **🔁 Auto-refresh** — re-fetches live data automatically (default every 2 min; toggle + interval in sidebar)

---

## 🗂️ Project Structure

```
commodity/
├── dashboard.py       # Main Streamlit app
├── requirements.txt   # Python dependencies
├── .venv/             # Virtual environment (auto-created)
└── README.md          # This file
```

---

## ⚙️ Prerequisites

- Python 3.10 or higher
- Internet connection (for live price & weather data)

Check your Python version:
```bash
python3 --version
```

---

## 🚀 Quick Start

### 1. Clone / navigate to the project folder

```bash
cd /Users/bansamay/Downloads/commodity
```

### 2. Create the virtual environment

```bash
python3 -m venv .venv
```

### 3. Activate the virtual environment

**macOS / Linux:**
```bash
source .venv/bin/activate
```

**Windows (Command Prompt):**
```cmd
.venv\Scripts\activate.bat
```

**Windows (PowerShell):**
```powershell
.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the dashboard

```bash
streamlit run dashboard.py
```

The dashboard opens automatically at **http://localhost:8501**

---

## 🔄 Stopping & Restarting

**Stop the server:**
Press `Ctrl + C` in the terminal.

**Restart (venv already set up):**
```bash
source .venv/bin/activate        # macOS/Linux
streamlit run dashboard.py
```

---

## 🌐 Running on a Custom Port

```bash
streamlit run dashboard.py --server.port 8080
```

---

## 🖥️ Running in Headless / Server Mode

Useful when deploying on a remote machine:

```bash
streamlit run dashboard.py --server.port 8501 --server.headless true
```

Then open `http://<server-ip>:8501` in your browser.

---

## 📦 Dependencies

| Package      | Purpose                          |
|--------------|----------------------------------|
| streamlit    | Web dashboard framework          |
| pandas       | Data manipulation                |
| numpy        | Numerical computations           |
| plotly       | Interactive charts               |
| yfinance     | Yahoo Finance NG futures prices  |
| requests     | OpenMeteo weather API calls      |
| streamlit-autorefresh | Soft auto-refresh on interval |

---

## 🔑 Data Sources

| Source        | Data                                   | API Key Required |
|---------------|----------------------------------------|------------------|
| Yahoo Finance | NG Futures (NG=F)                      | ❌ No            |
| OpenMeteo     | 14-day weather forecast                | ❌ No            |
| EIA v2        | Weekly storage & injection (Lower 48)  | ✅ Yes (free)    |
| EIA v2        | Dry production / consumption / power burn | ✅ Yes (free) |
| EIA-930       | US48 daily electricity demand          | ✅ Yes (free)    |

Without an EIA key, storage/fundamentals/electricity fall back to built-in models.

### EIA series & process codes used (validated against the live API)

| Endpoint | Facet | Meaning |
|----------|-------|---------|
| `natural-gas/stor/wkly` | `process=SWO`, `duoarea=R48`, series `NW2_EPG0_SWO_R48_BCF` | Weekly working-gas stock (injection = week-over-week diff) |
| `natural-gas/prod/sum`  | `process=FPD`, `duoarea=NUS` | Dry natural gas production (MMcf → Bcf/day) |
| `natural-gas/cons/sum`  | `process=VC0/VEU/VRS/VIN/VCS` | Total / power / residential / industrial / commercial consumption |
| `electricity/rto/daily-region-data` | `respondent=US48`, `type=D` | Daily Lower-48 electricity demand (MWh → GWh) |

### 5-Factor price recommendation

The recommendation engine combines five deterministic signals into a net score:

1. **Price momentum** — MA5 vs MA10
2. **Storage** — injection vs 5-year average
3. **Weather** — extreme temps drive heating/cooling demand
4. **Production (EIA)** — rising YoY dry output → bearish (oversupply)
5. **Demand (EIA)** — rising consumption / power burn YoY → bullish

---

## 🔁 Refreshing Data

**Auto-refresh:** Enabled by default. Use the **🔁 Auto-Refresh** controls in the sidebar to toggle it on/off and pick an interval (1 / 2 / 5 / 10 min). The app re-runs and re-fetches live data on that interval while preserving your sidebar selections.

**Manual refresh:** Click the **🔄 Refresh Now** button in the sidebar to immediately clear the cache and re-fetch all data.

**Caching:** Prices are cached for 2 minutes and weather for 15 minutes, so auto-refresh always pulls fresh data without hammering the APIs. The sidebar **Status** panel shows whether prices/weather are `🟢 Live` or `🟠 Fallback` and the last update time.

---

## ⚠️ Disclaimer

This dashboard is for **informational and educational purposes only**. It does not constitute financial or investment advice.
