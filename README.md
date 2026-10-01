# Intraday AI Prediction & News Intelligence Engine 📈🤖

An automated, zero-cost, mobile-friendly Progressive Web App (PWA) that ingests live market data, generates intraday equity trajectory forecasts using Google Gemini AI, and renders interactive visualization curves for Indian stock market indices and watchlists.

---

## 🏗️ Architecture & Operational Flow


┌─────────────────┐       ┌────────────────────────┐       ┌──────────────────────┐
│  cron-job.org   │ ──1──>│  GitHub Actions Runner │ ──2──>│  muthoot_poc Pipeline │
│ (15-min Trigger)│       │ (intraday_tracker.yml) │       │ (Python + yfinance)  │
└─────────────────┘       └────────────────────────┘       └──────────┬───────────┘
│ 3 (Fetch & Predict)
▼
┌─────────────────┐       ┌────────────────────────┐       ┌──────────────────────┐
│   GitHub Pages  │ <──5──│  public/data_store/*.json │ <──4──│ Google Gemini AI API │
│  (Static Web UI)│       │ (Auto-committed JSONs) │       │ (Market Trajectory)  │
└─────────────────┘       └────────────────────────┘       └──────────────────────┘


1. **Trigger Phase**: `cron-job.org` issues a scheduled REST API call every 15 minutes during Indian market hours (09:15 AM to 03:30 PM IST) to GitHub's `workflow_dispatch` endpoint.
2. **Execution Phase**: GitHub Actions spins up an Ubuntu runner (`.github/workflows/intraday_tracker.yml`) and sets up a Python 3.11 environment.
3. **Ingestion & AI Analysis Phase**: Python scripts (`muthoot_poc/archive_once.py` / `intraday_pipeline.py`) fetch live market price/volume data via `yfinance` and pass technical indicators/news sentiment to Gemini AI.
4. **Data Persistence**: The generated trajectory models and market data curves are exported into structured JSON files in `public/data_store/`.
5. **Deployment & UI Phase**: GitHub Actions commits updated JSON files back to `main` with `[skip ci]`. GitHub Pages automatically serves the static dynamic frontend (`index.html`) using Tailwind CSS and Chart.js.

---

## ⚡ Key Features

* **Real-Time Intraday Trajectory Curves**: Compares predicted AI trajectories with actual market volume/price movements across 15-minute intervals.
* **Strict Time-Gated IST Filtering**: Dynamically maps and formats market hours for Indian Standard Time (UTC+5:30).
* **Mobile-First Responsive PWA**: Optimized UI featuring horizontal scrolling stock watchlist tickers, mobile-friendly padded cards, and dynamic vertical grid stacking.
* **100% Zero-Cost Infrastructure**: Runs entirely on GitHub Actions, GitHub Pages free tier, and external cron orchestrators.
* **Zero Hardcoded Secrets**: Fully audited pipeline utilizing environment-level variable passing and GitHub Repository Secrets.

---

## 🛠️ Tooling & Tech Stack

| Category | Tool / Library | Role / Usage |
| :--- | :--- | :--- |
| **Frontend UI** | HTML5, Tailwind CSS, Chart.js | Responsive web UI, interactive stock trajectory charts, theme styling |
| **Backend Automation** | Python 3.11, `yfinance`, `pandas`, `requests` | Market data ingestion, technical analysis calculation, JSON generation |
| **AI Engine** | Google Gemini API | Predictive analysis and intraday stock momentum modeling |
| **CI/CD & Hosting** | GitHub Actions, GitHub Pages | Scheduled pipeline executions and static web app hosting |
| **External Trigger** | `cron-job.org` | High-precision 15-minute cron triggers bypassing GitHub queue delays |

---

## 🔑 Tokens, Secrets & Configuration

To keep the project secure and public-repo friendly, credentials are stored in environment variables and external managers:

### 1. GitHub Repository Secrets (`GEMINI_API_KEY`)
* **Purpose**: Authorizes backend Python scripts to query Google's Gemini AI model.
* **Location**: GitHub Repository > `Settings` > `Secrets and variables` > `Actions` > `Repository secrets`.
* **Usage in Workflow**:
  ```yaml
  env:
    GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}
	
	
2. GitHub Personal Access Token (PAT)
Purpose: Authorizes cron-job.org to trigger manual workflow execution via GitHub API.

Scope: Fine-Grained Token restricted to Intraday-Gemini repo with Actions: Read and write permission.

cron-job.org Endpoint:

URL: https://api.github.com/repos/flexuser/Intraday-Gemini/actions/workflows/intraday_tracker.yml/dispatches

Method: POST

Headers:

Authorization: Bearer <YOUR_GITHUB_PAT>

Accept: application/vnd.github+json

User-Agent: CronJobApp

Body: {"ref": "main"}

📜 Workflow Automations (intraday_tracker.yml)
Located at .github/workflows/intraday_tracker.yml:

name: Intraday Live Market Sync

on:
  schedule:
    - cron: '45 3 * * 1-5'      # 09:15 IST (Market Open)
    - cron: '*/15 4-9 * * 1-5'  # Every 15 mins from 09:30 IST to 15:15 IST
    - cron: '0 10 * * 1-5'      # 15:30 IST (Market Close)
  workflow_dispatch:

jobs:
  update-market-data:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install -r requirements.txt
      - run: python -m muthoot_poc.archive_once
      - run: |
          git config --global user.name "github-actions[bot]"
          git config --global user.email "41898282+github-actions[bot]@users.noreply.github.com"
          git add public/data_store/*.json
          git diff --staged --quiet || (git commit -m "auto: sync live intraday market curves [skip ci]" && git push)
		  
		  
💻 Local Development & Maintenance
Run Locally
To serve and preview the frontend dashboard on your local machine:

PowerShell
python -m http.server 8000
Then visit http://localhost:8000 in your web browser. Use Ctrl + F5 for hard refreshes.

Security Audit Scanner
Run this script to scan all files for hardcoded secrets or API key leaks:

PowerShell
python scan_secrets.py

---

### Step 3: Commit and Push to GitHub

Once `README.md` is generated or saved:
1. Open **GitHub Desktop**.
2. Commit `README.md` with the message: `docs: add project architecture and operational guide`.
3. Click **Push origin**.