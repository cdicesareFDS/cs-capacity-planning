# CS Workforce Analytics Dashboard

A Streamlit dashboard for visualizing customer success team capacity and activity metrics from Databricks.

## Local Development

1. **Clone the repo:**
   ```bash
   git clone <repo-url>
   cd cs-workforce-dashboard
   ```

2. **Create a Python virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up local credentials:**
   ```bash
   cp .streamlit/secrets.example.toml .streamlit/secrets.toml
   ```
   Then edit `.streamlit/secrets.toml` with your Databricks workspace host and SQL warehouse HTTP path.

5. **Run the app locally:**
   ```bash
   streamlit run sts_app.py
   ```
   Visit `http://localhost:8501` to view the dashboard.

## Databricks App Deployment

1. In your Databricks workspace, go to **Apps** → **Create App**
2. Select **Git folder** as the source
3. Connect your GitHub repo
4. Point to this folder as the app source
5. Databricks will automatically authenticate using your workspace credentials

The app will automatically refresh when you push changes to GitHub.

## Features

- **Date range filtering** — Adjust the analysis window by fiscal year
- **Department filtering** — View metrics for specific departments
- **Team Summary view** — Department-level aggregations and comparisons
- **Individual Employee view** — Per-person activity breakdown
- **Time allocation visualization** — Pie charts showing hours by activity type (Cases, Meetings, Calls, Emails, RPDs)
- **Export** — Download filtered data as CSV

## Data Source

The dashboard queries a Databricks workspace to pull:
- Case metrics (created, delegated, time spent)
- Meeting activity (unique meetings, clients met, hours)
- Call/Email activity (calls made, connection rate, time spent)
- RPD metrics (authored, commented, estimated hours)

All metrics are calculated based on business days (Mon-Fri, excluding US federal holidays).
