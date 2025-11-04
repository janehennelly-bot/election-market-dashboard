# Elections & the S&P 500 Dashboard

This project visualizes how U.S. presidential election cycles align with long-term stock market performance.  
It uses historical price data from **Yahoo Finance** and overlays election dates to explore whether markets react before or after election events.

---

## Features

- Interactive **Streamlit dashboard**.
- Choice of index: S&P 500 (`^GSPC`), Dow Jones (`^DJI`), or NASDAQ (`^IXIC`).
- Adjustable time range (5–40 years of history).
- Vertical markers for every U.S. presidential election since 1980.
- KPI metrics showing:
  - **Last Close**
  - **30-Day Change Before Election**
  - **30-Day Change After Election**
  - **Overall Growth Since Last Election**

---

##  Data Sources

- **Market data**: [Yahoo Finance API](https://finance.yahoo.com/)
- **Election dates**: `data/elections.csv` (included in this repository)

---

##  Insights

This dashboard demonstrates the timing of major market movements around election periods.  
While it does not establish causality, it offers a clear visual context to study whether markets tend to rise, fall, or stabilize around election cycles.

---

##  Installation and Usage

### 1. Clone this repository
```bash
git clone https://github.com/janehennelly-bot/election-market-dashboard.git
cd election-market-dashboard/app
