import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Elections & S&P 500", layout="wide")
st.title("Elections & the S&P 500: Minimal MVP")

symbol = st.sidebar.selectbox("Index", ["^GSPC", "^DJI", "^IXIC"], index=0)
years_back = st.sidebar.slider("Years of history", 5, 40, 25, 1)

elections = pd.read_csv("data/elections.csv", parse_dates=["election_date"]).sort_values("election_date")

@st.cache_data(show_spinner=False)
def load_prices(sym: str, years: int) -> pd.DataFrame:
    hist = yf.Ticker(sym).history(period=f"{years}y", interval="1d", auto_adjust=True)
    hist = hist.reset_index()
    out = hist[["Date", "Close"]].dropna().copy()
    out["Date"] = out["Date"].dt.tz_localize(None)
    out["Close"] = out["Close"].astype(float)
    return out





prices = load_prices(symbol, years_back)
# --- KPI cards: Election-specific metrics ------------------------------
import numpy as np
from datetime import timedelta

prices = prices.sort_values("Date").reset_index(drop=True)
prices["Return"] = prices["Close"].pct_change()

def pct(a, b):
    try:
        return float(np.round((a / b - 1) * 100, 2))
    except Exception:
        return np.nan

# Basic current close
last_close = float(prices["Close"].iloc[-1])
prev_close = float(prices["Close"].iloc[-2])

# Find latest election within the data range
in_range_elections = elections[
    (elections["election_date"] >= prices["Date"].min()) &
    (elections["election_date"] <= prices["Date"].max())
].sort_values("election_date")

if len(in_range_elections):
    last_elec_date = in_range_elections["election_date"].iloc[-1]
    # find closing price nearest the election day
    price_on_elec = prices.loc[prices["Date"] >= last_elec_date, "Close"].iloc[0]
else:
    last_elec_date = None
    price_on_elec = last_close

# 30 days BEFORE election
if last_elec_date:
    d30_pre = last_elec_date - timedelta(days=30)
    pre_close = prices.loc[prices["Date"] >= d30_pre, "Close"].iloc[0]
    pre_30d_change = pct(price_on_elec, pre_close)
else:
    pre_30d_change = np.nan

# 30 days AFTER election
if last_elec_date:
    d30_post = last_elec_date + timedelta(days=30)
    post_close = prices.loc[prices["Date"] >= d30_post, "Close"].iloc[0]
    post_30d_change = pct(post_close, price_on_elec)
else:
    post_30d_change = np.nan

# Since election
since_election = pct(last_close, price_on_elec)

# Display cards
k1, k2, k3, k4 = st.columns(4)
k1.metric("Last Close", f"{last_close:,.0f}", f"{pct(last_close, prev_close)}% 1D")

k2.metric("30 Days Before Election", f"{pre_30d_change}%")
k3.metric("30 Days After Election", f"{post_30d_change}%")

if last_elec_date is not None:
    k4.metric(f"Since {last_elec_date.date()}", f"{since_election}%")
else:
    k4.metric("Since Election", "N/A")







fig = go.Figure()


fig.add_trace(go.Scatter(
    x=prices["Date"],
    y=prices["Close"],
    mode="lines",
    name=symbol,
    line=dict(width=2)
))

for d in elections["election_date"]:
    if prices["Date"].min() <= d <= prices["Date"].max():
        fig.add_vline(x=d, line_dash="dash", opacity=0.5)

fig.update_layout(
    title=f"{symbol} with U.S. Presidential Election Days",
    xaxis_title="Date",
    yaxis_title="Close",
    legend_title="Index",
    hovermode="x unified"
)

st.plotly_chart(fig, use_container_width=True)

st.caption("MVP complete: price series + election markers. Next: cumulative returns, event study, volatility, and summary table.")
