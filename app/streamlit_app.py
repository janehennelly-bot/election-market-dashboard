import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Elections & S&P 500", layout="wide")
st.title("Elections & the S&P 500: Minimal MVP")

# --- Sector & Index Tickers ---
SECTOR_MAP = {
    "S&P 500 (Broad Index)": "^GSPC",
    "Dow Jones (30 Stocks)": "^DJI",
    "Technology (XLK)": "XLK",
    "Financials (XLF)": "XLF",
    "Energy (XLE)": "XLE",
    "Health Care (XLV)": "XLV",
    "Industrials (XLI)": "XLI",
    "Real Estate (XLRE)": "XLRE"
}
# --- New Sidebar Input ---
selection_name = st.sidebar.selectbox("Index / Sector ETF", list(SECTOR_MAP.keys()), index=0)
symbol = SECTOR_MAP[selection_name]
years_back = st.sidebar.slider("Years of history", 5, 40, 25, 1)

elections = pd.read_csv("data/elections.csv", parse_dates=["election_date"]).sort_values("election_date")

# --- Ensures the column name is capitalized for the final Summary DataFrame ---
if 'party' in elections.columns:
    elections = elections.rename(columns={'party': 'Party'})

# --- NEW LINE ADDED HERE ---
if 'party' in elections.columns:
    elections['party'] = elections['party'].astype(str).str.strip().str.lower()
# ---------------------------
@st.cache_data(show_spinner=False)
def load_prices(sym: str, years: int) -> pd.DataFrame:
    hist = yf.Ticker(sym).history(period=f"{years}y", interval="1d", auto_adjust=True)
    hist = hist.reset_index()
    out = hist[["Date", "Close"]].dropna().copy()
    out["Date"] = out["Date"].dt.tz_localize(None)
    out["Close"] = out["Close"].astype(float)
    return out





prices = load_prices(symbol, years_back)
# ---------- Helper utilities for election analysis ----------

import numpy as np
import pandas as pd
import plotly.express as px

# Ensure dtypes are clean
elections["election_date"] = pd.to_datetime(elections["election_date"]).dt.tz_localize(None)
prices["Date"] = pd.to_datetime(prices["Date"]).dt.tz_localize(None)

# Keep only columns we need, clean and sort
prices = prices[["Date", "Close"]].dropna().sort_values("Date").reset_index(drop=True)

# Fast lookup helpers on a trading-day index ----------------
# Use a DatetimeIndex so we can grab nearest trade day quickly
px_idx = prices.set_index("Date").sort_index()

def nearest_trade_day(dt: pd.Timestamp) -> pd.Timestamp:
    """Return the trading day in px_idx nearest to dt."""
    loc = px_idx.index.get_indexer([dt], method="nearest")[0]
    return px_idx.index[loc]

def value_on(dt: pd.Timestamp) -> float:
    """Close on nearest trading day to dt."""
    return float(px_idx.loc[nearest_trade_day(dt), "Close"])

def value_shifted(dt: pd.Timestamp, trading_days: int) -> float | None:
    """Close trading_days before/after the election day. Returns None if out of range."""
    center_pos = px_idx.index.get_indexer([nearest_trade_day(dt)], method="nearest")[0]
    target_pos = center_pos + trading_days
    if target_pos < 0 or target_pos >= len(px_idx.index):
        return None
    return float(px_idx.iloc[target_pos]["Close"])

def pct_change(a: float | None, b: float | None) -> float | None:
    """Return (b/a - 1) as percent; None if inputs missing."""
    if a is None or b is None or a == 0:
        return None
    return (b / a - 1.0) * 100.0

# ---------- Per-election summary (pre/post/1y & volatility) ----------

def build_election_summary(window_pre=30, window_post=30, window_1y=252, vol_window=60) -> pd.DataFrame:
    rows = []
    for _, r in elections.iterrows():
        d = r["election_date"]
        winner = r.get("winner", None)
        
        # --- Robustly pull the Party column (now capitalized) ---
        party = r["Party"] if "Party" in r.index else None
        
        # ... (rest of the function)

        # Prices at key offsets (trading-day shifts)
        p_before = value_shifted(d, -window_pre)
        p_at     = value_on(d)
        p_after  = value_shifted(d,  window_post)
        p_1y     = value_shifted(d,  window_1y)

        pre_30  = pct_change(p_before, p_at)              # % move into election
        post_30 = pct_change(p_at, p_after)               # % move after election
        post_1y = pct_change(p_at, p_1y)                  # % over ~1y after

        # Volatility: stdev of daily returns before vs after (± vol_window)
        try:
            center = nearest_trade_day(d)
            center_idx = px_idx.index.get_loc(center)

            pre_slice  = px_idx.iloc[max(0, center_idx - vol_window):center_idx + 1]["Close"].pct_change().dropna()
            post_slice = px_idx.iloc[center_idx:min(len(px_idx), center_idx + vol_window + 1)]["Close"].pct_change().dropna()
            vol_pre  = float(pre_slice.std() * np.sqrt(252) * 100.0) if len(pre_slice) > 3 else None   # Annualized %
            vol_post = float(post_slice.std() * np.sqrt(252) * 100.0) if len(post_slice) > 3 else None
            vol_chg  = (vol_post - vol_pre) if (vol_pre is not None and vol_post is not None) else None
        except Exception:
            vol_pre = vol_post = vol_chg = None

        rows.append({
            "Election": d.date(),
            "Winner": winner,
            "Party": party,
            "30d Before": pre_30,
            "30d After": post_30,
            "1y After": post_1y,
            "Vol Pre (ann%)": vol_pre,
            "Vol Post (ann%)": vol_post,
            "Vol Δ (pp)": vol_chg
        })

    out = pd.DataFrame(rows)
    return out
# ---------- New: Presidential Cycle Analysis ----------

def build_presidential_cycle(prices: pd.DataFrame, elections: pd.DataFrame) -> pd.DataFrame:
    # 1. Map all trading days to their 'Presidential Year' (1-4)
    cycle_map = {}
    
    # Iterate backwards for easier mapping of current term
    for i in range(len(elections) - 1, -1, -1):
        d_start = elections.iloc[i]["election_date"]
        # Find the next election date or use a future date (e.g., 2028-11-01) for the current cycle
        d_end = elections.iloc[i+1]["election_date"] if i + 1 < len(elections) else pd.to_datetime('2028-11-01')

        current_term_prices = prices[(prices["Date"] >= d_start) & (prices["Date"] < d_end)].copy()
        
        # Calculate the sequential trading day number starting from 0 on election day
        start_idx = prices.index[prices["Date"] == nearest_trade_day(d_start)].tolist()[0]
        
        for idx, row in current_term_prices.iterrows():
            # Get the position relative to the term start
            pos_in_full_prices = prices.index[prices["Date"] == row["Date"]].tolist()[0]
            trading_day_of_term = pos_in_full_prices - start_idx
            
            # 252 trading days is approximately 1 year
            # Year 1 (0 to 251), Year 2 (252 to 503), Year 3 (504 to 755), Year 4 (756 to 1007)
            pres_year = min(4, int(trading_day_of_term // 252) + 1)
            
            cycle_map[row["Date"]] = pres_year

    # 2. Join price data with Presidential Year
    cycle_df = prices.copy()
    cycle_df["PresidentialYear"] = cycle_df["Date"].map(cycle_map).fillna(method='ffill').fillna(method='bfill')
    cycle_df = cycle_df.dropna(subset=['PresidentialYear'])
    cycle_df["PresidentialYear"] = cycle_df["PresidentialYear"].astype(int).astype(str)
    
    # 3. Calculate Returns for each Presidential Year
    
    # Calculate daily returns
    cycle_df['DailyReturn'] = cycle_df['Close'].pct_change()
    
    # Calculate average annual return per Presidential Year
    yearly_returns = cycle_df.groupby(['PresidentialYear', cycle_df['Date'].dt.year])['DailyReturn'].sum().reset_index()
    avg_annual_returns = yearly_returns.groupby('PresidentialYear')['DailyReturn'].mean() * 100
    
    return avg_annual_returns.reset_index().rename(columns={'DailyReturn': 'Average Annual Return (%)'})

cycle_returns_df = build_presidential_cycle(prices, elections)

# ---------- End of New Presidential Cycle Function ----------
summary_df = build_election_summary()

# --- Executive KPI row: aggregated insights across elections ----
def safe_mean(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    return float(s.mean()) if len(s) else None

avg_pre  = safe_mean(summary_df["30d Before"])
avg_post = safe_mean(summary_df["30d After"])
avg_1y   = safe_mean(summary_df["1y After"])
avg_vol_change = safe_mean(summary_df["Vol Δ (pp)"])

st.markdown("## Election Impact Overview")

# KPIs (Key Performance Indicators)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Avg 30 Days Before Election", f"{avg_pre:.2f}%" if avg_pre is not None else "N/A", delta=f"{-avg_pre:.2f}% (Vol Change)" if avg_pre is not None else None, delta_color="inverse")
c2.metric("Avg 30 Days After Election", f"{avg_post:.2f}%" if avg_post is not None else "N/A", delta=f"{avg_post:.2f}%" if avg_post is not None else None)
c3.metric("Avg 1-Year After Election", f"{avg_1y:.2f}%" if avg_1y is not None else "N/A", delta=f"{avg_1y:.2f}%" if avg_1y is not None else None)
c4.metric("Avg Volatility Change", f"{avg_vol_change:.2f} pp" if avg_vol_change is not None else "N/A", delta=f"{avg_vol_change:.2f} pp", delta_color="inverse" if avg_vol_change and avg_vol_change > 0 else "normal")

st.markdown("---")

# Layout: Two columns for key charts, then a full-width summary table 
# --- Interactive Highlight Selector ---
election_options = ["(None - Show All)",] + summary_df["Election"].astype(str).tolist()
highlight_election = st.selectbox(
    "Highlight Individual Election Path:",
    options=election_options,
    index=0
)
col_chart_left, col_chart_right = st.columns([6, 4]) # Gives the left chart 60% and the right charts 40%

# ------------- LEFT COLUMN: Event study (average path −90 → +90) ----------
with col_chart_left:
    st.subheader("📈 Average Return Path Around Elections")

    window = 100  # Increase the window slightly for visual clarity
    aligned = []

    for _, r in elections.iterrows():
        d = r["election_date"]
        center = nearest_trade_day(d)
        center_idx = px_idx.index.get_loc(center)

        left = max(0, center_idx - window)
        right = min(len(px_idx) - 1, center_idx + window)

        seg = px_idx.iloc[left:right + 1].copy()
        
        # Calculate trading day offset and normalize
        center_pos = px_idx.index.get_loc(center)
        seg["TradingDaysFromElection"] = seg.index.map(lambda x: px_idx.index.get_loc(x) - center_pos)
        
        base = float(px_idx.iloc[center_idx]["Close"])
        seg["NormReturn"] = seg["Close"] / base - 1.0
        seg["ElectionDate"] = d.date() # For individual line display
        aligned.append(seg[["TradingDaysFromElection", "NormReturn", "ElectionDate"]])

    if aligned:
        # Combined DataFrame
        combined_df = pd.concat(aligned)
        
        # Calculate the average path
        avg_path = combined_df.groupby("TradingDaysFromElection")["NormReturn"].mean().reset_index()
        
        fig_ev = go.Figure()

        # Add individual election paths (faint lines for context)
        for date, group in combined_df.groupby("ElectionDate"):
            group = group[(group["TradingDaysFromElection"] >= -90) & (group["TradingDaysFromElection"] <= 90)]
            
            is_highlighted = (str(date) == highlight_election)
            line_color = 'orange' if is_highlighted else 'rgba(66, 133, 244, 0.2)'
            line_width = 3 if is_highlighted else 1

            fig_ev.add_trace(go.Scatter(
                x=group["TradingDaysFromElection"],
                y=group["NormReturn"] * 100.0,
                mode="lines",
                name=f"Election {date}" if is_highlighted else str(date),
                line=dict(width=line_width, color=line_color),
                hoverinfo='text',
                # Only show legend item for the highlighted line
                showlegend=is_highlighted,
                legendgroup="highlight"
            ))
            
        # Add the main average line (bold)
        fig_ev.add_trace(go.Scatter(
            x=avg_path["TradingDaysFromElection"],
            y=avg_path["NormReturn"] * 100.0,
            mode="lines",
            name="Average Path",
            line=dict(width=4, color='rgb(66, 133, 244)')
        ))
        
        fig_ev.add_vline(x=0, line_width=2, line_dash="dash", line_color="red", annotation_text="Election Day", annotation_position="top left")
        
        fig_ev.update_layout(
            xaxis_title="Trading Days from Election",
            yaxis_title="Average % Change (Cumulative)",
            hovermode="x unified",
            template="plotly_white",
            height=400,
            margin=dict(t=30, b=20)
        )
        st.plotly_chart(fig_ev, use_container_width=True)
    else:
        st.info("Not enough data to compute the event study for this index and history range.")

# ------------- RIGHT COLUMN: Party Comparison & Presidential Cycle ----------
with col_chart_right:
    # Use tabs for secondary charts in the right column
    right_tab1, right_tab2 = st.tabs(["🟥 Party Comparison", "📅 Presidential Cycle"])

    # Party Comparison
    with right_tab1:
        st.caption("Average Post-Election Returns by Party (1y & 30d)")
        
        # --- ROBUST CHECK & CLEANING ---
        party_col = 'Party' if 'Party' in summary_df.columns else ('party' if 'party' in summary_df.columns else None)
        
        if party_col and summary_df[party_col].notna().any():
            
            # Create a temporary column for clean grouping and visualization
            tmp_df = summary_df.copy()
            tmp_df["Party_Clean"] = tmp_df[party_col].astype(str).str.strip().str.capitalize()
            
            by_party = tmp_df.groupby("Party_Clean")[["30d After", "1y After"]].mean(numeric_only=True).reset_index()

            fig_party = px.bar(
                by_party.melt(id_vars="Party_Clean", var_name="Window", value_name="AvgReturn"),
                x="Party_Clean", y="AvgReturn", color="Window", barmode="group",
                color_discrete_map={'30d After': 'red', '1y After': 'blue'},
                labels={"Party_Clean": "Party", "AvgReturn": "Average Return (%)"},
                height=350,
                template="plotly_white"
            )
            fig_party.update_layout(margin=dict(t=20, b=20))
            st.plotly_chart(fig_party, use_container_width=True)
            
        else:
            st.warning("Your elections file is missing the required 'Party' column data. Please check `data/elections.csv`.")

    # Presidential Cycle
    with right_tab2:
        st.caption("Average Annualized Return by Presidential Term Year (Years 1-4)")
        fig_cycle = px.bar(
            cycle_returns_df,
            x="PresidentialYear", y="Average Annual Return (%)",
            labels={"PresidentialYear": "Presidential Year", "Average Annual Return (%)": "Avg. Annual Return (%)"},
            color="PresidentialYear",
            color_discrete_sequence=px.colors.qualitative.Plotly,
            height=350,
            template="plotly_white",
            title="The Four-Year Cycle"
        )
        fig_cycle.update_layout(showlegend=False, margin=dict(t=20, b=20))
        st.plotly_chart(fig_cycle, use_container_width=True)


st.markdown("---")

# Full-width Summary Table (formerly Tab 3)
st.subheader("🧾 Per-Election Summary Table (Including Volatility)")
show_cols = ["Election", "Winner", "Party", "30d Before", "30d After", "1y After", "Vol Pre (ann%)", "Vol Post (ann%)", "Vol Δ (pp)"]

# Use a larger font/better coloring for professional look
styled_df = summary_df[show_cols].sort_values("Election", ascending=False).style.format({
    "30d Before": "{:.2f}%",
    "30d After": "{:.2f}%",
    "1y After": "{:.2f}%",
    "Vol Pre (ann%)": "{:.2f}",
    "Vol Post (ann%)": "{:.2f}",
    "Vol Δ (pp)": "{:+.2f}"
}).background_gradient(subset=["Vol Δ (pp)"], cmap='RdYlGn', vmin=-10, vmax=10) # Color volatility change
# --- Add row highlighting based on user selection ---
if highlight_election != "(None - Show All)":
    # Find the index of the selected row
    # The 'Election' column is a date object, so ensure comparison is consistent
    highlight_index = summary_df[summary_df["Election"].astype(str) == highlight_election].index
    
    # Define the custom function to apply the style
    def highlight_row(row):
        # We use a dark yellow color (#555500) that works well on a dark background
        return ['background-color: #555500' if row.name in highlight_index else '' for _ in row]

    # Apply the new row style to the existing styled_df
    styled_df = styled_df.apply(highlight_row, axis=1)
st.dataframe(styled_df, use_container_width=True, height=450)


# ------------- TAB 4: Original price chart with markers (now a full tab again) ----------
st.markdown("---")
tab_price_chart = st.tabs(["📊 Full Price Chart with Election Markers"])

with tab_price_chart[0]:
    st.subheader(f"{symbol} Historical Price with U.S. Presidential Election Days")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=prices["Date"], y=prices["Close"],
        mode="lines", name=symbol, line=dict(width=2)
    ))
    for d in elections["election_date"]:
        if prices["Date"].min() <= d <= prices["Date"].max():
            fig.add_vline(x=d, line_dash="dash", opacity=0.5)
    fig.update_layout(
        xaxis_title="Date", yaxis_title="Close",
        hovermode="x unified", template="plotly_white",
        height=450
    )
    st.plotly_chart(fig, use_container_width=True)