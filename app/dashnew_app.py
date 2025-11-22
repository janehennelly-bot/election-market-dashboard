import pathlib
from functools import lru_cache

import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px

from dash import Dash, dcc, html, dash_table, Input, Output

# Basic app setup

app = Dash(__name__)
server = app.server  

app.title = "Elections & the S&P 500: Minimal MVP"

# Data locations
BASE_PATH = pathlib.Path(__file__).resolve().parent.parent
ELECTIONS_PATH = BASE_PATH / "data" / "elections.csv"

# Sector & Index Tickers 
SECTOR_MAP = {
    "S&P 500 (Broad Index)": "^GSPC",
    "Dow Jones (30 Stocks)": "^DJI",
    "Technology (XLK)": "XLK",
    "Financials (XLF)": "XLF",
    "Energy (XLE)": "XLE",
    "Health Care (XLV)": "XLV",
    "Industrials (XLI)": "XLI",
    "Real Estate (XLRE)": "XLRE",
}

# Elections data 
elections = pd.read_csv(ELECTIONS_PATH, parse_dates=["election_date"]).sort_values(
    "election_date"
)

# Ensure Party col is capitalized and clean
if "party" in elections.columns and "Party" not in elections.columns:
    elections = elections.rename(columns={"party": "Party"})

if "Party" in elections.columns:
    elections["Party"] = (
        elections["Party"].astype(str).str.strip().str.lower().str.capitalize()
    )

# Ensure datetime is timezone-naive
elections["election_date"] = pd.to_datetime(elections["election_date"]).dt.tz_localize(
    None
)

# Will be set inside callbacks whenever prices update
px_idx = None


# Prices and event-study utilities
@lru_cache(maxsize=16)
def load_prices(sym: str, years: int) -> pd.DataFrame:
    """
    Mirror Streamlit behavior: fetch up to `years` of daily prices, auto-adjusted.
    Cached so we don't hammer yfinance on every callback.
    """
    hist = yf.Ticker(sym).history(
        period=f"{years}y", interval="1d", auto_adjust=True
    ).reset_index()

    out = hist[["Date", "Close"]].dropna().copy()
    out["Date"] = pd.to_datetime(out["Date"]).dt.tz_localize(None)
    out["Close"] = out["Close"].astype(float)
    return out


def nearest_trade_day(dt: pd.Timestamp) -> pd.Timestamp:
    """Return the trading day in px_idx nearest to dt."""
    loc = px_idx.index.get_indexer([dt], method="nearest")[0]
    return px_idx.index[loc]


def value_on(dt: pd.Timestamp) -> float:
    """Close on nearest trading day to dt."""
    return float(px_idx.loc[nearest_trade_day(dt), "Close"])


def value_shifted(dt: pd.Timestamp, trading_days: int):
    """Close trading_days before/after the election day. Returns None if out of range."""
    center_pos = px_idx.index.get_indexer([nearest_trade_day(dt)], method="nearest")[0]
    target_pos = center_pos + trading_days
    if target_pos < 0 or target_pos >= len(px_idx.index):
        return None
    return float(px_idx.iloc[target_pos]["Close"])


def pct_change(a, b):
    """Return (b/a - 1) as percent; None if inputs missing."""
    if a is None or b is None or a == 0:
        return None
    return (b / a - 1.0) * 100.0


def build_election_summary(
    elections_df: pd.DataFrame,
    window_pre: int = 30,
    window_post: int = 30,
    window_1y: int = 252,
    vol_window: int = 60,
) -> pd.DataFrame:
    """
    Per-election summary: 30d before, 30d after, 1y after, and volatility change.
    Mirrors your Streamlit logic.
    """
    rows = []
    for _, r in elections_df.iterrows():
        d = r["election_date"]
        winner = r.get("winner", None)
        party = r.get("Party", None)

        # Prices at key offsets (trading-day shifts)
        p_before = value_shifted(d, -window_pre)
        p_at = value_on(d)
        p_after = value_shifted(d, window_post)
        p_1y = value_shifted(d, window_1y)

        pre_30 = pct_change(p_before, p_at)  # % move into election
        post_30 = pct_change(p_at, p_after)  # % move after election
        post_1y = pct_change(p_at, p_1y)  # % over 1y after

        # Volatility: stdev of daily returns before vs after (± vol_window)
        try:
            center = nearest_trade_day(d)
            center_idx = px_idx.index.get_loc(center)

            pre_slice = (
                px_idx.iloc[max(0, center_idx - vol_window) : center_idx + 1][
                    "Close"
                ]
                .pct_change()
                .dropna()
            )
            post_slice = (
                px_idx.iloc[center_idx : min(len(px_idx), center_idx + vol_window + 1)][
                    "Close"
                ]
                .pct_change()
                .dropna()
            )

            vol_pre = (
                float(pre_slice.std() * np.sqrt(252) * 100.0)
                if len(pre_slice) > 3
                else None
            )
            vol_post = (
                float(post_slice.std() * np.sqrt(252) * 100.0)
                if len(post_slice) > 3
                else None
            )
            vol_chg = (
                (vol_post - vol_pre)
                if (vol_pre is not None and vol_post is not None)
                else None
            )
        except Exception:
            vol_pre = vol_post = vol_chg = None

        rows.append(
            {
                "Election": d.date(),
                "Winner": winner,
                "Party": party,
                "30d Before": pre_30,
                "30d After": post_30,
                "1y After": post_1y,
                "Vol Pre (ann%)": vol_pre,
                "Vol Post (ann%)": vol_post,
                "Vol Δ (pp)": vol_chg,
            }
        )

    return pd.DataFrame(rows)


def build_presidential_cycle(prices: pd.DataFrame, elections_df: pd.DataFrame):
    """
    Your four-year presidential cycle logic, using trading-day counts.
    """
    cycle_map = {}

    for i in range(len(elections_df) - 1, -1, -1):
        d_start = elections_df.iloc[i]["election_date"]
        d_end = (
            elections_df.iloc[i + 1]["election_date"]
            if i + 1 < len(elections_df)
            else pd.to_datetime("2028-11-01")
        )

        current_term_prices = prices[
            (prices["Date"] >= d_start) & (prices["Date"] < d_end)
        ].copy()

        start_idx_list = prices.index[prices["Date"] == nearest_trade_day(d_start)]
        if not len(start_idx_list):
            continue
        start_idx = start_idx_list.tolist()[0]

        for _, row in current_term_prices.iterrows():
            pos_list = prices.index[prices["Date"] == row["Date"]]
            if not len(pos_list):
                continue
            pos_in_full_prices = pos_list.tolist()[0]
            trading_day_of_term = pos_in_full_prices - start_idx

            pres_year = min(4, int(trading_day_of_term // 252) + 1)
            cycle_map[row["Date"]] = pres_year

    cycle_df = prices.copy()
    cycle_df["PresidentialYear"] = (
        cycle_df["Date"].map(cycle_map).fillna(method="ffill").fillna(method="bfill")
    )
    cycle_df = cycle_df.dropna(subset=["PresidentialYear"])
    cycle_df["PresidentialYear"] = cycle_df["PresidentialYear"].astype(int).astype(str)

    cycle_df["DailyReturn"] = cycle_df["Close"].pct_change()

    yearly_returns = (
        cycle_df.groupby(["PresidentialYear", cycle_df["Date"].dt.year])["DailyReturn"]
        .sum()
        .reset_index()
    )
    avg_annual_returns = yearly_returns.groupby("PresidentialYear")[
        "DailyReturn"
    ].mean() * 100

    return (
        avg_annual_returns.reset_index().rename(
            columns={"DailyReturn": "Average Annual Return (%)"}
        )
    )


def safe_mean(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    return float(s.mean()) if len(s) else None


def color_for_vol_delta(v):
    """
    Approximate RdYlGn for Vol Δ (pp) in [-10, 10].
    Negative = greener, positive = redder.
    """
    if v is None or np.isnan(v):
        return "#000000"

    v = max(-10, min(10, v))
    # normalize to [0,1], where 0 -> green, 1 -> red
    t = (v + 10) / 20.0

    # 3-stop scale: green -> yellow -> red
    if t < 0.5:
        # green (0, 128, 0) to yellow (255, 255, 0)
        ratio = t / 0.5
        r = int(0 + ratio * (255 - 0))
        g = int(128 + ratio * (255 - 128))
        b = 0
    else:
        # yellow (255,255,0) to red (255,0,0)
        ratio = (t - 0.5) / 0.5
        r = 255
        g = int(255 - ratio * 255)
        b = 0

    return f"rgb({r},{g},{b})"


# Layout
app.layout = html.Div(
    style={
        "backgroundColor": "#0e1117",
        "color": "#f0f0f0",
        "minHeight": "100vh",
        "padding": "20px 40px",
        "fontFamily": "system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI'",
    },
    children=[
        html.Div(
            [
                html.H1(
                    "Elections & the S&P 500: Minimal MVP",
                    style={"marginBottom": "0.5rem"},
                ),
            ]
        ),
        html.Br(),
        # Controls row (sidebar-style but in a row)
        html.Div(
            style={"display": "flex", "gap": "40px", "flexWrap": "wrap"},
            children=[
                html.Div(
                    style={
                        "minWidth": "260px",
                        "maxWidth": "300px",
                        "backgroundColor": "#161a23",
                        "padding": "20px",
                        "borderRadius": "10px",
                    },
                    children=[
                        html.H4(
                            "Index / Sector ETF",
                            style={"fontSize": "0.9rem", "marginBottom": "6px"},
                        ),
                        dcc.Dropdown(
                            id="symbol-dropdown",
                            options=[
                                {"label": k, "value": v}
                                for k, v in SECTOR_MAP.items()
                            ],
                            value="^GSPC",
                            clearable=False,
                            style={"color": "#000000"},
                        ),
                        html.Br(),
                        html.H4(
                            "Years of history",
                            style={"fontSize": "0.9rem", "marginBottom": "0"},
                        ),
                        dcc.Slider(
                            id="years-slider",
                            min=5,
                            max=40,
                            step=1,
                            value=25,
                            marks={i: str(i) for i in range(5, 41, 5)},
                        ),
                    ],
                ),
            ],
        ),
        html.Br(),
        html.H2("Election Impact Overview"),
        html.Div(
            id="kpi-row",
            style={
                "display": "flex",
                "gap": "20px",
                "flexWrap": "wrap",
                "marginTop": "10px",
            },
            children=[
                html.Div(
                    id="kpi-pre",
                    style={
                        "flex": "1 1 200px",
                        "backgroundColor": "#161a23",
                        "padding": "15px 20px",
                        "borderRadius": "10px",
                    },
                ),
                html.Div(
                    id="kpi-post30",
                    style={
                        "flex": "1 1 200px",
                        "backgroundColor": "#161a23",
                        "padding": "15px 20px",
                        "borderRadius": "10px",
                    },
                ),
                html.Div(
                    id="kpi-post1y",
                    style={
                        "flex": "1 1 200px",
                        "backgroundColor": "#161a23",
                        "padding": "15px 20px",
                        "borderRadius": "10px",
                    },
                ),
                html.Div(
                    id="kpi-vol",
                    style={
                        "flex": "1 1 200px",
                        "backgroundColor": "#161a23",
                        "padding": "15px 20px",
                        "borderRadius": "10px",
                    },
                ),
            ],
        ),
        html.Hr(style={"borderColor": "#333"}),
        # Highlight selector
        html.Div(
            [
                html.Label(
                    "Highlight Individual Election Path:",
                    style={"fontWeight": "600"},
                ),
                dcc.Dropdown(
                    id="highlight-dropdown",
                    options=[],
                    value="NONE",
                    clearable=False,
                    style={"width": "420px", "color": "#000000"},
                ),
            ]
        ),
        html.Br(),
        # Main charts row
        html.Div(
            style={"display": "flex", "gap": "30px", "flexWrap": "wrap"},
            children=[
                html.Div(
                    style={"flex": "2 1 520px"},
                    children=[
                        html.H3("📈 Average Return Path Around Elections"),
                        dcc.Graph(
                            id="event-study-graph",
                            style={"height": "430px"},
                            config={"displaylogo": False},
                        ),
                    ],
                ),
                html.Div(
                    style={"flex": "1 1 360px"},
                    children=[
                        dcc.Tabs(
                            id="right-tabs",
                            value="party",
                            children=[
                                dcc.Tab(
                                    label="🟥 Party Comparison",
                                    value="party",
                                    children=[
                                        html.Div(
                                            style={"padding": "10px"},
                                            children=[
                                                html.Small(
                                                    "Average Post-Election Returns by Party (1y & 30d)"
                                                ),
                                                dcc.Graph(
                                                    id="party-graph",
                                                    style={"height": "340px"},
                                                    config={"displaylogo": False},
                                                ),
                                            ],
                                        )
                                    ],
                                ),
                                dcc.Tab(
                                    label="📅 Presidential Cycle",
                                    value="cycle",
                                    children=[
                                        html.Div(
                                            style={"padding": "10px"},
                                            children=[
                                                html.Small(
                                                    "Average Annualized Return by Presidential Term Year (Years 1–4)"
                                                ),
                                                dcc.Graph(
                                                    id="cycle-graph",
                                                    style={"height": "340px"},
                                                    config={"displaylogo": False},
                                                ),
                                            ],
                                        )
                                    ],
                                ),
                            ],
                        )
                    ],
                ),
            ],
        ),
        html.Hr(style={"borderColor": "#333", "marginTop": "25px"}),
        html.H3("🧾 Per-Election Summary Table (Including Volatility)"),
        dash_table.DataTable(
            id="summary-table",
            columns=[
                {"name": "Election", "id": "Election"},
                {"name": "Winner", "id": "Winner"},
                {"name": "Party", "id": "Party"},
                {"name": "30d Before", "id": "30d Before"},
                {"name": "30d After", "id": "30d After"},
                {"name": "1y After", "id": "1y After"},
                {"name": "Vol Pre (ann%)", "id": "Vol Pre (ann%)"},
                {"name": "Vol Post (ann%)", "id": "Vol Post (ann%)"},
                {"name": "Vol Δ (pp)", "id": "Vol Δ (pp)"},
            ],
            data=[],
            style_as_list_view=True,
            sort_action="native",
            page_size=20,
            style_header={
                "backgroundColor": "#11141c",
                "color": "#f0f0f0",
                "fontWeight": "600",
                "border": "1px solid #333",
            },
            style_cell={
                "backgroundColor": "#0e1117",
                "color": "#f0f0f0",
                "border": "1px solid #222",
                "padding": "6px 8px",
                "fontSize": "13px",
            },
            style_data_conditional=[],
        ),
    ],
)


# Callback
@app.callback(
    Output("kpi-pre", "children"),
    Output("kpi-post30", "children"),
    Output("kpi-post1y", "children"),
    Output("kpi-vol", "children"),
    Output("highlight-dropdown", "options"),
    Output("event-study-graph", "figure"),
    Output("party-graph", "figure"),
    Output("cycle-graph", "figure"),
    Output("summary-table", "data"),
    Output("summary-table", "style_data_conditional"),
    Input("symbol-dropdown", "value"),
    Input("years-slider", "value"),
    Input("highlight-dropdown", "value"),
)
def update_dashboard(symbol, years_back, highlight_value):
    global px_idx

    # Prices
    prices = load_prices(symbol, years_back)
    prices = prices[["Date", "Close"]].dropna().sort_values("Date").reset_index(
        drop=True
    )

    # Index by Date for fast lookup 
    px_idx = prices.set_index("Date").sort_index()

    # Summary DF & KPIs
    summary_df = build_election_summary(elections)

    avg_pre = safe_mean(summary_df["30d Before"])
    avg_post = safe_mean(summary_df["30d After"])
    avg_1y = safe_mean(summary_df["1y After"])
    avg_vol_change = safe_mean(summary_df["Vol Δ (pp)"])

    def kpi_block(label, value_str, delta_str=None, delta_positive=True):
        color = "#21ba45" if delta_positive else "#db2828"
        return html.Div(
            [
                html.Div(label, style={"fontSize": "0.8rem", "opacity": 0.8}),
                html.Div(
                    value_str,
                    style={"fontSize": "1.6rem", "fontWeight": "bold"},
                ),
                html.Div(
                    delta_str if delta_str is not None else "",
                    style={
                        "fontSize": "0.8rem",
                        "color": color,
                        "marginTop": "4px",
                    },
                ),
            ]
        )

    # KPI content (text + simple delta color)
    kpi_pre = kpi_block(
        "Avg 30 Days Before Election",
        f"{avg_pre:.2f}%" if avg_pre is not None else "N/A",
        delta_str=(
            f"{-avg_pre:.2f}% (Vol Change)" if avg_pre is not None else None
        ),
        delta_positive=(avg_pre is not None and avg_pre < 0),
    )

    kpi_post30 = kpi_block(
        "Avg 30 Days After Election",
        f"{avg_post:.2f}%" if avg_post is not None else "N/A",
        delta_str=(f"{avg_post:.2f}%" if avg_post is not None else None),
        delta_positive=(avg_post is not None and avg_post > 0),
    )

    kpi_post1y = kpi_block(
        "Avg 1-Year After Election",
        f"{avg_1y:.2f}%" if avg_1y is not None else "N/A",
        delta_str=(f"{avg_1y:.2f}%" if avg_1y is not None else None),
        delta_positive=(avg_1y is not None and avg_1y > 0),
    )

    kpi_vol = kpi_block(
        "Avg Volatility Change",
        f"{avg_vol_change:.2f} pp" if avg_vol_change is not None else "N/A",
        delta_str=(
            f"{avg_vol_change:.2f} pp" if avg_vol_change is not None else None
        ),
        # Higher vol_change => worse (red)
        delta_positive=(avg_vol_change is not None and avg_vol_change <= 0),
    )

    # Highlight dropdown options
    election_options = [
        {"label": "(None - Show All)", "value": "NONE"}
    ] + [
        {"label": str(d), "value": str(d)}
        for d in summary_df["Election"].astype(str).tolist()
    ]

    # Normalized highlight selection
    highlighted_date_str = (
        None if highlight_value in (None, "", "NONE") else str(highlight_value)
    )

    # Event-study chart
    window = 100
    aligned = []

    for _, r in elections.iterrows():
        d = r["election_date"]
        center = nearest_trade_day(d)
        center_idx = px_idx.index.get_loc(center)

        left = max(0, center_idx - window)
        right = min(len(px_idx) - 1, center_idx + window)

        seg = px_idx.iloc[left : right + 1].copy()

        center_pos = px_idx.index.get_loc(center)
        seg["TradingDaysFromElection"] = seg.index.map(
            lambda x: px_idx.index.get_loc(x) - center_pos
        )

        base = float(px_idx.iloc[center_idx]["Close"])
        seg["NormReturn"] = seg["Close"] / base - 1.0
        seg["ElectionDate"] = d.date()
        aligned.append(seg[["TradingDaysFromElection", "NormReturn", "ElectionDate"]])

    fig_ev = go.Figure()
    if aligned:
        combined_df = pd.concat(aligned)
        avg_path = (
            combined_df.groupby("TradingDaysFromElection")["NormReturn"]
            .mean()
            .reset_index()
        )

        # Individual paths
        for date, group in combined_df.groupby("ElectionDate"):
            group = group[
                (group["TradingDaysFromElection"] >= -90)
                & (group["TradingDaysFromElection"] <= 90)
            ]
            date_str = str(date)
            is_highlighted = (highlighted_date_str == date_str)

            line_color = "orange" if is_highlighted else "rgba(66,133,244,0.25)"
            line_width = 3 if is_highlighted else 1

            fig_ev.add_trace(
                go.Scatter(
                    x=group["TradingDaysFromElection"],
                    y=group["NormReturn"] * 100.0,
                    mode="lines",
                    name=f"Election {date_str}" if is_highlighted else date_str,
                    line=dict(width=line_width, color=line_color),
                    hoverinfo="text",
                    showlegend=is_highlighted,
                    legendgroup="highlight",
                )
            )

        # Average path
        fig_ev.add_trace(
            go.Scatter(
                x=avg_path["TradingDaysFromElection"],
                y=avg_path["NormReturn"] * 100.0,
                mode="lines",
                name="Average Path",
                line=dict(width=4, color="rgb(66,133,244)"),
            )
        )

    fig_ev.add_vline(
        x=0,
        line_width=2,
        line_dash="dash",
        line_color="red",
    )
    fig_ev.update_layout(
        xaxis_title="Trading Days from Election",
        yaxis_title="Average % Change (Cumulative)",
        hovermode="x unified",
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        height=400,
        margin=dict(t=30, b=40, l=60, r=20),
    )

    # Party comparison chart
    party_col = (
        "Party"
        if "Party" in summary_df.columns
        else ("party" if "party" in summary_df.columns else None)
    )

    if party_col and summary_df[party_col].notna().any():
        tmp_df = summary_df.copy()
        tmp_df["Party_Clean"] = (
            tmp_df[party_col].astype(str).str.strip().str.capitalize()
        )

        by_party = (
            tmp_df.groupby("Party_Clean")[["30d After", "1y After"]]
            .mean(numeric_only=True)
            .reset_index()
        )

        df_melt = by_party.melt(
            id_vars="Party_Clean", var_name="Window", value_name="AvgReturn"
        )

        fig_party = px.bar(
            df_melt,
            x="Party_Clean",
            y="AvgReturn",
            color="Window",
            barmode="group",
            labels={
                "Party_Clean": "Party",
                "AvgReturn": "Average Return (%)",
                "Window": "Window",
            },
            template="plotly_dark",
            height=340,
        )
        fig_party.update_layout(
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
            margin=dict(t=30, b=40, l=40, r=10),
        )
    else:
        fig_party = go.Figure()
        fig_party.update_layout(
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
            annotations=[
                dict(
                    text="Missing Party data in elections.csv",
                    x=0.5,
                    y=0.5,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(color="white"),
                )
            ],
        )

    # Presidential cycle chart
    cycle_returns_df = build_presidential_cycle(prices, elections)

    fig_cycle = px.bar(
        cycle_returns_df,
        x="PresidentialYear",
        y="Average Annual Return (%)",
        labels={
            "PresidentialYear": "Presidential Year",
            "Average Annual Return (%)": "Avg. Annual Return (%)",
        },
        color="PresidentialYear",
        template="plotly_dark",
        height=340,
    )
    fig_cycle.update_layout(
        showlegend=False,
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        margin=dict(t=30, b=40, l=40, r=10),
        title_text="The Four-Year Cycle",
        title_font_size=16,
    )

    # Summary table: formatted strings and row coloring
    show_cols = [
        "Election",
        "Winner",
        "Party",
        "30d Before",
        "30d After",
        "1y After",
        "Vol Pre (ann%)",
        "Vol Post (ann%)",
        "Vol Δ (pp)",
    ]

    display_df = summary_df[show_cols].sort_values("Election", ascending=False).copy()

    # keep numeric copy for coloring
    num_vol_delta = display_df["Vol Δ (pp)"].values

    # Format numbers as strings
    def fmt_pct(x):
        return "None" if x is None or pd.isna(x) else f"{x:.2f}%"

    def fmt_plain(x):
        return "None" if x is None or pd.isna(x) else f"{x:.2f}"

    display_df["30d Before"] = display_df["30d Before"].apply(fmt_pct)
    display_df["30d After"] = display_df["30d After"].apply(fmt_pct)
    display_df["1y After"] = display_df["1y After"].apply(fmt_pct)
    display_df["Vol Pre (ann%)"] = display_df["Vol Pre (ann%)"].apply(fmt_plain)
    display_df["Vol Post (ann%)"] = display_df["Vol Post (ann%)"].apply(fmt_plain)
    display_df["Vol Δ (pp)"] = display_df["Vol Δ (pp)"].apply(
        lambda x: "None" if x is None or pd.isna(x) else f"{x:+.2f}"
    )

    table_data = display_df.to_dict("records")

    # style_data_conditional for heatmap + row highlight
    styles = []

    # Vol Δ heat colors
    for i, v in enumerate(num_vol_delta):
        color = color_for_vol_delta(v) if v is not None and not pd.isna(v) else None
        if color:
            styles.append(
                {
                    "if": {"row_index": i, "column_id": "Vol Δ (pp)"},
                    "backgroundColor": color,
                    "color": "#000000",
                }
            )

    # Row highlight based on selection
    if highlighted_date_str is not None:
        for i, rec in enumerate(table_data):
            if str(rec["Election"]) == highlighted_date_str:
                styles.append(
                    {
                        "if": {"row_index": i},
                        "backgroundColor": "#555500",
                    }
                )

    return (
        kpi_pre,
        kpi_post30,
        kpi_post1y,
        kpi_vol,
        election_options,
        fig_ev,
        fig_party,
        fig_cycle,
        table_data,
        styles,
    )


# Main
if __name__ == "__main__":
    app.run(debug=True)
