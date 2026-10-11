"""
Portfolio Stress Tester
-----------------------
Would your portfolio have survived 2008? Enter a few tickers and weights and get:
  - Crisis replays (2008, COVID 2020, 2022 rate hikes) vs the S&P 500
  - Value at Risk (VaR) and Expected Shortfall (CVaR)
  - Monte Carlo simulation of possible futures
  - Efficient frontier with max-Sharpe and min-volatility portfolios

Educational use only. Not financial advice.
Run with:  streamlit run app.py
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from scipy.optimize import minimize

st.set_page_config(page_title="Portfolio Stress Tester", page_icon="📉", layout="wide")

TRADING_DAYS = 252
START_DATE = "2006-01-01"
BENCHMARK = "SPY"

CRISES = {
    "2008 Financial Crisis": ("2007-10-09", "2009-03-09"),
    "COVID-19 Crash": ("2020-02-19", "2020-03-23"),
    "2022 Rate Hikes": ("2022-01-03", "2022-10-12"),
}


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Downloading market data...", ttl=3600)
def load_prices(tickers: tuple, start: str) -> pd.DataFrame:
    """Adjusted close prices for the given tickers (one column per ticker)."""
    raw = yf.download(list(tickers), start=start, auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        return pd.DataFrame()
    prices = raw["Close"]
    if isinstance(prices, pd.Series):  # single ticker edge case
        prices = prices.to_frame(name=tickers[0])
    return prices.dropna(how="all").dropna(axis=1, how="all")


def parse_portfolio(df: pd.DataFrame):
    """Turn the editable table into a normalized weight Series (sums to 1)."""
    df = df.dropna(subset=["Ticker"]).copy()
    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    df = df[(df["Ticker"] != "") & (df["Weight (%)"] > 0)]
    if df.empty:
        return None
    df = df.groupby("Ticker", as_index=False)["Weight (%)"].sum()
    return pd.Series(
        df["Weight (%)"].values / df["Weight (%)"].sum(), index=df["Ticker"].values
    )


# ----------------------------------------------------------------------------
# Risk maths
# ----------------------------------------------------------------------------
def max_drawdown(growth: pd.Series) -> float:
    """Largest peak-to-trough fall of a growth curve (returned as a negative number)."""
    return float((growth / growth.cummax() - 1).min())


def var_cvar(returns: pd.Series, confidence: float):
    """
    Historical 1-day VaR and CVaR (Expected Shortfall), as positive loss fractions.
    VaR  = the loss you'd only exceed on (1 - confidence) of days.
    CVaR = the average loss on those worst days.
    """
    cutoff = np.percentile(returns, (1 - confidence) * 100)
    tail = returns[returns <= cutoff]
    return float(-cutoff), float(-tail.mean())


def crisis_replay(prices: pd.DataFrame, weights: pd.Series, start: str, end: str):
    """
    Replay the portfolio through a historical crisis window.
    Tickers that did not exist then are dropped and the rest are re-weighted.
    """
    window = prices[weights.index].loc[start:end].dropna(axis=1)
    if len(window) < 5 or window.shape[1] == 0:
        return None
    w = weights[window.columns]
    coverage = float(w.sum())
    w = w / w.sum()

    daily = window.pct_change(fill_method=None).dropna().dot(w)
    growth = (1 + daily).cumprod()
    growth = pd.concat([pd.Series([1.0], index=[window.index[0]]), growth])
    return {
        "growth": growth,
        "total_return": float(growth.iloc[-1] - 1),
        "max_drawdown": max_drawdown(growth),
        "coverage": coverage,
    }


def monte_carlo(port_returns: pd.Series, horizon: int, n_sims: int, seed: int = 42):
    """
    Simulate portfolio paths with normally distributed daily returns drawn from the
    portfolio's historical mean and volatility. Returns an array (horizon+1, n_sims)
    of growth multipliers starting at 1.0.
    """
    rng = np.random.default_rng(seed)
    mu, sigma = port_returns.mean(), port_returns.std()
    shocks = rng.normal(mu, sigma, size=(horizon, n_sims))
    return np.vstack([np.ones(n_sims), np.cumprod(1 + shocks, axis=0)])


def efficient_frontier(returns: pd.DataFrame, n_random: int = 4000, seed: int = 1):
    """Random portfolios plus the optimizer-found max-Sharpe and min-volatility mixes."""
    mean = returns.mean().values * TRADING_DAYS
    cov = returns.cov().values * TRADING_DAYS
    k = len(mean)

    rng = np.random.default_rng(seed)
    w_rand = rng.dirichlet(np.ones(k), size=n_random)
    r_rand = w_rand @ mean
    v_rand = np.sqrt(np.einsum("ij,jk,ik->i", w_rand, cov, w_rand))

    constraints = {"type": "eq", "fun": lambda w: w.sum() - 1}
    bounds = [(0.0, 1.0)] * k
    x0 = np.ones(k) / k

    def vol(w):
        return np.sqrt(w @ cov @ w)

    def neg_sharpe(w):
        return -(w @ mean) / vol(w)

    max_sharpe = minimize(neg_sharpe, x0, bounds=bounds, constraints=constraints, method="SLSQP").x
    min_vol = minimize(vol, x0, bounds=bounds, constraints=constraints, method="SLSQP").x

    return {
        "rand_ret": r_rand,
        "rand_vol": v_rand,
        "rand_sharpe": r_rand / v_rand,
        "max_sharpe": max_sharpe,
        "min_vol": min_vol,
        "mean": mean,
        "cov": cov,
    }


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


# ----------------------------------------------------------------------------
# Sidebar inputs
# ----------------------------------------------------------------------------
st.sidebar.title("📉 Your portfolio")
default_table = pd.DataFrame(
    {"Ticker": ["AAPL", "SPY", "GLD"], "Weight (%)": [40.0, 30.0, 30.0]}
)
edited = st.sidebar.data_editor(
    default_table,
    num_rows="dynamic",
    use_container_width=True,
    hide_index=True,
    column_config={
        "Ticker": st.column_config.TextColumn("Ticker"),
        "Weight (%)": st.column_config.NumberColumn("Weight (%)", min_value=0.0, step=1.0),
    },
)
capital = st.sidebar.number_input("Portfolio value ($)", min_value=100, value=10_000, step=1_000)
confidence = st.sidebar.slider("VaR confidence level", 0.90, 0.99, 0.95, 0.01)
lookback = st.sidebar.slider("Risk lookback (years)", 1, 15, 5)
horizon = st.sidebar.slider("Simulation horizon (trading days)", 21, 756, 252, 21)
n_sims = st.sidebar.selectbox("Monte Carlo paths", [1_000, 5_000, 10_000], index=1)

# ----------------------------------------------------------------------------
# Load + prepare data
# ----------------------------------------------------------------------------
st.title("Would your portfolio survive the next crash?")
st.caption(
    "Educational tool, not financial advice. Based on historical data from Yahoo Finance; "
    "past performance does not predict future results."
)

weights = parse_portfolio(edited)
if weights is None:
    st.info("Add at least one ticker with a positive weight in the sidebar.")
    st.stop()

requested = list(weights.index)
prices = load_prices(tuple(sorted(set(requested + [BENCHMARK]))), START_DATE)
if prices.empty:
    st.error("Could not download market data. Check your internet connection and tickers.")
    st.stop()

missing = [t for t in requested if t not in prices.columns]
if missing:
    st.warning(f"No data found for: {', '.join(missing)}. They were removed from the portfolio.")
tickers = [t for t in requested if t in prices.columns]
if not tickers:
    st.error("None of the tickers returned data.")
    st.stop()
weights = weights[tickers] / weights[tickers].sum()

returns_all = prices[tickers].pct_change(fill_method=None).dropna()
cutoff_date = returns_all.index[-1] - pd.DateOffset(years=lookback)
returns = returns_all.loc[cutoff_date:]
if len(returns) < 60:
    st.error("Not enough overlapping history for these tickers. Try fewer or older tickers.")
    st.stop()

port_ret = returns.dot(weights)
growth = (1 + port_ret).cumprod()

tab_risk, tab_crisis, tab_mc, tab_ef = st.tabs(
    ["Risk overview", "Crisis replay", "Monte Carlo", "Efficient frontier"]
)

# ----------------------------------------------------------------------------
# Tab 1: risk overview
# ----------------------------------------------------------------------------
with tab_risk:
    n_days = len(port_ret)
    cagr = growth.iloc[-1] ** (TRADING_DAYS / n_days) - 1
    ann_vol = port_ret.std() * np.sqrt(TRADING_DAYS)
    sharpe = (port_ret.mean() * TRADING_DAYS) / ann_vol
    mdd = max_drawdown(growth)
    var, cvar = var_cvar(port_ret, confidence)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Annual return (CAGR)", pct(cagr))
    c2.metric("Annual volatility", pct(ann_vol))
    c3.metric("Sharpe ratio (rf = 0)", f"{sharpe:.2f}")
    c4.metric("Max drawdown", pct(mdd))

    c5, c6 = st.columns(2)
    c5.metric(f"1-day VaR ({confidence:.0%})", f"${var * capital:,.0f}", pct(-var), delta_color="off")
    c6.metric(f"1-day CVaR ({confidence:.0%})", f"${cvar * capital:,.0f}", pct(-cvar), delta_color="off")

    one_in = int(round(1 / (1 - confidence)))
    st.markdown(
        f"**In plain English:** on roughly 1 trading day in {one_in}, this portfolio has lost "
        f"**{pct(var)} (\\${var * capital:,.0f})** or more. On those bad days, the average loss "
        f"was **{pct(cvar)} (\\${cvar * capital:,.0f})**."
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=growth.index, y=growth * capital, name="Portfolio", line=dict(width=2.5)))
    if BENCHMARK in prices.columns:
        bench = prices[BENCHMARK].loc[growth.index[0]:].dropna()
        fig.add_trace(
            go.Scatter(x=bench.index, y=bench / bench.iloc[0] * capital, name="S&P 500 (SPY)",
                       line=dict(dash="dash"))
        )
    fig.update_layout(title=f"Growth of ${capital:,.0f} over the last {lookback} years",
                      yaxis_title="Value ($)", hovermode="x unified", height=420)
    st.plotly_chart(fig, use_container_width=True)

    st.caption("Weights are assumed to be rebalanced daily. VaR uses historical simulation.")

# ----------------------------------------------------------------------------
# Tab 2: crisis replay
# ----------------------------------------------------------------------------
with tab_crisis:
    st.subheader("How would this portfolio have done?")
    cols = st.columns(len(CRISES))
    for col, (name, (start, end)) in zip(cols, CRISES.items()):
        with col:
            st.markdown(f"**{name}**")
            result = crisis_replay(prices, weights, start, end)
            if result is None:
                st.warning("Your tickers have no data for this period.")
                continue
            m1, m2 = st.columns(2)
            m1.metric("Total return", pct(result["total_return"]))
            m2.metric("Max drawdown", pct(result["max_drawdown"]))
            if result["coverage"] < 0.999:
                st.caption(
                    f"Only {result['coverage']:.0%} of your portfolio existed then. "
                    "The rest was dropped and the remainder re-weighted."
                )

            f = go.Figure()
            f.add_trace(go.Scatter(x=result["growth"].index, y=result["growth"], name="Portfolio",
                                   line=dict(width=2.5)))
            if BENCHMARK in prices.columns:
                b = prices[BENCHMARK].loc[start:end].dropna()
                if len(b) > 1:
                    f.add_trace(go.Scatter(x=b.index, y=b / b.iloc[0], name="SPY", line=dict(dash="dash")))
            f.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10),
                            yaxis_title="Growth of $1", legend=dict(orientation="h", y=-0.2))
            st.plotly_chart(f, use_container_width=True)

# ----------------------------------------------------------------------------
# Tab 3: Monte Carlo
# ----------------------------------------------------------------------------
with tab_mc:
    paths = monte_carlo(port_ret, horizon, n_sims)
    x = np.arange(paths.shape[0])
    p5, p25, p50, p75, p95 = (np.percentile(paths, q, axis=1) * capital for q in (5, 25, 50, 75, 95))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=p95, line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=p5, fill="tonexty", name="5th-95th percentile",
                             line=dict(width=0), opacity=0.25))
    fig.add_trace(go.Scatter(x=x, y=p75, line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=p25, fill="tonexty", name="25th-75th percentile",
                             line=dict(width=0), opacity=0.45))
    fig.add_trace(go.Scatter(x=x, y=p50, name="Median", line=dict(width=3)))
    fig.add_hline(y=capital, line_dash="dot", annotation_text="Starting value")
    fig.update_layout(title=f"{n_sims:,} simulated futures over {horizon} trading days",
                      xaxis_title="Trading days", yaxis_title="Portfolio value ($)", height=460)
    st.plotly_chart(fig, use_container_width=True)

    final = paths[-1] * capital
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Median outcome", f"${np.median(final):,.0f}")
    m2.metric("Bad case (5th pct)", f"${np.percentile(final, 5):,.0f}")
    m3.metric("Good case (95th pct)", f"${np.percentile(final, 95):,.0f}")
    m4.metric("Chance of losing money", pct(float((final < capital).mean())))
    st.caption(
        "Daily returns are modelled as normally distributed using this portfolio's historical "
        "mean and volatility. Real markets have fatter tails, so treat this as an optimistic baseline."
    )

# ----------------------------------------------------------------------------
# Tab 4: efficient frontier
# ----------------------------------------------------------------------------
with tab_ef:
    if len(tickers) < 2:
        st.info("Add at least two tickers to see the efficient frontier.")
    else:
        ef = efficient_frontier(returns)
        w_user = weights.values
        user_ret = float(w_user @ ef["mean"])
        user_vol = float(np.sqrt(w_user @ ef["cov"] @ w_user))

        def point(w):
            return float(w @ ef["mean"]), float(np.sqrt(w @ ef["cov"] @ w))

        ms_ret, ms_vol = point(ef["max_sharpe"])
        mv_ret, mv_vol = point(ef["min_vol"])

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=ef["rand_vol"], y=ef["rand_ret"], mode="markers", name="Random portfolios",
            marker=dict(size=5, color=ef["rand_sharpe"], colorscale="Viridis", showscale=True,
                        colorbar=dict(title="Sharpe"), opacity=0.7)))
        fig.add_trace(go.Scatter(x=[user_vol], y=[user_ret], mode="markers", name="Your portfolio",
                                 marker=dict(size=18, symbol="star", color="red", line=dict(width=1, color="white"))))
        fig.add_trace(go.Scatter(x=[ms_vol], y=[ms_ret], mode="markers", name="Max Sharpe",
                                 marker=dict(size=14, symbol="diamond", color="gold", line=dict(width=1, color="black"))))
        fig.add_trace(go.Scatter(x=[mv_vol], y=[mv_ret], mode="markers", name="Min volatility",
                                 marker=dict(size=14, symbol="square", color="white", line=dict(width=1, color="black"))))
        fig.update_layout(title="Risk vs return", xaxis_title="Annual volatility",
                          yaxis_title="Annual expected return", xaxis_tickformat=".0%",
                          yaxis_tickformat=".0%", height=500)
        st.plotly_chart(fig, use_container_width=True)

        table = pd.DataFrame(
            {"Your mix": weights.values, "Max Sharpe": ef["max_sharpe"], "Min volatility": ef["min_vol"]},
            index=tickers,
        )
        st.dataframe(table.style.format("{:.1%}"), use_container_width=True)
        st.caption(
            "Optimal weights are calculated on past data only and shift with the lookback window. "
            "They illustrate the trade-off between risk and return and are not a recommendation."
        )
