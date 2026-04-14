"""Interactive Portfolio Analytics Application — Financial Data Analytics II"""
import warnings; warnings.filterwarnings("ignore")

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats
from scipy.optimize import minimize

st.set_page_config(page_title="Portfolio Analytics", page_icon="📈",
                   layout="wide", initial_sidebar_state="expanded")

# ─── Core statistics ──────────────────────────────────────────────────────────

def series_stats(r: pd.Series, rf: float) -> dict:
    """All scalar stats for a single daily return series."""
    rf_d, v = rf / 252, r.std() * 252 ** 0.5
    exc = r.mean() * 252 - rf
    down_std = r[r < rf_d].std() * 252 ** 0.5
    wealth = (1 + r).cumprod()
    return dict(
        ret=r.mean() * 252, vol=v,
        sharpe=exc / v if v else np.nan,
        sortino=exc / down_std if down_std else np.nan,
        mdd=((wealth - wealth.cummax()) / wealth.cummax()).min(),
        skew=r.skew(), kurt=r.kurt(), rmin=r.min(), rmax=r.max(),
    )

def fmt(s: dict) -> dict:
    """Format stats dict for display."""
    return {"Ann. Return": f"{s['ret']:.2%}", "Ann. Volatility": f"{s['vol']:.2%}",
            "Sharpe Ratio": f"{s['sharpe']:.3f}", "Sortino Ratio": f"{s['sortino']:.3f}",
            "Max Drawdown": f"{s['mdd']:.2%}"}

def risk_contribution(w: np.ndarray, cov: np.ndarray) -> np.ndarray:
    pv = w @ cov @ w
    return (w * (cov @ w)) / pv if pv > 0 else np.zeros_like(w)

def drawdown_series(r: pd.Series) -> pd.Series:
    w = (1 + r).cumprod()
    return (w - w.cummax()) / w.cummax()

# ─── Optimization (cached) ────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def run_optimizations(mu_list, cov_list, rf: float):
    mu, cov = np.array(mu_list), np.array(cov_list)
    n = len(mu)
    w0, bnds = np.full(n, 1/n), [(0, 1)] * n
    eq = [{"type": "eq", "fun": lambda w: w.sum() - 1}]

    gmv = minimize(lambda w: w @ cov @ w, w0, method="SLSQP", bounds=bnds, constraints=eq,
                   options={"maxiter": 1000, "ftol": 1e-12})
    def neg_sharpe(w):
        v = np.sqrt(w @ cov @ w * 252)
        return -(w @ mu * 252 - rf) / v if v > 1e-12 else np.inf
    tan = minimize(neg_sharpe, w0, method="SLSQP", bounds=bnds, constraints=eq,
                   options={"maxiter": 1000, "ftol": 1e-12})
    return (gmv.success, gmv.x.tolist()), (tan.success, tan.x.tolist())

@st.cache_data(ttl=3600)
def efficient_frontier(mu_list, cov_list, n_pts=120):
    mu, cov = np.array(mu_list), np.array(cov_list)
    n = len(mu); w0 = np.full(n, 1/n); bnds = [(0, 1)] * n
    vols, rets = [], []
    for t in np.linspace((mu * 252).min(), (mu * 252).max(), n_pts):
        res = minimize(lambda w: w @ cov @ w, w0, method="SLSQP", bounds=bnds,
                       constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1},
                                    {"type": "eq", "fun": lambda w, t=t: w @ mu * 252 - t}],
                       options={"maxiter": 500})
        if res.success and res.fun >= 0:
            vols.append(float(np.sqrt(res.fun * 252))); rets.append(float(t))
    return vols, rets

# ─── Data download (cached) ───────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def download_prices(tickers_tuple, start: str, end: str):
    data, bench, bench_ok = {}, None, True
    for t in list(tickers_tuple) + ["^GSPC"]:
        try:
            raw = yf.download(t, start=start, end=end, auto_adjust=True,
                  progress=False)
            if raw.empty or len(raw) < 50:
                bench_ok = bench_ok and t != "^GSPC"; continue
            close = raw["Close"]
            if isinstance(close, pd.DataFrame): close = close.iloc[:, 0]
            data[t] = close.rename(t)
        except Exception:
            if t == "^GSPC": bench_ok = False

    failed = [t for t in tickers_tuple if t not in data]
    if failed: return None, None, failed, bench_ok, None

    prices = pd.concat([data[t] for t in tickers_tuple], axis=1)
    bad = prices.columns[prices.isnull().mean() > 0.05].tolist()
    if bad: return None, None, bad, bench_ok, None

    prices = prices.dropna()
    warn = None
    if "^GSPC" in data:
        bench = data["^GSPC"].reindex(prices.index).ffill()
    return prices, bench, [], bench_ok, warn

# ─── Sidebar inputs ───────────────────────────────────────────────────────────

st.sidebar.title("📈 Portfolio Analytics")
st.sidebar.markdown("---")
st.sidebar.header("Configuration")

ticker_raw = st.sidebar.text_area("Tickers (3–10, comma-separated)", value="AAPL, MSFT, GOOGL, AMZN, NVDA")
c1, c2 = st.sidebar.columns(2)
start_date = c1.date_input("Start", value=pd.Timestamp("2019-01-01"))
end_date   = c2.date_input("End",   value=pd.Timestamp("2024-12-31"))
rf_rate = st.sidebar.number_input("Risk-free rate (annual %)", 0.0, 25.0, 2.0, 0.1) / 100
if st.sidebar.button("🚀 Run Analysis", type="primary", use_container_width=True):
    st.session_state.has_run = True
run = st.session_state.get("has_run", False)

with st.sidebar.expander("ℹ️ About & Methodology"):
    st.markdown("""
**Data:** Yahoo Finance (`yfinance`) · adjusted closing prices
**Returns:** Simple (arithmetic) daily returns
**Annualization:** ×252 trading days
**Risk-free rate:** annualized ÷ 252 for daily conversion
**Sharpe:** (Ann. excess return) / (Ann. volatility)
**Sortino:** (Ann. excess return) / (Ann. downside deviation)
**Portfolio variance:** w′Σw — full quadratic form
**Optimization:** SLSQP, no short-selling (w∈[0,1], Σw=1)
**Efficient frontier:** constrained optimization at each target return
**Benchmark:** S&P 500 (^GSPC) — comparison only, excluded from optimization
""")

# ─── Validation & data load ───────────────────────────────────────────────────

st.title("📊 Interactive Portfolio Analytics")

if if not run:
    st.session_state.has_run = False
    st.info("👈 Configure your portfolio in the sidebar and click **Run Analysis**."); st.stop()); st.stop()

tickers = list(dict.fromkeys(t.strip().upper() for t in ticker_raw.replace(";",",").split(",") if t.strip()))

errors = []
if len(tickers) < 3: errors.append("At least **3** tickers required.")
if len(tickers) > 10: errors.append("No more than **10** tickers allowed.")
if (pd.Timestamp(end_date) - pd.Timestamp(start_date)).days < 730:
    errors.append("Date range must be **at least 2 years**.")
if start_date >= end_date: errors.append("Start date must be before end date.")
for e in errors: st.error(f"❌ {e}")
if errors: st.stop()

with st.spinner("📥 Downloading data from Yahoo Finance…"):
    prices, bench_ser, failed, bench_ok, warn_msg = download_prices(
        tuple(tickers), start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))

if failed:
    st.error(f"❌ Ticker(s) failed to download or had insufficient data: **{', '.join(failed)}**. "
             "Check symbols and try again."); st.stop()
if prices is None:
    st.error("❌ Could not load data. Check your inputs and try again."); st.stop()
if not bench_ok:
    st.warning("⚠️ S&P 500 (^GSPC) data unavailable — benchmark comparisons omitted.")
if warn_msg:
    st.warning(warn_msg)

# Precompute core objects
returns   = prices.pct_change().dropna()
assets    = list(returns.columns)
n         = len(assets)
mu        = returns.mean()
cov       = returns.cov()
mu_l, cov_l = mu.values.tolist(), [row.tolist() for row in cov.values]
eq_w      = np.full(n, 1/n)

bench_r: pd.Series | None = None
if bench_ser is not None:
    bench_r = bench_ser.pct_change().dropna().reindex(returns.index).dropna()

st.success(f"✅ Loaded **{', '.join(assets)}** | "
           f"{returns.index[0].date()} → {returns.index[-1].date()} | "
           f"{len(returns):,} trading days")

# Run optimizations (cached)
with st.spinner("⚙️ Optimizing portfolios…"):
    (gmv_ok, gmv_wl), (tan_ok, tan_wl) = run_optimizations(mu_l, cov_l, rf_rate)

gmv_w = np.array(gmv_wl) if gmv_ok else eq_w.copy()
tan_w = np.array(tan_wl) if tan_ok else eq_w.copy()
if not gmv_ok: st.warning("⚠️ GMV optimization did not converge — results approximate.")
if not tan_ok: st.warning("⚠️ Tangency optimization did not converge — results approximate.")

# ─── Shared UI helpers ────────────────────────────────────────────────────────

def show_metrics(w: np.ndarray):
    """Display 5 portfolio metrics in st.metric columns."""
    s = series_stats(returns @ w, rf_rate)
    cols = st.columns(5)
    for col, (k, v) in zip(cols, fmt(s).items()): col.metric(k, v)

def weight_bar(w: np.ndarray, title: str, color: str):
    fig = go.Figure(go.Bar(x=assets, y=w, marker_color=color))
    fig.update_layout(title=title, xaxis_title="Asset", yaxis_title="Weight",
                      yaxis_tickformat=".0%", margin=dict(t=40, b=20))
    return fig

def prc_bar(w: np.ndarray, title: str, color: str):
    prc = risk_contribution(w, cov.values)
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Weight", x=assets, y=w, marker_color=color, opacity=0.8))
    fig.add_trace(go.Bar(name="Risk Contribution", x=assets, y=prc, marker_color="crimson", opacity=0.8))
    fig.update_layout(title=title, xaxis_title="Asset", yaxis_title="Fraction",
                      barmode="group", yaxis_tickformat=".0%", margin=dict(t=40))
    return fig

def wealth_trace(r: pd.Series, name: str, **kw) -> go.Scatter:
    return go.Scatter(x=r.index, y=(10_000 * (1 + r).cumprod()).values, name=name, mode="lines", **kw)

# ─── TABS ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Exploratory Analysis", "⚠️ Risk Analysis",
    "🔗 Correlation & Covariance", "💼 Portfolio Optimization",
    "🔬 Estimation Window Sensitivity"])

# ════════════════════ TAB 1 – EXPLORATORY ANALYSIS ═══════════════════════════
with tab1:
    st.header("Exploratory Analysis")

    # Summary stats table
    st.subheader("Summary Statistics")
    all_r = {**{t: returns[t] for t in assets}, **({"S&P 500": bench_r} if bench_r is not None else {})}
    rows = {}
    for name, r in all_r.items():
        s = series_stats(r, rf_rate)
        rows[name] = {"Ann. Return": f"{s['ret']:.2%}", "Ann. Volatility": f"{s['vol']:.2%}",
                      "Skewness": f"{s['skew']:.3f}", "Excess Kurtosis": f"{s['kurt']:.3f}",
                      "Min Daily": f"{s['rmin']:.2%}", "Max Daily": f"{s['rmax']:.2%}"}
    st.dataframe(pd.DataFrame(rows).T, use_container_width=True)

    st.markdown("---")

    # Cumulative wealth index
    st.subheader("Cumulative Wealth Index ($10,000 Initial Investment)")
    sel = st.multiselect("Assets to display:", assets, default=assets, key="wealth_ms")
    fig_w = go.Figure([wealth_trace(returns[t], t) for t in sel])
    if bench_r is not None:
        fig_w.add_trace(wealth_trace(bench_r, "S&P 500", line=dict(dash="dash", color="black", width=2)))
    fig_w.update_layout(title="Cumulative Wealth Index", xaxis_title="Date",
                        yaxis_title="Value ($)", hovermode="x unified")
    st.plotly_chart(fig_w, use_container_width=True)

    st.markdown("---")

    # Return distribution
    st.subheader("Return Distribution")
    d1, d2 = st.columns([1, 3])
    with d1:
        dtick = st.selectbox("Stock:", assets, key="dist_sel")
        dview = st.radio("Chart type:", ["Histogram + Normal", "Q-Q Plot"])
    with d2:
        r_s = returns[dtick].dropna()
        if dview == "Histogram + Normal":
            xs = np.linspace(r_s.min(), r_s.max(), 300)
            fig_d = go.Figure([
                go.Histogram(x=r_s, nbinsx=70, histnorm="probability density",
                             name="Returns", opacity=0.65, marker_color="steelblue"),
                go.Scatter(x=xs, y=stats.norm.pdf(xs, r_s.mean(), r_s.std()),
                           mode="lines", name="Normal Fit", line=dict(color="red", width=2))])
            fig_d.update_layout(title=f"Return Distribution – {dtick}",
                                xaxis_title="Daily Return", yaxis_title="Density")
        else:
            (tq, sq), (slope, intercept, _) = stats.probplot(r_s, dist="norm")
            lx = np.array([tq.min(), tq.max()])
            fig_d = go.Figure([
                go.Scatter(x=tq, y=sq, mode="markers", name="Sample",
                           marker=dict(color="steelblue", size=4)),
                go.Scatter(x=lx, y=slope * lx + intercept, mode="lines",
                           name="Normal Reference", line=dict(color="red", width=2))])
            fig_d.update_layout(title=f"Q-Q Plot – {dtick}",
                                xaxis_title="Theoretical Quantiles", yaxis_title="Sample Quantiles")
        st.plotly_chart(fig_d, use_container_width=True)

# ════════════════════ TAB 2 – RISK ANALYSIS ══════════════════════════════════
with tab2:
    st.header("Risk Analysis")

    # Rolling volatility
    st.subheader("Rolling Annualized Volatility")
    rwin = st.select_slider("Window (trading days):", [30, 60, 90, 120], value=60)
    fig_rv = go.Figure([go.Scatter(x=returns[t].rolling(rwin).std().index,
                                   y=(returns[t].rolling(rwin).std() * 252**0.5).values,
                                   name=t, mode="lines") for t in assets])
    fig_rv.update_layout(title=f"Rolling {rwin}-Day Annualized Volatility",
                         xaxis_title="Date", yaxis_title="Ann. Volatility",
                         yaxis_tickformat=".0%", hovermode="x unified")
    st.plotly_chart(fig_rv, use_container_width=True)

    st.markdown("---")

    # Drawdown
    st.subheader("Drawdown Analysis")
    dd_t = st.selectbox("Stock:", assets, key="dd_sel")
    dd_s = drawdown_series(returns[dd_t])
    st.metric("Maximum Drawdown", f"{dd_s.min():.2%}")
    fig_dd = go.Figure(go.Scatter(x=dd_s.index, y=dd_s.values, mode="lines", fill="tozeroy",
                                  name=dd_t, line=dict(color="crimson"),
                                  fillcolor="rgba(220,20,60,0.15)"))
    fig_dd.update_layout(title=f"Drawdown from Peak – {dd_t}", xaxis_title="Date",
                         yaxis_title="Drawdown", yaxis_tickformat=".0%")
    st.plotly_chart(fig_dd, use_container_width=True)

    st.markdown("---")

    # Risk-adjusted metrics
    st.subheader("Risk-Adjusted Performance Metrics")
    st.caption(f"Risk-free rate: {rf_rate:.2%} annualized")
    all_r2 = {**{t: returns[t] for t in assets}, **({"S&P 500": bench_r} if bench_r is not None else {})}
    risk_rows = {name: fmt(series_stats(r, rf_rate)) for name, r in all_r2.items()}
    st.dataframe(pd.DataFrame(risk_rows).T, use_container_width=True)

# ════════════════════ TAB 3 – CORRELATION & COVARIANCE ═══════════════════════
with tab3:
    st.header("Correlation & Covariance Analysis")

    # Heatmap
    st.subheader("Pairwise Correlation Heatmap")
    corr = returns.corr()
    fig_hm = go.Figure(go.Heatmap(z=corr.values, x=assets, y=assets,
                                   colorscale="RdBu_r", zmid=0, zmin=-1, zmax=1,
                                   text=np.round(corr.values, 2), texttemplate="%{text}",
                                   colorbar=dict(title="Correlation")))
    fig_hm.update_layout(title="Pairwise Correlation Matrix",
                         xaxis_title="Asset", yaxis_title="Asset")
    st.plotly_chart(fig_hm, use_container_width=True)

    st.markdown("---")

    # Rolling correlation
    st.subheader("Rolling Pairwise Correlation")
    c1, c2, c3 = st.columns(3)
    sa = c1.selectbox("Stock A:", assets, index=0, key="rc_a")
    sb = c2.selectbox("Stock B:", [t for t in assets if t != sa], index=0, key="rc_b")
    rcw = c3.selectbox("Window (days):", [30, 60, 90, 120], index=1, key="rc_w")
    rc = returns[sa].rolling(rcw).corr(returns[sb])
    fig_rc = go.Figure(go.Scatter(x=rc.index, y=rc.values, mode="lines", line=dict(color="steelblue")))
    fig_rc.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig_rc.update_layout(title=f"Rolling {rcw}-Day Correlation: {sa} vs {sb}",
                         xaxis_title="Date", yaxis_title="Correlation")
    st.plotly_chart(fig_rc, use_container_width=True)

    # Covariance matrix
    with st.expander("📋 Show Annualized Covariance Matrix"):
        st.caption("Annualized covariance matrix (daily covariance × 252)")
        st.dataframe((cov * 252).style.format("{:.6f}"),
             use_container_width=True)

# ════════════════════ TAB 4 – PORTFOLIO OPTIMIZATION ═════════════════════════
with tab4:
    st.header("Portfolio Construction & Optimization")

    # Equal-weight
    st.subheader("Equal-Weight (1/N) Portfolio")
    show_metrics(eq_w)
    st.markdown("---")

    # GMV & Tangency
    st.subheader("Optimized Portfolios")
    lc, rc = st.columns(2)
    for col, label, w, color in [(lc, "Global Minimum Variance (GMV)", gmv_w, "steelblue"),
                                  (rc, "Maximum Sharpe Ratio (Tangency)", tan_w, "darkorange")]:
        with col:
            st.markdown(f"**{label}**")
            show_metrics(w)
            st.plotly_chart(weight_bar(w, f"{label.split('(')[1].rstrip(')')} Weights", color),
                            use_container_width=True)
    st.markdown("---")

    # Risk contribution
    st.subheader("Risk Contribution Decomposition")
    st.info("**Percentage Risk Contribution (PRC):** each bar shows what fraction of total portfolio "
            "variance comes from each asset (sums to 100%). A stock with 10% weight but 25% PRC is a "
            "disproportionate source of volatility — due to high individual volatility or strong "
            "positive correlation with the rest of the portfolio.")
    pc1, pc2 = st.columns(2)
    pc1.plotly_chart(prc_bar(gmv_w, "GMV: Weight vs. Risk Contribution", "steelblue"),
                     use_container_width=True)
    pc2.plotly_chart(prc_bar(tan_w, "Tangency: Weight vs. Risk Contribution", "darkorange"),
                     use_container_width=True)
    st.markdown("---")

    # Custom portfolio
    st.subheader("Custom Portfolio Builder")
    st.markdown("Adjust the sliders — weights are **normalized** to sum to 100%.")
    scols = st.columns(min(n, 5))
    raw_w = np.array([scols[i % 5].slider(t, 0.0, 1.0, round(1/n, 2), 0.01, key=f"cw_{t}")
                      for i, t in enumerate(assets)])
    custom_w = raw_w / raw_w.sum() if raw_w.sum() > 0 else eq_w.copy()

    st.markdown("**Normalized weights:**")
    st.dataframe(pd.DataFrame({"Asset": assets, "Weight": [f"{w:.2%}" for w in custom_w]})
                 .set_index("Asset").T, use_container_width=True)
    show_metrics(custom_w)
    st.markdown("---")

    # Efficient frontier
    st.subheader("Efficient Frontier & Capital Allocation Line")
    st.markdown("The **efficient frontier** traces portfolios with maximum return per unit of risk. "
                "The **Capital Allocation Line (CAL)** runs from the risk-free rate through the "
                "Tangency portfolio — the steepest achievable risk-return tradeoff.")

    with st.spinner("Generating efficient frontier…"):
        ef_v, ef_r = efficient_frontier(mu_l, cov_l)

    fig_ef = go.Figure()
    if ef_v:
        fig_ef.add_trace(go.Scatter(x=ef_v, y=ef_r, mode="lines", name="Efficient Frontier",
                                    line=dict(color="navy", width=2.5)))

    # Individual stocks
    for t in assets:
        fig_ef.add_trace(go.Scatter(x=[returns[t].std() * 252**0.5], y=[returns[t].mean() * 252],
                                    mode="markers+text", name=t, text=[t],
                                    textposition="top center", marker=dict(size=8)))

    # Benchmark
    if bench_r is not None:
        fig_ef.add_trace(go.Scatter(x=[bench_r.std() * 252**0.5], y=[bench_r.mean() * 252],
                                    mode="markers+text", name="S&P 500", text=["S&P 500"],
                                    textposition="top center",
                                    marker=dict(size=11, symbol="square", color="black")))

    # Named portfolios
    for pname, pw, pcolor, psym in [("Equal-Weight", eq_w, "green", "star"),
                                     ("GMV", gmv_w, "blue", "diamond"),
                                     ("Tangency", tan_w, "red", "triangle-up"),
                                     ("Custom", custom_w, "purple", "circle")]:
        s = series_stats(returns @ pw, rf_rate)
        fig_ef.add_trace(go.Scatter(x=[s["vol"]], y=[s["ret"]],
                                    mode="markers+text", name=pname, text=[pname],
                                    textposition="top center",
                                    marker=dict(size=14, symbol=psym, color=pcolor,
                                                line=dict(width=1, color="black"))))

    # CAL
    ts = series_stats(returns @ tan_w, rf_rate)
    if ts["vol"] > 0:
        slope_cal = (ts["ret"] - rf_rate) / ts["vol"]
        cal_v = np.linspace(0, max(max(ef_v) if ef_v else 0, ts["vol"]) * 1.4, 200)
        fig_ef.add_trace(go.Scatter(x=cal_v.tolist(), y=(rf_rate + slope_cal * cal_v).tolist(),
                                    mode="lines", name="Capital Allocation Line",
                                    line=dict(color="orange", width=2, dash="dash")))

    fig_ef.update_layout(title="Efficient Frontier with Capital Allocation Line",
                         xaxis_title="Annualized Volatility", yaxis_title="Annualized Return",
                         xaxis_tickformat=".0%", yaxis_tickformat=".0%", hovermode="closest")
    st.plotly_chart(fig_ef, use_container_width=True)
    st.markdown("---")

    # Portfolio comparison
    st.subheader("Portfolio Comparison")
    all_ports = {"Equal-Weight": eq_w, "GMV": gmv_w, "Tangency": tan_w, "Custom": custom_w}
    port_colors = {"Equal-Weight": "green", "GMV": "blue", "Tangency": "red", "Custom": "purple"}

    fig_cmp = go.Figure([wealth_trace(returns @ pw, nm, line=dict(color=port_colors[nm]))
                         for nm, pw in all_ports.items()])
    if bench_r is not None:
        fig_cmp.add_trace(wealth_trace(bench_r, "S&P 500",
                                        line=dict(dash="dash", color="black", width=2)))
    fig_cmp.update_layout(title="Portfolio Cumulative Wealth ($10,000 Initial)",
                          xaxis_title="Date", yaxis_title="Value ($)", hovermode="x unified")
    st.plotly_chart(fig_cmp, use_container_width=True)

    # Summary table
    comp = {nm: fmt(series_stats(returns @ pw, rf_rate)) for nm, pw in all_ports.items()}
    if bench_r is not None:
        comp["S&P 500"] = fmt(series_stats(bench_r, rf_rate))
    st.dataframe(pd.DataFrame(comp).T, use_container_width=True)

# ════════════════════ TAB 5 – ESTIMATION WINDOW SENSITIVITY ══════════════════
with tab5:
    st.header("Estimation Window Sensitivity Analysis")
    st.info("**Why this matters:** Mean-variance optimization is highly sensitive to input estimates — "
            "small changes in historical returns or covariances can produce dramatically different weights. "
            "The tables below show how GMV and Tangency portfolios shift across different lookback "
            "windows, illustrating that results are only as stable as the data used to produce them.")

    total_yrs = len(returns) / 252
    win_map = {k: v for k, v in {"1 Year": 252, "3 Years": 756,
                                   "5 Years": 1260, "Full Sample": None}.items()
               if v is None or total_yrs >= v / 252}

    sel_wins = st.multiselect("Lookback windows:", list(win_map.keys()),
                               default=list(win_map.keys()), key="est_w")
    if not sel_wins:
        st.warning("Select at least one window.")
    else:
        gmv_tbl, tan_tbl = {}, {}
        with st.spinner("Computing across estimation windows…"):
            for wn in sel_wins:
                nd = win_map[wn]
                wr = returns if nd is None else returns.iloc[-nd:]
                if len(wr) < 60:
                    st.warning(f"⚠️ Insufficient data for **{wn}** — skipping."); continue
                wmu, wcov = wr.mean().values.tolist(), [r.tolist() for r in wr.cov().values]
                (gok, gwl), (tok, twl) = run_optimizations(wmu, wcov, rf_rate)
                if gok:
                    gw = np.array(gwl)
                    gmv_tbl[wn] = {**{a: f"{gw[i]:.1%}" for i, a in enumerate(assets)},
                                   "Ann. Return": f"{gw @ np.array(wmu) * 252:.2%}",
                                   "Ann. Volatility": f"{np.sqrt(gw @ np.array(wcov) @ gw * 252):.2%}"}
                if tok:
                    tw = np.array(twl)
                    tv = np.sqrt(tw @ np.array(wcov) @ tw * 252)
                    tr = tw @ np.array(wmu) * 252
                    tan_tbl[wn] = {**{a: f"{tw[i]:.1%}" for i, a in enumerate(assets)},
                                   "Ann. Return": f"{tr:.2%}",
                                   "Ann. Volatility": f"{tv:.2%}",
                                   "Sharpe Ratio": f"{(tr - rf_rate) / tv:.3f}" if tv else "N/A"}

        for label, tbl, color in [("Global Minimum Variance (GMV)", gmv_tbl, "steelblue"),
                                    ("Tangency", tan_tbl, "darkorange")]:
            if not tbl: continue
            st.subheader(f"{label} Portfolio Across Estimation Windows")
            st.dataframe(pd.DataFrame(tbl).T, use_container_width=True)
            fig_s = go.Figure([
                go.Bar(name=wn,
                       x=assets,
                       y=[float(tbl[wn][a].strip("%")) / 100 for a in assets])
                for wn in tbl])
            fig_s.update_layout(title=f"{label} Weights by Estimation Window",
                                xaxis_title="Asset", yaxis_title="Weight",
                                barmode="group", yaxis_tickformat=".0%")
            st.plotly_chart(fig_s, use_container_width=True)
