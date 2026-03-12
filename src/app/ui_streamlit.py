from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh
from streamlit.components.v1 import html
import time

from analytics.simulated_portfolio import (
    append_portfolio_snapshot,
    build_open_positions_with_market,
    execute_buy,
    execute_sell_position,
    fetch_latest_prices,
    load_closed_trades,
    load_open_positions,
    load_operations,
    load_portfolio_history,
    load_portfolio_state,
    monte_carlo_gbm_portfolio,
    reset_portfolio,
)
from features.feature_explanations import DEFAULT_DISPLAY_FEATURES, FEATURE_EXPLANATIONS
from runtime_config import get_env_str, get_predictions_path, load_runtime_env


RENAME_MAP = {
    "empresa": "Empresa",
    "ticker": "Ticker",
    "accion": "Accion",
    "prob_subida": "Probabilidad subida",
    "retorno_esperado": "Retorno esperado",
    "riesgo": "Riesgo",
    "ratio_retorno_riesgo": "Ratio retorno/riesgo",
    "weight": "Peso",
    "senal_final": "Senal final",
    "explicacion_retorno": "Explicacion retorno",
    "confianza": "Confianza",
    "confianza_pct": "Confianza %",
    "signal_strength": "Fuerza senal",
    "semaforo_icono": "Icono",
    "semaforo": "Semaforo",
    "return_1d": "Retorno diario",
    "sma_5": "SMA 5",
    "sma_20": "SMA 20",
    "sma_50": "SMA 50",
    "volatility_20": "Volatilidad 20d",
    "rsi_14": "RSI 14",
    "macd": "MACD",
    "macd_signal": "MACD signal",
    "bb_upper": "Bollinger superior",
    "bb_lower": "Bollinger inferior",
    "momentum_10": "Momentum 10d",
    "volume_change_1d": "Cambio volumen 1d",
}


def _safe_confidence(value: float) -> float:
    if pd.isna(value):
        return 0.5
    return float(value)


def _safe_value(value: float | int | None) -> float:
    if value is None or pd.isna(value):
        return 0.0
    return float(value)


def classify_semaphore(confidence: float, action: str) -> str:
    if action == "Comprar":
        if confidence >= 0.65:
            return "VERDE"
        if confidence >= 0.55:
            return "AMARILLO"
        return "ROJO"
    if confidence <= 0.35:
        return "ROJO"
    if confidence <= 0.45:
        return "AMARILLO"
    return "NEUTRO"


def semaphore_color(label: str) -> str:
    if label == "VERDE":
        return "#00C853"
    if label == "AMARILLO":
        return "#FFD54F"
    if label == "ROJO":
        return "#FF5252"
    return "#90A4AE"


def semaphore_icon(label: str) -> str:
    if label == "VERDE":
        return "🟢"
    if label == "AMARILLO":
        return "🟡"
    if label == "ROJO":
        return "🔴"
    return "⚪"


def trend_arrow(momentum: float) -> str:
    if momentum > 0:
        return "▲"
    if momentum < 0:
        return "▼"
    return "→"


def prepare_display_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["confianza"] = out.get("confianza", out.get("proba_up", 0.5)).apply(_safe_confidence)
    out["prob_subida"] = out.get("prob_subida", out["confianza"]).apply(_safe_confidence)
    out["confianza_pct"] = (out["confianza"] * 100).round(2)
    out["signal_strength"] = (out["confianza"] - 0.5).abs().mul(2).round(4)
    out["momentum_10"] = out.get("momentum_10", 0.0).apply(_safe_value)
    out["retorno_esperado"] = pd.to_numeric(out.get("retorno_esperado", 0.0), errors="coerce").fillna(0.0)
    out["riesgo"] = pd.to_numeric(
        out.get("riesgo", out.get("volatility_20", 1e-6)),
        errors="coerce",
    ).fillna(1e-6).clip(lower=1e-6)
    out["ratio_retorno_riesgo"] = pd.to_numeric(
        out.get("ratio_retorno_riesgo", out["retorno_esperado"] / out["riesgo"]),
        errors="coerce",
    ).fillna(0.0)
    out["semaforo"] = out.apply(
        lambda row: classify_semaphore(row["confianza"], str(row.get("accion", ""))),
        axis=1,
    )
    out["semaforo_icono"] = out["semaforo"].apply(semaphore_icon)
    out["senal_final"] = out["accion"].astype(str) + " | " + out["semaforo"].astype(str)
    out["prob_bar"] = out["prob_subida"].clip(lower=0.0, upper=1.0)
    out["retorno_bar"] = ((out["retorno_esperado"].clip(-0.03, 0.03) + 0.03) / 0.06).clip(0.0, 1.0)
    return out


def inject_finance_theme() -> None:
    st.markdown(
        """
        <style>
        html, body, [class*="css"]  {
            font-family: "Inter", "Segoe UI", Arial, sans-serif;
        }
        .kpiCard {
            border-radius: 14px;
            padding: 12px 14px;
            border: 1px solid #1e2a35;
            background: linear-gradient(135deg, #0f1720 0%, #111b28 100%);
        }
        .kpiLabel {
            font-size: 0.84rem;
            color: #9fb3c8;
            margin-bottom: 4px;
        }
        .kpiValue {
            font-size: 1.7rem;
            font-weight: 700;
            color: #e8f0f7;
        }
        .rankCard {
            border-radius: 12px;
            padding: 10px 12px;
            margin-bottom: 8px;
            border: 1px solid #223140;
            background: #0f1a26;
        }
        .rankTitle {
            font-size: 0.98rem;
            font-weight: 600;
            color: #e8f0f7;
        }
        .rankSub {
            font-size: 0.84rem;
            color: #9fb3c8;
        }
        .barWrap {
            width: 100%;
            height: 9px;
            border-radius: 8px;
            background: #223140;
            margin-top: 6px;
        }
        .barFill {
            height: 9px;
            border-radius: 8px;
            background: linear-gradient(90deg, #00C853 0%, #37d67a 100%);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_summary_cards(df: pd.DataFrame) -> None:
    buy_count = int((df["accion"] == "Comprar").sum())
    neutral_count = int(df["semaforo"].isin(["AMARILLO", "NEUTRO"]).sum())
    negative_count = int((df["semaforo"] == "ROJO").sum())
    avg_confidence = float(df["confianza"].mean() * 100) if len(df) else 0.0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"<div class='kpiCard'><div class='kpiLabel'>📈 Señales compra</div><div class='kpiValue' style='color:#00C853'>{buy_count}</div></div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"<div class='kpiCard'><div class='kpiLabel'>📊 Señales neutrales</div><div class='kpiValue' style='color:#FFD54F'>{neutral_count}</div></div>",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"<div class='kpiCard'><div class='kpiLabel'>📉 Señales negativas</div><div class='kpiValue' style='color:#FF5252'>{negative_count}</div></div>",
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f"<div class='kpiCard'><div class='kpiLabel'>🤖 Confianza media</div><div class='kpiValue'>{avg_confidence:.1f}%</div></div>",
            unsafe_allow_html=True,
        )


def render_feature_explanations(df: pd.DataFrame) -> None:
    with st.expander("Para que sirve cada variable"):
        for feature in DEFAULT_DISPLAY_FEATURES:
            if feature in df.columns and feature in FEATURE_EXPLANATIONS:
                st.markdown(f"- **{feature}**: {FEATURE_EXPLANATIONS[feature]}")


def render_model_recommendation_reason(row: pd.Series) -> None:
    st.subheader("Por que el modelo recomienda esta accion")
    title = str(row.get("explanation_title", "")).strip()
    points_raw = str(row.get("explanation_points", "")).strip()
    if title:
        st.markdown(f"**{title}**")
    if points_raw:
        for point in [p.strip() for p in points_raw.split("||") if p.strip()]:
            st.markdown(f"- {point}")
    else:
        st.info("No hay explicacion disponible para esta empresa.")


def _format_seconds(seconds: int) -> str:
    sec = max(int(seconds), 0)
    minutes, s = divmod(sec, 60)
    hours, m = divmod(minutes, 60)
    if hours > 0:
        return f"{hours} h {m} min {s} s"
    if minutes > 0:
        return f"{minutes} min {s} s"
    return f"{s} s"


def render_refresh_countdown(
    portfolio_next_refresh_in_s: int,
    market_next_refresh_in_s: int,
) -> None:
    portfolio_initial = max(int(portfolio_next_refresh_in_s), 0)
    market_initial = max(int(market_next_refresh_in_s), 0)
    portfolio_period = 20
    market_period = 3600
    portfolio_text = _format_seconds(portfolio_initial)
    market_text = _format_seconds(market_initial)
    html(
        f"""
        <div style="border:1px solid #223140;border-radius:10px;padding:10px 14px;margin:6px 0 14px 0;background:#0f1a26;">
          <div style="font-size:0.82rem;color:#9fb3c8;margin-bottom:6px;">Data Refresh Countdown</div>
          <div style="display:flex;gap:24px;flex-wrap:wrap;">
            <div style="font-size:0.92rem;color:#e8f0f7;">
              <span style="color:#64B5F6;font-weight:600;">Portfolio refresh in:</span>
              <span id="portfolio-counter">{portfolio_text}</span>
            </div>
            <div style="font-size:0.92rem;color:#e8f0f7;">
              <span style="color:#FFD54F;font-weight:600;">Market data refresh in:</span>
              <span id="market-counter">{market_text}</span>
            </div>
          </div>
        </div>
        <script>
          const portfolioPeriod = {portfolio_period};
          const marketPeriod = {market_period};
          const portfolioStart = {portfolio_initial};
          const marketStart = {market_initial};
          const portfolioEl = document.getElementById("portfolio-counter");
          const marketEl = document.getElementById("market-counter");
          const fmt = (sec) => {{
            sec = Math.max(0, Math.floor(sec));
            const h = Math.floor(sec / 3600);
            const m = Math.floor((sec % 3600) / 60);
            const s = sec % 60;
            if (h > 0) return `${{h}} h ${{m}} min ${{s}} s`;
            if (m > 0) return `${{m}} min ${{s}} s`;
            return `${{s}} s`;
          }};
          let p = portfolioStart % portfolioPeriod;
          let m = marketStart % marketPeriod;
          const tick = () => {{
            if (portfolioEl) portfolioEl.innerText = fmt(p);
            if (marketEl) marketEl.innerText = fmt(m);
            p = (p - 1 + portfolioPeriod) % portfolioPeriod;
            m = (m - 1 + marketPeriod) % marketPeriod;
          }};
          tick();
          setInterval(tick, 1000);
        </script>
        """,
        height=92,
    )


@st.cache_data(show_spinner=False, ttl=3600)
def load_predictions_hourly(predictions_path: str) -> tuple[pd.DataFrame, float]:
    path = Path(predictions_path)
    if not path.exists():
        return pd.DataFrame(), time.time()
    return pd.read_csv(path), time.time()


@st.cache_data(show_spinner=False, ttl=3600)
def load_feature_history(config_path: str) -> tuple[pd.DataFrame, float]:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    feature_path = Path(cfg["data"]["feature_path"]) / "feature_table.parquet"
    if not feature_path.exists():
        return pd.DataFrame(), time.time()
    hist = pd.read_parquet(feature_path)
    hist["date"] = pd.to_datetime(hist["date"])
    return hist.sort_values(["ticker", "date"]).reset_index(drop=True), time.time()


@st.cache_data(show_spinner=False, ttl=20)
def load_market_prices_cached(tickers: tuple[str, ...]) -> tuple[dict[str, float], float]:
    return fetch_latest_prices(list(tickers)), time.time()


def render_investment_simulator(history_df: pd.DataFrame, universe_df: pd.DataFrame) -> None:
    st.markdown("---")
    st.subheader("Mi Portfolio Simulado")

    if history_df.empty:
        st.info("No hay historico para simular. Ejecuta el pipeline de entrenamiento.")
        return

    company_map = (
        universe_df[["empresa", "ticker"]]
        .drop_duplicates()
        .sort_values(["empresa", "ticker"])
        .reset_index(drop=True)
    )
    option_labels = [f"{row.empresa} ({row.ticker})" for row in company_map.itertuples(index=False)]
    label_to_ticker = {f"{row.empresa} ({row.ticker})": row.ticker for row in company_map.itertuples(index=False)}
    ticker_to_company = {row.ticker: row.empresa for row in company_map.itertuples(index=False)}

    with st.expander("Configuracion de cuenta", expanded=False):
        state = load_portfolio_state()
        st.caption(f"Capital inicial actual: {state.initial_capital:,.2f} EUR | Efectivo: {state.cash:,.2f} EUR")
        c1, c2 = st.columns([2, 1])
        with c1:
            new_capital = st.number_input(
                "Nuevo capital inicial (reinicia portfolio)",
                min_value=100.0,
                value=float(state.initial_capital),
                step=500.0,
            )
        with c2:
            if st.button("Reiniciar portfolio"):
                reset_portfolio(initial_capital=float(new_capital))
                st.success("Portfolio reiniciado.")
                st.rerun()

    if "portfolio_autorefresh_enabled" not in st.session_state:
        st.session_state["portfolio_autorefresh_enabled"] = True
    auto_col, text_col = st.columns([1, 3])
    with auto_col:
        st.session_state["portfolio_autorefresh_enabled"] = st.toggle(
            "Auto-refresh 20s",
            value=st.session_state["portfolio_autorefresh_enabled"],
            key="portfolio_autorefresh_toggle",
        )
    with text_col:
        st.caption(
            "Actualiza precios y metricas cada 20s sin perder estado de portfolio. "
            "Solo refresca datos de mercado y recalculos."
        )

    if st.session_state["portfolio_autorefresh_enabled"]:
        st_autorefresh(interval=20_000, key="portfolio_20s_refresh")

    refresh = st.button("Refrescar precios ahora")

    operations = load_operations()
    open_positions = load_open_positions()
    op_tickers = open_positions["ticker"].dropna().astype(str).unique().tolist() if not open_positions.empty else []
    all_tickers = sorted(set(company_map["ticker"].dropna().unique().tolist()) | set(op_tickers))
    if refresh:
        load_market_prices_cached.clear()
    prices, portfolio_fetched_at = load_market_prices_cached(tuple(all_tickers))
    st.session_state["portfolio_last_refresh_epoch"] = float(portfolio_fetched_at)

    st.markdown("### Comprar posiciones")
    b1, b2, b3 = st.columns([2, 1, 1])
    with b1:
        buy_label = st.selectbox("Empresa a comprar", options=option_labels, key="buy_label")
    with b2:
        amount_eur = st.number_input("Capital a invertir (EUR)", min_value=50.0, value=500.0, step=50.0)
    with b3:
        if st.button("Comprar", type="primary"):
            buy_ticker = label_to_ticker[buy_label]
            buy_company = str(company_map.loc[company_map["ticker"] == buy_ticker, "empresa"].iloc[0])
            price = float(prices.get(buy_ticker, 0.0))
            if price <= 0:
                st.error("No se pudo obtener precio de mercado para comprar.")
            else:
                result = execute_buy(
                    buy_ticker,
                    amount_eur=amount_eur,
                    market_price=price,
                    company_name=buy_company,
                )
                if result["ok"]:
                    st.success(result["message"])
                    st.rerun()
                else:
                    st.error(result["message"])

    positions_df = build_open_positions_with_market(open_positions, prices=prices, company_map=ticker_to_company)

    st.markdown("### Vender posiciones")
    if positions_df.empty:
        st.info("No hay posiciones abiertas para vender.")
    else:
        st.caption("Cada boton vende SOLO esa compra (posicion individual).")
        for row in positions_df.sort_values("buy_timestamp", ascending=False).itertuples(index=False):
            c1, c2, c3, c4, c5, c6 = st.columns([2.0, 1.0, 1.0, 1.0, 1.2, 0.9])
            c1.markdown(f"**{row.empresa} ({row.ticker})**  \nID: `{row.position_id}`")
            c2.metric("Compra", f"{float(row.avg_cost):.4f}")
            c3.metric("Actual", f"{float(row.current_price):.4f}")
            c4.metric("Acciones", f"{float(row.shares):.4f}")
            with c5:
                shares_to_sell = st.number_input(
                    "Vender",
                    min_value=0.0,
                    max_value=float(row.shares),
                    value=float(row.shares),
                    step=max(float(row.shares) / 20.0, 0.0001),
                    key=f"sell_shares_{row.position_id}",
                    help="Cantidad de acciones a vender de esta posicion",
                )
            with c6:
                if st.button("Vender", key=f"sell_{row.position_id}"):
                    price = float(prices.get(row.ticker, 0.0))
                    if price <= 0:
                        st.error("No se pudo obtener precio de mercado para vender.")
                    else:
                        result = execute_sell_position(
                            position_id=str(row.position_id),
                            market_price=price,
                            shares_to_sell=float(shares_to_sell),
                        )
                        if result["ok"]:
                            st.success(result["message"])
                            st.rerun()
                        else:
                            st.error(result["message"])

    state = load_portfolio_state()
    operations = load_operations()
    open_positions = load_open_positions()
    positions_df = build_open_positions_with_market(open_positions, prices=prices, company_map=ticker_to_company)
    positions_value = float(positions_df["current_value"].sum()) if not positions_df.empty else 0.0
    total_value = state.cash + positions_value
    total_return = (total_value / state.initial_capital - 1.0) if state.initial_capital > 0 else 0.0
    append_portfolio_snapshot(state.cash, positions_value)

    realized_pnl = float(pd.to_numeric(operations.get("realized_pnl", 0.0), errors="coerce").fillna(0.0).sum())
    unrealized_pnl = float(positions_df["unrealized_pnl"].sum()) if not positions_df.empty else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Valor total cartera", f"{total_value:,.2f} EUR")
    c2.metric("Rentabilidad total", f"{total_return * 100:.2f}%")
    c3.metric("PnL realizado", f"{realized_pnl:,.2f} EUR")
    c4.metric("PnL no realizado", f"{unrealized_pnl:,.2f} EUR")

    hist = load_portfolio_history()
    if not hist.empty:
        curve = go.Figure()
        curve.add_trace(
            go.Scatter(
                x=hist["timestamp"],
                y=hist["total_value"],
                mode="lines",
                name="Valor cartera",
                line={"color": "#00C853", "width": 2},
            )
        )
        curve.update_layout(
            template="plotly_dark",
            hovermode="x unified",
            height=300,
            margin={"l": 20, "r": 20, "t": 20, "b": 20},
        )
        st.plotly_chart(curve, use_container_width=True)

    st.caption("Posiciones abiertas")
    if positions_df.empty:
        st.info("No hay posiciones abiertas.")
    else:
        ranking_cols = ["ticker", "retorno_esperado", "riesgo", "ratio_retorno_riesgo"]
        merge_ref = universe_df[ranking_cols].drop_duplicates(subset=["ticker"])
        positions_df = positions_df.merge(merge_ref, on="ticker", how="left")
        st.session_state["portfolio_positions"] = positions_df.copy()
        st.dataframe(
            positions_df[
                [
                    "position_id",
                    "empresa",
                    "ticker",
                    "buy_timestamp",
                    "avg_cost",
                    "shares",
                    "current_price",
                    "invested",
                    "current_value",
                    "unrealized_pnl",
                    "unrealized_return",
                ]
            ],
            width="stretch",
            column_config={
                "position_id": st.column_config.TextColumn("ID"),
                "buy_timestamp": st.column_config.DatetimeColumn("Fecha compra"),
                "shares": st.column_config.NumberColumn("Acciones", format="%.4f"),
                "avg_cost": st.column_config.NumberColumn("Precio compra", format="%.4f"),
                "current_price": st.column_config.NumberColumn("Precio actual", format="%.4f"),
                "invested": st.column_config.NumberColumn("Invertido", format="%.2f"),
                "current_value": st.column_config.NumberColumn("Valor actual", format="%.2f"),
                "unrealized_pnl": st.column_config.NumberColumn("PnL no realizado", format="%.2f"),
                "unrealized_return": st.column_config.NumberColumn("Rentab. posicion", format="%.2%"),
            },
        )

    st.caption("Operaciones historicas (compras y ventas)")
    if operations.empty:
        st.info("No hay operaciones registradas.")
    else:
        st.dataframe(
            operations.sort_values("timestamp", ascending=False),
            width="stretch",
            column_config={
                "shares": st.column_config.NumberColumn("Acciones", format="%.4f"),
                "price": st.column_config.NumberColumn("Precio", format="%.4f"),
                "amount": st.column_config.NumberColumn("Importe", format="%.2f"),
                "realized_pnl": st.column_config.NumberColumn("PnL realizado", format="%.2f"),
            },
        )

    st.caption("Historico de operaciones cerradas")
    closed_df = load_closed_trades()
    if closed_df.empty:
        st.info("No hay operaciones cerradas todavia.")
    else:
        st.dataframe(
            closed_df[
                [
                    "position_id",
                    "ticker",
                    "empresa",
                    "buy_timestamp",
                    "sell_timestamp",
                    "buy_price",
                    "sell_price",
                    "realized_pnl",
                    "realized_return",
                ]
            ].sort_values("sell_timestamp", ascending=False),
            width="stretch",
            column_config={
                "position_id": st.column_config.TextColumn("ID"),
                "buy_timestamp": st.column_config.DatetimeColumn("Fecha compra"),
                "sell_timestamp": st.column_config.DatetimeColumn("Fecha venta"),
                "buy_price": st.column_config.NumberColumn("Precio compra", format="%.4f"),
                "sell_price": st.column_config.NumberColumn("Precio venta", format="%.4f"),
                "realized_pnl": st.column_config.NumberColumn("Beneficio/Perdida", format="%.2f"),
                "realized_return": st.column_config.NumberColumn("Rentab. realizada", format="%.2%"),
            },
        )

    st.markdown("### Escenarios futuros (Monte Carlo)")
    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        horizon_days = int(st.slider("Horizonte (dias)", min_value=30, max_value=365, value=126, step=5))
    with mc2:
        n_sims = int(st.slider("Numero de simulaciones", min_value=500, max_value=5000, value=2000, step=500))
    with mc3:
        lookback_days = int(st.slider("Ventana calibracion", min_value=60, max_value=504, value=252, step=21))

    if positions_df.empty:
        st.info("Abre al menos una posicion para simular escenarios futuros.")
    else:
        mc = monte_carlo_gbm_portfolio(
            history_df=history_df,
            positions_df=positions_df,
            horizon_days=horizon_days,
            n_sims=n_sims,
            lookback_days=lookback_days,
        )
        terminal_values = mc["terminal_values"]
        paths = mc["paths"]
        stats = mc["stats"]
        if len(terminal_values) == 0:
            st.info("No hay datos suficientes para simulacion Monte Carlo.")
        else:
            pess = float(stats["p05"])
            mid = float(stats["p50"])
            opt = float(stats["p95"])
            mean_val = float(stats["mean"])
            s1, s2, s3 = st.columns(3)
            s1.metric("Escenario pesimista (p5)", f"{pess:,.2f} EUR")
            s2.metric("Escenario medio (p50)", f"{mid:,.2f} EUR")
            s3.metric("Escenario optimista (p95)", f"{opt:,.2f} EUR")
            st.metric("Valor esperado", f"{mean_val:,.2f} EUR")

            # Show subset of trajectories + confidence bands
            path_fig = go.Figure()
            n_show = min(40, paths.shape[0])
            x_axis = list(range(paths.shape[1]))
            for i in range(n_show):
                path_fig.add_trace(
                    go.Scatter(
                        x=x_axis,
                        y=paths[i],
                        mode="lines",
                        line={"color": "rgba(100,181,246,0.15)", "width": 1},
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )
            p05_path = np.percentile(paths, 5, axis=0)
            p50_path = np.percentile(paths, 50, axis=0)
            p95_path = np.percentile(paths, 95, axis=0)
            mean_path = np.mean(paths, axis=0)
            path_fig.add_trace(
                go.Scatter(
                    x=x_axis,
                    y=p95_path,
                    mode="lines",
                    line={"width": 0},
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
            path_fig.add_trace(
                go.Scatter(
                    x=x_axis,
                    y=p05_path,
                    mode="lines",
                    fill="tonexty",
                    fillcolor="rgba(66,165,245,0.20)",
                    line={"width": 0},
                    name="Banda confianza 5-95%",
                )
            )
            path_fig.add_trace(
                go.Scatter(
                    x=x_axis,
                    y=p50_path,
                    mode="lines",
                    line={"color": "#FFD54F", "width": 2},
                    name="Trayectoria mediana",
                )
            )
            path_fig.add_trace(
                go.Scatter(
                    x=x_axis,
                    y=mean_path,
                    mode="lines",
                    line={"color": "#00C853", "width": 2},
                    name="Valor esperado",
                )
            )
            path_fig.update_layout(
                template="plotly_dark",
                height=340,
                margin={"l": 20, "r": 20, "t": 20, "b": 20},
                xaxis_title="Dia simulado",
                yaxis_title="Valor cartera (EUR)",
            )
            st.plotly_chart(path_fig, use_container_width=True)

            hist_fig = go.Figure()
            hist_fig.add_trace(
                go.Histogram(
                    x=terminal_values,
                    nbinsx=60,
                    marker_color="#42A5F5",
                    opacity=0.85,
                    name="Distribucion valores futuros",
                )
            )
            hist_fig.update_layout(
                template="plotly_dark",
                height=320,
                margin={"l": 20, "r": 20, "t": 20, "b": 20},
                xaxis_title="Valor futuro cartera (EUR)",
                yaxis_title="Frecuencia",
            )
            st.plotly_chart(hist_fig, use_container_width=True)


def _apply_time_range(df: pd.DataFrame, range_label: str) -> pd.DataFrame:
    if df.empty:
        return df
    max_date = df["date"].max()
    if range_label == "1M":
        cutoff = max_date - pd.DateOffset(months=1)
    elif range_label == "3M":
        cutoff = max_date - pd.DateOffset(months=3)
    elif range_label == "6M":
        cutoff = max_date - pd.DateOffset(months=6)
    elif range_label == "1Y":
        cutoff = max_date - pd.DateOffset(years=1)
    else:
        cutoff = df["date"].min()
    return df[df["date"] >= cutoff].copy()


def build_company_analysis_chart(company_df: pd.DataFrame, title: str, range_label: str = "1Y") -> go.Figure:
    plot_df = company_df.sort_values("date").copy()
    plot_df = _apply_time_range(plot_df, range_label=range_label)
    if plot_df.empty:
        return go.Figure()

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.62, 0.18, 0.20],
        subplot_titles=(
            f"{title} | Precio, SMA y Bollinger",
            "Volumen",
            "RSI (14)",
        ),
    )

    fig.add_trace(
        go.Candlestick(
            x=plot_df["date"],
            open=plot_df["Open"],
            high=plot_df["High"],
            low=plot_df["Low"],
            close=plot_df["Close"],
            name="Precio",
            increasing_line_color="#00C853",
            decreasing_line_color="#FF5252",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    for col_name, color, label in [
        ("sma_5", "#42A5F5", "SMA 5"),
        ("sma_20", "#FFD54F", "SMA 20"),
        ("sma_50", "#AB47BC", "SMA 50"),
    ]:
        if col_name in plot_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=plot_df["date"],
                    y=plot_df[col_name],
                    mode="lines",
                    name=label,
                    line={"width": 1.6, "color": color},
                ),
                row=1,
                col=1,
            )

    if "bb_upper" in plot_df.columns and "bb_lower" in plot_df.columns:
        fig.add_trace(
            go.Scatter(
                x=plot_df["date"],
                y=plot_df["bb_upper"],
                mode="lines",
                line={"width": 1, "color": "#90A4AE"},
                name="Bollinger superior",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=plot_df["date"],
                y=plot_df["bb_lower"],
                mode="lines",
                fill="tonexty",
                line={"width": 1, "color": "#90A4AE"},
                fillcolor="rgba(144,164,174,0.15)",
                name="Bollinger inferior",
            ),
            row=1,
            col=1,
        )

    vol_colors = [
        "#00C853" if c >= o else "#FF5252"
        for c, o in zip(plot_df["Close"], plot_df["Open"])
    ]
    fig.add_trace(
        go.Bar(
            x=plot_df["date"],
            y=plot_df["Volume"],
            marker_color=vol_colors,
            name="Volumen",
            opacity=0.85,
        ),
        row=2,
        col=1,
    )

    if "rsi_14" in plot_df.columns:
        fig.add_trace(
            go.Scatter(
                x=plot_df["date"],
                y=plot_df["rsi_14"],
                mode="lines",
                line={"width": 1.8, "color": "#26C6DA"},
                name="RSI 14",
            ),
            row=3,
            col=1,
        )
        fig.add_hline(y=70, line_dash="dash", line_color="#FF5252", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="#FFD54F", row=3, col=1)

    fig.update_layout(
        template="plotly_dark",
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
        legend={"orientation": "h", "y": 1.02, "x": 0},
        margin={"l": 30, "r": 30, "t": 60, "b": 20},
        height=900,
    )
    fig.update_xaxes(
        rangeselector={
            "buttons": [
                {"count": 1, "label": "1M", "step": "month", "stepmode": "backward"},
                {"count": 3, "label": "3M", "step": "month", "stepmode": "backward"},
                {"count": 6, "label": "6M", "step": "month", "stepmode": "backward"},
                {"count": 1, "label": "1Y", "step": "year", "stepmode": "backward"},
                {"step": "all", "label": "MAX"},
            ]
        },
        showspikes=True,
    )
    fig.update_yaxes(title_text="Precio", row=1, col=1)
    fig.update_yaxes(title_text="Volumen", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1, range=[0, 100])
    return fig


load_runtime_env()
PREDICTIONS_PATH = get_predictions_path()
CONFIG_PATH = get_env_str("CONFIG_PATH", "config/settings.yaml")

st.set_page_config(page_title="IBEX 35 Predictor", layout="wide")
inject_finance_theme()
st.title("IBEX 35 Intelligence Dashboard")
st.caption("Estilo terminal financiero: señales, ranking y contexto técnico.")

if not PREDICTIONS_PATH.exists():
    st.warning("No hay predicciones disponibles. Ejecuta primero el pipeline de predicción.")
else:
    df, market_fetched_at = load_predictions_hourly(str(PREDICTIONS_PATH))
    if df.empty:
        st.warning("No hay predicciones disponibles. Ejecuta primero el pipeline de predicción.")
        st.stop()
    st.session_state["market_last_refresh_epoch"] = float(market_fetched_at)
    now_epoch = time.time()
    portfolio_last = float(st.session_state.get("portfolio_last_refresh_epoch", now_epoch))
    market_last = float(st.session_state.get("market_last_refresh_epoch", now_epoch))
    render_refresh_countdown(
        portfolio_next_refresh_in_s=max(0, int(20 - (now_epoch - portfolio_last))),
        market_next_refresh_in_s=max(0, int(3600 - (now_epoch - market_last))),
    )
    df = df.sort_values(["date", "ticker"])
    df = prepare_display_dataframe(df)
    latest_date = df["date"].max()
    st.caption(f"Fecha de señal: {latest_date}")

    companies = sorted(df["empresa"].dropna().unique().tolist()) if "empresa" in df.columns else []
    selected_company = st.selectbox(
        "Empresa",
        options=["Todas"] + companies,
        key="analysis_selected_company",
    )
    filtered = df.copy()
    if selected_company != "Todas":
        filtered = filtered[filtered["empresa"] == selected_company]

    history_df, _ = load_feature_history(CONFIG_PATH)
    render_summary_cards(filtered)
    render_investment_simulator(history_df=history_df, universe_df=df)
    st.markdown("---")

    display_cols = [
        "empresa",
        "ticker",
        "accion",
        "semaforo_icono",
        "semaforo",
        "confianza",
        "confianza_pct",
        "signal_strength",
    ] + [c for c in DEFAULT_DISPLAY_FEATURES if c in filtered.columns]

    st.subheader("Resumen tecnico por empresa")
    st.dataframe(filtered[display_cols].rename(columns=RENAME_MAP), width="stretch")
    render_feature_explanations(filtered)

    st.markdown("---")
    st.subheader("Analisis por empresa")

    hist_df = history_df
    if hist_df.empty:
        st.warning("No se encontro historial en feature_table.parquet. Ejecuta el pipeline de entrenamiento.")
    else:
        if selected_company == "Todas":
            chart_ticker = st.selectbox(
                "Selecciona ticker para analisis",
                options=sorted(filtered["ticker"].dropna().unique().tolist()),
                key="analysis_selected_ticker",
            )
            chart_company = (
                filtered.loc[filtered["ticker"] == chart_ticker, "empresa"].iloc[0]
                if len(filtered.loc[filtered["ticker"] == chart_ticker]) > 0
                else chart_ticker
            )
        else:
            company_rows = filtered[filtered["empresa"] == selected_company]
            chart_ticker = company_rows["ticker"].iloc[0]
            chart_company = selected_company

        company_history = hist_df[hist_df["ticker"] == chart_ticker].copy()
        if company_history.empty:
            st.info("No hay historico suficiente para esta empresa.")
        else:
            range_label = st.radio(
                "Rango temporal",
                options=["1M", "3M", "6M", "1Y", "MAX"],
                index=3,
                horizontal=True,
                key="analysis_range",
            )
            fig = build_company_analysis_chart(
                company_history,
                f"{chart_company} ({chart_ticker})",
                range_label=range_label,
            )
            st.plotly_chart(fig, use_container_width=True)

        selected_row = filtered[filtered["ticker"] == chart_ticker]
        if not selected_row.empty:
            st.markdown("---")
            render_model_recommendation_reason(selected_row.iloc[0])
            if "explicacion_retorno" in selected_row.columns:
                st.markdown("**Explicacion del retorno esperado:**")
                st.markdown(f"- {selected_row.iloc[0]['explicacion_retorno']}")

    st.markdown(
        "<div style='margin-top:20px;text-align:right;font-size:0.78rem;color:#8EA3B8;'>Author: Sergio Robledo Blanco</div>",
        unsafe_allow_html=True,
    )
