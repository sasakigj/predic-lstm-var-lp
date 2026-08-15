"""
============================================================
  LSTM vs VAR — Prédiction de Séries Financières
  Amine SEMLALI | Master ESEF | Université de Lorraine
============================================================

Module 1 : Récupération des données et modèle VAR baseline
"""

import numpy as np
import pandas as pd
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────
# 1. RÉCUPÉRATION DES DONNÉES
# ─────────────────────────────────────────────────────────

def fetch_market_data(tickers=("AAPL", "^GSPC", "^VIX", "^TNX"), period="3y"):
    """
    Récupère les données de plusieurs actifs pour construire
    un système VAR multivarié.

    AAPL  : action cible à prédire
    ^GSPC : S&P 500 (marché global)
    ^VIX  : indice de volatilité implicite
    ^TNX  : taux 10 ans US (taux sans risque)
    """
    tickers = list(tickers)

    raw = yf.download(
        tickers, period=period, interval="1d",
        group_by="ticker", auto_adjust=True,
        threads=False, progress=False
    )

    data = {}
    for ticker in tickers:
        try:
            if len(tickers) == 1:
                series = raw["Close"]
            else:
                series = raw[ticker]["Close"]
            if series.dropna().shape[0] > 0:
                data[ticker] = series
            else:
                print(f"Ticker {ticker} : aucune donnée récupérée")
        except Exception as e:
            print(f"Erreur {ticker}: {e}")

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df = df.dropna()
    if len(df) > 0:
        df.index = df.index.tz_localize(None)
    return df


# indicateurs techniques

def compute_rsi(series, period=14):
    """Relative Strength Index."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_macd(series, fast=12, slow=26, signal=9):
    """Moving Average Convergence Divergence."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_bollinger(series, window=20, n_std=2):
    """Bandes de Bollinger."""
    mid = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = mid + n_std * std
    lower = mid - n_std * std
    pct_b = (series - lower) / (upper - lower)
    return upper, mid, lower, pct_b


def build_features(df, target_col="AAPL"):
    """
    Construit un jeu de features techniques et macro
    à partir des données brutes multivariées.
    """
    feat = pd.DataFrame(index=df.index)

    for col in df.columns:
        feat[f"{col}_ret"] = np.log(df[col] / df[col].shift(1))

    feat["rsi"] = compute_rsi(df[target_col])
    macd, macd_sig, macd_hist = compute_macd(df[target_col])
    feat["macd"] = macd
    feat["macd_signal"] = macd_sig
    feat["macd_hist"] = macd_hist
    _, _, _, pct_b = compute_bollinger(df[target_col])
    feat["bollinger_pctb"] = pct_b

    feat["volatility_10d"] = feat[f"{target_col}_ret"].rolling(10).std()
    feat["volatility_30d"] = feat[f"{target_col}_ret"].rolling(30).std()

    feat["momentum_5d"] = df[target_col].pct_change(5)
    feat["momentum_20d"] = df[target_col].pct_change(20)

    feat["price"] = df[target_col]

    feat = feat.dropna()
    return feat


# 3. VAR

def fit_var_model(df, maxlags=15, ic="aic"):
    """
    Ajuste un modèle VAR(p) multivarié avec sélection
    automatique de l'ordre optimal par critère d'information.
    """
    from statsmodels.tsa.api import VAR

    returns = np.log(df / df.shift(1)).dropna()

    model = VAR(returns)
    order_selection = model.select_order(maxlags=maxlags)
    best_lag = getattr(order_selection, ic)

    if best_lag == 0:
        best_lag = 1

    fitted = model.fit(best_lag)
    return fitted, best_lag, returns


def var_forecast(fitted_model, returns, target_col, last_price,
                  horizon=1, n_test_points=60):
    """
    Génère des prévisions rolling du modèle VAR sur un
    ensemble de test.

    Avec horizon > 1, la prédiction porte sur le rendement
    log CUMULÉ sur les `horizon` prochains jours (et non plus
    seulement sur le jour suivant), pour une comparaison
    équitable avec le LSTM à horizon variable.
    """
    lag = fitted_model.k_ar
    predictions = []
    actuals = []

    n_obs = len(returns)
    target_idx = returns.columns.get_loc(target_col)
    start_idx = n_obs - n_test_points - horizon

    for t in range(start_idx, n_obs - horizon):
        history = returns.iloc[t - lag:t].values
        try:
            fc = fitted_model.forecast(history, steps=horizon)
            pred_cum_ret = fc[:, target_idx].sum()
        except:
            pred_cum_ret = 0.0

        actual_cum_ret = returns[target_col].iloc[t + 1: t + 1 + horizon].sum()

        predictions.append(pred_cum_ret)
        actuals.append(actual_cum_ret)

    return np.array(predictions), np.array(actuals)


if __name__ == "__main__":
    df = fetch_market_data()
    print(df.tail())
    feat = build_features(df)
    print(feat.tail())
