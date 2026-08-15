"""
============================================================
  LSTM vs VAR vs Projections Locales — Prédiction Financière
  Amine SEMLALI | Master ESEF | Université de Lorraine
============================================================

Module 4 : Projections locales (Jordà, 2005)

Contrairement au VAR itéré (module1_var.py), qui construit la
prévision à horizon h en itérant h fois le modèle un-pas-en-avant
(propageant ainsi toute erreur de spécification à tous les horizons),
la projection locale estime DIRECTEMENT, en une seule régression,
la relation entre l'information disponible à t et le rendement
cumulé à t+h :

    y_{t+h} = alpha_h + beta_h' x_t + eps_{t+h}

Les résidus suivent structurellement un processus MA(h-1) (Jordà,
2005), ce qui impose l'usage d'erreurs-types HAC (Newey-West) avec
une fenêtre L = h, plutôt que des erreurs-types OLS classiques.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")


def build_lp_dataset(returns, target_col, horizon, n_lags=4):
    """
    Construit le jeu de données pour la régression de projection locale
    à l'horizon h.

    Cible   : y_{t+h} = somme des rendements de t+1 à t+horizon (rendement
              cumulé), même convention que var_forecast pour une
              comparaison équitable entre modèles.
    Régresseurs : valeurs courantes et retardées (n_lags au total,
              incluant t) de TOUTES les variables du système, connues
              à la date t.
    """
    cols = list(returns.columns)
    n_obs = len(returns)
    rows_X, rows_y, idx = [], [], []

    for t in range(n_lags - 1, n_obs - horizon):
        y_t = returns[target_col].iloc[t + 1: t + 1 + horizon].sum()

        x_t = []
        for lag in range(0, n_lags):
            x_t.extend(returns.iloc[t - lag][cols].values)

        rows_X.append(x_t)
        rows_y.append(y_t)
        idx.append(returns.index[t])

    feature_names = [f"{c}_lag{lag}" for lag in range(n_lags) for c in cols]
    X = pd.DataFrame(rows_X, columns=feature_names, index=idx)
    y = pd.Series(rows_y, name="y_cum", index=idx)
    return X, y


def fit_local_projection(returns, target_col, horizon, n_lags=4, n_test_points=60):
    """
    Estime une projection locale (Jordà, 2005) pour prédire directement
    le rendement cumulé à horizon h, SANS itération (contrairement au
    VAR itéré). Utilise des erreurs-types HAC (Newey-West, L=h) pour
    corriger l'autocorrélation MA(h-1) structurelle des résidus
    multi-horizon.

    Retourne :
        preds, actuals : rolling forecast sur le test set (même format
                          que var_forecast, pour comparaison directe)
        results         : objet statsmodels ajusté sur le train set
                          (coefficients, t-stats HAC, R², etc.)
    """
    X, y = build_lp_dataset(returns, target_col, horizon, n_lags)

    n_obs = len(X)
    train_end = n_obs - n_test_points
    if train_end < n_lags * len(returns.columns) + 5:
        raise ValueError(
            "Pas assez d'observations pour estimer la projection locale "
            "avec ces paramètres (horizon/n_lags trop élevés pour la "
            "taille de l'échantillon)."
        )

    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_test, y_test = X.iloc[train_end:], y.iloc[train_end:]

    X_train_c = sm.add_constant(X_train)
    model = sm.OLS(y_train, X_train_c)

    # Erreurs-types HAC (Newey-West), fenêtre L = h : règle simple
    # recommandée pour couvrir la dépendance MA(h-1) théorique des
    # résidus (cf. cours, "L = h couvre la dépendance théorique")
    results = model.fit(cov_type="HAC", cov_kwds={"maxlags": max(horizon, 1)})

    X_test_c = sm.add_constant(X_test, has_constant="add")
    preds = results.predict(X_test_c)

    return preds.values, y_test.values, results


def summarize_lp_coefficients(results, target_col, n_display=8):
    """
    Produit un tableau de synthèse des coefficients les plus significatifs,
    dans l'esprit du tableau de la diapositive 50 du cours (coefficient,
    erreur-type HAC, t-stat, étoiles de significativité).
    """
    params = results.params
    tvalues = results.tvalues
    pvalues = results.pvalues
    std_errs = results.bse

    rows = []
    for name in params.index:
        if name == "const":
            continue
        stars = ""
        if pvalues[name] < 0.01:
            stars = "***"
        elif pvalues[name] < 0.05:
            stars = "**"
        elif pvalues[name] < 0.10:
            stars = "*"
        rows.append({
            "Variable": name,
            "Coefficient": params[name],
            "Erreur std (HAC)": std_errs[name],
            "t-stat": tvalues[name],
            "Signif.": stars,
        })

    df = pd.DataFrame(rows)
    df["abs_t"] = df["t-stat"].abs()
    df = df.sort_values("abs_t", ascending=False).drop(columns="abs_t")
    return df.head(n_display).reset_index(drop=True)


if __name__ == "__main__":
    np.random.seed(3)
    n = 500
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    returns_data = np.random.multivariate_normal(
        mean=[0.0005, 0.0003, 0.0, 0.0],
        cov=[[0.0004, 0.0002, -0.0001, 0.00005],
             [0.0002, 0.0002, -0.00008, 0.00003],
             [-0.0001, -0.00008, 0.001, -0.00002],
             [0.00005, 0.00003, -0.00002, 0.00001]],
        size=n
    )
    returns = pd.DataFrame(returns_data, index=dates,
                           columns=["AAPL", "^GSPC", "^VIX", "^TNX"])

    HORIZON = 6
    preds, actuals, results = fit_local_projection(
        returns, target_col="AAPL", horizon=HORIZON, n_lags=4, n_test_points=60
    )

    print(f"Nombre de prédictions LP: {len(preds)}")
    print(f"R² (in-sample, train): {results.rsquared:.4f}")
    print()
    print("=== Tableau des coefficients les plus significatifs ===")
    print(summarize_lp_coefficients(results, "AAPL"))
    print()
    rmse = np.sqrt(np.mean((preds - actuals)**2))
    print(f"RMSE test set: {rmse:.5f}")
    print()
    print("=== MODULE PROJECTIONS LOCALES : OK ===")
