"""
============================================================
  LSTM vs VAR — Prédiction de Séries Financières
  Amine SEMLALI | Master ESEF | Université de Lorraine
============================================================

Module 3 : Test de Diebold-Mariano (1995)
"""

import numpy as np
from scipy import stats


def diebold_mariano_test(actual, pred1, pred2, h=1, loss="squared"):
    """
    Test de Diebold-Mariano pour comparer statistiquement
    la précision prédictive de deux modèles.

    H0 : les deux modèles ont la même précision prédictive
    H1 : les précisions diffèrent significativement

    Paramètres
    ----------
    actual : valeurs réelles observées
    pred1  : prédictions du modèle 1 (ex: LSTM)
    pred2  : prédictions du modèle 2 (ex: VAR)
    h      : horizon de prévision (pour l'ajustement de la variance)
    loss   : "squared" (MSE) ou "absolute" (MAE)

    Retourne
    --------
    dm_stat  : statistique de test (~ N(0,1) sous H0)
    p_value  : p-value bilatérale
    """
    actual = np.asarray(actual).flatten()
    pred1 = np.asarray(pred1).flatten()
    pred2 = np.asarray(pred2).flatten()

    e1 = actual - pred1
    e2 = actual - pred2

    if loss == "squared":
        d = e1**2 - e2**2
    else:
        d = np.abs(e1) - np.abs(e2)

    n = len(d)
    d_mean = np.mean(d)

    # Variance de long terme avec correction Newey-West
    # pour tenir compte de l'autocorrélation (horizon h)
    gamma_0 = np.var(d, ddof=1)
    var_d = gamma_0

    for lag in range(1, h):
        gamma_lag = np.mean(
            (d[lag:] - d_mean) * (d[:-lag] - d_mean)
        )
        var_d += 2 * gamma_lag

    var_d = max(var_d, 1e-12)  # éviter division par zéro
    dm_stat = d_mean / np.sqrt(var_d / n)

    # Correction de Harvey, Leybourne & Newbold (1997) pour petits échantillons
    hln_correction = np.sqrt(
        (n + 1 - 2*h + h*(h-1)/n) / n
    )
    dm_stat_corrected = dm_stat * hln_correction

    # p-value bilatérale (loi de Student avec n-1 ddl)
    p_value = 2 * (1 - stats.t.cdf(np.abs(dm_stat_corrected), df=n - 1))

    return dm_stat_corrected, p_value


def interpret_dm_test(dm_stat, p_value, model1_name="LSTM", model2_name="VAR",
                       alpha=0.05):
    """
    Génère une interprétation textuelle claire du résultat du test.
    """
    significant = p_value < alpha

    if not significant:
        conclusion = (
            f"Différence NON significative (p={p_value:.4f} > {alpha}). "
            f"On ne peut pas conclure que {model1_name} et {model2_name} "
            f"ont des précisions prédictives différentes."
        )
    else:
        better = model1_name if dm_stat < 0 else model2_name
        conclusion = (
            f"Différence STATISTIQUEMENT SIGNIFICATIVE (p={p_value:.4f} < {alpha}). "
            f"Le modèle {better} prédit significativement mieux."
        )

    return {
        "dm_statistic": dm_stat,
        "p_value": p_value,
        "significant": significant,
        "conclusion": conclusion,
    }


def compute_metrics(actual, predicted):
    """RMSE, MAE, MAPE standards."""
    actual = np.asarray(actual).flatten()
    predicted = np.asarray(predicted).flatten()

    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    mae = np.mean(np.abs(actual - predicted))

    # MAPE : on évite la division par zéro
    mask = np.abs(actual) > 1e-8
    if mask.sum() > 0:
        mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
    else:
        mape = np.nan

    # Directional accuracy : le modèle prédit-il le bon sens de variation ?
    dir_actual = np.sign(actual)
    dir_pred = np.sign(predicted)
    directional_accuracy = np.mean(dir_actual == dir_pred) * 100

    return {
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape,
        "Directional_Accuracy": directional_accuracy,
    }


if __name__ == "__main__":
    np.random.seed(42)
    n = 200
    actual = np.random.randn(n) * 0.01

    # Modèle 1 : légèrement meilleur (bruit plus faible)
    pred1 = actual + np.random.randn(n) * 0.005
    # Modèle 2 : moins bon (bruit plus fort)
    pred2 = actual + np.random.randn(n) * 0.012

    dm_stat, p_val = diebold_mariano_test(actual, pred1, pred2)
    result = interpret_dm_test(dm_stat, p_val, "LSTM", "VAR")

    print("=== Test de Diebold-Mariano ===")
    print(f"Statistique DM : {dm_stat:.4f}")
    print(f"P-value        : {p_val:.4f}")
    print(result["conclusion"])
    print()

    metrics1 = compute_metrics(actual, pred1)
    metrics2 = compute_metrics(actual, pred2)
    print("=== Métriques Modèle 1 (LSTM simulé) ===")
    print(metrics1)
    print("=== Métriques Modèle 2 (VAR simulé) ===")
    print(metrics2)
    print()
    print("Module Diebold-Mariano fonctionne correctement.")
