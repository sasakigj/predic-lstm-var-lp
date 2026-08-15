# LSTM vs VAR vs Projections Locales — Prédiction de Séries Financières

Amine SEMLALI | Master ESEF | Université de Lorraine
aminesemlalicontact@gmail.com

---

## Présentation

Ce projet compare trois approches de prédiction de rendements financiers :

- **VAR multivarié (itéré)** : modèle économétrique classique, sélection d'ordre par AIC, utilisant plusieurs actifs corrélés (action cible, indice de marché, VIX, taux 10 ans)
- **Projections locales (Jordà, 2005)** : estimation directe, horizon par horizon, de la relation entre l'information disponible et le rendement cumulé futur, sans itération
- **LSTM avec mécanisme d'attention** : réseau de neurones récurrent bidirectionnel avec attention additive, entraîné sur des features techniques et macro-financières

La comparaison est validée statistiquement par le **test de Diebold-Mariano (1995)**, appliqué en comparaisons pairwise entre les trois modèles, pour déterminer si les différences de précision prédictive observées sont significatives ou dues au hasard.

---

## Installation

```bash
pip install streamlit torch statsmodels yfinance pandas numpy scikit-learn plotly scipy
streamlit run lstm_var_dashboard.py
```

---

## Structure du projet

| Fichier | Contenu |
|---|---|
| `module1_var.py` | Récupération des données (yfinance), feature engineering (RSI, MACD, Bollinger), modèle VAR |
| `module2_lstm.py` | Architecture LSTM bidirectionnel + attention, loss hybride MSE/directionnelle, entraînement avec early stopping |
| `module3_dm_test.py` | Test de Diebold-Mariano, métriques de performance (RMSE, MAE, MAPE, directional accuracy) |
| `module4_lp.py` | Projections locales (Jordà, 2005), régression HAC, table de coefficients |
| `lstm_var_dashboard.py` | Dashboard Streamlit assemblant l'ensemble du pipeline |

---

## Horizon de prédiction

Les trois modèles prédisent le **rendement log cumulé** sur un horizon de N jours (paramétrable), et non plus seulement le rendement du lendemain. C'est un choix méthodologique : le signal exploitable dans les rendements journaliers est très faible (proche de l'hypothèse de marché efficient), alors qu'un horizon plus long (5 à 20 jours) capture mieux les dynamiques de momentum et de retour à la moyenne.

Pour le VAR, l'horizon est géré par sommation des prévisions pas-à-pas sur N jours. Pour la projection locale, la cible est directement régressée à l'horizon N. Pour le LSTM, la cible d'entraînement est le rendement cumulé calculé à partir des prix bruts.

---

## Modèle VAR (itéré)

Système d'équations sur les rendements log de quatre actifs corrélés (action cible, S&P 500, VIX, taux 10 ans US). L'ordre du modèle est sélectionné automatiquement par critère d'information (AIC). Les prévisions à horizon h sont générées en **itérant** h fois le modèle un-pas-en-avant, puis en sommant les h prévisions.

Cette approche a une limite documentée : toute erreur de spécification de la dynamique de court terme se propage à tous les horizons par itération.

---

## Projections Locales (Jordà, 2005)

Sur les conseils de mon professeur d'économétrie (M2 ESEF, Université de Lorraine), j'ai ajouté ce troisième modèle pour corriger la limite du VAR itéré décrite ci-dessus.

### Principe

Plutôt que d'itérer, la projection locale estime **directement**, en une seule régression, la relation entre l'information disponible à la date t et le rendement cumulé à t+h :

```
y_{t+h} = alpha_h + beta_h' x_t + eps_{t+h}
```

### Spécificité économétrique : erreurs-types HAC

Les résidus d'une projection locale suivent structurellement un processus **MA(h-1)** dès que h > 1 (les chocs de t+1 à t+h contaminent tous le résidu à l'horizon h). Les erreurs-types OLS classiques sont donc invalides. Le module utilise des erreurs-types **HAC (Newey-West)** avec une fenêtre L = h, comme recommandé dans la littérature (règle simple L = h, qui couvre exactement la dépendance théorique).

### Contenu de `module4_lp.py`

- `build_lp_dataset` : construction du jeu de régresseurs (valeurs courantes et retardées de toutes les variables du système)
- `fit_local_projection` : estimation OLS avec erreurs-types HAC, rolling forecast sur le test set
- `summarize_lp_coefficients` : tableau des coefficients les plus significatifs (t-stats, étoiles de significativité), dans l'esprit des sorties économétriques classiques

---

## Modèle LSTM avec attention

Architecture :

- LSTM bidirectionnel, 2 couches, dropout
- Couche d'attention additive (Bahdanau-style) qui pondère l'importance de chaque pas de temps de la séquence d'entrée
- Tête de régression (MLP) pour la prédiction du rendement cumulé à horizon N

L'entraînement utilise un early stopping sur la validation loss et un scheduler de learning rate.

### Loss hybride MSE / directionnelle

Un LSTM entraîné en MSE pur sur des rendements financiers converge souvent vers des prédictions proches de zéro : c'est mathématiquement optimal quand le signal est faible, mais peu utile en pratique (le modèle ne "s'engage" jamais sur un sens de variation).

La loss utilisée ici combine :

```
loss = alpha * MSE + (1 - alpha) * pénalité_directionnelle
```

La pénalité directionnelle est une **hinge loss à marge** :

```
penalty = mean( relu(margin - pred * sign(actual)) )
```

Le point important : la marge est strictement positive, ce qui empêche le modèle d'atteindre une pénalité nulle en prédisant toujours zéro (contrairement à une formulation naïve basée sur `tanh(k*pred)` qui s'annule exactement en zéro). La marge est recalculée à chaque batch comme une fraction de l'écart-type des rendements observés, pour s'adapter automatiquement à l'échelle de l'horizon choisi.

Le paramètre `alpha` est réglable dans le dashboard : `alpha=1.0` revient à une MSE pure, `alpha` plus faible force davantage le modèle à privilégier le bon sens de variation.

---

## Test de Diebold-Mariano

Le test compare les erreurs quadratiques de deux modèles sur l'échantillon de test :

```
d_t = e_t(modèle 1)^2 - e_t(modèle 2)^2

H0 : E[d_t] = 0   (précisions équivalentes)
H1 : E[d_t] != 0  (précisions différentes)
```

La statistique de test suit une loi normale standard sous H0, avec correction de Newey-West pour l'autocorrélation liée à l'horizon (essentielle dès que l'horizon dépasse 1 jour) et correction de Harvey-Leybourne-Newbold pour les échantillons de taille finie.

Le dashboard applique ce test en trois comparaisons pairwise : LSTM vs VAR, LSTM vs Projection Locale, VAR vs Projection Locale.

---

## Contenu du dashboard

**Prédictions** : comparaison visuelle des prédictions des trois modèles contre les valeurs réelles, courbes d'apprentissage du LSTM

**Métriques** : RMSE, MAE, MAPE, directional accuracy pour les trois modèles

**Test de Diebold-Mariano** : statistique de test, p-value et conclusion pour chaque comparaison pairwise

**Attention** : heatmap des poids d'attention appris par le LSTM, montrant quels jours passés influencent le plus la prédiction

**Projections Locales** : R² in-sample et table des coefficients les plus significatifs de la régression HAC

---

## Stack technique

Python 3.x, PyTorch, statsmodels, yfinance, scikit-learn, plotly, streamlit, scipy
