"""
============================================================
  LSTM vs VAR — Prédiction de Séries Financières
  Amine SEMLALI | Master ESEF | Université de Lorraine
============================================================

Dashboard principal Streamlit.
Assemble : données marché, feature engineering, VAR,
LSTM+Attention, test de Diebold-Mariano.

Pour lancer, voir README.md !
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

from module1_var import (
    fetch_market_data, build_features, fit_var_model, var_forecast
)
from module2_lstm import (
    create_sequences, train_test_split_temporal, scale_features,
    train_lstm_model, predict_lstm
)
from module3_dm_test import (
    diebold_mariano_test, interpret_dm_test, compute_metrics
)
from module4_lp import (
    fit_local_projection, summarize_lp_coefficients
)


def safe_pick_winner(candidates, prefer_max=False):
    """
    Choisit le meilleur modèle sur une métrique, en ignorant les valeurs
    NaN (ex: MAPE indéfini si les rendements réels sont proches de zéro).
    Retourne None si aucune valeur valide n'est disponible.
    """
    valid = {k: v for k, v in candidates.items() if v is not None and not np.isnan(v)}
    if not valid:
        return None
    return max(valid, key=valid.get) if prefer_max else min(valid, key=valid.get)


def main():
    st.set_page_config(page_title="LSTM vs VAR", layout="wide")

    st.markdown("""
    <style>
    .metric-card {
        background: linear-gradient(135deg, #1A2035, #1E2D45);
        border: 1px solid #2A4070;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
    .metric-value { font-size: 26px; font-weight: bold; color: #4FC3F7; }
    .metric-label { font-size: 12px; color: #90CAF9; margin-top: 4px; }
    .info-box {
        background: #1A2035;
        border-left: 3px solid #4FC3F7;
        padding: 12px 16px;
        border-radius: 4px;
        margin: 8px 0;
        font-size: 13px;
        color: #B0BEC5;
    }
    .verdict-box {
        background: #1B2E1B;
        border-left: 3px solid #4CAF50;
        padding: 14px 16px;
        border-radius: 4px;
        margin: 12px 0;
        font-size: 14px;
        color: #C8E6C9;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("LSTM (Attention) vs VAR — Prédiction de Rendements Financiers")
    st.markdown("""<div class="info-box">
    Comparaison rigoureuse entre un modèle économétrique classique (<b>VAR multivarié</b>)
    et un modèle de deep learning (<b>LSTM avec mécanisme d'attention</b>) pour la prédiction
    de rendements financiers, avec test statistique de <b>Diebold-Mariano</b> pour valider
    la significativité de la différence de performance.
    </div>""", unsafe_allow_html=True)

    # ── Configuration ──
    st.sidebar.header("Configuration")
    ticker_target = st.sidebar.text_input("Actif cible", "AAPL")
    period = st.sidebar.selectbox("Historique", ["1y", "2y", "3y", "5y"], index=2)
    seq_length = st.sidebar.slider("Longueur de séquence (jours)", 10, 60, 20)
    horizon = st.sidebar.slider(
        "Horizon de prédiction (jours)", 1, 20, 5,
        help="Nombre de jours à l'avance. Un horizon plus long donne "
             "généralement un signal plus exploitable qu'à J+1."
    )
    n_test = st.sidebar.slider("Taille du test set", 30, 120, 60)
    epochs = st.sidebar.slider("Epochs LSTM", 20, 200, 80)
    hidden_dim = st.sidebar.slider("Dimension cachée LSTM", 16, 128, 64)
    alpha = st.sidebar.slider(
        "Poids MSE vs directionnel (alpha)", 0.0, 1.0, 0.7, 0.05,
        help="1.0 = MSE pur. Plus proche de 0, plus le modèle est "
             "pénalisé s'il se trompe de sens de variation."
    )
    with st.sidebar.expander("Régularisation avancée"):
        dropout = st.slider("Dropout", 0.0, 0.6, 0.2, 0.05,
                            help="Augmenter si le modèle overfit rapidement")
        weight_decay = st.select_slider(
            "Weight decay", options=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
            value=1e-5, format_func=lambda x: f"{x:.0e}"
        )

    if "results" not in st.session_state:
        st.session_state.results = None

    col_run, col_reset = st.sidebar.columns([2, 1])
    run_clicked = col_run.button("Lancer l'analyse", use_container_width=True)
    if col_reset.button("↺", use_container_width=True, help="Réinitialiser les résultats"):
        st.session_state.results = None
        st.rerun()

    if run_clicked:
        try:
            with st.spinner("Récupération des données de marché..."):
                df = fetch_market_data(
                    tickers=(ticker_target, "^GSPC", "^VIX", "^TNX"), period=period
                )

            if df is None or len(df) < 200:
                st.error("Données insuffisantes. Vérifiez le ticker ou la période.")
                return

            st.success(f"{len(df)} observations récupérées pour {ticker_target}")

            with st.spinner("Feature engineering..."):
                feat = build_features(df, target_col=ticker_target)

            # ── VAR (itéré) ──
            with st.spinner("Ajustement du modèle VAR..."):
                fitted_var, best_lag, returns = fit_var_model(df, maxlags=15)
                var_preds, var_actuals = var_forecast(
                    fitted_var, returns, ticker_target, df[ticker_target].iloc[-1],
                    horizon=horizon, n_test_points=n_test
                )

            # ── Projections Locales (Jordà, 2005) ──
            with st.spinner("Estimation des projections locales (Jordà)..."):
                try:
                    lp_preds, lp_actuals, lp_results = fit_local_projection(
                        returns, target_col=ticker_target, horizon=horizon,
                        n_lags=4, n_test_points=n_test
                    )
                    lp_available = True
                except Exception as e:
                    lp_available = False
                    st.warning(f"Projections locales indisponibles pour ces paramètres : {e}")

            # ── LSTM ──
            with st.spinner("Entraînement du LSTM avec attention (peut prendre 1-2 min)..."):
                target_col = f"{ticker_target}_ret"
                X, y, feature_cols = create_sequences(
                    feat, target_col=target_col, seq_length=seq_length, horizon=horizon
                )
                X_train, X_test, y_train, y_test = train_test_split_temporal(X, y, test_size=n_test)

                n_val = min(40, max(5, len(X_train) // 5))
                X_train_fit, X_val = X_train[:-n_val], X_train[-n_val:]
                y_train_fit, y_val = y_train[:-n_val], y_train[-n_val:]

                # Standardisation des features (fit sur train uniquement)
                X_train_fit, X_val, X_test, feature_scaler = scale_features(
                    X_train_fit, X_val, X_test
                )

                model, history, device = train_lstm_model(
                    X_train_fit, y_train_fit, X_val, y_val,
                    n_features=X.shape[2], hidden_dim=hidden_dim,
                    epochs=epochs, alpha=alpha, dropout=dropout,
                    weight_decay=weight_decay
                )
                lstm_preds, attn_weights = predict_lstm(model, X_test, device)

            # ── Alignement des séries pour comparaison ──
            if lp_available:
                min_len = min(len(var_preds), len(lstm_preds), len(y_test), len(lp_preds))
            else:
                min_len = min(len(var_preds), len(lstm_preds), len(y_test))

            var_preds_aligned = var_preds[-min_len:]
            var_actuals_aligned = var_actuals[-min_len:]
            lstm_preds_aligned = lstm_preds[-min_len:]
            lstm_actuals_aligned = y_test[-min_len:]

            results_dict = {
                "var_preds": var_preds_aligned,
                "var_actuals": var_actuals_aligned,
                "lstm_preds": lstm_preds_aligned,
                "lstm_actuals": lstm_actuals_aligned,
                "history": history,
                "attn_weights": attn_weights,
                "best_lag": best_lag,
                "ticker": ticker_target,
                "seq_length": seq_length,
                "horizon": horizon,
                "lp_available": lp_available,
            }

            if lp_available:
                results_dict["lp_preds"] = lp_preds[-min_len:]
                results_dict["lp_actuals"] = lp_actuals[-min_len:]
                results_dict["lp_results"] = lp_results

            st.session_state.results = results_dict
            st.success("Analyse terminée !")
        except Exception as e:
            st.error(
                f"Une erreur est survenue pendant l'analyse : {e}\n\n"
                "Causes fréquentes : ticker invalide, historique trop court "
                "pour les paramètres choisis (horizon, longueur de séquence), "
                "ou données de marché indisponibles pour la période sélectionnée. "
                "Essayez de réduire l'horizon, la longueur de séquence, ou d'augmenter "
                "l'historique récupéré."
            )
            st.session_state.results = None

    # ── Affichage des résultats ──
    if st.session_state.results is not None:
        r = st.session_state.results

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Prédictions", "Métriques", "Test de Diebold-Mariano", "Attention",
            "Projections Locales"
        ])

        # TAB 1 — Prédictions
        with tab1:
            st.subheader(f"Prédictions de rendement cumulé à {r['horizon']} jour(s) — {r['ticker']}")
            fig = go.Figure()
            x_axis = list(range(len(r["lstm_actuals"])))
            fig.add_trace(go.Scatter(x=x_axis, y=r["lstm_actuals"], name="Valeurs réelles",
                                      line=dict(color="#FFFFFF", width=2)))
            fig.add_trace(go.Scatter(x=x_axis, y=r["lstm_preds"], name="LSTM (Attention)",
                                      line=dict(color="#4FC3F7", width=2)))
            fig.add_trace(go.Scatter(x=x_axis, y=r["var_preds"], name=f"VAR({r['best_lag']})",
                                      line=dict(color="#EF5350", width=2, dash="dash")))
            if r.get("lp_available"):
                fig.add_trace(go.Scatter(x=x_axis, y=r["lp_preds"], name="Projection Locale (Jordà)",
                                          line=dict(color="#66BB6A", width=2, dash="dot")))
            fig.update_layout(
                paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
                font=dict(color="#B0BEC5"), height=450,
                xaxis=dict(title="Jour (test set)", gridcolor="#1E2D45"),
                yaxis=dict(title=f"Rendement log cumulé ({r['horizon']}j)", gridcolor="#1E2D45"),
                legend=dict(bgcolor="#1A2035")
            )
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Courbes d'apprentissage LSTM")

            best_ep = r["history"]["best_epoch"]
            total_ep = r["history"]["total_epochs"]

            if r["history"]["overfitting_warning"]:
                st.warning(
                    f"⚠️ Le meilleur modèle a été trouvé très tôt (epoch {best_ep}/{total_ep}). "
                    f"C'est le signe d'un overfitting rapide : le modèle est probablement trop "
                    f"complexe pour la quantité de données disponibles. Essayez d'augmenter le "
                    f"dropout / weight decay, de réduire la dimension cachée, ou d'utiliser "
                    f"plus d'historique."
                )
            else:
                st.success(f"✅ Meilleur modèle trouvé à l'epoch {best_ep}/{total_ep} — apprentissage progressif sain.")

            fig_hist = go.Figure()
            fig_hist.add_trace(go.Scatter(y=r["history"]["train_loss"], name="Train Loss",
                                           line=dict(color="#4FC3F7")))
            fig_hist.add_trace(go.Scatter(y=r["history"]["val_loss"], name="Validation Loss",
                                           line=dict(color="#FFA726")))
            fig_hist.add_vline(
                x=best_ep, line=dict(color="#4CAF50", width=2, dash="dash"),
                annotation_text=f"Meilleur modèle (epoch {best_ep})"
            )
            fig_hist.update_layout(
                paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
                font=dict(color="#B0BEC5"), height=300,
                xaxis=dict(title="Epoch", gridcolor="#1E2D45"),
                yaxis=dict(title="Loss hybride (MSE + directionnelle)", gridcolor="#1E2D45"),
                legend=dict(bgcolor="#1A2035")
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        # TAB 2 — Métriques
        with tab2:
            st.subheader("Métriques de performance comparées")

            m_lstm = compute_metrics(r["lstm_actuals"], r["lstm_preds"])
            m_var = compute_metrics(r["var_actuals"], r["var_preds"])

            metrics_dict = {
                "LSTM (Attention)": m_lstm,
                f"VAR({r['best_lag']}) itéré": m_var,
            }
            if r.get("lp_available"):
                m_lp = compute_metrics(r["lp_actuals"], r["lp_preds"])
                metrics_dict["Projection Locale (Jordà)"] = m_lp

            df_metrics = pd.DataFrame(metrics_dict).T
            st.dataframe(df_metrics.style.format("{:.4f}"), use_container_width=True)

            n_cols = 4
            c1, c2, c3, c4 = st.columns(n_cols)
            metric_names = ["RMSE", "MAE", "MAPE", "Directional_Accuracy"]
            for col, mname in zip([c1, c2, c3, c4], metric_names):
                with col:
                    candidates = {k: v[mname] for k, v in metrics_dict.items()}
                    winner = safe_pick_winner(candidates, prefer_max=(mname == "Directional_Accuracy"))
                    winner_display = winner if winner else "N/A"
                    st.markdown(f"""<div class="metric-card">
                        <div class="metric-value" style="color:#4CAF50">{winner_display}</div>
                        <div class="metric-label">Meilleur sur {mname}</div>
                    </div>""", unsafe_allow_html=True)

        # TAB 3 — Diebold-Mariano
        with tab3:
            st.subheader("Test de Diebold-Mariano (1995)")
            st.markdown("""<div class="info-box">
            H0 : les deux modèles ont la même précision prédictive.<br>
            H1 : la précision diffère significativement.<br>
            Si p-value < 0.05, la différence de performance observée n'est pas due au hasard.
            </div>""", unsafe_allow_html=True)

            comparisons = [("LSTM", r["lstm_actuals"], r["lstm_preds"], "VAR itéré", r["var_preds"])]
            if r.get("lp_available"):
                comparisons.append(("LSTM", r["lstm_actuals"], r["lstm_preds"], "Projection Locale", r["lp_preds"]))
                comparisons.append(("VAR itéré", r["var_actuals"], r["var_preds"], "Projection Locale", r["lp_preds"]))

            for name1, actual1, pred1, name2, pred2 in comparisons:
                st.markdown(f"#### {name1} vs {name2}")
                dm_stat, p_value = diebold_mariano_test(actual1, pred1, pred2, h=r["horizon"])
                result = interpret_dm_test(dm_stat, p_value, name1, name2)

                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"""<div class="metric-card">
                        <div class="metric-value">{dm_stat:.4f}</div>
                        <div class="metric-label">Statistique DM</div>
                    </div>""", unsafe_allow_html=True)
                with c2:
                    st.markdown(f"""<div class="metric-card">
                        <div class="metric-value">{p_value:.4f}</div>
                        <div class="metric-label">P-value</div>
                    </div>""", unsafe_allow_html=True)

                st.markdown(f"""<div class="verdict-box">
                <b>Conclusion :</b> {result['conclusion']}
                </div>""", unsafe_allow_html=True)
                st.markdown("---")

            st.markdown(f"""
            <div class="info-box">
            La statistique DM est calculée sur la différence des erreurs quadratiques entre
            les deux modèles, avec correction de Newey-West (horizon h={r['horizon']}) et
            correction de Harvey-Leybourne-Newbold pour petits échantillons. Cette correction
            est essentielle dès que h > 1 car les erreurs de prévision cumulées sur plusieurs
            jours sont autocorrélées.
            </div>
            """, unsafe_allow_html=True)

        # TAB 4 — Attention
        with tab4:
            st.subheader("Poids d'attention du LSTM")
            st.markdown("""<div class="info-box">
            Le mécanisme d'attention apprend à pondérer l'importance de chaque jour de la
            séquence d'entrée pour la prédiction finale. Cette heatmap montre, pour un
            échantillon de prédictions du test set, quels jours ont le plus influencé le modèle.
            </div>""", unsafe_allow_html=True)

            n_show = min(20, r["attn_weights"].shape[0])
            fig_attn = go.Figure(data=go.Heatmap(
                z=r["attn_weights"][:n_show],
                colorscale="Blues",
                colorbar=dict(title="Poids"),
            ))
            fig_attn.update_layout(
                paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
                font=dict(color="#B0BEC5"), height=400,
                xaxis=dict(title=f"Position dans la séquence (0 = il y a {r['seq_length']} jours)"),
                yaxis=dict(title="Échantillon de prédiction (test set)"),
            )
            st.plotly_chart(fig_attn, use_container_width=True)

            avg_attn = r["attn_weights"].mean(axis=0)
            fig_avg = go.Figure()
            fig_avg.add_trace(go.Bar(
                x=list(range(len(avg_attn))), y=avg_attn,
                marker_color="#4FC3F7"
            ))
            fig_avg.update_layout(
                title="Attention moyenne par position temporelle",
                paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
                font=dict(color="#B0BEC5"), height=300,
                xaxis=dict(title="Jours dans le passé", gridcolor="#1E2D45"),
                yaxis=dict(title="Poids d'attention moyen", gridcolor="#1E2D45"),
            )
            st.plotly_chart(fig_avg, use_container_width=True)

        # TAB 5 — Projections Locales
        with tab5:
            st.subheader("Projections Locales (Jordà, 2005)")
            st.markdown("""<div class="info-box">
            Contrairement au VAR itéré, qui construit la prévision à horizon h en
            itérant h fois le modèle un-pas-en-avant, la projection locale estime
            <b>directement</b>, en une seule régression, la relation entre l'information
            disponible à t et le rendement cumulé à t+h. Les erreurs-types sont corrigées
            par la méthode de Newey-West (HAC) avec une fenêtre L=h, pour tenir compte
            de l'autocorrélation MA(h-1) structurelle des résidus multi-horizon.
            </div>""", unsafe_allow_html=True)

            if not r.get("lp_available"):
                st.warning("Les projections locales n'ont pas pu être estimées pour cette configuration.")
            else:
                lp_results = r["lp_results"]

                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"""<div class="metric-card">
                        <div class="metric-value">{lp_results.rsquared:.4f}</div>
                        <div class="metric-label">R² (in-sample, train)</div>
                    </div>""", unsafe_allow_html=True)
                with c2:
                    st.markdown(f"""<div class="metric-card">
                        <div class="metric-value">{r['horizon']}</div>
                        <div class="metric-label">Fenêtre HAC (L = horizon)</div>
                    </div>""", unsafe_allow_html=True)

                st.markdown("---")
                st.subheader("Coefficients les plus significatifs")
                coef_table = summarize_lp_coefficients(lp_results, r["ticker"])
                st.dataframe(
                    coef_table.style.format({
                        "Coefficient": "{:.4f}",
                        "Erreur std (HAC)": "{:.4f}",
                        "t-stat": "{:.3f}",
                    }),
                    use_container_width=True
                )
                st.markdown("""
                <div class="info-box">
                Lecture : les variables affichées sont les valeurs courantes et retardées
                (lag0 à lag3) de chaque actif du système. Un coefficient assorti d'étoiles
                (* p&lt;0.10, ** p&lt;0.05, *** p&lt;0.01) indique une contribution
                statistiquement significative à la prévision du rendement cumulé à
                l'horizon choisi, une fois l'autocorrélation des résidus prise en compte.
                </div>
                """, unsafe_allow_html=True)

    else:
        st.info("Configure les paramètres dans la barre latérale et lance l'analyse.")

    st.markdown("---")
    st.markdown("""<div style="text-align:center; color:#546E7A; font-size:12px; padding: 8px;">
    Amine SEMLALI · Master ESEF · Université de Lorraine · aminesemlalicontact@gmail.com
    </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
