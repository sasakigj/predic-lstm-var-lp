"""
============================================================
  LSTM vs VAR — Prédiction de Séries Financières
  Amine SEMLALI | Master ESEF | Université de Lorraine
============================================================

Module 2 : LSTM avec mécanisme d'attention (PyTorch)
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

# 1. Préparation des séquences

def scale_features(X_train, X_val, X_test):
    """
    Standardise les features (moyenne=0, écart-type=1), en calculant
    les statistiques UNIQUEMENT sur le train set (pour éviter tout
    lookahead bias), puis en les appliquant à train/val/test.

    C'est une étape cruciale : sans elle, des features à échelles très
    différentes (RSI ~0-100, rendements ~0.01, volatilité ~0.001...)
    déséquilibrent les gradients et poussent le LSTM vers des
    prédictions plates et peu informatives.

    X_train, X_val, X_test : arrays de forme (n_samples, seq_length, n_features)
    """
    n_features = X_train.shape[2]

    scaler = StandardScaler()
    # On aplatit (samples * seq_length, n_features) pour fitter le scaler
    X_train_flat = X_train.reshape(-1, n_features)
    scaler.fit(X_train_flat)

    def apply_scaler(X):
        shape = X.shape
        X_flat = X.reshape(-1, n_features)
        X_scaled = scaler.transform(X_flat)
        return X_scaled.reshape(shape)

    return apply_scaler(X_train), apply_scaler(X_val), apply_scaler(X_test), scaler


def create_sequences(features_df, target_col="AAPL_ret", seq_length=20, horizon=1):
    """
    Transforme les données en séquences (X, y) pour le LSTM.
    X : fenêtre glissante de seq_length pas de temps, toutes features
    y : rendement log CUMULÉ sur les `horizon` prochains jours,
        calculé à partir du prix brut : log(P[t+h] / P[t])

    Avec horizon=1 on retrouve le comportement d'origine (rendement
    du lendemain). Un horizon plus long lisse le bruit journalier
    et donne au modèle un signal plus exploitable.
    """
    feature_cols = [c for c in features_df.columns if c != "price"]
    data = features_df[feature_cols].values
    prices = features_df["price"].values

    X, y = [], []
    n = len(data)
    for i in range(n - seq_length - horizon):
        X.append(data[i:i + seq_length])
        p_start = prices[i + seq_length - 1]
        p_end = prices[i + seq_length - 1 + horizon]
        cum_log_return = np.log(p_end / p_start)
        y.append(cum_log_return)

    return np.array(X), np.array(y), feature_cols


def train_test_split_temporal(X, y, test_size=60):
    """Split temporel : le test set est toujours à la fin (pas de shuffle)."""
    X_train, X_test = X[:-test_size], X[-test_size:]
    y_train, y_test = y[:-test_size], y[-test_size:]
    return X_train, X_test, y_train, y_test


# 2. Loss hybride : MSE + pénalité directionnelle

class HybridDirectionalLoss(nn.Module):
    """
    Combine l'erreur quadratique classique (MSE) avec une pénalité
    de type "hinge" qui force le modèle à s'ENGAGER sur un sens de
    variation (hausse/baisse), pas seulement une amplitude proche
    de la moyenne.

    loss = alpha * MSE + (1 - alpha) * pénalité_directionnelle

    ATTENTION : une formulation naïve du type relu(-tanh(k*pred)*sign(actual))
    s'annule exactement quand pred=0, ce qui crée une solution dégénérée
    où le modèle apprend à toujours prédire 0 pour éviter toute pénalité.
    On utilise donc une hinge loss AVEC MARGE :

        penalty = relu(margin - pred * sign(actual))

    À pred=0, la pénalité vaut `margin` (strictement positive), ce qui
    élimine cette solution triviale et force le modèle à s'engager.
    La marge est recalculée à chaque batch comme une fraction de
    l'écart-type des rendements observés, pour s'adapter automatiquement
    à l'échelle des cibles (rendement journalier ou cumulé sur N jours).
    """
    def __init__(self, alpha=0.7, margin_ratio=0.15):
        super().__init__()
        self.alpha = alpha
        self.margin_ratio = margin_ratio
        self.mse = nn.MSELoss()

    def forward(self, pred, actual):
        mse_loss = self.mse(pred, actual)

        margin = self.margin_ratio * torch.std(actual).detach()
        margin = torch.clamp(margin, min=1e-6)

        actual_sign = torch.sign(actual)
        directional_penalty = torch.mean(
            torch.relu(margin - pred * actual_sign)
        )

        return self.alpha * mse_loss + (1 - self.alpha) * directional_penalty


# 3. Architecture LSTM + attention

class AttentionLayer(nn.Module):
    """
    Mécanisme d'attention additive (Bahdanau-style).
    Apprend à pondérer chaque pas de temps de la séquence
    LSTM selon son importance pour la prédiction finale.
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, hidden_dim)
        self.context_vector = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_output):
        energy = torch.tanh(self.attn(lstm_output))
        attention_weights = torch.softmax(
            self.context_vector(energy), dim=1
        )
        weighted_output = lstm_output * attention_weights
        context = torch.sum(weighted_output, dim=1)
        return context, attention_weights.squeeze(-1)


class LSTMAttentionModel(nn.Module):
    """
    LSTM bidirectionnel + couche d'attention + tête de régression.
    Prédit le rendement cumulé à horizon h à partir d'une fenêtre
    de features.
    """
    def __init__(self, n_features, hidden_dim=64, n_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
            bidirectional=True,
        )
        self.attention = AttentionLayer(hidden_dim * 2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        context, attn_weights = self.attention(lstm_out)
        output = self.fc(context)
        return output.squeeze(-1), attn_weights


# 4. Entraînement

def train_lstm_model(X_train, y_train, X_val, y_val,
                      n_features, hidden_dim=64, n_layers=2,
                      epochs=100, lr=0.001, batch_size=32, patience=15,
                      alpha=0.7, dropout=0.2, weight_decay=1e-5):
    """
    Entraîne le modèle LSTM+Attention avec early stopping.

    alpha : poids de la composante MSE dans la loss hybride.
    dropout, weight_decay : réglages de régularisation, à augmenter
        si le modèle overfit rapidement (best_epoch très bas).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LSTMAttentionModel(n_features, hidden_dim, n_layers, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = HybridDirectionalLoss(alpha=alpha)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.FloatTensor(y_val).to(device)

    n_samples = len(X_train_t)
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    best_state = None
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_samples)
        epoch_loss = 0.0

        for i in range(0, n_samples, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = X_train_t[idx], y_train_t[idx]

            optimizer.zero_grad()
            preds, _ = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(idx)

        epoch_loss /= n_samples

        model.eval()
        with torch.no_grad():
            val_preds, _ = model(X_val_t)
            val_loss = criterion(val_preds, y_val_t).item()

        scheduler.step(val_loss)
        history["train_loss"].append(epoch_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    history["best_epoch"] = best_epoch
    history["total_epochs"] = len(history["train_loss"])
    history["overfitting_warning"] = best_epoch < max(3, 0.15 * history["total_epochs"])

    return model, history, device


def predict_lstm(model, X_test, device):
    """Génère les prédictions et les poids d'attention sur le test set."""
    model.eval()
    X_test_t = torch.FloatTensor(X_test).to(device)
    with torch.no_grad():
        preds, attn_weights = model(X_test_t)
    return preds.cpu().numpy(), attn_weights.cpu().numpy()


if __name__ == "__main__":
    np.random.seed(0)
    n_features = 13
    X = np.random.randn(300, 20, n_features).astype(np.float32)
    y = np.random.randn(300).astype(np.float32) * 0.01

    X_train, X_test, y_train, y_test = train_test_split_temporal(X, y, test_size=60)
    X_train, X_val = X_train[:-40], X_train[-40:]
    y_train, y_val = y_train[:-40], y_train[-40:]

    model, history, device = train_lstm_model(
        X_train, y_train, X_val, y_val, n_features, epochs=10, alpha=0.7
    )
    preds, attn = predict_lstm(model, X_test, device)
    print("Predictions shape:", preds.shape)
    print("Attention weights shape:", attn.shape)
    print("Modèle LSTM+Attention (loss hybride) fonctionne correctement.")
