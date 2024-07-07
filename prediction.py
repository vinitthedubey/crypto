import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
import joblib

cryptos = ["BTC-USD", "ETH-USD", "DOGE-USD", "USDT-USD", "BNB-USD"]

def train_and_save_best_model(crypto):
    btc = pd.read_csv(f"{crypto}_combined.csv", index_col=0, parse_dates=True)
    
    train = btc.iloc[:-200]
    test = btc.iloc[-200:]
    
    predictors = ["close", "volume", "open", "high", "low", "edit_count", "sentiment", "neg_sentiment"]
    
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, min_samples_split=50, random_state=1),
        "XGBoost": XGBClassifier(random_state=1, learning_rate=.1, n_estimators=200),
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "GradientBoostingClassifier": GradientBoostingClassifier(n_estimators=100, random_state=1),
        "KNeighbors": KNeighborsClassifier(n_neighbors=5)
    }
    
    best_model = None
    best_score = 0
    
    for model_name, model in models.items():
        model.fit(train[predictors], train["target"])
        preds = model.predict(test[predictors])
        score = precision_score(test["target"], preds)
        print(f"{model_name} Precision for {crypto}: {score}")
        
        if score > best_score:
            best_score = score
            best_model = model
    
    print(f"Best Model for {crypto}: {best_model.__class__.__name__} with precision {best_score}")
    
    joblib.dump(best_model, f"{crypto}_best_model.pkl")
    return best_model, best_score

def run():
    for crypto in cryptos:
        train_and_save_best_model(crypto)
