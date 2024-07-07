from flask import Flask, render_template, request, redirect
import pandas as pd
import joblib
from data_preparation import rundata
from prediction import run
app = Flask(__name__)

cryptos = ["BTC-USD", "ETH-USD", "DOGE-USD", "USDT-USD", "BNB-USD"]

@app.route("/",methods=['GET','POST'])
def home():
    if request.method=="GET":
        return render_template("homepage.html")
    elif request.method=="POST":
        rundata()
        run()
        return redirect('index')

@app.route('/index')
def index():
    return render_template('index.html', cryptos=cryptos)

@app.route('/predict', methods=['POST'])
def predict():
    crypto = request.form['crypto']
    model = joblib.load(f"{crypto}_best_model.pkl")
    data = pd.read_csv(f"{crypto}_combined.csv", index_col=0, parse_dates=True).iloc[-1:]
    predictors = ["close", "volume", "open", "high", "low", "edit_count", "sentiment", "neg_sentiment"]
    prediction = model.predict(data[predictors])
    result = "Buy" if prediction[0] == 1 else "Don't Buy"
    return render_template('result.html', result=result, crypto=crypto)

if __name__ == '__main__':
    app.run(debug=True)
