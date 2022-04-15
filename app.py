import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model_log= pickle.load(open('model_log.pkl', 'rb'))

df = pd.read_csv('diabetes.csv')
X=df.drop(['Outcome'],axis=1)

y=df['Outcome']

@app.route('/')
def home():
    return render_template('resultp.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    float_features = [int(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    prediction = model_log.predict(final_features)

    if prediction == 1:
        pred = "You have Diabetes."
    elif prediction == 0:
        pred = "You don't have diabetes."
    output = pred

    return render_template('resultp.html', prediction_text='{}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
