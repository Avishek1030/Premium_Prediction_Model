from flask import Flask, request, jsonify, render_template
import numpy as np
from sklearn.ensemble import  RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import pickle
import warnings
warnings.filterwarnings('ignore')

model=pickle.load(open(r"C:\Users\AVISHEK PASWAN\Desktop\Avishek\FinalRFModel.pkl",'rb'))
scaler=pickle.load(open(r"C:\Users\AVISHEK PASWAN\Desktop\Avishek\scaler.pkl",'rb'))

app=Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data=request.form
        print(data)
        scaled_data=scaler.transform(np.array([[data['anual_income'],data['health_score'],data['credit_score'],data['exercise_frequency']]]))
        prediction=model.predict(scaled_data)
        return render_template('index.html',predicted_value=f"The premium will be {prediction[0].round(2)}", method='get')
    except Exception as e:
        return jsonify({'error': str(e)})
if __name__ == '__main__':
    app.run(debug=True)
