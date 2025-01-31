from flask import Flask, render_template, request
import numpy as np
import pickle

# Load the trained model
model = pickle.load(open("credit_card_fraud_model.pkl", "rb"))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=["POST"])
def predict():
    try:
        # Get user input from the form
        input_features = [float(request.form[f'V{i}']) for i in range(1, 29)]
        input_features.insert(0, float(request.form['Time']))   # Add 'Time' at index 0
        input_features.append(float(request.form['Amount']))    # Add 'Amount' at the end

        # Convert to NumPy array and reshape for prediction
        features_array = np.array([input_features]).reshape(1, -1)

        # Make prediction (1 = Fraud, 0 = Legitimate)
        prediction = model.predict(features_array)[0]
        result = "🚨 Fraudulent Transaction Detected!" if prediction == 1 else "✅ Legitimate Transaction"

        return render_template("index.html", prediction=result)

    except Exception as e:
        return render_template("index.html", prediction=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
