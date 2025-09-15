from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model
model = pickle.load(open("model.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form values
        loan_amount = float(request.form['loan_amount'])
        interest_rate = float(request.form['interest_rate'])
        annual_income = float(request.form['annual_income'])
        loan_term = int(request.form['loan_term'])

        # Example feature array (update if your model expects differently)
        features = np.array([[loan_amount, interest_rate, annual_income, loan_term]])
        prediction = model.predict(features)[0]
        
        # Try to get probability if available
        try:
            confidence = round(np.max(model.predict_proba(features)[0]) * 100, 2)
        except:
            confidence = "N/A"

        # Format prediction as expected by the template
        risk_level = "High" if prediction == 0 else "Low"
        prediction_text = f"Default Risk: {risk_level}"
        
        # Create dictionary with all values for the template
        input_values = {
            'Loan Amount': f"${loan_amount:,.2f}",
            'Interest Rate': f"{interest_rate}%",
            'Annual Income': f"${annual_income:,.2f}",
            'Loan Term': f"{loan_term} months"
        }
        
        # Create message based on risk assessment
        if risk_level == "High":
            message = "Based on the provided information, this loan has a higher risk of default. Key factors may include the loan amount relative to income, high interest rate, or extended loan term."
        else:
            message = "Based on the provided information, this loan has a lower risk of default. The loan amount, interest rate, and term appear manageable relative to the income."

        return render_template(
            'result.html',
            prediction=prediction_text,
            input_values=input_values,
            message=message
        )

    except Exception as e:
        return render_template(
            'result.html',
            prediction="Error in prediction",
            input_values={"Error": str(e)},
            message="An error occurred during prediction."
        )

if __name__ == "__main__":
    app.run(debug=True)
