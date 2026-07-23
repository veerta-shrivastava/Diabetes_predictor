from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('diabetes_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [float(x) for x in request.form.values()]
        prediction = model.predict([np.array(features)])

        result = "Diabetic" if prediction[0] == 1 else "Not Diabetic"

        return render_template(
            'index.html',
            prediction_text=f'Result: {result}',
            form=request.form
        )

    except Exception as e:
        return render_template(
            'index.html',
            prediction_text=f'Error: {str(e)}',
            form=request.form
        )

# def predict():
#     try:
#         features = [float(x) for x in request.form.values()]
#         prediction = model.predict([np.array(features)])
#         result = "Diabetic" if prediction[0] == 1 else "Not Diabetic"
#         return render_template('index.html', prediction_text=f'Result: {result}')
#     except Exception as e:
#         return render_template('index.html', prediction_text=f'Error: {str(e)}')
@app.route('/predict', methods=['GET'])
def predict_page():
    return render_template('index.html')
if __name__ == '__main__':
    app.run(debug=True)
