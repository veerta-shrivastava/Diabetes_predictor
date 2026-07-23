# 🩺 Diabetes Predictor

A machine learning-powered web application that predicts whether a person is likely to have diabetes based on key health parameters. The application provides an intuitive web interface built with Flask, allowing users to enter medical details and receive instant predictions.

---

## 🚀 Features

* Predicts the likelihood of diabetes using a trained Machine Learning model.
* Simple and user-friendly web interface.
* Instant prediction results.
* Handles invalid inputs gracefully.
* Built with Flask for lightweight deployment.

---

## 🛠️ Tech Stack

### Frontend

* HTML5
* CSS3

### Backend

* Flask (Python)

### Machine Learning

* Scikit-learn
* NumPy
* Joblib

---

## 📂 Project Structure

```text
Diabetes_predictor/
│── static/
│   └── index.css
│── templates/
│   └── index.html
│── app.py
│── diabetes_model.pkl
│── requirements.txt
│── .gitignore
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/veerta-shrivastava/Diabetes_predictor.git
cd Diabetes_predictor
```

### 2. Create a virtual environment (Recommended)

```bash
python -m venv .venv
```

Activate it:

**Windows**

```bash
.venv\Scripts\activate
```

**Linux/macOS**

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
python app.py
```

Open your browser and visit:

```
http://127.0.0.1:5000
```

---

## 📋 Input Parameters

The model predicts diabetes based on the following health attributes:

* Pregnancies
* Glucose Level
* Blood Pressure
* Skin Thickness
* Insulin Level
* BMI
* Diabetes Pedigree Function
* Age

---

## 🧠 Machine Learning Model

The trained model is stored as:

```
diabetes_model.pkl
```

It is loaded using Joblib and used to generate predictions from user-provided medical information.

---


---

## 📦 Dependencies

* Flask
* NumPy
* Scikit-learn
* SciPy
* Joblib
* Jinja2
* Werkzeug

Install all dependencies using:

```bash
pip install -r requirements.txt
```

---

## 🔮 Future Improvements

* User authentication.
* Prediction history.
* Probability score for predictions.
* Interactive charts and visualizations.
* Model comparison with multiple algorithms.
* Deployment using Render or Railway.
* Responsive mobile-friendly interface.

---

