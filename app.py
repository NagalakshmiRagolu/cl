import numpy as np
from flask import Flask, render_template, request
import pickle
import os

app = Flask(__name__)

# ==============================
# Load trained models
# ==============================
models = {
    "knn": None,
    "naive_bayes": None,
    "dt": None,
    "rf": None,
    "adaboost": None,
    "gb": None,
    "xgb": None,
    "svc": None
}

# Load all models if available
for key, filename in {
    "knn": "KNN_model.pkl",
    "naive_bayes": "Naive_bayes_model.pkl",
    "dt": "DT_model.pkl",
    "rf": "RF_model.pkl",
    "adaboost": "adaboost_model.pkl",
    "gb": "GB_model.pkl",
    "xgb": "XB_model.pkl",
    "svc": "SVC_model.pkl"
}.items():
    path = os.path.join("models", filename)
    if os.path.exists(path):
        with open(path, "rb") as f:
            models[key] = pickle.load(f)
    else:
        print(f"⚠️ {filename} not found in models/")

# ==============================
# Routes
# ==============================
@app.route("/")
def home():
    return render_template("index.html")

# ---------- Prediction Routes ----------
@app.route("/predict/<model_name>", methods=["GET", "POST"])
def predict(model_name):
    model = models.get(model_name)
    if request.method == "GET":
        return render_template("predict.html", model_name=model_name.upper())

    if model is None:
        return render_template("predict.html", model_name=model_name.upper(), result="⚠️ Model not available. Train first!")

    try:
        sepal_length = float(request.form["sepal_length"])
        sepal_width = float(request.form["sepal_width"])
        petal_length = float(request.form["petal_length"])
        petal_width = float(request.form["petal_width"])

        # Add dummy first column if model expects 5 features
        try:
            input_features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
            # Check feature number dynamically
            if model.n_features_in_ > input_features.shape[1]:
                extra = model.n_features_in_ - input_features.shape[1]
                input_features = np.hstack([np.zeros((1, extra)), input_features])
        except AttributeError:
            # If model does not have n_features_in_ attribute, just use 4 features
            input_features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

        prediction = model.predict(input_features)[0]
        iris_map = {0: "Iris-setosa", 1: "Iris-versicolor", 2: "Iris-virginica"}
        species = iris_map.get(prediction, "Unknown")

        return render_template(
            "predict.html",
            model_name=model_name.upper(),
            result=f"🌸 Predicted Species: {species}"
        )
    except Exception as e:
        return render_template("predict.html", model_name=model_name.upper(), result=f"Error: {e}")


@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

if __name__ == "__main__":
    app.run(debug=True)
