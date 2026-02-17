from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# Load dataset
data = pd.read_csv("diabetes.csv")

X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# Train model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


@app.route("/", methods=["GET", "POST"])
def index():
    result = ""
    if request.method == "POST":
        input_data = [[
            int(request.form["preg"]),
            int(request.form["glu"]),
            int(request.form["bp"]),
            int(request.form["skin"]),
            int(request.form["ins"]),
            float(request.form["bmi"]),
            float(request.form["dpf"]),
            int(request.form["age"])
        ]]

        prediction = model.predict(input_data)

        if prediction[0] == 1:
            result = "⚠️ Patient has Diabetes"
        else:
            result = "✅ Patient is Healthy"

    return render_template("index.html", result=result)


if __name__ == "__main__":
    app.run(debug=True)
