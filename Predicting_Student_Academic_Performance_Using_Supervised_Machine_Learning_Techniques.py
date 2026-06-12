# Generated from: Predicting_Student_Academic_Performance_Using_Supervised_Machine_Learning_Techniques.ipynb
# Converted at: 2026-06-12T15:40:26.092Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# <a href="https://colab.research.google.com/github/jawadwali3454/AI-AND-ML/blob/main/Predicting_Student_Academic_Performance_Using_Supervised_Machine_Learning_Techniques.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>


from google.colab import files

uploaded = files.upload()

import zipfile

with zipfile.ZipFile('student.zip', 'r') as zip_ref:
    zip_ref.extractall('dataset')

import os

os.listdir('dataset')

import os

print(os.listdir())
print(os.listdir('dataset'))

import pandas as pd

data = pd.read_csv('dataset/student-mat.csv', sep=';')

data.head()

data.describe()

data.isnull().sum()

data.info()

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8,5))
sns.histplot(data['G3'], bins=20, kde=True)
plt.title("Distribution of G3 (Final Grades)")
plt.show()

plt.figure(figsize=(6,4))
sns.boxplot(x=data['studytime'], y=data['G3'])
plt.title("Study Time vs Final Grade")
plt.show()

plt.figure(figsize=(6,4))
sns.scatterplot(x=data['absences'], y=data['G3'])
plt.title("Absences vs Final Grade")
plt.show()

plt.figure(figsize=(12,8))
sns.heatmap(data.corr(numeric_only=True),
            annot=True,
            cmap='coolwarm')
plt.title("Feature Correlation Matrix")
plt.show()

plt.figure(figsize=(6,4))
sns.scatterplot(x=data['G1'], y=data['G3'])
plt.title("G1 vs G3")
plt.show()

plt.figure(figsize=(6,4))
sns.scatterplot(x=data['G2'], y=data['G3'])
plt.title("G2 vs G3")
plt.show()

X = data[['studytime', 'failures', 'absences', 'G1', 'G2']]
y = data['G3']

# # **Train-Test Split**


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# **Model Training (Linear Regression)**


from sklearn.linear_model import LinearRegression

lr_model = LinearRegression()

lr_model.fit(X_train, y_train)

# Prediction
y_pred_lr = lr_model.predict(X_test)

# **Model Evaluation**
# 
# 


from sklearn.metrics import mean_absolute_error, r2_score

print("Linear Regression Results")
print("MAE:", mean_absolute_error(y_test, y_pred_lr))
print("R2 Score:", r2_score(y_test, y_pred_lr))

# **Actual vs Predicted (Linear Regression)**


import matplotlib.pyplot as plt

plt.figure(figsize=(7,5))

plt.scatter(y_test, y_pred_lr, color='blue', alpha=0.6)

plt.xlabel("Actual Grades (G3)")
plt.ylabel("Predicted Grades (G3)")
plt.title("Linear Regression: Actual vs Predicted")

plt.show()

import numpy as np
import matplotlib.pyplot as plt

X_simple = data[['G2']]
y_simple = data['G3']

from sklearn.linear_model import LinearRegression

model_simple = LinearRegression()
model_simple.fit(X_simple, y_simple)

y_line = model_simple.predict(X_simple)

plt.figure(figsize=(7,5))
plt.scatter(X_simple, y_simple, color='blue', alpha=0.5)
plt.plot(X_simple, y_line, color='red')

plt.xlabel("G2 (Second Period Grade)")
plt.ylabel("G3 (Final Grade)")
plt.title("Best Fit Line: G2 vs G3")

plt.show()

results = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred_lr
})

results.head(10)

results["Error"] = results["Actual"] - results["Predicted"]
results.head(10)

# **Random Forest Regressor**


from sklearn.ensemble import RandomForestRegressor

# **Model Train**


rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

rf_model.fit(X_train, y_train)

# **Prediction**


y_pred_rf = rf_model.predict(X_test)

# **Evaluation**


from sklearn.metrics import mean_absolute_error, r2_score

print("Random Forest Results")
print("MAE:", mean_absolute_error(y_test, y_pred_rf))
print("R2 Score:", r2_score(y_test, y_pred_rf))

import matplotlib.pyplot as plt

plt.figure(figsize=(7,5))

plt.scatter(y_test, y_pred_rf, alpha=0.6)

plt.xlabel("Actual G3")
plt.ylabel("Predicted G3")
plt.title("Random Forest: Actual vs Predicted")

plt.show()

# **COMPARING**


import pandas as pd

results = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest"],
    "MAE": [
        mean_absolute_error(y_test, y_pred_lr),
        mean_absolute_error(y_test, y_pred_rf)
    ],
    "R2 Score": [
        r2_score(y_test, y_pred_lr),
        r2_score(y_test, y_pred_rf)
    ]
})

results

import matplotlib.pyplot as plt
import numpy as np

models = ["Linear Regression", "Random Forest"]

mae_scores = [
    mean_absolute_error(y_test, y_pred_lr),
    mean_absolute_error(y_test, y_pred_rf)
]

r2_scores = [
    r2_score(y_test, y_pred_lr),
    r2_score(y_test, y_pred_rf)
]

x = np.arange(len(models))

plt.figure(figsize=(8,5))

# MAE Bar
plt.bar(x - 0.2, mae_scores, width=0.4, label="MAE")

# R2 Bar
plt.bar(x + 0.2, r2_scores, width=0.4, label="R2 Score")

plt.xticks(x, models)
plt.title("Model Comparison: MAE vs R2 Score")
plt.legend()

plt.show()

import numpy as np

# Example student data
new_student = np.array([[2, 0, 4, 10, 12]])

prediction = rf_model.predict(new_student)

print("Predicted Final Grade (G3):", prediction[0])

# **Multiple Students Test**


new_data = np.array([
    [1, 2, 10, 8, 9],
    [3, 0, 2, 14, 15],
    [2, 1, 6, 11, 12]
])

predictions = rf_model.predict(new_data)

print("Predictions:", predictions)