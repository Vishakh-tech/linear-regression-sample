import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
import numpy as np

# Load dataset
data = pd.read_csv('data/Startups.csv')

X = data[['R&D Spend', 'Administration', 'Marketing Spend', 'State']]
y = data['Profit']

# Preprocessing and modeling pipeline
preprocessor = ColumnTransformer(
    transformers=[('state', OneHotEncoder(drop='first'), ['State'])],
    remainder='passthrough'
)
pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('regressor', LinearRegression())
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

print("R² score:", r2_score(y_test, y_pred))
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(-1, 1), y_test.values.reshape(-1, 1)), axis=1))
