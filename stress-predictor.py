# Step 1: Install and import required libraries
try:
    import gdown
except ImportError:
    import os
    os.system('pip install gdown')
    import gdown

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Step 2: Download the dataset from Google Drive
file_id = '1ZJ_Q2HCLtUSA6MDjZmLZtSrUx-TFxrnC'
url = f'https://drive.google.com/uc?id={file_id}'
output = 'mental_health.csv'
gdown.download(url, output, quiet=False)

# Step 3: Load the dataset
df = pd.read_csv(output)
print("\nDataset loaded successfully!")
print("Available columns:\n", df.columns.tolist())

# Step 4: Automatically detect the stress column
target = next((col for col in df.columns if 'stress' in col.lower()), None)
if not target:
    raise KeyError("No stress-related column found in the dataset.")

print(f"\nUsing target column: {target}")

# Step 5: Clean the data
# Fill missing numerical values
num_cols = ['Age', 'Sleep Duration', 'Physical Activity', 'Screen Time']
for col in num_cols:
    if col in df.columns:
        df[col].fillna(df[col].median(), inplace=True)

# Encode categorical features
cat_cols = ['Gender', 'Occupation']
le = LabelEncoder()
for col in cat_cols:
    if col in df.columns:
        df[col] = le.fit_transform(df[col].astype(str))

# Drop rows with missing target
df = df.dropna(subset=[target])

# Step 6: Encode the target labels
# Example mapping: Low=0, Medium=1, High=2
target_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
df[target] = df[target].map(target_mapping)

# Step 7: Define features and target
features = [col for col in ['Age', 'Gender', 'Occupation', 'Sleep Duration', 'Physical Activity', 'Screen Time'] if col in df.columns]
X = df[features]
y = df[target]

# Step 8: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 9: Train models
# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
pred_lr = lr.predict(X_test)

# Lasso Regression
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
pred_lasso = lasso.predict(X_test)

# Step 10: Evaluate models
print("\nModel Performance:")
print("Linear Regression R² Score:", r2_score(y_test, pred_lr))
print("Linear Regression RMSE:", np.sqrt(mean_squared_error(y_test, pred_lr)))
print("Lasso Regression R² Score:", r2_score(y_test, pred_lasso))
print("Lasso Regression RMSE:", np.sqrt(mean_squared_error(y_test, pred_lasso)))
