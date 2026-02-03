# first_ML_project.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

# Load data
print("Loading data...")
df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv')
print(f"Data shape: {df.shape}")
print(df.head())

# Data preparation
y = df['logS']
X = df.drop('logS', axis=1)

# Data splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)
print(f"\nTraining set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

# Linear Regression
print("\n" + "="*50)
print("Linear Regression Model")
print("="*50)

lr = LinearRegression()
lr.fit(X_train, y_train)

y_lr_train_pred = lr.predict(X_train)
y_lr_test_pred = lr.predict(X_test)

lr_train_mse = mean_squared_error(y_train, y_lr_train_pred)
lr_train_r2 = r2_score(y_train, y_lr_train_pred)
lr_test_mse = mean_squared_error(y_test, y_lr_test_pred)
lr_test_r2 = r2_score(y_test, y_lr_test_pred)

print(f'Training MSE: {lr_train_mse:.6f}')
print(f'Training R2: {lr_train_r2:.6f}')
print(f'Test MSE: {lr_test_mse:.6f}')
print(f'Test R2: {lr_test_r2:.6f}')

# Random Forest
print("\n" + "="*50)
print("Random Forest Model")
print("="*50)

rf = RandomForestRegressor(max_depth=2, random_state=100)
rf.fit(X_train, y_train)

y_rf_train_pred = rf.predict(X_train)
y_rf_test_pred = rf.predict(X_test)

rf_train_mse = mean_squared_error(y_train, y_rf_train_pred)
rf_train_r2 = r2_score(y_train, y_rf_train_pred)
rf_test_mse = mean_squared_error(y_test, y_rf_test_pred)
rf_test_r2 = r2_score(y_test, y_rf_test_pred)

print(f'Training MSE: {rf_train_mse:.6f}')
print(f'Training R2: {rf_train_r2:.6f}')
print(f'Test MSE: {rf_test_mse:.6f}')
print(f'Test R2: {rf_test_r2:.6f}')

# Model comparison
print("\n" + "="*50)
print("Model Comparison")
print("="*50)
print(f"{'Model':<20} {'Train MSE':<12} {'Train R2':<12} {'Test MSE':<12} {'Test R2':<12}")
print("-"*68)
print(f"{'Linear Regression':<20} {lr_train_mse:.6f}    {lr_train_r2:.6f}    {lr_test_mse:.6f}    {lr_test_r2:.6f}")
print(f"{'Random Forest':<20} {rf_train_mse:.6f}    {rf_train_r2:.6f}    {rf_test_mse:.6f}    {rf_test_r2:.6f}")

# Data visualization for Linear Regression
print("\n" + "="*50)
print("Generating visualization...")
print("="*50)

plt.figure(figsize=(10, 5))

# Linear Regression predictions vs actual
plt.subplot(1, 2, 1)
plt.scatter(x=y_train, y=y_lr_train_pred, c="#7CAE00", alpha=0.3, label='Training Data')
plt.scatter(x=y_test, y=y_lr_test_pred, c="#F8766D", alpha=0.3, label='Test Data')
z = np.polyfit(np.concatenate([y_train, y_test]), np.concatenate([y_lr_train_pred, y_lr_test_pred]), 1)
p = np.poly1d(z)
y_all = np.concatenate([y_train, y_test])
plt.plot(y_all, p(y_all), '#00BFC4', linewidth=2)
plt.xlabel('Experimental LogS')
plt.ylabel('Predicted LogS')
plt.title('Linear Regression: Predicted vs Experimental')
plt.legend()
plt.grid(True, alpha=0.3)

# Random Forest predictions vs actual
plt.subplot(1, 2, 2)
plt.scatter(x=y_train, y=y_rf_train_pred, c="#7CAE00", alpha=0.3, label='Training Data')
plt.scatter(x=y_test, y=y_rf_test_pred, c="#F8766D", alpha=0.3, label='Test Data')
z = np.polyfit(np.concatenate([y_train, y_test]), np.concatenate([y_rf_train_pred, y_rf_test_pred]), 1)
p = np.poly1d(z)
plt.plot(y_all, p(y_all), '#00BFC4', linewidth=2)
plt.xlabel('Experimental LogS')
plt.ylabel('Predicted LogS')
plt.title('Random Forest: Predicted vs Experimental')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('model_predictions.png', dpi=300, bbox_inches='tight')
plt.show()

# Feature importance for Random Forest
print("\n" + "="*50)
print("Random Forest Feature Importance")
print("="*50)

feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance)

plt.figure(figsize=(8, 5))
plt.barh(feature_importance['feature'], feature_importance['importance'], color='#619CFF')
plt.xlabel('Feature Importance')
plt.title('Random Forest Feature Importance')
plt.gca().invert_yaxis()  # Highest importance at top
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*50)
print("Analysis complete!")
print("="*50)
print("\nSummary:")
print("-" * 40)
print(f"1. Linear Regression performs better on test data (R2 = {lr_test_r2:.4f})")
print(f"2. Random Forest has lower performance (R2 = {rf_test_r2:.4f}) with max_depth=2")
print(f"3. Most important feature: {feature_importance.iloc[0]['feature']}")
print(f"4. Visualizations saved as 'model_predictions.png' and 'feature_importance.png'")