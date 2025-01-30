import pandas as pd  
import numpy as np 
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf




warnings.simplefilter(action='ignore', category=FutureWarning)

# Load the dataset  
df = pd.read_csv("climate_change_dataset.csv")  

# Display basic info  
"""print(df.info())  
print(df.head())  """

# Check for missing values  
"""print(df.isnull().sum())  """
# Fill numerical missing values with mean
df.fillna(df.select_dtypes(include=['number']).mean(), inplace=True)

# Fill categorical missing values with mode (most frequent value)
df["Country"].fillna(df["Country"].mode()[0], inplace=True)

# Drop rows with too many missing values (optional) 

# Select only numeric columns
numeric_cols = df.select_dtypes(include=[np.number])

# Compute IQR only for numeric columns
Q1 = numeric_cols.quantile(0.25)
Q3 = numeric_cols.quantile(0.75)
IQR = Q3 - Q1

# Apply IQR filter only on numeric data
df_filtered = df[~((numeric_cols < (Q1 - 1.5 * IQR)) | (numeric_cols > (Q3 + 1.5 * IQR))).any(axis=1)]

# Display the new dataframe
# Global Temperature Trends
print(df_filtered.shape)  # Check if rows were removed
plt.figure(figsize=(12, 6))
sns.lineplot(x=df["Year"], y=df["Avg Temperature (°C)"], ci=None)
plt.xlabel("Year")
plt.ylabel("Avg Temperature (°C)")
plt.title("Global Temperature Trends Over Time")
#plt.show()

#CO₂ Emissions by Country
top_countries = df.groupby("Country")["CO2 Emissions (Tons/Capita)"].mean().sort_values(ascending=False).head(10)
top_countries.plot(kind="bar", figsize=(12,6), color="red")
plt.xlabel("Country")
plt.ylabel("Average CO2 Emissions (Tons/Capita)")
plt.title("Top 10 Countries by Average CO2 Emissions")
#plt.show()

plt.figure(figsize=(10,6))
sns.heatmap(numeric_cols.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Between Climate Factors")
#plt.show()


# Select features and target variable
features = df[['Year', 'CO2 Emissions (Tons/Capita)', 'Rainfall (mm)', 'Population', 'Renewable Energy (%)', 'Forest Area (%)']]
target = df['Avg Temperature (°C)']  # Example: Predicting temperature

# Split data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)



# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict future values
y_pred = model.predict(X_test)


rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)


print("Linear Regression Performance:")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

poly = PolynomialFeatures(2)  # Define PolynomialFeatures explicitly
poly_model = make_pipeline(poly, LinearRegression())

# Fit with multiple features
X = df[['Year', 'CO2 Emissions (Tons/Capita)']]
y = df['Avg Temperature (°C)']
poly_model.fit(X, y)

mean_co2 = df['CO2 Emissions (Tons/Capita)'].mean()

# Prepare future data
future_data = pd.DataFrame([[year, mean_co2] for year in range(2025, 2051)], columns=['Year', 'CO2 Emissions (Tons/Capita)'])

# Transform the future data using the same PolynomialFeatures
future_data_poly = poly.transform(future_data)

# Predict future trends
future_predictions = poly_model.named_steps['linearregression'].predict(future_data_poly)

# Create a DataFrame with the future years and their predictions
future_years = range(2025, 2051)
predictions_df = pd.DataFrame({'Year': future_years, 'Predicted Avg Temperature (°C)': future_predictions})

# Plot the historical temperature data
plt.figure(figsize=(12, 6))
sns.lineplot(x=df['Year'], y=df['Avg Temperature (°C)'], label='Historical Avg Temperature', color='blue')

# Plot the future predictions
sns.lineplot(x=predictions_df['Year'], y=predictions_df['Predicted Avg Temperature (°C)'], label='Predicted Avg Temperature', color='red', linestyle='--')

# Add labels and title
plt.xlabel("Year")
plt.ylabel("Avg Temperature (°C)")
plt.title("Global Temperature Trends: Historical vs Predicted (2025-2050)")
plt.legend()

# Show the plot
plt.show()


temperature_data = df['Avg Temperature (°C)']
# Train ARIMA model on temperature data
adf_result = adfuller(temperature_data)
print(f"ADF Statistic: {adf_result[0]}")
print(f"p-value: {adf_result[1]}")

# If the p-value is greater than 0.05, the series is non-stationary and differencing is needed
# Let's try first difference to make it stationary
temperature_data_diff = temperature_data.diff().dropna()

# Plot ACF and PACF to determine parameters p, d, q
plot_acf(temperature_data_diff)
plot_pacf(temperature_data_diff)
plt.show()

# Based on ACF and PACF, choose p, d, q values
# For example, let's use p=1, d=1, q=1 as a starting point
model = ARIMA(temperature_data, order=(1, 1, 1))

# Fit the model
arima_model = model.fit()

# Forecast the future temperature values (for the next 25 years)
forecast_steps = 25  # Predicting for 25 years (2025-2050)
forecast = arima_model.forecast(steps=forecast_steps)

# Create a DataFrame to hold the predictions
forecast_years = pd.date_range(start='2025', periods=forecast_steps, freq='Y')
forecast_df = pd.DataFrame({'Year': forecast_years, 'Predicted Avg Temperature (°C)': forecast})

# Plot the historical and forecasted values
plt.figure(figsize=(12, 6))
##plt.plot(df.index, temperature_data, label='Historical Avg Temperature', color='blue')
#plt.plot(forecast_df['Year'], forecast_df['Predicted Avg Temperature (°C)'], label='Predicted Avg Temperature', color='red', linestyle='--')
plt.xlabel("Year")
plt.ylabel("Avg Temperature (°C)")
plt.title("Global Temperature Forecasting with ARIMA")
plt.legend()
plt.show()

# Print the forecasted values
#print(forecast_df)