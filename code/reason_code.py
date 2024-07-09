import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm

# Load dataset
p = 'car_train-data.csv'
df = pd.read_csv(p)

# Standardize column names to lowercase
df.columns = list(map(str.lower, df.columns))

# Extract 'company' and 'model' from the data
df['company'] = df['name'].apply(lambda x: x.split(' ')[0])
df['model'] = df['name'].apply(lambda x: ' '.join(x.split(' ')[1:]))

# Remove units from numeric columns and convert to numeric types
df['mileage'] = df['mileage'].str.replace('km/kg', '').str.replace('kmpl', '')
df['engine'] = df['engine'].str.replace('cc', '')
df['power'] = df['power'].str.replace('bhp', '')

df['mileage'] = pd.to_numeric(df['mileage'], errors='coerce')
df['engine'] = pd.to_numeric(df['engine'], errors='coerce')
df['power'] = pd.to_numeric(df['power'], errors='coerce')
df['power'].fillna(df['power'].mean(), inplace=True)

# Create dummy variables for categorical features
df['fuel_diesel'] = (df['fuel_type'] == 'Diesel').astype(int)
df['fuel_petrol'] = (df['fuel_type'] == 'Petrol').astype(int)
df['transmission_automatic'] = (df['transmission'] == 'Automatic').astype(int)
df['owner_first'] = (df['owner_type'] == 'First').astype(int)
df['owner_second'] = (df['owner_type'] == 'Second').astype(int)

# Define feature columns
features = ['year', 'mileage', 'engine', 'power', 'fuel_petrol', 
            'transmission_automatic', 'seats']

# Filter out rows with invalid price values and select relevant columns
df = df[df.price>0]
df['id'] = range(1, len(df) + 1)

# Define key columns
IDs = ['id', 'company', 'model', 'price']
df = df[IDs + features]
df[features] = df[features].fillna(df[features].mean())

X = df.copy()
y=df[['price']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 31)

#construct key data frame
key_df = X_train[IDs]

# Apply range standardization (Min-Max Scaling)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train[features])
X_test_scaled = scaler.transform(X_test[features])

# Build and train the linear regression model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Print coefficients of the model
print("Coefficients:")
for feature, coef in zip(features, model.coef_[0]):
    print(f"{feature}: {coef}")

# Evaluate the model using R-squared
r_squared = model.score(X_test_scaled, y_test)
print(f"R-squared: {r_squared}")

# Align indices for y_train and X_train_scaled
y_train = y_train.reset_index(drop=True)
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=features)
X_train_scaled_df = sm.add_constant(X_train_scaled_df)

# Build the OLS model using statsmodels
ols_model = sm.OLS(y_train, X_train_scaled_df).fit()

# Print p-values of the features
print("\nP-values:")
print(ols_model.pvalues)

# Print the summary of the OLS model
print("\nModel Summary:")
print(ols_model.summary())

original_pred_price = model.predict(X_train_scaled)
# Initialize a DataFrame to store the elasticities
elasticities = pd.DataFrame(columns=['Variable', 'Elasticity'])

# Calculate elasticities for each feature
for feature in features:
    X_perturbed = X_train_scaled.copy()
    perturbation = X_perturbed[:, features.index(feature)] * 0.01  # 1% perturbation
    X_perturbed[:, features.index(feature)] += perturbation
    
    # Calculate the new predicted prices
    new_pred_price = model.predict(X_perturbed)
    
    # Calculate the percentage change in predictions
    change_in_pred = ((new_pred_price - original_pred_price) / original_pred_price).mean()
    
    # Calculate the elasticity
    elasticity = change_in_pred / 0.01  # Because we used a 1% perturbation
    elasticities = pd.concat([elasticities, pd.DataFrame({'Variable': [feature], 'Elasticity': [elasticity]})], ignore_index=True)

# Create DataFrame for individual impacts
impact_df = pd.DataFrame(index=df.index)
for feature in features:
    avg_x = df[feature].mean()
    elasticity_value = elasticities[elasticities['Variable'] == feature]['Elasticity'].values[0]
    impact_df[feature] = elasticity_value * (df[feature] - avg_x) / avg_x

# Identify the top two variables for each car
def find_top_impacts(row):
    impacts = row[features]
    top_impacts = impacts.abs().nlargest(2)
    result = pd.concat([top_impacts.index.to_series(index=['Top1_Var', 'Top2_Var']),
                        top_impacts.reset_index(drop=True).rename({0: 'Top1_Value', 1: 'Top2_Value'})])
    return result

reason_df = impact_df.apply(find_top_impacts, axis=1)

# Attach key columns
reason_df = pd.concat([key_df, reason_df], axis = 1)

# Display the reason codes for the first 10 cars
print(reason_df.head(8))

#########SHAP################
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import shap

p = 'car_train-data.csv'
df = pd.read_csv(p)
df.columns = map(str.lower, df.columns)
df['company'] = df['name'].apply(lambda x: x.split(' ')[0])
df['model'] = df['name'].apply(lambda x: ' '.join(x.split(' ')[1:]))
df['mileage'] = df['mileage'].str.replace('km/kg', '').str.replace('kmpl', '')
df['engine'] = df['engine'].str.replace('cc', '')
df['power'] = df['power'].str.replace('bhp', '')
df[['mileage', 'engine', 'power']] = df[['mileage', 'engine', 'power']].apply(pd.to_numeric, errors='coerce')
df['power'].fillna(df['power'].mean(), inplace=True)
Create Dummy Variables and Define Features:
Dummy variables are created for categorical features.
The target and features are defined, and rows with invalid prices are filtered out.

df['fuel_diesel'] = (df['fuel_type'] == 'Diesel').astype(int)
df['fuel_petrol'] = (df['fuel_type'] == 'Petrol').astype(int)
df['transmission_automatic'] = (df['transmission'] == 'Automatic').astype(int)
df['owner_first'] = (df['owner_type'] == 'First').astype(int)
df['owner_second'] = (df['owner_type'] == 'Second').astype(int)
features = ['year', 'mileage', 'engine', 'power', 'fuel_petrol', 'transmission_automatic', 'seats']
df = df[df['price'] > 0]
df = df[features + ['price', 'company', 'model']]
df[features] = df[features].fillna(df[features].mean())