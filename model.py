
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import pickle
df= pd.read_csv('daily_electricity_data.csv')
df.head()
df.tail()
df.info()
df.isnull().sum()
from sklearn.preprocessing import StandardScaler

# Assuming your data is loaded into a pandas DataFrame named 'df'

renewable_energy = ['Wind', 'Hydroelectric', 'Solar', 'Biomass']

# Create a StandardScaler object
scaler = StandardScaler()

# Copy the DataFrame to avoid modifying the original
renewable_data = df.copy()

# Standardize renewable energy features
renewable_data[renewable_energy] = scaler.fit_transform(df[renewable_energy])

# Filter Hydroelectric (optional, modify condition as needed)
filtered_hydroelectric = renewable_data[renewable_data['Hydroelectric'] <= 2]

# Print original and standardized data descriptions
print("Original Data Description:")
print(df.describe())

print("\nStandardized Data Description:")
print(renewable_data.describe())

# Create histograms for all renewable energy features
for col in renewable_energy:
  plt.figure()  # Create a separate figure for each feature
  sns.histplot(data=renewable_data, x=col)
  plt.title(f"Histogram of Standardized {col}")
  plt.xlabel(f"{col} (Standardized)")
  plt.ylabel("Frequency")
  plt.show()

# Optional: Histogram for filtered Hydroelectric (if applicable)
if len(filtered_hydroelectric) > 0:
  plt.figure()
  sns.histplot(data=filtered_hydroelectric, x='Hydroelectric')
  plt.title("Histogram of Standardized Hydroelectric (<= 2)")
  plt.xlabel("Hydroelectric (Standardized)")
  plt.ylabel("Frequency")
  plt.show()
else:
  print("No data for Hydroelectric after filtering (all values > 2)")
from sklearn.preprocessing import StandardScaler

# Assuming your data is loaded into a pandas DataFrame named 'df'

non_renewable_energy = ['Nuclear', 'Oil and Gas', 'Coal']

# Create a StandardScaler object
scaler = StandardScaler()

# Copy the DataFrame to avoid modifying the original
non_renewable_data = df.copy()

# Standardize non-renewable energy features
non_renewable_data[non_renewable_energy] = scaler.fit_transform(df[non_renewable_energy])

# Filter Nuclear (keep between 0 and 0.5)
filtered_nuclear = non_renewable_data[
    (non_renewable_data['Nuclear'] >= 0) & (non_renewable_data['Nuclear'] <= 0.5)]

# Filter Coal (keep between -2 and 2)
filtered_coal = non_renewable_data[
    (non_renewable_data['Coal'] >= -2) & (non_renewable_data['Coal'] <= 2)]

# Print original and standardized data descriptions
print("Original Data Description:")
print(df.describe())

print("\nStandardized Data Description:")
print(non_renewable_data.describe())

# Create histograms for non-renewable energy features with filtering applied
for col in non_renewable_energy:
  plt.figure()  # Create a separate figure for each feature
  if col == 'Nuclear':
    # Use filtered_nuclear for Nuclear histogram
    sns.histplot(data=filtered_nuclear, x=col)
    plt.title(f"Histogram of Standardized {col} (0 to 0.5)")
  elif col == 'Coal':
    # Use filtered_coal for Coal histogram
    sns.histplot(data=filtered_coal, x=col)
    plt.title(f"Histogram of Standardized {col} (-2 to 2)")
  else:
    # Use non_renewable_data for Oil and Gas histogram
    sns.histplot(data=non_renewable_data, x=col)
    plt.title(f"Histogram of Standardized {col}")
  plt.xlabel(f"{col} (Standardized)")
  plt.ylabel("Frequency")
  plt.show()
import plotly.graph_objects as go

# Assuming you have sample data in a DataFrame named 'df' (replace with your actual data)
# ... (your data definition)

# Define lists of renewable and non-renewable energy sources
renewable_energy = ["Solar", "Biomass", "Wind", "Hydroelectric"]
non_renewable_energy = ["Nuclear", "Oil and Gas", "Coal"]

# Create traces for renewable and non-renewable energy consumption (assuming 'Consumption' column represents production)
trace_renewable = go.Scatter(
    x=df["DateTime"],
    y=df[renewable_energy].sum(axis=1),  # Sum consumption (production) of all renewables
    name='Renewable Energy',
    mode='lines+markers',
    line=dict(color='green'),
    marker=dict(color='green')
)

trace_non_renewable = go.Scatter(
    x=df["DateTime"],
    y=df[non_renewable_energy].sum(axis=1),  # Sum consumption (production) of all non-renewables
    name='Non-Renewable Energy',
    mode='lines+markers',
    line=dict(color='red'),
    marker=dict(color='red')
)

# Create the layout with desired figure size
layout = go.Layout(
    title='Comparison of Renewable and Non-Renewable Energy Production',
    xaxis_title='DateTime',
    yaxis_title='Production (units)',  # Corrected to 'Production'
    width=1100,
    height=500
)

# Combine traces and layout in a figure
fig = go.Figure(data=[trace_renewable, trace_non_renewable], layout=layout)

# Display the graph
fig.show()
import plotly.graph_objects as go

# Assuming your data is loaded into a DataFrame named 'df'
# Assuming 'Datetime', 'Production', and 'Consumption' are columns in your DataFrame

# Select relevant data (correct typos)
df_plot = df[['DateTime', 'Production', 'Consumption']]

# Convert Datetime to a format suitable for plotly.graph_objects (optional)
# If Datetime is already in a compatible format (e.g., datetime64), you might not need conversion
df_plot['DateTime'] = pd.to_datetime(df_plot['DateTime'])  # Example conversion

# Create line chart traces for Production and Consumption
production_trace = go.Scatter(
    x=df_plot['DateTime'],
    y=df_plot['Production'],
    mode='lines',
    name='Production'
)
consumption_trace = go.Scatter(
    x=df_plot['DateTime'],
    y=df_plot['Consumption'],
    mode='lines',
    name='Consumption'
)

# Combine traces into a figure
fig = go.Figure(data=[production_trace, consumption_trace])

# Set titles and labels
fig.update_layout(
    title='Production vs. Consumption over Time',
    xaxis_title='DateTime',
    yaxis_title='Values'
)

# Display the comparison chart
fig.show()
import plotly.graph_objects as go
# Aggregate production and consumption by energy source
energy_sources = df.drop(columns=['DateTime', 'Consumption', 'Production']).columns.tolist()
production = df[energy_sources].sum()
consumption = df[energy_sources].sum()

# Create pie chart for production
production_fig = go.Figure(data=[go.Pie(labels=energy_sources, values=production, hole=.3)])
production_fig.update_layout(title_text="Energy Production")

# Create pie chart for consumption
consumption_fig = go.Figure(data=[go.Pie(labels=energy_sources, values=consumption, hole=.3)])
consumption_fig.update_layout(title_text="Energy Consumption")

# Show the plots
production_fig.show()
consumption_fig.show()
# Sum production and consumption of renewable and non-renewable sources
renewable_sources = ["Solar", "Biomass", "Wind", "Hydroelectric"]
non_renewable_sources = ["Nuclear", "Oil and Gas", "Coal"]

renewable_production = df[renewable_sources].sum().sum()
non_renewable_production = df[non_renewable_sources].sum().sum()

renewable_consumption = df[renewable_sources].sum().sum()
non_renewable_consumption = df[non_renewable_sources].sum().sum()

# Create pie chart for production
production_fig = go.Figure()
production_fig.add_trace(go.Pie(labels=["Renewable", "Non-Renewable"],
                                 values=[renewable_production, non_renewable_production],
                                 hole=.3,
                                 marker_colors=['green', 'red']))
production_fig.update_layout(title_text="Energy Production")

# Create pie chart for consumption
consumption_fig = go.Figure()
consumption_fig.add_trace(go.Pie(labels=["Renewable", "Non-Renewable"],
                                  values=[renewable_consumption, non_renewable_consumption],
                                  hole=.3,
                                  marker_colors=['green', 'red']))
consumption_fig.update_layout(title_text="Energy Consumption")

# Show the plots
production_fig.show()
consumption_fig.show()
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Convert 'DateTime' column to datetime type
df['DateTime'] = pd.to_datetime(df['DateTime'])

# Splitting the dataset into features and target variable
X = df.drop(['DateTime', 'Consumption', 'Production'], axis=1)
y = df['Consumption']  # We are predicting consumption

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create linear regression object
model = LinearRegression()

# Train the model using the training sets
model.fit(X_train, y_train)

# Make predictions using the testing set
y_pred = model.predict(X_test)

# Model evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R^2 Score:", r2)
from sklearn.ensemble import RandomForestRegressor

# Create Random Forest regressor object
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model using the training sets
rf_model.fit(X_train, y_train)

# Make predictions using the testing set
y_pred_rf = rf_model.predict(X_test)

# Model evaluation
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print("Random Forest Mean Squared Error:", mse_rf)
print("Random Forest R^2 Score:", r2_rf)
from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Create the grid search object
grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=42),
                           param_grid=param_grid,
                           cv=5,  # 5-fold cross-validation
                           scoring='r2',  # Use R^2 score for evaluation
                           n_jobs=-1)  # Use all available cores

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Get the best parameters
best_params = grid_search.best_params_
print("Best Parameters:", best_params)

# Use the best estimator found by GridSearchCV
best_rf_model = grid_search.best_estimator_

# Make predictions using the testing set
y_pred_best = best_rf_model.predict(X_test)

# Model evaluation
mse_best = mean_squared_error(y_test, y_pred_best)
r2_best = r2_score(y_test, y_pred_best)

print("Tuned Random Forest Mean Squared Error:", mse_best)
print("Tuned Random Forest R^2 Score:", r2_best)


# Scatter plot of actual vs. predicted values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_best, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], linestyle='--', color='red')
plt.title('Actual vs. Predicted Values')
plt.xlabel('Actual Consumption')
plt.ylabel('Predicted Consumption')
plt.show()

pickle.dump(best_rf_model,open('model.pkl','wb'))
