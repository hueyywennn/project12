#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv("car_price.csv")
df


# In[3]:


df.isnull().sum()


# In[4]:


df.describe()


# In[5]:


sns.set_style("dark")
sns.set_context("talk")
sns.set_palette("pastel")
sns.despine(left=True, bottom=True)
sns.set(font="monospace")

plt.figure(figsize=(10, 5))
sns.histplot(df["Price"], bins=30, kde=True)
plt.title("Distribution of Car Prices")
plt.show()


# In[6]:


sns.set_style("dark")
sns.set_context("talk")
sns.set_palette("pastel")
sns.despine(left=True, bottom=True)
sns.set(font="monospace")

plt.figure(figsize=(10, 5))
sns.boxplot(x=df["Fuel_Type"], y=df["Price"])
plt.title("Car Price vs Fuel Type")
plt.show()


# In[7]:


sns.set_style("dark")
sns.set_context("talk")
sns.set_palette("pastel")
sns.despine(left=True, bottom=True)
sns.set(font="monospace")

plt.figure(figsize=(10, 5))
sns.scatterplot(x=df["Mileage"], y=df["Price"], hue=df["Transmission"])
plt.title("Price vs Mileage with Transmission Type")
plt.show()


# In[8]:


sns.set_style("dark")
sns.set_context("talk")
sns.set_palette("pastel")
sns.despine(left=True, bottom=True)
sns.set(font="monospace")

plt.figure(figsize=(10, 5))
sns.boxplot(x=df["Transmission"], y=df["Price"])
plt.title("Car Price vs Transmission Type")
plt.show()


# In[9]:


sns.set_style("dark")
sns.set_context("talk")
sns.set_palette("pastel")
sns.despine(left=True, bottom=True)
sns.set(font="monospace")

plt.figure(figsize=(10, 5))
sns.scatterplot(x=df["Year"], y=df["Price"], hue=df["Fuel_Type"])
plt.title("Price vs Year with Fuel Type")
plt.show()


# In[10]:


sns.set_style("dark")
sns.set_context("talk")
sns.set_palette("pastel")
sns.despine(left=True, bottom=True)
sns.set(font="monospace")

plt.figure(figsize=(10, 5))
numerical_df = df.select_dtypes(include=[np.number])  # Select only numerical columns
sns.heatmap(numerical_df.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()


# In[11]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# In[12]:


categorical_features = ["Brand", "Model", "Fuel_Type", "Transmission"]
numerical_features = ["Year", "Engine_Size", "Mileage", "Doors", "Owner_Count"]

target = "Price"


# In[13]:


categorical_transformer = OneHotEncoder(handle_unknown='ignore')
numerical_transformer = StandardScaler()

preprocessor = ColumnTransformer([
    ('num', numerical_transformer, numerical_features),
    ('cat', categorical_transformer, categorical_features)
])


# In[14]:


#Multiple models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
}


# In[15]:


X = df.drop(columns=[target])
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[16]:


results = {}
model_pipelines = {}
for name, model in models.items():
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    accuracy = r2 * 100  # Using R² as a proxy for accuracy
    
    results[name] = {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2, "MAPE": mape, "Accuracy": accuracy}
    model_pipelines[name] = pipeline


# In[17]:


results_df = pd.DataFrame(results).T
print(results_df)


# In[18]:


best_metrics = results_df.idxmin()  # For MAE, MSE, RMSE, MAPE (lower is better)
best_metrics["R2"] = results_df["R2"].idxmax()  # For R², higher is better
best_metrics["Accuracy"] = results_df["Accuracy"].idxmax()  # For Accuracy, higher is better
print("Best models per metric:")
print(best_metrics)


# In[19]:


best_model_name = best_metrics["R2"]
best_model_pipeline = model_pipelines[best_model_name]


# In[20]:


def predict_price():
    input_data = {}
    input_data["Brand"] = input("Enter Brand: ")
    input_data["Model"] = input("Enter Model: ")
    input_data["Year"] = int(input("Enter Year: "))
    input_data["Engine_Size"] = float(input("Enter Engine Size: "))
    input_data["Fuel_Type"] = input("Enter Fuel Type (Petrol/Diesel/Hybrid/Electric): ")
    input_data["Transmission"] = input("Enter Transmission (Manual/Automatic/Semi-Automatic): ")
    input_data["Mileage"] = int(input("Enter Mileage: "))
    input_data["Doors"] = int(input("Enter Number of Doors: "))
    input_data["Owner_Count"] = int(input("Enter Number of Previous Owners: "))
    
    input_df = pd.DataFrame([input_data])
    predicted_price = best_model_pipeline.predict(input_df)[0]
    print(f"Predicted Price: {predicted_price}")

predict_price()


# In[21]:


metrics = ["Accuracy", "MAE", "MSE", "R2"]
for metric in metrics:
    plt.figure(figsize=(8, 5))
    bars = results_df[metric].plot(kind='bar', color='skyblue')
    best_model = best_metrics[metric]
    best_index = list(results_df.index).index(best_model)
    bars.patches[best_index].set_color('red')  # Highlight best model in red
    
    # Annotate best model
    plt.text(best_index, results_df[metric].iloc[best_index] + (0.05 * results_df[metric].iloc[best_index]),
             f"Best: {best_model}", ha='center', fontsize=10, fontweight='bold', color='red')
    
    plt.title(f"Model Comparison - {metric}")
    plt.ylabel(metric)
    plt.xticks(rotation=45)
    plt.show()

