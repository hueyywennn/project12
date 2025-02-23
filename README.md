# Used Car Price Estimator
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

### Project Overview
to predict the market value of a car based on features such as make, model, year, mileage, and other relevant factors using machine learning techniques.

### Dataset 
(https://www.kaggle.com/datasets/asinow/car-price-dataset/data)

| Name        | Description                                                                                  | Example Values                             |
|----------------|----------------------------------------------------------------------------------------------|--------------------------------------------|
| Brand          | Specifies the brand of the car (e.g., Toyota, BMW, Ford).                                    | "Toyota", "BMW", "Mercedes"               |
| Model          | Specifies the model of the car (e.g., Corolla, Focus, X5).                                   | "Corolla", "Focus", "X5"                  |
| Year           | The production year of the car. Newer years typically indicate higher prices.               | 2005, 2018, 2023                          |
| Engine_Size    | Specifies the engine size in liters (L). Larger engines generally correlate with higher prices. | 1.6, 2.0, 3.5                             |
| Fuel_Type      | Indicates the type of fuel used by the car:                                                  | Petrol, Diesel, Hybrid, Electric          |
| Transmission   | The type of transmission in the car:                                                         | Manual, Automatic, Semi-Automatic         |
| Mileage        | The total distance the car has traveled, measured in kilometers. Lower mileage generally indicates a higher price. | 15,000, 75,000, 230,000                    |
| Doors          | The number of doors in the car. Commonly 2, 3, 4, or 5 doors.                                | 2, 3, 4, 5                                |
| Owner_Count    | The number of previous owners of the car. Fewer owners generally indicate a higher price.    | 1, 2, 3, 4                                |
| Price          | The estimated selling price of the car. It is calculated based on several factors such as production year, engine size, mileage, fuel type, and transmission. | 5,000, 15,000, 30,000                     |

### Features
1. **Data Cleaning**
  -	Checked missing values: none
2. **Exploratory Data Analysis (EDA)**
  -	Statistical Summaries: mean, median, variance, and standard deviation
  -	Correlation Analysis: heatmap correlations
  -	Distribution Plots: histograms, scatter plots
  -	Outlier Detection: boxplot
3. **Machine Learning Models**
  -	Data Normalization: Standard Scaler, One Hot Encoder
  -	Classification Algorithms: linear regression, decision trees, and random forest
  -	Predictive Modeling: Classification
  -	Test sizes: 20%
4. **Interactive Visualizations**
  -	Classification Visualization: Model accuracy across different algorithms
  -	User input prompt

## Tools used
1. **Programming Language** 
  - Python
2. **Libraries**
  - pandas, numpy, scikit-learn, matplotlib
3. **Visualization Tools**
  - seaborn
