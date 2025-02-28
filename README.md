# Used Car Price Estimator
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

### Project Overview
This project aims to predict the market value of used cars based on various attributes such as brand, model, year, mileage, fuel type, transmission, and engine size. Machine learning techniques are employed to build predictive models that estimate car prices accurately.

### Dataset 
[Dataset](https://www.kaggle.com/datasets/asinow/car-price-dataset/data)

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

## Project Objectives
1. **Data Cleaning & Preprocessing**: Handling missing values, data normalization, and feature encoding.
2. **Exploratory Data Analysis (EDA)**: Understanding the relationship between various features and car prices.
3. **Feature Engineering**: Selecting and transforming features for better model performance.
4. **Model Training & Evaluation**: Implementing and comparing multiple machine learning models to predict car prices accurately.
5. **Hyperparameter Tuning**: Optimizing model parameters to enhance accuracy and performance.
6. **User Input Prompt**: Allowing users to input car details and get a predicted price.

## Machine Learning Models Used
- **Linear Regression**: The baseline model to predict prices based on a linear feature relationship.
- **Decision Trees**: Tree-based model to capture complex relationships in data.
- **Random Forest Regressor**: Ensemble learning method to improve prediction accuracy.
- **Standard Scaler**: Used for feature normalization.
- **One Hot Encoder**: Encoding categorical variables for machine learning models.

## Technologies Used
- **Programming Language**: Python
- **Libraries**: Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn
- **Data Processing**: Pandas, NumPy for data manipulation
- **Visualization Tools**: Matplotlib, Seaborn for data exploration

## Project Workflow
1. **Data Ingestion**: Load and preprocess the dataset.
2. **Exploratory Data Analysis (EDA)**: Analyze and visualize relationships between variables.
3. **Feature Engineering**: Encode categorical variables and normalize numerical data.
4. **Model Training**: Train Linear Regression, Decision Trees, and Random Forest models.
5. **Hyperparameter Tuning**: Optimize models to improve accuracy.
6. **Model Evaluation**: Compare model performance using evaluation metrics.
7. **User Input Prompt**: Allow users to input car details and receive a predicted price.
8. **Results Interpretation**: Derive insights and recommendations from predictions.
