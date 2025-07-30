# OLA-Ensemble-Business-Case
# OLA Driver Churn Prediction: An Ensemble Learning Approach

## Project Overview

This repository contains a comprehensive data science case study focused on predicting driver churn for Ola, a leading ride-sharing platform. Driver retention is a critical business challenge for Ola, as high churn rates lead to increased acquisition costs and potential service disruptions. This project leverages historical driver data to build and evaluate machine learning models, providing actionable insights and recommendations to enhance driver loyalty and reduce attrition.

## Problem Statement

Recruiting and retaining drivers presents a significant challenge for Ola. Driver churn is high, and drivers can easily switch to competitors based on fluctuating rates. As the company expands, this high churn becomes increasingly problematic. Acquiring new drivers is costly, and frequent driver departures negatively impact organizational morale. This study, conducted from the perspective of a data scientist within Ola's Analytics Department, aims to predict whether a driver will leave the company based on their monthly attributes from 2019 and 2020. These attributes include:

* **Demographics:** City, Age, Gender.
* **Tenure Information:** Date of Joining, Last Working Date.
* **Historical Performance Data:** Quarterly Rating, Monthly Business Acquired, Grade, Income.

The primary objective is to utilize ensemble learning techniques to predict driver churn, evaluate model performance, and deliver actionable insights for improving driver retention strategies.

## Dataset

The analysis is based on the `ola_driver_scaler.csv` dataset, which contains monthly information for a segment of Ola drivers.

**Key Columns:**
* `MMM-YY`: Reporting Month and Year
* `Driver_ID`: Unique identifier for each driver
* `Age`: Age of the driver
* `Gender`: Driver's gender (Male: 0, Female: 1)
* `City`: City code of operation
* `Education_Level`: Driver's education level (0: 10+, 1: 12+, 2: Graduate)
* `Income`: Average monthly income
* `Dateofjoining`: Date when the driver joined Ola
* `LastWorkingDate`: Most recent or final day the driver worked (critical for churn definition)
* `Joining Designation`: Driver's designation at joining
* `Grade`: Driver's assigned grade at reporting time
* `Total Business Value`: Monthly total monetary business value (negative values indicate adjustments)
* `Quarterly Rating`: Quarterly driver rating (1-5, higher is better)

## Project Purpose

The purpose of this case study is twofold:
1.  **Business Impact:** To provide Ola with a predictive model that can identify drivers at high risk of churning, enabling proactive intervention and reducing costly driver attrition.
2.  **Technical Skill Showcase:** To demonstrate proficiency in key data science methodologies, including:
    * Exploratory Data Analysis (EDA)
    * Advanced Missing Value Imputation (KNN Imputation)
    * Feature Engineering for time-series and categorical data
    * Handling Class Imbalance
    * Feature Encoding (Target Encoding, Binary Mapping)
    * Ensemble Learning (Bagging - Random Forest, Boosting - XGBoost)
    * Hyperparameter Tuning (`RandomizedSearchCV`)
    * Comprehensive Model Evaluation (Classification Reports, Confusion Matrices, ROC AUC Curves, Feature Importance)

## Methodology

The project followed a structured data science pipeline:

1.  **Data Loading & Initial Exploration:** Loaded the dataset, examined its structure, data types, and summarized key statistics.
2.  **Data Preprocessing:**
    * Handled missing values using a combination of forward-fill (for time-series consistency) and K-Nearest Neighbors (KNN) imputation.
    * Transformed date columns into datetime objects.
    * Aggregated monthly data to create a driver-centric dataset, essential for churn prediction.
3.  **Feature Engineering:**
    * Created new features such as `Grade_change` and `Income_increase` to capture driver progression.
    * Derived `Tenure_Month` to quantify driver experience.
    * Defined the target variable (`Churn`) based on the presence of `LastWorkingDate`.
4.  **Data Preparation for Modeling:**
    * Applied binary mapping for 'Yes'/'No' features.
    * Used Target Encoding for the 'City' categorical feature.
    * Split the data into training and testing sets, ensuring stratification to maintain class proportions.
    * Addressed class imbalance using the `class_weight='balanced'` parameter in models.
    * *Note: Feature scaling was not applied, as tree-based models are generally robust to feature scales.*
5.  **Model Building:**
    * **Decision Tree Classifier:** Implemented as a baseline.
    * **Random Forest Classifier (Bagging):** A robust ensemble model.
    * **XGBoost Classifier (Boosting):** A high-performance gradient boosting model.
    * Hyperparameter tuning was performed for all models using `RandomizedSearchCV` to optimize performance and prevent overfitting.
6.  **Results Evaluation:**
    * Models were evaluated using comprehensive metrics: Classification Reports (Precision, Recall, F1-Score), Confusion Matrices, and ROC AUC curves.
    * Feature importance analysis was conducted to identify key drivers of churn for each model.

## Key Outcomes & Insights

The analysis revealed several critical factors influencing driver churn:

* **Driver Tenure is Paramount:** Drivers with shorter tenure (`Months_employed`, `Frequency_of_ride`) are significantly more prone to churning. This highlights the importance of early engagement and support for new recruits.
* **Performance Drives Retention:** Drivers who consistently deliver higher `Total Business Value` and achieve better `Quarterly Ratings` are more likely to remain active. Performance-based incentives could reinforce loyalty.
* **Demographics Less Influential:** Basic demographic data (`Age`, `Gender`, `Education_Level`) and changes in `Grade` or `Income` did not show as strong a correlation with churn as tenure and performance.
* **No Seasonal Churn:** Driver departures appear to be distributed evenly throughout the year, suggesting no specific seasonal patterns requiring targeted interventions.
* **Tuned XGBoost as Best Performer:** The **Tuned XGBoost Classifier** demonstrated the strongest predictive power with an **ROC AUC of 0.85**. It also achieved a strong F1-score of **0.84** for the majority class (non-churners) and a competitive **0.66** for the minority class (churners), indicating a good balance in identifying both loyal and at-risk drivers.

## Actionable Recommendations

Based on these findings, we recommend the following strategies for Ola:

1.  **Implement a "New Driver Success Program":** Focus resources on drivers within their first 3-6 months, offering mentorship, clear feedback, and support to reduce early churn.
2.  **Establish Performance-Based Incentives:** Reward drivers directly for high `Quarterly Ratings` and `Total Business Value` through bonuses or preferential ride assignments.
3.  **Proactive Performance Intervention:** Develop an automated alert system to identify drivers with declining `Quarterly Rating` or `Total Business Value`, followed by targeted support or personalized feedback.
4.  **Enhance Onboarding Clarity:** Improve initial training to clearly explain `Joining_Designation` and potential career progression paths within Ola to boost driver morale and long-term commitment.
5.  **Continuous Model Monitoring:** Regularly monitor the churn prediction model's performance with new data and retrain it periodically to adapt to changing driver behavior and market conditions.

## Repository Structure
