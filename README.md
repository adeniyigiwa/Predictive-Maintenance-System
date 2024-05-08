# Proactive IT Management with Predictive Maintenance System

## Description
This notebook leverages machine learning algorithms to implement predictive maintenance for proactive IT management. The objective is to predict equipment failure types based on historical data, thereby reducing downtime and minimizing maintenance costs.

## Setup
1. **Import Libraries:**
   - `numpy`, `pandas` for data manipulation.
   - `matplotlib.pyplot`, `seaborn` for data visualization.
   - `sklearn` for machine learning models and evaluation metrics.
   - `imbalanced-learn` for handling imbalanced datasets.

2. **Google Drive Mount:**  
   If using Google Colab, mount your dataset via Google Drive.

3. **Load Dataset:**
   Import the predictive maintenance dataset using `pd.read_csv` from the specified path.

## Exploratory Data Analysis
1. **General Information:**  
   - Use `.info()` and `.head()` to understand data types, missing values, and data samples.

2. **Failure Type Distribution:**  
   - Analyze the distribution of `Failure Type` and `Target` variables using value counts and visualizations.

3. **Feature Visualization:**  
   - Visualize feature relationships using `sns.pairplot`.
   - Plot visualizations of categorical features (`Type`, `Target`, etc.) to identify patterns.

## Data Preprocessing
1. **Encode Categorical Features:**  
   - **Ordinal Encoding** for the `Type` feature (L/M/H).  
   - **Label Encoding** for the `Failure Type` feature.

2. **Train-Test Split:**  
   - Use `train_test_split` to split data into training (80%) and testing (20%) sets.

3. **Imbalanced Dataset Handling:**  
   - Use **SMOTE** to balance the training data.

## Model Training and Evaluation
1. **Logistic Regression:**  
   - Train and evaluate a logistic regression model.
   - Report training and test accuracies, and generate a classification report.

2. **Decision Tree:**  
   - Train and evaluate a decision tree model.
   - Report training and test accuracies, and generate a classification report.

3. **Random Forest:**  
   - Train and evaluate a random forest classifier.
   - Report training and test accuracies, and generate a classification report.

4. **Ensemble Models:**  
   - **Voting Classifier:**  
     - Combine RandomForest, KNeighbors, and SVC in a voting classifier.
     - Report training and test accuracies, and generate a classification report.

   - **Stacking Classifier:**  
     - Stack RandomForest, KNeighbors, and SVC with a final logistic regression estimator.
     - Report training and test accuracies, and generate a classification report.

## Model Optimization
1. **GridSearchCV:**  
   - Use GridSearchCV with cross-validation to tune individual model hyperparameters.
   - Incorporate optimized models in ensemble voting classifiers.

## Evaluation Metrics
1. **Classification Reports:**  
   - Generate comprehensive classification reports for each model.

2. **Confusion Matrix:**  
   - Visualize the confusion matrix for stacking model predictions.

3. **Confidence Intervals:**  
   - Compute confidence intervals for the ensemble model's accuracy using bootstrapping.

## Conclusion
- This notebook provides a systematic approach to predictive maintenance by building, tuning, and evaluating machine learning classifiers.
- The final stacking ensemble demonstrates high classification accuracy across failure types, emphasizing the advantages of model ensembling in predictive maintenance.
