# Disease Prediction using Machine Learning

## Project Overview
This project is a machine learning model that predicts various diseases based on patient symptoms. The dataset used for training includes various symptoms as features, with the goal of accurately predicting the disease using machine learning algorithms.

## Dataset
The dataset consists of two main files:
- Training.csv: Used for training the machine learning model.
- Testing.csv: Used for testing the model's accuracy.

### Features:
- Symptoms: A total of 132 symptoms are included as features in the dataset.
- Target: The target variable is the prognosis, which indicates the disease diagnosed based on the symptoms.

## Requirements
- Python 3.x
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- mlxtend

## Installation
1. Clone the repository:
   
    git clone https://github.com/yourusername/disease-prediction-ml.git
    
2. Navigate to the project directory:
   
    cd disease-prediction-ml
    
3. Install the required packages:
   
    pip install -r requirements.txt
    
## Model Training
The model is trained using the K-Nearest Neighbors (KNN) algorithm. A GridSearchCV is used to find the best parameters for the model. The selected best model is then evaluated on the testing dataset.

### Model Evaluation
- Confusion Matrix: Visualized using Seaborn and mlxtend to show the accuracy of predictions.
- Accuracy Score: Achieved 100% accuracy on the testing dataset.
- Classification Report: Shows precision, recall, and F1-score for each disease.

### Code Snippet
`python
# Model Training
clf_knn = KNeighborsClassifier()
parametrs_knn = {'n_neighbors': [1, 3, 5, 7, 9, 11], 'metric': ['euclidean', 'manhattan', 'chebyshev']}
grid_clf_knn = GridSearchCV(clf_knn, parametrs_knn, cv=6, n_jobs=-1)
grid_clf_knn.fit(X_train, y_train)

# Model Evaluation
best_model_knn = grid_clf_knn.best_estimator_
y_pred_knn = best_model_knn.predict(X_test)
cm_knn = confusion_matrix(y_test, y_pred_knn)
print("Accuracy score for model: ", accuracy_score(y_test, y_pred_knn))
print("Classification report for model: \n", classification_report(y_test, y_pred_knn))