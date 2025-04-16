# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import joblib
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import  classification_report, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import  roc_curve, roc_auc_score
from sklearn.model_selection import learning_curve




# %% [markdown]
# ## 1. DATA PREPROCESSING

# %%
data = pd.read_csv('heart_disease.csv') # loading the data
data.head() # first five rows

# %%
data.tail() # last 5 rows


# %%
data.shape # no. of rows and columns

# %%
# getting informationa About the data
data.info()

# %%
# checking for missing values
data.isnull().sum()

# %%
data.describe() # An approchable measure for the data

# %%
data['target'].value_counts()

# %% [markdown]
# 1 REPRESENTS DEFECTIVE HEART

# %% [markdown]
# 0 REPRESENTS HEALTHY HEART

# %% [markdown]
# ## 2. SPLITTING THE FEATURES AND THE TARGET

# %%
X = data.drop(columns= 'target', axis=1)
Y = data['target']

# %%
print(X)

# %%
print(Y)

# %% [markdown]
# ## 3.SPLITTING THE DATA INTO TRAINING AND TEST DATA

# %%
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2, stratify= Y, random_state=2)

# %%
print(X.shape,X_test.shape, X_train.shape)

# %% [markdown]
# ## 4.TRAINING THE MODEL

# %% [markdown]
# NAIVE BAYES THEOREM

# %%


# Initialize the Naive Bayes model
nb_model = GaussianNB()

# Fit the model on the training data
nb_model.fit(X_train, Y_train)

# Make predictions on the test data
Y_pred_nb = nb_model.predict(X_test)

# Evaluate the accuracy of the model
accuracy_nb = accuracy_score(Y_test, Y_pred_nb)
print(f"Naive Bayes Accuracy: {accuracy_nb * 100:.2f}%")

# Detailed classification report
report_nb = classification_report(Y_test, Y_pred_nb, output_dict=True)
print("Classification Report for Naive Bayes:")
print(classification_report(Y_test, Y_pred_nb))

# Confusion matrix
conf_matrix_nb = confusion_matrix(Y_test, Y_pred_nb)
print("Confusion Matrix for Naive Bayes:")
print(conf_matrix_nb)

# 1. Plotting the Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_nb, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Predicted Negative', 'Predicted Positive'],
            yticklabels=['Actual Negative', 'Actual Positive'])
plt.title('Confusion Matrix for Naive Bayes')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# 2. Plotting the ROC Curve
Y_prob_nb = nb_model.predict_proba(X_test)[:, 1]  # Get probabilities for the positive class
fpr_nb, tpr_nb, thresholds_nb = roc_curve(Y_test, Y_prob_nb)
roc_auc_nb = roc_auc_score(Y_test, Y_prob_nb)

plt.figure(figsize=(8, 6))
plt.plot(fpr_nb, tpr_nb, color='blue', label=f'ROC Curve (area = {roc_auc_nb:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for Naive Bayes')
plt.legend(loc='lower right')
plt.show()

# 3. Bar Chart for Classification Report Metrics
# Convert keys to a list for indexing
labels_nb = list(report_nb.keys())

# Prepare data for bar plot
precision_values_nb = [report_nb[label]['precision'] for label in labels_nb[:-3]]  # Exclude 'accuracy', 'macro avg', 'weighted avg'
recall_values_nb = [report_nb[label]['recall'] for label in labels_nb[:-3]]
f1_values_nb = [report_nb[label]['f1-score'] for label in labels_nb[:-3]]

# Prepare x locations for bars
x_nb = range(len(labels_nb) - 3)  # Exclude the last three labels

plt.figure(figsize=(10, 6))
plt.bar(x_nb, precision_values_nb, width=0.2, label='Precision', color='blue', align='center')
plt.bar([p + 0.2 for p in x_nb], recall_values_nb, width=0.2, label='Recall', color='orange', align='center')
plt.bar([p + 0.4 for p in x_nb], f1_values_nb, width=0.2, label='F1 Score', color='green', align='center')

plt.xlabel('Classes')
plt.ylabel('Scores')
plt.title('Classification Report Metrics for Naive Bayes')
plt.xticks([p + 0.2 for p in x_nb], labels_nb[:-3], rotation=0)  # Centered labels
plt.legend()
plt.tight_layout()
plt.show()

# 4. Plotting the Learning Curve for Naive Bayes
train_sizes, train_scores, test_scores = learning_curve(nb_model, X_train, Y_train, 
                                                          train_sizes=np.linspace(0.1, 1.0, 10),
                                                          cv=5, n_jobs=-1)

train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)

plt.figure(figsize=(12, 6))
plt.plot(train_sizes, train_scores_mean, label='Training Score', marker='o')
plt.plot(train_sizes, test_scores_mean, label='Cross-Validation Score', marker='o')

plt.title('Learning Curve for Naive Bayes')
plt.xlabel('Training Size')
plt.ylabel('Score')
plt.legend()
plt.grid()
plt.show()

joblib.dump(nb_model, 'nb_model.pkl')

# %% [markdown]
# RANDOM FOREST CLASSIFIER

# %%


# Initialize the Random Forest model
rf_model = RandomForestClassifier(random_state=42)

# Hyperparameter tuning using Grid Search
rf_params = {
    'n_estimators': [50, 100],
    'max_depth': [None, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'bootstrap': [True]
}

# GridSearchCV for hyperparameter tuning
rf_grid = GridSearchCV(rf_model, rf_params, cv=3, scoring='accuracy')
rf_grid.fit(X_train, Y_train)
best_rf_model = rf_grid.best_estimator_

# Make predictions on the test data
Y_pred_rf = best_rf_model.predict(X_test)

# Evaluate the accuracy of the model
accuracy_rf = accuracy_score(Y_test, Y_pred_rf)
print(f"Random Forest Accuracy: {accuracy_rf * 100:.2f}%")

# Detailed classification report
print("Classification Report for Random Forest:")
print(classification_report(Y_test, Y_pred_rf))

# Confusion matrix
print("Confusion Matrix for Random Forest:")
conf_matrix = confusion_matrix(Y_test, Y_pred_rf)
print(conf_matrix)

# Visualizing Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.title('Confusion Matrix for Random Forest')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Plotting Learning Curves
train_sizes = np.linspace(0.1, 0.9, 9)  # Values between 0.1 and 0.9
train_scores = []
test_scores = []

for train_size in train_sizes:
    # Ensure train_size is interpreted correctly
    X_train_subset, _, Y_train_subset, _ = train_test_split(X_train, Y_train, train_size=train_size, random_state=42)
    rf_model.fit(X_train_subset, Y_train_subset)
    train_scores.append(rf_model.score(X_train_subset, Y_train_subset))
    test_scores.append(rf_model.score(X_test, Y_test))

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores, label='Training Accuracy', marker='o')
plt.plot(train_sizes, test_scores, label='Testing Accuracy', marker='o')
plt.title('Learning Curves for Random Forest')
plt.xlabel('Training Size')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.show()

# Visualizing Class Balance in Training Set
plt.figure(figsize=(10, 5))
sns.countplot(x=Y_train, palette='viridis')
plt.title('Class Distribution in Training Set')
plt.xlabel('Classes')
plt.ylabel('Count')
plt.xticks(ticks=[0, 1], labels=['Negative', 'Positive'])
plt.show()

# Visualizing Class Balance in Test Set
plt.figure(figsize=(10, 5))
sns.countplot(x=Y_test, palette='viridis')
plt.title('Class Distribution in Test Set')
plt.xlabel('Classes')
plt.ylabel('Count')
plt.xticks(ticks=[0, 1], labels=['Negative', 'Positive'])
plt.show()

# Save the tuned Random Forest model
joblib.dump(best_rf_model, 'rf_model_tuned.pkl')


# %% [markdown]
# FEATURE SCALING (FOR LOGISTIC REGRESSION, SVM, GRADIENT BOOSTING)

# %%


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# %% [markdown]
# LOGISTIC REGRESSION

# %%

# Fit the logistic regression model
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, Y_train)

# Make predictions on the test data
Y_pred = logreg.predict(X_test)

# Evaluate the accuracy of the model
accuracy = accuracy_score(Y_test, Y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Detailed classification report
report = classification_report(Y_test, Y_pred, output_dict=True)
print(classification_report(Y_test, Y_pred))

# Confusion matrix
conf_matrix = confusion_matrix(Y_test, Y_pred)
print("Confusion Matrix:\n", conf_matrix)

# 1. Plotting the Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Predicted Negative', 'Predicted Positive'],
            yticklabels=['Actual Negative', 'Actual Positive'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# 2. Plotting the ROC Curve
Y_prob = logreg.predict_proba(X_test)[:, 1]  # Get probabilities for the positive class
fpr, tpr, thresholds = roc_curve(Y_test, Y_prob)
roc_auc = roc_auc_score(Y_test, Y_prob)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# 3. Bar Chart for Classification Report Metrics
# Convert keys to a list for indexing
labels = list(report.keys())

# Prepare data for bar plot
precision_values = [report[label]['precision'] for label in labels[:-3]]  # Exclude 'accuracy', 'macro avg', 'weighted avg'
recall_values = [report[label]['recall'] for label in labels[:-3]]
f1_values = [report[label]['f1-score'] for label in labels[:-3]]

# Prepare x locations for bars
x = range(len(labels) - 3)  # Exclude the last three labels

plt.figure(figsize=(10, 6))
plt.bar(x, precision_values, width=0.2, label='Precision', color='blue', align='center')
plt.bar([p + 0.2 for p in x], recall_values, width=0.2, label='Recall', color='orange', align='center')
plt.bar([p + 0.4 for p in x], f1_values, width=0.2, label='F1 Score', color='green', align='center')

plt.xlabel('Classes')
plt.ylabel('Scores')
plt.title('Classification Report Metrics')
plt.xticks([p + 0.2 for p in x], labels[:-3], rotation=0)  # Centered labels
plt.legend()
plt.tight_layout()
plt.show()

# 4. Learning Curve for Logistic Regression
train_sizes, train_scores, test_scores = learning_curve(
    logreg, X_train, Y_train, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10), scoring='accuracy'
)

# Calculate the mean and standard deviation for training and test scores
train_scores_mean = train_scores.mean(axis=1)
test_scores_mean = test_scores.mean(axis=1)
train_scores_std = train_scores.std(axis=1)
test_scores_std = test_scores.std(axis=1)

# Plotting the learning curve
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores_mean, 'o-', color='blue', label='Training Score')
plt.plot(train_sizes, test_scores_mean, 'o-', color='orange', label='Cross-Validation Score')

# Plot the std deviation as a shaded area
plt.fill_between(train_sizes, 
                 train_scores_mean - train_scores_std, 
                 train_scores_mean + train_scores_std, 
                 color='blue', alpha=0.1)
plt.fill_between(train_sizes, 
                 test_scores_mean - test_scores_std, 
                 test_scores_mean + test_scores_std, 
                 color='orange', alpha=0.1)

plt.title('Learning Curve for Logistic Regression')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy Score')
plt.legend(loc='best')
plt.grid()
plt.show()


joblib.dump(logreg,'logreg.pkl')

# %% [markdown]
# SUPPORT VECTOR MACHINES (SVM)

# %%


# Initialize the SVM model
svm_model = SVC(kernel='linear', probability=True)  # Set probability=True for ROC curve

# Fit the model on the training data
svm_model.fit(X_train, Y_train)

# Make predictions on the test data
Y_pred_svm = svm_model.predict(X_test)

# Evaluate the accuracy of the model
accuracy_svm = accuracy_score(Y_test, Y_pred_svm)
print(f"SVM Accuracy: {accuracy_svm * 100:.2f}%")

# Detailed classification report
report_svm = classification_report(Y_test, Y_pred_svm, output_dict=True)
print("Classification Report for SVM:")
print(classification_report(Y_test, Y_pred_svm))

# Confusion matrix
conf_matrix_svm = confusion_matrix(Y_test, Y_pred_svm)
print("Confusion Matrix for SVM:")
print(conf_matrix_svm)

# 1. Plotting the Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_svm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Predicted Negative', 'Predicted Positive'],
            yticklabels=['Actual Negative', 'Actual Positive'])
plt.title('Confusion Matrix for SVM')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# 2. Plotting the ROC Curve
Y_prob_svm = svm_model.predict_proba(X_test)[:, 1]  # Get probabilities for the positive class
fpr_svm, tpr_svm, thresholds_svm = roc_curve(Y_test, Y_prob_svm)
roc_auc_svm = roc_auc_score(Y_test, Y_prob_svm)

plt.figure(figsize=(8, 6))
plt.plot(fpr_svm, tpr_svm, color='blue', label=f'ROC Curve (area = {roc_auc_svm:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for SVM')
plt.legend(loc='lower right')
plt.show()

# 3. Bar Chart for Classification Report Metrics
# Convert keys to a list for indexing
labels_svm = list(report_svm.keys())

# Prepare data for bar plot
precision_values_svm = [report_svm[label]['precision'] for label in labels_svm[:-3]]  # Exclude 'accuracy', 'macro avg', 'weighted avg'
recall_values_svm = [report_svm[label]['recall'] for label in labels_svm[:-3]]
f1_values_svm = [report_svm[label]['f1-score'] for label in labels_svm[:-3]]

# Prepare x locations for bars
x_svm = range(len(labels_svm) - 3)  # Exclude the last three labels

plt.figure(figsize=(10, 6))
plt.bar(x_svm, precision_values_svm, width=0.2, label='Precision', color='blue', align='center')
plt.bar([p + 0.2 for p in x_svm], recall_values_svm, width=0.2, label='Recall', color='orange', align='center')
plt.bar([p + 0.4 for p in x_svm], f1_values_svm, width=0.2, label='F1 Score', color='green', align='center')

plt.xlabel('Classes')
plt.ylabel('Scores')
plt.title('Classification Report Metrics for SVM')
plt.xticks([p + 0.2 for p in x_svm], labels_svm[:-3], rotation=0)  # Centered labels
plt.legend()
plt.tight_layout()
plt.show()

# 4. Learning Curve for SVM
train_sizes, train_scores, test_scores = learning_curve(
    svm_model, X_train, Y_train, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10), scoring='accuracy'
)

# Calculate the mean and standard deviation for training and test scores
train_scores_mean = train_scores.mean(axis=1)
test_scores_mean = test_scores.mean(axis=1)
train_scores_std = train_scores.std(axis=1)
test_scores_std = test_scores.std(axis=1)

# Plotting the learning curve
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores_mean, 'o-', color='blue', label='Training Score')
plt.plot(train_sizes, test_scores_mean, 'o-', color='orange', label='Cross-Validation Score')

# Plot the std deviation as a shaded area
plt.fill_between(train_sizes, 
                 train_scores_mean - train_scores_std, 
                 train_scores_mean + train_scores_std, 
                 color='blue', alpha=0.1)
plt.fill_between(train_sizes, 
                 test_scores_mean - test_scores_std, 
                 test_scores_mean + test_scores_std, 
                 color='orange', alpha=0.1)

plt.title('Learning Curve for SVM')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy Score')
plt.legend(loc='best')
plt.grid()
plt.show()


joblib.dump(svm_model, 'svm_model.pkl')

# %% [markdown]
# GRADIENT BOOSTING

# %%
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score
from sklearn.model_selection import learning_curve
import numpy as np

# Initialize the Gradient Boosting model
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)

# Fit the model on the training data
gb_model.fit(X_train, Y_train)

# Make predictions on the test data
Y_pred_gb = gb_model.predict(X_test)

# Evaluate the accuracy of the model
accuracy_gb = accuracy_score(Y_test, Y_pred_gb)
print(f"Gradient Boosting Accuracy: {accuracy_gb * 100:.2f}%")

# Detailed classification report
report_gb = classification_report(Y_test, Y_pred_gb, output_dict=True)
print("Classification Report for Gradient Boosting:")
print(classification_report(Y_test, Y_pred_gb))

# Confusion matrix
conf_matrix_gb = confusion_matrix(Y_test, Y_pred_gb)
print("Confusion Matrix for Gradient Boosting:")
print(conf_matrix_gb)

# Plotting the Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_gb, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Predicted Negative', 'Predicted Positive'],
            yticklabels=['Actual Negative', 'Actual Positive'])
plt.title('Confusion Matrix for Gradient Boosting')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Plotting the ROC Curve
Y_prob_gb = gb_model.predict_proba(X_test)[:, 1]
fpr_gb, tpr_gb, thresholds_gb = roc_curve(Y_test, Y_prob_gb)
roc_auc_gb = roc_auc_score(Y_test, Y_prob_gb)

plt.figure(figsize=(8, 6))
plt.plot(fpr_gb, tpr_gb, color='blue', label=f'ROC Curve (area = {roc_auc_gb:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for Gradient Boosting')
plt.legend(loc='lower right')
plt.show()

# Learning Curve for Gradient Boosting
train_sizes, train_scores, test_scores = learning_curve(gb_model, X_train, Y_train, cv=5, scoring='accuracy', n_jobs=-1)
train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)

plt.figure(figsize=(8, 6))
plt.plot(train_sizes, train_scores_mean, label='Training Score', color='blue')
plt.plot(train_sizes, test_scores_mean, label='Cross-Validation Score', color='orange')
plt.title('Learning Curve for Gradient Boosting')
plt.xlabel('Training Examples')
plt.ylabel('Score')
plt.legend(loc='best')
plt.grid()
plt.show()


joblib.dump(gb_model, 'gb_model.pkl')

# %% [markdown]
# ## 4.MODEL EVALUATION

# %%
# Function to calculate and print metrics for each model
def print_final_results(model_name, Y_test, Y_pred):
    accuracy = accuracy_score(Y_test, Y_pred)
    precision = precision_score(Y_test, Y_pred)
    recall = recall_score(Y_test, Y_pred)
    f1 = f1_score(Y_test, Y_pred)

    # Print the results in a clean format
    print(f"{model_name}:")
    print("-" * 21)
    print(f"Accuracy:      {accuracy * 100:.2f}%")
    print(f"Precision:     {precision:.2f}")
    print(f"Recall:        {recall:.2f}")
    print(f"F1-Score:      {f1:.2f}\n")

# Assuming the models are already trained and predictions are made
# Logistic Regression
print_final_results("Logistic Regression", Y_test, Y_pred)

# Naive Bayes
print_final_results("Naive Bayes", Y_test, Y_pred_nb)

# SVM
print_final_results("SVM", Y_test, Y_pred_svm)

# Random Forest
print_final_results("Random Forest", Y_test, Y_pred_rf)

# Gradient Boosting
print_final_results("Gradient Boosting", Y_test, Y_pred_gb)


# %% [markdown]
# ## 5. BUILDING A PREDICTIVE SYSTEM

# %%
# Define input data as a tuple
input_data = (65,0,2,140,417,1,0,157,0,0.8,2,1,2)

# Convert the input data to a NumPy array
input_data_as_numpy_array = np.asarray(input_data)

# Reshaping the array to match the model's input shape (1, -1) means 1 row and as many columns as needed
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# Make a prediction using the Random Forest model
indication = best_rf_model.predict(input_data_reshaped)

# Print the prediction result
print(indication)
if (indication[0]==0):
    print("It's very unlikely that you have a Heart Disease")
else:
    print("You should consult a doctor, you may have a Heart Disease")

# %% [markdown]
# ## 6.SAVING THE TRAINED MODEL AND DEPLOYING

# %%
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import joblib

# Load models
best_rf_model = joblib.load('rf_model_tuned.pkl')  # Replace with your model filenames
gb_model = joblib.load('gb_model.pkl')
logreg = joblib.load('logreg.pkl')
nb_model = joblib.load('nb_model.pkl')
svm_model = joblib.load('svm_model.pkl')

# Create a title for the app
st.title("Heart Disease Prediction App")

# Create input fields for user inputs
age = st.number_input("Age", min_value=0, max_value=120, value=30)
sex = st.selectbox("Sex", options=["Female (0)", "Male (1)"])  # 0 for female, 1 for male
cp = st.selectbox("Chest Pain Type", options=["Typical Angina (0)", "Atypical Angina (1)", "Non-Anginal Pain (2)", "Asymptomatic (3)"])  # 0-3
trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0, value=120)
chol = st.number_input("Cholesterol (mg/dl)", min_value=0, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=["No (0)", "Yes (1)"])
restecg = st.selectbox("Resting Electrocardiographic Results", options=["Normal (0)", "Abnormal (1)", "Probable or definite left ventricular hypertrophy (2)"])
thalach = st.number_input("Maximum Heart Rate Achieved", min_value=0, value=150)
exang = st.selectbox("Exercise Induced Angina", options=["No (0)", "Yes (1)"])
oldpeak = st.number_input("Oldpeak", value=1.0)
slope = st.selectbox("Slope of the Peak Exercise ST Segment", options=["Upsloping (0)", "Flat (1)", "Downsloping (2)"])
ca = st.selectbox("Number of Major Vessels (0-3)", options=[0, 1, 2, 3])
thal = st.selectbox("Thalassemia", options=["Normal (0)", "Fixed Defect (1)", "Reversible Defect (2)", "Unknown (3)"])

# Select the model to use for prediction
model_choice = st.selectbox("Select Prediction Model", options=["Random Forest", "Gradient Boosting", "Logistic Regression", "Naive Bayes", "SVM"])

# Create a button to make the prediction
if st.button("Predict"):
    # Encode inputs
    sex = int(sex.split(' ')[-1])  # Extract numerical value from string
    cp = int(cp.split(' ')[-1])  # Extract numerical value from string
    fbs = int(fbs.split(' ')[-1])  # Extract numerical value from string
    restecg = int(restecg.split(' ')[-1])  # Extract numerical value from string
    exang = int(exang.split(' ')[-1])  # Extract numerical value from string
    slope = int(slope.split(' ')[-1])  # Extract numerical value from string
    thal = int(thal.split(' ')[-1])  # Extract numerical value from string

    # Prepare the input data as a numpy array
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

    # Scale the input data
    scaler = StandardScaler()
    input_data_scaled = scaler.fit_transform(input_data)  # Scale based on the same method used for training

    # Select the model based on user choice
    if model_choice == "Random Forest":
        model = rf_model
    elif model_choice == "Gradient Boosting":
        model = gb_model
    elif model_choice == "Logistic Regression":
        model = logreg
    elif model_choice == "Naive Bayes":
        model = nb_model
    else:  # SVM
        model = svm_model

    # Make a prediction
    prediction = model.predict(input_data_scaled)

    # Prepare output data for display
    result_df = pd.DataFrame({
        "Input Feature": [
            "Age", "Sex (0 = Female, 1 = Male)", "Chest Pain Type (0-3)", "Resting BP (mm Hg)", 
            "Cholesterol (mg/dl)", "Fasting Blood Sugar (0 = No, 1 = Yes)", 
            "Resting ECG Results (0-2)", "Max Heart Rate", "Exercise Induced Angina (0 = No, 1 = Yes)", 
            "Oldpeak", "Slope (0-2)", "No. of Major Vessels (0-3)", "Thalassemia (0-3)"
        ],
        "Your Input": [
            age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal
        ],
        "Normal Range": [
            "18-120", "0-1", "0-3", "<120", "<200", "0 or 1", "0-2", ">=70", "0 or 1", "<5", "0-2", "0-3", "0-3"
        ]
    })

    # Display the prediction result
    st.subheader("Prediction Result")
    if prediction[0] == 1:
        st.success("You should consult a doctor, you may have a Heart Disease.")
    else:
        st.success("The patient is unlikely to have Heart disease.")

    # Display user inputs and normal values
    st.subheader("Your Inputs vs Normal Values")
    st.write(result_df)

    # Optionally, you can provide a way to download results as CSV
    csv = result_df.to_csv(index=False)
    st.download_button("Download Results as CSV", csv, "results.csv")





