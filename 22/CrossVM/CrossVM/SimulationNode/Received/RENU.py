

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Pre-Modeling Tasks

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Modeling

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB


# Evaluation and comparision of all the models

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score, confusion_matrix, precision_recall_fscore_support
from sklearn.metrics import roc_auc_score,auc,f1_score
from sklearn.metrics import precision_recall_curve

df = pd.read_csv("/content/data (1).csv")
df

df['diagnosis'].value_counts()

df.head()

df.columns

df.describe().T

df.info()

df.drop(['Unnamed: 32','id'], axis=1, inplace=True)

df.shape

df['diagnosis'] = df['diagnosis'].astype('category').cat.codes
df['diagnosis']

df.corr()['diagnosis'].sort_values()

import seaborn as sns

cols = ['diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
        'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean']

sns.pairplot(data=df[cols], hue='diagnosis', palette='Set1', height=5, aspect=2)

#Exploring dataset:
#sns.pairplot(df, kind="scatter", hue="diagnosis")
#plt.show()

X = df.drop('diagnosis', axis=1).values
y = df['diagnosis'].values

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.25, random_state=101)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train.shape

X_test.shape

from sklearn.preprocessing import LabelEncoder

models = []
Z = [SVC(), DecisionTreeClassifier(), KNeighborsClassifier(), GaussianNB()]

X = ["SVC", "DecisionTreeClassifier", "KNeighborsClassifier","NaiveBayes"]

# Convert target variable to numeric labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

for i in range(len(Z)):
    model = Z[i]
    model.fit(X_train, y_train_encoded)
    pred = model.predict(X_test)
    models.append(accuracy_score(pred, y_test_encoded))

# Print the accuracy scores
for model_name, accuracy in zip(X, models):
    print(model_name + " Accuracy: {:.3f}".format(accuracy))

from tabulate import tabulate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

models = []
models1 = []
models2 = []
models3 = []
models4 = []

for i in range(len(Z)):
    model = Z[i]
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    models.append(accuracy_score(pred, y_test))
    models1.append(precision_score(pred, y_test))
    models2.append(recall_score(pred, y_test))
    models3.append(f1_score(pred, y_test))
    models4.append(roc_auc_score(pred, y_test))

d = {
    "Algorithm": Z,
    "Accuracy": models,
    "Precision": models1,
    "Recall": models2,
    "F1-Score": models3,
    "AUC": models4
}

table = tabulate(d, headers="keys", tablefmt="grid")
print(table)

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# ...

# Print confusion matrices
for i in range(len(Z)):
    print(f"\nConfusion Matrix - {Z[i]}")
    print(conf_matrices[i])

# Plot confusion matrices for all results
plt.figure(figsize=(12, 10))
for i in range(len(Z)):
    plt.subplot(2, 2, i+1)
    sns.heatmap(conf_matrices[i], annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {Z[i]}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
plt.tight_layout()
plt.show()

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# Split the data into training and test sets
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the SVC model
svc_model = SVC()

# Define the hyperparameter grid for SVC
svc_param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'sigmoid'],
    'gamma': [0.1, 1, 10]
}

# Perform grid search with cross-validation for SVC
svc_grid_search = GridSearchCV(estimator=svc_model, param_grid=svc_param_grid, cv=5)
svc_grid_search.fit(X_train, y_train)

# Print the best hyperparameters for SVC
#print("Best Hyperparameters for SVC: ", svc_grid_search.best_params_)

# Evaluate the SVC model with the best hyperparameters
svc_best_model = svc_grid_search.best_estimator_
svc_accuracy = svc_best_model.score(X_test, y_test)
#print("Accuracy for SVC: ", svc_accuracy)

# Define the Naive Bayes model
nb_model = GaussianNB()

# Perform grid search with cross-validation for Naive Bayes (no hyperparameters to tune)

# Fit the Naive Bayes model
nb_model.fit(X_train, y_train)

# Evaluate the Naive Bayes model
nb_accuracy = nb_model.score(X_test, y_test)
#print("Accuracy for Naive Bayes: ", nb_accuracy)

# Define the Decision Tree Classifier model
dt_model = DecisionTreeClassifier()

# Define the hyperparameter grid for Decision Tree Classifier
dt_param_grid = {
    'max_depth': [None, 1, 2, 3],
    'min_samples_split': [2, 3, 4]
}

# Perform grid search with cross-validation for Decision Tree Classifier
dt_grid_search = GridSearchCV(estimator=dt_model, param_grid=dt_param_grid, cv=5)
dt_grid_search.fit(X_train, y_train)

# Print the best hyperparameters for Decision Tree Classifier
#print("Best Hyperparameters for Decision Tree Classifier: ", dt_grid_search.best_params_)

# Evaluate the Decision Tree Classifier model with the best hyperparameters
dt_best_model = dt_grid_search.best_estimator_
dt_accuracy = dt_best_model.score(X_test, y_test)
#print("Accuracy for Decision Tree Classifier: ", dt_accuracy)

# Define the K-Nearest Neighbors Classifier model
knn_model = KNeighborsClassifier()

# Define the hyperparameter grid for K-Nearest Neighbors Classifier
knn_param_grid = {
    'n_neighbors': [3, 5, 7],
    'weights': ['uniform', 'distance']
}

# Perform grid search with cross-validation for K-Nearest Neighbors Classifier
knn_grid_search = GridSearchCV(estimator=knn_model, param_grid=knn_param_grid, cv=5)
knn_grid_search.fit(X_train, y_train)

# Print the best hyperparameters for K-Nearest Neighbors
# Print the best hyperparameters for SVC
#print("Best Hyperparameters for KNN: ", knn_grid_search.best_params_)

# Evaluate the SVC model with the best hyperparameters
knn_best_model = knn_grid_search.best_estimator_
knn_accuracy = knn_best_model.score(X_test, y_test)
#print("Accuracy for SVC: ", knn_accuracy)

import pandas as pd
from tabulate import tabulate
# Create a dictionary to store the results
results = {
    'Model': ['SVC', 'Naive Bayes', 'Decision Tree', 'K-Nearest Neighbors'],
    'Best Parameters': [svc_grid_search.best_params_, {}, dt_grid_search.best_params_, knn_grid_search.best_params_],
    'Accuracy': [svc_accuracy, nb_accuracy, dt_accuracy, knn_accuracy]
}

# Create a DataFrame from the results dictionary
df_results = pd.DataFrame(results)


table = tabulate(df_results, headers="keys", tablefmt="grid")
print(table)

from sklearn.metrics import confusion_matrix
import seaborn as sns

# Define a function to plot confusion matrix using heatmap
def plot_confusion_matrix(cm, labels, ax):
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

# Create a 2x2 grid of subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Calculate confusion matrix for SVC
svc_cm = confusion_matrix(y_test, svc_best_model.predict(X_test))
plot_confusion_matrix(svc_cm, labels=["class_0", "class_1"], ax=axes[0, 0])
axes[0, 0].set_title("Confusion Matrix for SVC")

# Calculate confusion matrix for Naive Bayes
nb_cm = confusion_matrix(y_test, nb_model.predict(X_test))
plot_confusion_matrix(nb_cm, labels=["class_0", "class_1"], ax=axes[0, 1])
axes[0, 1].set_title("Confusion Matrix for Naive Bayes")

# Calculate confusion matrix for Decision Tree Classifier
dt_cm = confusion_matrix(y_test, dt_best_model.predict(X_test))
plot_confusion_matrix(dt_cm, labels=["class_0", "class_1"], ax=axes[1, 0])
axes[1, 0].set_title("Confusion Matrix for Decision Tree Classifier")

# Calculate confusion matrix for K-Nearest Neighbors Classifier
knn_cm = confusion_matrix(y_test, knn_best_model.predict(X_test))
plot_confusion_matrix(knn_cm, labels=["class_0", "class_1"], ax=axes[1, 1])
axes[1, 1].set_title("Confusion Matrix for K-Nearest Neighbors Classifier")

# Adjust the layout and spacing of subplots
plt.tight_layout()

# Show the plot
plt.show()

from sklearn.metrics import classification_report, roc_auc_score
import pandas as pd

# Define the SVC model with probability=True
svc_model = SVC(probability=True)

# Perform grid search with cross-validation for SVC
svc_grid_search = GridSearchCV(estimator=svc_model, param_grid=svc_param_grid, cv=5)
svc_grid_search.fit(X_train, y_train)

# Get the best SVC model with the best hyperparameters
svc_best_model = svc_grid_search.best_estimator_

# Calculate classification report for SVC
svc_report = classification_report(y_test, svc_best_model.predict(X_test), output_dict=True)
svc_auc = roc_auc_score(y_test, svc_best_model.predict_proba(X_test)[:, 1])

# Calculate classification report for Naive Bayes
nb_report = classification_report(y_test, nb_model.predict(X_test), output_dict=True)
nb_auc = roc_auc_score(y_test, nb_model.predict_proba(X_test)[:, 1])

# Calculate classification report for Decision Tree Classifier
dt_report = classification_report(y_test, dt_best_model.predict(X_test), output_dict=True)
dt_auc = roc_auc_score(y_test, dt_best_model.predict_proba(X_test)[:, 1])

# Calculate classification report for K-Nearest Neighbors Classifier
knn_report = classification_report(y_test, knn_best_model.predict(X_test), output_dict=True)
knn_auc = roc_auc_score(y_test, knn_best_model.predict_proba(X_test)[:, 1])

# Create a dictionary to store the results
results = {
    'Model': ['SVC', 'Naive Bayes', 'Decision Tree', 'K-Nearest Neighbors'],
    'Precision': [svc_report['weighted avg']['precision'], nb_report['weighted avg']['precision'],
                  dt_report['weighted avg']['precision'], knn_report['weighted avg']['precision']],
    'Recall': [svc_report['weighted avg']['recall'], nb_report['weighted avg']['recall'],
               dt_report['weighted avg']['recall'], knn_report['weighted avg']['recall']],
    'F1-score': [svc_report['weighted avg']['f1-score'], nb_report['weighted avg']['f1-score'],
                 dt_report['weighted avg']['f1-score'], knn_report['weighted avg']['f1-score']],
    'AUC': [svc_auc, nb_auc, dt_auc, knn_auc]
}

# Create a DataFrame from the results dictionary
df_results = pd.DataFrame(results)


table = tabulate(df_results, headers="keys", tablefmt="grid")
print(table)
#plt.

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ...

# Plotting the results
fig = plt.figure(figsize=(12, 8))
fig.suptitle('Model Performance')

# Create subplots for each metric
metrics = ['Precision', 'Recall', 'F1-Score', 'AUC']

for i, metric in enumerate(metrics):
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.set_title(metric)

    # Generate X, Y, Z data for the plot
    models = d['Algorithm']
    scores = d[metric]
    xpos = np.arange(len(models))
    ypos = np.ones_like(xpos)
    zpos = np.zeros_like(xpos)
    dx = 0.8
    dy = 0.8
    dz = scores

    # Plot the bars with different colors
    colors = plt.cm.viridis(np.linspace(0, 1, len(scores)))
    for x, y, z, color in zip(xpos, ypos, dz, colors):
        ax.bar3d(x, y, zpos[0], dx, dy, z, color=color)

    # Set the labels and limits
    ax.set_xticks(np.arange(len(models)))
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.set_ylim([0, 2])
    ax.set_zlim([0, 1])
    ax.set_xlabel('Model')
    ax.set_ylabel(metric)
    ax.set_zlabel('Score')

    # Add value labels to the bars
    for x, y, z, score in zip(xpos, ypos, dz, scores):
        ax.text(x + dx / 2, y + dy, z, f'{score:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

# Calculate the average score for each model
average_scores = np.mean([d[metric] for metric in metrics], axis=0)

# Find the best model based on average score
best_model_idx = np.argmax(average_scores)
best_model = models[best_model_idx]
best_average_score = average_scores[best_model_idx]

# Print the best model and its average score
print(f"Best Model: {best_model}")
print(f"Average Score: {best_average_score:.3f}")