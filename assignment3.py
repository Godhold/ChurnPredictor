# -*- coding: utf-8 -*-
"""Assignment3.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ShWZU4tRchn62z0nBCLwCnLWbKprH-Zp
"""

import pandas as pd
import numpy as np
from google.colab import drive
drive.mount('/content/drive')
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

pip show keras

pip install tensorflow scikit-learn

data=pd.read_csv('/content/drive/MyDrive/Colab Notebooks/CustomerChurn_dataset.csv')
data.info()



data=data.drop('customerID',axis=1)

features=data.drop('Churn',axis=1)

target=data['Churn']

features.isnull().sum()

x_categorical1=features.select_dtypes(include='object')
target

categorical_columns = x_categorical1.select_dtypes(include='object').columns

column_transformer = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), categorical_columns)],
    remainder='passthrough'
)

x_encoded = column_transformer.fit_transform(x_categorical1)


mutual_info = mutual_info_classif(x_encoded, target)

mutual_info_df = pd.DataFrame(mutual_info, index=column_transformer.get_feature_names_out(x_categorical1.columns), columns=['Mutual Information'])

pd.set_option('display.max_rows', None)

print(mutual_info_df.sort_values(by='Mutual Information', ascending=False))

from sklearn.feature_selection import chi2
X_cat = column_transformer.fit_transform(x_categorical1)


chi2_stat, p_values = chi2(X_cat, target)

# Get the column names after one-hot encoding
encoded_columns = column_transformer.get_feature_names_out(input_features=categorical_columns)

# Create a DataFrame to display p-values
chi2_df = pd.DataFrame({'Chi2 Statistic': chi2_stat, 'P-Value': p_values}, index=encoded_columns)

# Display the DataFrame
print(chi2_df.sort_values(by='P-Value'))

selected_features=['Contract','Partner','Dependents','PaperlessBilling','PaymentMethod','TotalCharges','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','InternetService','Churn']

dataset = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/CustomerChurn_dataset.csv', usecols=selected_features)
dataset.info()

dataset['ServiceFeatures'] = dataset['InternetService'] + '_' + \
                                dataset['OnlineSecurity'] + '_' + \
                                dataset['OnlineBackup'] + '_' + \
                                dataset['DeviceProtection'] + '_' + \
                                dataset['TechSupport']

# Drop the original columns if needed
dataset = dataset.drop(['InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport'], axis=1)

dataset.info()

dataset=pd.DataFrame(dataset)

label_encoder = LabelEncoder()
encoded_dataset = dataset.copy()

columns_to_encode = ['Partner', 'Dependents', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'TotalCharges', 'ServiceFeatures']

for column in columns_to_encode:
    encoded_dataset[column] = label_encoder.fit_transform(encoded_dataset[column])

encoded_dataset

new_scaler = StandardScaler()
scaled_encoded_dataset = new_scaler.fit_transform(encoded_dataset)

encoded_dataset

"""encoded_dataset"""

label_encoder = LabelEncoder()
encoded_df = dataset.copy()  # Create a copy of the original DataFrame

# Encode the 'Churn' column using label encoding
encoded_df['Churn'] = label_encoder.fit_transform(encoded_df['Churn'])
encoded_df = encoded_df[['Churn']]

encoded_df

X_train, X_test, y_train, y_test = train_test_split(encoded_dataset, encoded_df, test_size=0.2, random_state=42)

feature_names = list(X_train.columns)

feature_names

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(X_train_scaled)

print(encoded_df.shape)

# Define the MLP model using the Functional API
input_layer = Input(shape=(X_train_scaled.shape[1],))
hidden_layer1 = Dense(64, activation='relu')(input_layer)
hidden_layer2 = Dense(32, activation='relu')(hidden_layer1)
output_layer = Dense(1, activation='sigmoid')(hidden_layer2)

# Create the model
model = Model(inputs=input_layer, outputs=output_layer)

# Compile the model with binary_crossentropy loss for binary classification
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# Train the model
model.fit(X_train_scaled,encoded_df, epochs=10, batch_size=32, validation_split=0.1)

model.summary()

# Make predictions on the test set
predictions = model.predict(X_test_scaled)
predictions_binary = (predictions > 0.5).astype(int)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions_binary)
print(f'Test Accuracy: {accuracy}')

param_grid = {
    'hidden_layer_sizes': [(64,), (128,), (64, 32), (128, 64)],
    'activation': ['relu', 'tanh'],
    'learning_rate_init': [0.001, 0.01, 0.1],
}
mlp = MLPClassifier(max_iter=100)

# Set up GridSearchCV
grid_search = GridSearchCV(mlp, param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# Perform the search
grid_search.fit(X_train_scaled, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_

# Train the best model on the entire training set
best_model = grid_search.best_estimator_
best_model.fit(X_train_scaled, y_train)

# Evaluate on the test set
test_accuracy = best_model.score(X_test_scaled, y_test)
print(f'Test Accuracy: {test_accuracy}')
print(best_params)

# Make predictions on the test set
test_predictions = best_model.predict(X_test_scaled)

test_predictions

from sklearn.metrics import confusion_matrix

# Generate a confusion matrix
conf_matrix = confusion_matrix(y_test, test_predictions)
print(conf_matrix)

from sklearn.metrics import classification_report

# Generate a classification report
class_report = classification_report(y_test, test_predictions)
print(class_report)

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Calculate ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, test_predictions)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()

from joblib import dump
dump(scaler, '/content/drive/My Drive/Colab Notebooks/churn_scalerx.pkl')

model.save('/content/drive/My Drive/Colab Notebooks/modelx.h5')