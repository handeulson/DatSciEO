import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from dataset import TreeClassifPreprocessedDataset

data = []
labels = []
# Specify data folder direction
data_dir = r'C:\Users\User\DatSciEO-class\data\1123_delete_nan_samples_nanmean_B2'
ds = TreeClassifPreprocessedDataset(data_dir)
for data_, label_ in ds:
    data.append(data_)
    labels.append(label_)



# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Initialize an SVM classifier
clf = SVC()

# Train the classifier
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")