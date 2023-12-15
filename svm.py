import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pandas as pd

data = pd.read_csv('clean.csv')

# Plot the distribution of classes before applying the model
plt.figure(figsize=(10, 5))
data['Label'].value_counts().plot(kind='bar')
plt.title('Distribution of Classes before Applying the Model')
plt.show()

# Encoding the labels
le = LabelEncoder()
data['Label'] = le.fit_transform(data['Label'])

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(data['full_text'], data['Label'], test_size=0.2, random_state=42)

# Vectorizing the text data
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Training the Naive Bayes classifier
#nb_classifier = MultinomialNB()
#nb_classifier.fit(X_train_vec, y_train)
#
## Predicting on the test set
#y_pred = nb_classifier.predict(X_test_vec)

svm = SVC(kernel = "rbf")
svm.fit(X_train_vec,y_train)
y_pred = svm.predict(X_test_vec)

# Plot the distribution of predicted classes
plt.figure(figsize=(10, 5))
pd.Series(y_pred).value_counts().plot(kind='bar')
plt.title('Distribution of Predicted Classes')
plt.show()

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, zero_division=1)

print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Classification Report:\n{class_report}")

# Apply the model to the entire dataset
X_full = vectorizer.transform(data['full_text'])
data['predicted_label'] = svm.predict(X_full)

# print the count for each class
c_before = data['Label'].value_counts()
print(f"Before:\n{c_before}")

# print the count for each class
c_after = data['predicted_label'].value_counts()
print(f"After:\n{c_after}")

# Plot the distribution of classes after applying the model
plt.figure(figsize=(10, 5))
data['predicted_label'].value_counts().plot(kind='bar')
plt.title('Distribution of Classes after Applying the Model')
plt.show()

# show the data which is misclassified
# data['Label'] = le.inverse_transform(data['Label'])
# data['predicted_label'] = le.inverse_transform(data['predicted_label'])
# misclassified = data[data['Label'] != data['predicted_label']]
# print(misclassified)

# display data that has changes to the label
data['Label'] = le.inverse_transform(data['Label'])
data['predicted_label'] = le.inverse_transform(data['predicted_label'])
changed = data[data['Label'] != data['predicted_label']]
print(changed)

# export the change to a csv file
changed.to_csv('changed_naive_bayes.csv', index=False)

# Export the DataFrame to a CSV file
data.to_csv('classified_naive_bayes.csv', index=False)