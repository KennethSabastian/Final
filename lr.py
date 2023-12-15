import pandas as pd
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA

# Load the data
df = pd.read_csv('finale.csv')

# Preprocess the text
df['text'] = df['full_text'].apply(lambda x: re.sub(r'\W+', ' ', x.lower()))

# Convert the text to numerical features
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['text'])

# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, df['Label'], test_size=0.2, random_state=42)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, zero_division=1))

# Apply the model to the entire dataset
df['predicted_label'] = model.predict(X)

# display data that has changes to the label and save it to a csv file
df_changed = df[df['Label'] != df['predicted_label']]
df_changed.to_csv('changed_lr.csv', index=False)


# Save the classified data
df.to_csv('classified.csv', index=False)

# Reduce the dimensionality of the features to 2D
pca = PCA(n_components=2, random_state=42)
reduced_features = pca.fit_transform(X.toarray())

# Create a DataFrame for the reduced features
df_reduced = pd.DataFrame(reduced_features, columns=['PC1', 'PC2'])

# Add the predicted labels to the DataFrame
df_reduced['predicted_label'] = df['predicted_label']

# Map the labels to numbers
label_mapping = {'Bully': 0, 'Nobully': 1}
df_reduced['predicted_label'] = df_reduced['predicted_label'].map(label_mapping)

# Drop rows with missing values
df_reduced = df_reduced.dropna(subset=['predicted_label'])

# Check that all values in 'predicted_label' are in 'label_mapping'
assert set(df_reduced['predicted_label']).issubset(set(label_mapping.values()))

# Create a scatter plot
plt.figure(figsize=(8, 6))
scatter = plt.scatter(df_reduced['PC1'], df_reduced['PC2'], c=df_reduced['predicted_label'])
plt.title('Scatter plot of Classified Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(*scatter.legend_elements(), title='Classes')
plt.show()