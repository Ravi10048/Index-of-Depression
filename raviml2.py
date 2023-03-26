import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
from pandas_profiling import ProfileReport

# Load the data from CSV file
data = pd.read_csv('indian_people_dataset.csv')

# Report=ProfileReport(data)
# Report.to_file("indian_people_dataset.html")

from sklearn import preprocessing

# # Creating labelEncoder
le = preprocessing.LabelEncoder()

print(data.head())
# # Converting string labels into numbers.
data['Gender']=le.fit_transform(data['Gender'])

data['Family history']=le.fit_transform(data['Family history'])
data['Cultural factors']=le.fit_transform(data['Cultural factors'])
data['Trauma history ']=le.fit_transform(data['Trauma history'])
data['Treatment history ']=le.fit_transform(data['Treatment history'])
data['Suicidal ideation ']=le.fit_transform(data['Suicidal ideation'])
data['Medications ']=le.fit_transform(data['Medications'])
data['Substance use ']=le.fit_transform(data['Substance use'])

# save=data.to_csv("encoded1.csv")
X = data.iloc[:, 1:-1].values  # Input features
y = data['Level of depression']  # Target variable


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)

# Create a Random Forest classifier with 100 trees
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the Random Forest classifier on the training data
clf.fit(X_train, y_train)

# plt.plot(X_train,y_train)
# plt.show()

# plt.plot(X_test,y_test)
# plt.show()

# Predict the target variable for the test data
y_pred = clf.predict(X_test)

# plt.plot(y_test,y_pred)
# plt.show()

# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Display the confusion matrix and classification report
print("Confusion Matrix:")
print(cm)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Compute and display the accuracy score
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)

y_train_pred = clf.predict(X_train)
# Compute and display the accuracy score
acc1 = accuracy_score(y_train, y_train_pred)
print("Accuracy:", acc1)


# Plot the confusion matrix as a heatmap
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(np.unique(y)))
plt.xticks(tick_marks, np.unique(y), rotation=45)
plt.yticks(tick_marks, np.unique(y))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()







'''  Name: The name of the individual
Age: The age of the individual (in years)
Gender: The gender of the individual (Male, Female, Other)
Sadness: The degree to which the individual is experiencing sadness, on a scale of 1-5
Loss of interest: The degree to which the individual has lost interest in activities, on a scale of 1-5
Sleep disturbances: The degree to which the individual is experiencing sleep disturbances, on a scale of 1-5
Appetite changes: The degree to which the individual is experiencing changes in appetite, on a scale of 1-5
Hopelessness: The degree to which the individual is experiencing hopelessness, on a scale of 1-5
Stressful life events: The number of stressful life events the individual has experienced recently
Family history: Whether the individual has a family history of depression (Yes/No)
Social support: The degree to which the individual feels supported by others, on a scale of 1-5
Physical health problems: The degree to which the individual is experiencing physical health problems, on a scale of 1-5
Substance use: Whether the individual is using substances (e.g. drugs, alcohol) that could be contributing to their depression symptoms (Yes/No)
Medications: Whether the individual is taking medication to manage their depression symptoms (Yes/No)
Anxiety symptoms: The degree to which the individual is experiencing anxiety symptoms, on a scale of 1-5
Suicidal ideation: Whether the individual has had thoughts of suicide (Yes/No)
Treatment history: The individual's history of seeking treatment for depression (e.g. therapy, medication)
Trauma history: Whether the individual has a history of trauma (e.g. abuse, neglect) that could be contributing to their depression symptoms (Yes/No)
Financial stress: The degree to which the individual is experiencing financial stress, on a scale of 1-5
Cultural factors: Whether cultural factors (e.g. stigma surrounding mental health) are impacting the individual's depression symptoms (Yes/No)
Level of depression: The severity of the individual's depression symptoms, on a scale of 1-5'''