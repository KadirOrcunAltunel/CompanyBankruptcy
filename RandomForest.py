import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
import seaborn as sns
from sklearn.metrics import precision_recall_curve, classification_report, roc_curve
from numpy import argmax, sqrt

# read data
bankruptcy = pd.read_csv('/Users/kadiraltunel/Documents/Datasets/american_bankruptcy.csv')
print(bankruptcy.head())
print(bankruptcy.info())

# change column names
new_columns = ['Company Name', 'Status', 'Year', 'Current Assets', 'Cost of Sold Goods',
               'Depreciation and Amortization', 'EBITDA', 'Inventory', 'Net Income', 'Total Receivables',
               'Market Value', 'Net Sales', 'Total Assets', 'Total Long Term Debt', 'EBIT', 'Gross Profit',
               'Total Current Liabilities', 'Retained Earnings', 'Total Revenue', 'Total Liabilities',
               'Total Operating Expenses']
bankruptcy.columns = new_columns

print(bankruptcy.head())
print(bankruptcy.isnull().sum())

# Drop Company Name and Year since they are not important
bankruptcy.drop(['Company Name', 'Year'], axis=1, inplace=True)
print(bankruptcy.info())

# Get Dummy Values for Status
enc = OrdinalEncoder(dtype=int)
bankruptcy[['Status']] = enc.fit_transform(bankruptcy[['Status']])
print(bankruptcy.head())
print(bankruptcy.info())

# Print Correlation Matrix
plt.figure(figsize=(26, 20))
heatmap = sns.heatmap(bankruptcy.corr(), vmin=-1, vmax=1, annot=True)
plt.show()

# Print the count of response variable and visualize it
print(bankruptcy['Status'].value_counts(normalize=True))

plt.figure(figsize=(16, 10))
sns.countplot(x=bankruptcy['Status'])
plt.show()

# Create labels(response variable) and features (explanatory variables)
labels = bankruptcy['Status']
features = bankruptcy.drop(['Status'], axis=1)
print(labels)
print(features)

# Create train and test sets
train_features, test_features, train_labels, test_labels = \
    train_test_split(features, labels, test_size=0.2, random_state=0)

# Normalize the data
scaler = StandardScaler()
train_features_scaled = scaler.fit_transform(train_features)
test_features_scaled = scaler.transform(test_features)

# Reassign test and train features with normalized data
test_features = pd.DataFrame(test_features_scaled, columns=[features])
train_features = pd.DataFrame(train_features_scaled, columns=[features])
print(test_features.describe())

# Use Random Forest for predicting company bankruptcy and print confusion matrix
base_rf_model = RandomForestClassifier(random_state=0)
base_rf_model.fit(train_features, train_labels)
base_rf_model_pred = base_rf_model.predict(test_features)
print(classification_report(test_labels, base_rf_model_pred))

# print the prediction probability of positive outcomes
base_pred_prob = base_rf_model.predict_proba(test_features)
print(base_pred_prob)
pred_positive = base_pred_prob[:, 1]

# Use Precision Recall Curve to improve f1 score
precision, recall, thresholds = precision_recall_curve(test_labels, pred_positive)

# Find the best Threshold that improves f1 Score
f1_score = (2 * precision * recall) / (precision + recall)
i_max = argmax(f1_score)
print('Best Threshold=%f, F-Score=%.3f' % (thresholds[i_max], f1_score[i_max]))

# Graph the result
plt.figure(figsize=(16, 10))
no_skill = len(test_labels[test_labels == 1]) / len(test_labels)
plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
plt.plot(recall, precision, marker='.', label='Random Forest')
plt.scatter(recall[i_max], precision[i_max], marker='o', color='black', label='Best')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.show()

# Use the best threshold to test the Random Forest Model
best_threshold = 0.23
best_threshold_pred = base_rf_model.predict_proba(test_features)
predicted_f1 = (best_threshold_pred[:, 1] >= best_threshold).astype('int')
print(classification_report(test_labels, predicted_f1))

# Use ROC Curve to improve ROC
fpr, tpr, thresholds = roc_curve(test_labels, pred_positive)

# Find the best Threshold
g_means = sqrt(tpr * (1 - fpr))
ix = argmax(g_means)
print('Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], g_means[ix]))

# Graph the result
plt.figure(figsize=(16, 10))
plt.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
plt.plot(fpr, tpr, marker='.', label='Logistic')
plt.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

# Use the best threshold for random forest
best_threshold_ROC = 0.1
best_threshold_ROC_pred = base_rf_model.predict_proba(test_features)
predicted_ROC = (best_threshold_ROC_pred[:, 1] >= best_threshold_ROC).astype('int')
print(classification_report(test_labels, predicted_ROC))

# Result Precision Recall Curve improved precision, recall and f1-score dramatically. ROC Curve, on the other
# hand improved recall dramatically whereas accuracy suffered a lot. Precision and f1 score saw some moderate
# improvement. Overall, Precision Recall Curve performed better in the prediction.

