import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split

pd.set_option('display.max_columns', None)


# Data preparation
dataframe = pd.read_csv('https://sololearn.com/uploads/files/titanic.csv')
dataframe['male'] = dataframe['Sex'] == 'male'

X = dataframe[['Pclass', 'male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']].values
y = dataframe['Survived'].values

X_train, X_test, y_train, y_test = train_test_split(X, y)
#print(X)
#print(Y)


# Regression Model Building
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict
y_pred_proba = model.predict_proba(X_test)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:,1])


"""
result = model.predict([test_dataset])
print(result)
print(model.score(X_test, y_test))
"""

# plot
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('1 - specificity')
plt.ylabel('sensitivity')
plt.savefig("lao")
plt.show()
