# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 17:43:39 2024

@author: thomas
"""

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_absolute_error, confusion_matrix
import shap
import plotly.graph_objects as go
# %%

data = pd.read_csv("wine+quality/winequality-red.csv", sep = ";")


# %% correlation of features 
correlation_matrix = data.corr().abs()

fig, ax = plt.subplots(figsize=(10,10))   
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", ax = ax)
plt.show()

# %% histogram of quality distribution
sns.countplot(x='quality', data=data)
plt.show()


# %% parallel coordinates plot

fig = go.Figure(go.Parcoords(
    line=dict(color=data['quality'], colorscale = "jet"),
    dimensions=[dict(label=col, values=data[col]) for col in data.columns]
))

fig.show(renderer = "browser")

# %% boxplots
for column in data.columns[:-1]:  
    sns.boxplot(x='quality', y=column, data=data)
    plt.show()


#%% normalize data
scaler = StandardScaler()
X = data.iloc[:, :-1]  
y = data['quality']    
X_scaled = scaler.fit_transform(X)

data_scaled = pd.DataFrame(X_scaled,columns = X.columns)
data_scaled["quality"] = y

#%% Data preparation

#also develope aggregated label
mapping = {3: 1, 4: 1, 5: 2, 6: 3, 7:4, 8: 4}
data_scaled["quality_agg"] = data_scaled["quality"].map(mapping)

data_train,data_test = train_test_split(data_scaled,test_size = 0.1, stratify=data_scaled["quality"])


X_train = data_train.iloc[:,:-2]
X_test = data_test.iloc[:,:-2]
y_train = data_train["quality"]
y_test = data_test["quality"]

y_train_agg = data_train["quality_agg"]
y_test_agg = data_test["quality_agg"]

#%% Random forest on true label

model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print(f"mean absolute error of classification: {mean_absolute_error(y_test,y_pred)}")

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

y_temp = y_test.to_frame().reset_index()
class_8_indices = y_temp[y_temp["quality"] == 8].index
class_8_index = list(model.classes_).index(8)
shap_values_class_8 = shap_values[:,:,class_8_index]
shap_values_class_8 = shap_values_class_8[class_8_indices]

X_test_class_8 = X_test.reset_index().drop(["index"],axis = 1).loc[class_8_indices]


# shap.summary_plot(
#     shap_values_class_8, 
#     X_test_class_8, 
#     feature_names=X_test.columns,
#     #plot_type="bar",
#     show = False
# )
# plt.title("Feature contributions to class 8 prediction")
# plt.show()

cm = confusion_matrix(y_test, y_pred, labels = [3,4,5,6,7,8])
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[3,4,5,6,7,8], yticklabels=[3,4,5,6,7,8])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

# %% Random Forest on aggregated label

model = RandomForestClassifier()
model.fit(X_train, y_train_agg)
y_pred = model.predict(X_test)
print(classification_report(y_test_agg, y_pred))
print(f"mean absolute error of classification: {mean_absolute_error(y_test_agg,y_pred)}")

cm = confusion_matrix(y_test_agg, y_pred, labels = [1,2,3,4])
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1, 2, 3], yticklabels=[0, 1, 2, 3])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

