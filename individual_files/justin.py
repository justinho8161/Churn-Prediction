from sklearn.model_selection import GridSearchCV
import warnings
import model as fxns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
import requests
from sklearn.ensemble.partial_dependence import plot_partial_dependence
from sklearn.ensemble.partial_dependence import partial_dependence
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import pdb
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
df = pd.read_csv('data/churn.csv')
clean = fxns.Cleaning()
clean.fit(df)
df = clean.all(scale=True)

#drops datetime type columns

df = clean.drop_dt()
#drops our data leakage columns
df = df.drop(['length_of_use', 'early_churn'], axis=1)


X = df[df.columns.difference(['last_trip_datetime','signup_date_datetime','churn'])]
y = df['churn'].values
X_train, X_test, y_train, y_test = train_test_split(X, y)


rf = RandomForestClassifier(n_estimators=500, oob_score=True)
rf.fit(X_train, y_train)
print("\n11: accuracy score:", rf.score(X_test, y_test))
print("    out of bag score:", rf.oob_score_)
rf.feature_importances_
rf.feature


new_thing = zip(rf.feature_importances_,X_train.columns)
rf.feature_importances_
plt.barh(X_train.columns,rf.feature_importances_)


plt.ylim(y_ind.min() + 0.5, y_ind.max() + 0.5)
plt.yticks(y_ind, feature_names)
plt.xlabel('Relative feature importances')
plt.ylabel('Features');




plot_partial_dependence(rf, X_train, rf.feature_importances_,feature_names = X_train.columns,figsize=(12,10))
plt.tight_layout();

plot_partial_dependence(model, X_train, top10_colindex,
                        feature_names = all_feature_names,
                        figsize=(12,10))
plt.tight_layout();


plot_partial_dependence(model, X_train, top10_colindex,
                        feature_names = all_feature_names,
                        figsize=(12,10))
plt.tight_layout();
