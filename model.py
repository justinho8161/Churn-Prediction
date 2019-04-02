import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from matplotlib import pyplot as plt
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble.partial_dependence import plot_partial_dependence

class Cleaning():
	def __init__(self):
		pass

	def fit(self, df):
		self.df = df

	def dt_and_churn(self):
		copy = self.df.copy()
		copy['last_trip_datetime'] = pd.to_datetime(copy["last_trip_date"])
		copy["signup_date_datetime"] = pd.to_datetime(copy["signup_date"])
		copy = copy.drop(columns=["last_trip_date", "signup_date"])
		copy["churn"] = copy['last_trip_datetime'].dt.month < 6
		copy.churn = copy.churn.astype(int)
		return copy

	def driver_rating(self, mode=True):
		copy = self.df.copy()
		if mode==True:
			val = copy['avg_rating_of_driver'][copy['avg_rating_of_driver'] != np.NaN].mode()[0]
			copy['avg_rating_of_driver'] = copy['avg_rating_of_driver'].replace(np.NaN, val)
			return copy
		if mode==False:
			val = copy['avg_rating_of_driver'][copy['avg_rating_of_driver'] != np.NaN].mean()
			copy['avg_rating_of_driver'] = copy['avg_rating_of_driver'].replace(np.NaN, val)
		return copy
	def early_churn(self):
		copy = self.df.copy()
		copy['length_of_use'] = copy['last_trip_datetime'] - copy['signup_date_datetime']
		copy['length_of_use'] = copy['length_of_use'].dt.days
		copy['early_churn'] = copy['length_of_use'] < 3
		copy.early_churn = copy.early_churn.astype(int)
		return copy
	
	def dummy(self):
		copy = self.df.copy()
		copy.luxury_car_user = copy.luxury_car_user.astype(int)
		new = pd.get_dummies(copy[['city', 'phone', 'luxury_car_user']], drop_first=True)
		copy = copy.drop(['city', 'phone', 'luxury_car_user'], axis=1)
		newdf = pd.concat([copy, new], axis=1)
		return newdf

	def scaler(self):
		copy = self.df.copy()
		X = copy[['avg_dist', 'avg_rating_by_driver', 'avg_rating_of_driver',
    		'avg_surge', 'surge_pct', 'trips_in_first_30_days',
    		'weekday_pct', 'length_of_use']].values
		scale = StandardScaler()
		scale.fit(X)
		copy[['avg_dist', 'avg_rating_by_driver', 'avg_rating_of_driver',
    		'avg_surge', 'surge_pct', 'trips_in_first_30_days',
    		'weekday_pct', 'length_of_use']] = scale.transform(X)
		return copy
	
	def drop_dt(self):
		copy = self.df.copy()
		copy = copy.drop(['last_trip_datetime', 'signup_date_datetime'], axis=1)
		return copy

	def all(self, scale=False):
		self.df = self.dt_and_churn()
		self.df = self.driver_rating(mode=True)
		self.df = self.early_churn()
		self.df = self.df.dropna(axis=0)
		self.df = self.dummy()
		if scale==True:
			self.df = self.scaler()
		return self.df

class EDA():
	def __init__(self):
		pass

	def fit(self, df):
		self.df = df
		return self

	def pie_plot(self, col):
		counts = self.df[col].value_counts()
		plt.pie(counts, labels=counts.index)
		plt.title(col)
		plt.show()

	def cat_charter(self):
		copy = self.df.copy()
		newdf = np.empty((len(self.df.columns),2))
		copy = copy.replace('None or Unspecified', np.NaN)
		for i, column in enumerate(copy.columns):
			appends =[len(copy[column].unique()), copy[column].count()] 
			newdf[i,:] = appends
		result = pd.DataFrame(newdf, index =[name for name in copy.columns], 
			columns = ['Unique_Values', 'Non_null_count'])
		return result

	def scatplot(self, col1, col2):
		plt.scatter(self.df[col1], self.df[col2])
		plt.xlabel(col1)
		plt.ylabel(col2)
		plt.show()

class Model():
	def __init__(self):
		pass
	
	def fit(self, df):
		self.df = df
		self.y = self.df.churn.values
		self.X = self.df.drop('churn', axis=1).values
		self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, 
			self.y, test_size=0.2)
		self.names = df.drop('churn',axis=1).columns
		return self
	
	def CV(self, algorithm):
		cv = np.sqrt(-1*cross_val_score(algorithm, self.X_train, self.y_train, n_jobs = -1, 
			scoring='neg_mean_squared_log_error', cv=10))
		return cv

	def fit_algorithm(self, algorithm):
		alg = algorithm()
		alg.fit(X_train, y_train)
		alg.score()
	
	def importance(self, algorithm):
		feature_importances = algorithm.feature_importances_
		top10_colindex = np.argsort(feature_importances)[::-1][0:10]
		#names = self.names[top10_colindex]
		# print(names)
		feature_importances = feature_importances[top10_colindex]
		feature_importances = feature_importances / np.sum(feature_importances)
		y_ind = np.arange(9, -1, -1) # 9 to 0
		fig = plt.figure(figsize=(8, 8))
		plt.barh(y_ind, feature_importances, height = 0.3, align='center')
		plt.ylim(y_ind.min() + 0.5, y_ind.max() + 0.5)
		plt.yticks(y_ind, self.names)
		plt.xlabel('Relative feature importances')
		plt.ylabel('Features')

		# print("1) Sorted features, their relative importances, and their indices:" )
		# for fn, fi, indx in zip(self.names, feature_importances, top10_colindex):
		# 	print("{0:<30s} | {1:6.3f} | {2}".format(fn, fi, indx))
	
	def plot_partial_dependencies(self, algorithm):
		feature_importances = algorithm.feature_importances_
		top10_colindex = np.argsort(feature_importances)[::-1][0:10]
		plot_partial_dependence(algorithm, self.X_train, features=top10_colindex, feature_names = self.names, figsize=(12,10))
		plt.tight_layout()












