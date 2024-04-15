# -*- coding: utf-8 -*-
"""108_ML-LAB_TEST1.ipynb

### **QUESTION NUMBER 1**
"""

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn import tree
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score,homogeneity_score,completeness_score

df = pd.read_csv('/content/parkinsons.data')
df.head()

print("SHAPE OF THE DATASET",df.shape)
print("Checking whether null values are there or not")
df.isnull().sum()

df.info()

x = df.drop(columns=['name', 'status'], axis=1)
y = df['status']

print(x)

print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

scaler = StandardScaler()
scaler.fit(x_train)
X_train = scaler.transform(x_train)
X_test = scaler.transform(x_test)

def cm_display(actual,predicted):
  confusion_matrix = metrics.confusion_matrix(actual, predicted)
  cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
  cm_display.plot()
  plt.show()

def roc_curves(actual,predicted):
  fpr, tpr, threshold = metrics.roc_curve(actual,predicted)
  roc_auc = metrics.auc(fpr, tpr)

  plt.title('Receiver Operating Characteristic')
  plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
  plt.legend(loc = 'lower right')
  plt.plot([0, 1], [0, 1],'r--')
  plt.xlim([0, 1])
  plt.ylim([0, 1])
  plt.ylabel('True Positive Rate')
  plt.xlabel('False Positive Rate')
  plt.show()

"""### **USING LOGISTIC REGRESSION**"""

lr_model = LogisticRegression(random_state=0)
lr_model.fit(X_train,y_train)

lr_y_pred = lr_model.predict(X_test)

print("The accuracy of logistic regression model is ",accuracy_score(y_test, lr_y_pred))
cm_display(y_test,lr_y_pred)

roc_curves(y_test,lr_y_pred)

"""# **USING SUPPORT VECTOR MACHINES**"""

svm_model = svm.SVC(kernel='linear')
svm_model.fit(X_train,y_train)

svm_y_pred = svm_model.predict(X_test)

svm_accuracy = accuracy_score(y_test, svm_y_pred)
print("The accuracy of the svm model is ",svm_accuracy)
cm_display(y_test,svm_y_pred)

roc_curves(y_test,svm_y_pred)

"""### **K NEAREST NEIGHBORS **"""

knn_classifier = KNeighborsClassifier(n_neighbors=3)
knn_classifier.fit(X_train,y_train)

knn_y_pred = knn_classifier.predict(X_test)

knn_accuracy = accuracy_score(y_test, knn_y_pred)
print("The accuracy of the knn model is ",knn_accuracy)
cm_display(y_test,knn_y_pred)

roc_curves(y_test,knn_y_pred)

"""### NAIVE **BAYES**"""

from sklearn.naive_bayes import GaussianNB

nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

nb_y_pred = nb_model.predict(X_test)

nb_accuracy = accuracy_score(y_test, nb_y_pred)
print("The accuracy of the naive bayes model is ",nb_accuracy)
cm_display(y_test,nb_y_pred)

roc_curves(y_test,nb_y_pred)

"""### **PERCEPTRON LEARNING ALGORITHM**"""

# Build the Perceptron Model
class Perceptron:

	def __init__(self, num_inputs, learning_rate=0.01):
		# Initialize the weight and learning rate
		self.weights = np.random.rand(num_inputs + 1)
		self.learning_rate = learning_rate

	# Define the first linear layer
	def linear(self, inputs):
		Z = inputs @ self.weights[1:].T + + self.weights[0]
		return Z

	# Define the Heaviside Step function.
	def Heaviside_step_fn(self, z):
		if z >= 0:
			return 1
		else:
			return 0

	# Define the Prediction
	def predict(self, inputs):
		Z = self.linear(inputs)
		try:
			pred = []
			for z in Z:
				pred.append(self.Heaviside_step_fn(z))
		except:
			return self.Heaviside_step_fn(Z)
		return pred

	# Define the Loss function
	def loss(self, prediction, target):
		loss = (prediction-target)
		return loss

	#Define training
	def train(self, inputs, target):
		prediction = self.predict(inputs)
		error = self.loss(prediction, target)
		self.weights[1:] += self.learning_rate * error * inputs
		self.weights[0] += self.learning_rate * error

	# Fit the model
	def fit(self, X, y, num_epochs):
		for epoch in range(num_epochs):
			for inputs, target in zip(X, y):
				self.train(inputs, target)

perceptron = Perceptron(num_inputs=X_train.shape[1])
perceptron.fit(X_train, y_train, num_epochs=10)

pla_y_pred = pred = perceptron.predict(X_test)

pla_accuracy = accuracy_score(y_test, pla_y_pred)
print("The accuracy of the pla model is ",pla_accuracy)
cm_display(y_test,pla_y_pred)

roc_curves(y_test,pla_y_pred)

"""### **MULTI LAYER PERCEPTRON**"""

import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation
import matplotlib.pyplot as plt

model = Sequential([


      # dense layer 1
    Dense(256, activation='sigmoid'),

    # dense layer 2
    Dense(128, activation='sigmoid'),

      # output layer
    Dense(10, activation='sigmoid'),
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10,
          validation_split=0.2)

results = model.evaluate(x_test,  y_test, verbose = 0)
print('test loss, test acc:', results)

"""## **CLUSTERING**"""

model = KMeans(n_clusters=2)
y_pred = model.fit_predict(x)

wcss = model.inertia_
print("Within-Cluster Sum of Squares (WCSS):", wcss)
klabels = model.labels_
silhouette=silhouette_score(x,klabels)

print("Silhouette Score [More towards 1 indicates better working]",silhouette)

"""### **RESULTS**

SUPPORT VECTOR MACHINE WAS FOUND TO BE THE SUITABLE ONE FOR ME

High Accuracy: An accuracy of 87.18% indicates that the SVM model correctly predicts the target variable (either 1 - PARKINSON AFFECTED or 0 - HEALTHY ) for approximately 87.18% of the samples in your dataset. High accuracy is generally desirable as it reflects the model's ability to make correct predictions.

AUC Score: The AUC score of 0.78 suggests that the SVM model has good discriminatory power. A higher AUC value indicates that the model is better at distinguishing between the positive and negative classes. An AUC of 0.78 is reasonably good and indicates that the SVM model performs well in terms of classifying the data points.
"""
