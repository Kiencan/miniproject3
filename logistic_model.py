import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing

# import tensorflow as tf

import joblib

train_df = pd.read_csv('../archive/emnist-balanced-train.csv', header=None)
label_map = pd.read_csv("../archive/emnist-balanced-mapping.txt", 
                        delimiter = ' ', 
                        index_col=0, 
                        header=None)

#Initialising an empty dictionary
label_dictionary = {}

#Running a loop for ASCII equivalent to character conversion
for index, label in enumerate(label_map):
    label_dictionary[index] = chr(label)

train_df_new = train_df[train_df[0].isin(np.arange(0, 36))]
x_train = train_df_new.loc[:, 1:]
y_train = train_df_new.loc[:, 0]

def flip_and_rotate(image):
    W = 28
    H = 28
    image = image.reshape(W, H)
    image = np.fliplr(image)
    image = np.rot90(image)
    return image
x_train = np.apply_along_axis(flip_and_rotate, 1, x_train.values)
x_train = x_train.reshape(-1, 784)
x_train = x_train.astype('float32') / 255
x_train = x_train.astype('float32')


X_train, X_test, Y_train, Y_test = train_test_split(x_train, y_train, test_size= 0.2, random_state=0)
number_of_classes = Y_train.nunique()


def onehotencoding(y):
    y_one = np.zeros((y.shape[0], number_of_classes))
    for i, label in enumerate(y):
        y_one[i, label] = 1
    return y_one

Y_train_onehot = onehotencoding(Y_train)

class SoftmaxRegression:
    def __init__(self,lr=0.01,epochs=100,batch_size=64):
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.w = None
        self.loss_list = None
        
    def sofmax (self,z):
        e_z = np.exp(z)
        return e_z/e_z.sum(axis=1,keepdims=True)

    def predict(self,X):
        return self.sofmax(np.dot(X,self.w))

    def loss(self,X,y):
        y_pred = self.predict(X)
        return -np.sum(y*np.log(y_pred))/len(y)

    def mini_batch_gradient_descent(self,X,y):
        for epoch in range(self.epochs):
            for i in range(0,len(X),self.batch_size):
                X_batch = X[i:i+self.batch_size]
                y_batch = y[i:i+self.batch_size]
                y_pred = self.predict(X_batch)
                self.w = self.w - self.lr*np.dot(X_batch.T,y_pred-y_batch)/len(X_batch)
            self.loss_list.append(self.loss(X,y))

    def fit(self,X,y):
        X = np.hstack((X,np.ones((X.shape[0],1))))
        self.w = np.random.randn(X.shape[1],y.shape[1])
        self.loss_list = []
        self.mini_batch_gradient_descent(X,y)

    def evaluate(self,X,y):
        X = np.hstack((X,np.ones((X.shape[0],1))))
        y_pred = self.predict(X)
        y_pred = np.argmax(y_pred,axis=1)
        print('accuracy: ',np.sum(y_pred==y)/len(y))
        cm = confusion_matrix(y,y_pred)
        print(cm)
        sns.heatmap(cm,annot=True,fmt='d')
        plt.show()
        return y_pred
    

# model = SoftmaxRegression(0.01, 1000, batch_size=64)
# model.fit(X_train, Y_train_onehot)
# label_pred = model.evaluate(X_test,Y_test)
# print(label_pred)

# joblib.dump(model, 'log_model.sav')

model = LogisticRegression(solver='lbfgs')
model.fit(X_train, Y_train)

label_pred = model.predict(X_test)
#accuracy
accuracy = np.sum(label_pred == Y_test) / len(Y_test)
print(f'Accuracy: {accuracy * 100:.2f}%')

joblib.dump(model, 'logistic_model.sav')

