import pandas as pd, numpy as np, math
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import random,math
import matplotlib.pyplot as plt
df1=pd.read_csv('Desktop/Machine Learning/data_banknote_authentication.csv')

df1=shuffle(df1)
X=df1.iloc[:,0:4]
Y=df1.iloc[:,4]
print(Y.value_counts())
inputnode=4
hiddennode=4

print("Number of input nodes",inputnode)
print("Number of hidden nodes",hiddennode-2)
def sigmoid(x):#sigmoid function
  return 1 / (1 + math.exp(-x))


'''Splitting the data into training,validation and test datasets and assigning initial weights
 for the neurons in hidden and output layers'''
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,Y,test_size=0.33, random_state=42)
Xval,Xtest,Yval,Ytest=train_test_split(Xtest,Ytest,test_size=0.33, random_state=42)
Xtrain = np.concatenate([Xtrain, np.ones(len(Xtrain))[:, np.newaxis]], axis=1)
Ytrain=np.array(list(Ytrain))
Xval = np.concatenate([Xval, np.ones(len(Xval))[:, np.newaxis]], axis=1)
Yval=np.array(list(Yval))
Xtest = np.concatenate([Xtest, np.ones(len(Xtest))[:, np.newaxis]], axis=1)
Ytest=np.array(list(Ytest))
Wlayer1=np.random.random_sample((inputnode,5))#1st layer Weights
Woutput=np.random.random_sample((inputnode+1))#2nd layer Weights


'''Training on the training data to update the weights'''

niter=500
for p in range(niter):
    for l in range(Xtrain.shape[0]):
          #     print(l)
      o1=np.matmul(Wlayer1,Xtrain[l,:])#1st layer
      o1=[sigmoid(i) for i in o1]#Activation
      o1.append(1)
      o1=np.array(o1)
      #     print(o1)
      o2=sigmoid(np.matmul(Woutput,o1))
      delk=o2*(1-o2)*(Ytrain[l]-o2)
          #     print(delk)
      Woutput=Woutput-0.01*delk*o1
      delh=Woutput*delk*o1*(1-o1)
     
          #     print(delh.shape)
          #     print(Wlayer1.shape)
      Wlayer1=Wlayer1-0.01*delh* Xtrain[l,:]
      
'''Testing the learned parameters on the validation set'''

output=[]
for l in range(Xval.shape[0]):  
  o1=np.matmul(Wlayer1,Xval[l,:])
  o1=[sigmoid(i) for i in o1]
  o1=[1 if i>0.75 else 0 if i<0.25 else i for i in o1]
  o1.append(1)
  o1=np.array(o1)
  o2=np.matmul(Woutput,o1)
  output.append(sigmoid(o2))


  

'''Checking accuracy of validation for various thresholds values'''

thresh=0.94
prediction=[1 if i <thresh else 0 for i in output]
print("Validation prediction\n",prediction)
print("correctly classified\n",sum([1 if i==j else 0 for i,j in zip(prediction,Yval)]))
print("Number of misclassified\t",sum([1 if i!=j else 0 for i,j in zip(prediction,Yval)]))
print("Accuracy of Validation\t",sum([1 if i==j else 0 for i,j in zip(prediction,Yval)])/len(Yval))

'''Using the learned parameters and the threshold values from validation on test data'''

output=[]
for l in range(Xtest.shape[0]):
  o1=np.matmul(Wlayer1,Xtest[l,:])
  o1=[sigmoid(i) for i in o1]
  o1=[1 if i>0.75 else 0 if i<0.25 else i for i in o1]
  o1.append(1)
  o1=np.array(o1)
  o2=np.matmul(Woutput,o1)
  output.append(sigmoid(o2))
 

'''Checking the accuracy of the predicted test data'''

thresh=0.79
print("Threshold selected according to validation set")
prediction=[1 if i <thresh else 0 for i in output]
print("Test prediction\n",prediction)
print("correctly classified\t",sum([1 if i==j else 0 for i,j in zip(prediction,Ytest)]))
print("misclassified\t",sum([1 if i!=j else 0 for i,j in zip(prediction,Ytest)]))
print("Accuracy of test\n",sum([1 if i==j else 0 for i,j in zip(prediction,Ytest)])/len(Ytest))