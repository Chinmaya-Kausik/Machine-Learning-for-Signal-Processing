import PIL
import np
import os
import math

from PIL import Image

#Make a list of names of all the files for images
listOfImages = os.listdir('Assignment_1/Fischer_faces_data/Training')

#Set up lists to take in data for happy and sad images
happy = []
sad = []

#Auxiliary function to be used to flatten the image data array
def flatten(l):
  a = []
  for i in l:
    a += i
  return a

#Load data from the image files into the right lists
for i in listOfImages:
  if "happy" in i:
    happy.append(flatten(np.asarray(Image.open('Assignment_1/Fischer_faces_data/Training/' + i).convert()).tolist()))
  else:
    sad.append(flatten(np.asarray(Image.open('Assignment_1/Fischer_faces_data/Training/' + i).convert()).tolist()))

#List with all the data together
total = happy + sad 

#Function for multiplying matrices
def mul(a,b):
  if(len(a[0]) != len(b)):
    print("The matrices to be multiplied have the wrong sizes")
    return None
  else: 
    prod = np.zeros((len(a), len(b[0]))).tolist()
    for i in range(len(a)):
      for j in range(len(b[0])):
        prod[i][j] = sum([a[i][k]*b[k][j] for k in range(len(a[0]))])
    return prod

#Function to find the mean vector of a list of vectors
def meanVector(l):
  mean = [0 for i in range(len(l[0]))]
  for i in l:
    mean = np.add(mean,i).tolist()
  mean = [j/len(l) for j in mean]
  return mean

#Finding the (eigenvalue, eigenvector) pairs for the covariance matrix of high dimensional data
def highDEigPairs(l):
  n = len(l) 
  m = meanVector(l)
  X = [np.subtract(i,m).tolist() for i in l]
  Xt = np.transpose(X).tolist()
  S = [[j/n for j in i] for i in mul(X, Xt)]
  (eig, eigv) = np.linalg.eig(S)
  eigpairs = [(eig[i], eigv[i]) for i in range(len(eig))]
  eigpairs = sorted(eigpairs, key = (lambda x: x[0]), reverse = True)
  eigpairs = [(i[0], [j/math.sqrt(n*i[0]) for j in mul([i[1]], X)[0]]) for i in eigpairs]
  return eigpairs

#Find total and in-class mean vectors
mean = meanVector(total)
m1 = meanVector(happy)
m2 = meanVector(sad) 

#Find the projection matrix
eigenPairs = highDEigPairs(total)
Wt = [i[1] for i in eigenPairs]
W = np.transpose(Wt).tolist()

#Recenter vectors about the means
Xtotal = [np.subtract(i,mean).tolist() for i in total]
Xhappy = [np.subtract(i,m1).tolist() for i in happy]
Xsad = [np.subtract(i,m2).tolist() for i in sad]

#Find projected output from PCA
Ytotal = mul(Xtotal, W)
Yhappy = mul(Xhappy, W)
Ysad = mul(Xsad, W)

#Because all means of rows (happy, sad, total) are zero by virtue of the PCA procedure, we don't need to subtract anything from the rows of the Y's

#Function to find the covariance matrix of a list of vectors
def covarianceMatrix(l):
  S = np.zeros((len(l[0]), len(l[0]))).tolist()
  m = meanVector(l)
  for i in l:
    S = makelist(np.add(S, mul(np.subtract(np.transpose(i), m).tolist(),np.subtract(i,m).tolist())))
  S = [[j/len(l) for j in i] for i in S]
  return S