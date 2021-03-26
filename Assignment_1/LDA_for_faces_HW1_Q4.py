import PIL
import np
import os
import math

from PIL import Image
listOfImages = os.listdir('Assignment_1/Fischer_faces_data/Training')

happy = []
sad = []


def flatten(l):
  a = []
  for i in l:
    a += i
  return a

im = Image.open('Assignment_1/Fischer_faces_data/Training/subject01.happy.gif')


for i in listOfImages:
  if "happy" in i:
    happy.append(flatten(np.asarray(Image.open('Assignment_1/Fischer_faces_data/Training/' + i).convert()).tolist()))
  else:
    sad.append(flatten(np.asarray(Image.open('Assignment_1/Fischer_faces_data/Training/' + i).convert()).tolist()))

total = happy + sad 

def makelist(l):
  return [i.tolist() for i in l].tolist()

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

print(len(total), [len(i) for i in total])

def meanVector(l):
  mean = [0 for i in range(len(l[0]))]
  for i in l:
    mean = np.add(mean,i).tolist()
  mean = [j/len(l) for j in mean]
  return mean



def covarianceMatrix(l):
  S = np.zeros((len(l[0]), len(l[0]))).tolist()
  m = meanVector(l)
  for i in l:
    S = makelist(np.add(S, mul(np.subtract(np.transpose(i), m).tolist(),np.subtract(i,m).tolist())))
  S = [[j/len(l) for j in i] for i in S]
  return S


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


mean = meanVector(total)
m1 = meanVector(happy)
m2 = meanVector(sad) 

eigenPairs = highDEigPairs(total)
Wt = [i[1] for i in eigenPairs]
W = np.transpose(Wt).tolist()
Xtotal = [np.subtract(i,mean).tolist() for i in total]
Xhappy = [np.subtract(i,m1).tolist() for i in happy]
Xsad = [np.subtract(i,m2).tolist() for i in sad]
Ytotal = mul(Xtotal, W)
Yhappy = mul(Xhappy, W)
Ysad = mul(Xsad, W)

