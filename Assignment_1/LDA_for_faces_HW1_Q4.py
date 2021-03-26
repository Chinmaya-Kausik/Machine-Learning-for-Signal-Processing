import PIL
import np
import os

from PIL import Image
listOfImages = os.listdir('Assignment_1/Fischer_faces_data/Training')

happy = []
sad = []


def flatten(l):
  a = []
  for i in l:
    a += i
  return [a]

im = Image.open('Assignment_1/Fischer_faces_data/Training/subject01.happy.gif')


for i in listOfImages:
  if "happy" in i:
    happy.append(flatten(np.asarray(Image.open('Assignment_1/Fischer_faces_data/Training/' + i).convert()).tolist()))
  else:
    sad.append(flatten(np.asarray(Image.open('Assignment_1/Fischer_faces_data/Training/' + i).convert()).tolist()))

total = happy + sad 


def mul(a,b):
  if len(a[0]) != len(b):
    print("The matrices to be multiplied have the wrong sizes")
    return None
  else: 
    prod = np.zeros((len(a), len(b[0])))
    for i in range(len(a)):
      for j in range(len(b[0])):
        prod[i][j] = sum([a[i][k]*b[k][j] for k in range(len(a[0]))])
    return prod


def meanVector(l):
  mean = [0 for i in range(len(l[0]))]
  for i in l:
    mean = np.add(mean,i)
  mean = [i/len(l) for i in mean]
  return mean

mean = meanVector(total)
m1 = meanVector(happy)
m2 = meanVector(sad) 


def covarianceMatrix(l):
  S = np.zeros((len(l[0]), len(l[0])))
  m = meanVector(l)
  for i in l:
    S = np.add(S, mul(np.subtract(np.transpose(i), m),np.subtract(i,m)))
  S = [i/len(l) for i in S]
  return S


def highDCovMat(l):
  n = len(l) 
  m = meanVector(l)
  X = mul(l, np.transpose(l))
  return None