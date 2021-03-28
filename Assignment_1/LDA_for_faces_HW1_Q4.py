import PIL
import np
import os
import math
#import matplotlib.pyplot as plt (Uncomment when plotting)

from PIL import Image

#Make a list of names of all the files for the images
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
  (eigval, eigvect) = np.linalg.eig(S)
  eigpairs = [(eigval[i], eigvect[i]) for i in range(len(eigval))]
  eigpairs = sorted(eigpairs, key = (lambda x: x[0]), reverse = True)
  eigpairs = [(i[0], [j/math.sqrt(n*i[0]) for j in mul([i[1]], X)[0]]) for i in eigpairs]
  return eigpairs

#Find total and in-class mean vectors
mean = meanVector(total)

#Find the projection matrix
eigenPairs = highDEigPairs(total)
Wt = [i[1] for i in eigenPairs]
W = np.transpose(Wt).tolist()
print("PCA Projection matrix found")

#Recenter vectors about the means
Xtotal = [np.subtract(i,mean).tolist() for i in total]
Xhappy = [np.subtract(i,mean).tolist() for i in happy]
Xsad = [np.subtract(i,mean).tolist() for i in sad]
print("Recentered data computed")

#Find projected output from PCA
Ytotal = mul(Xtotal, W)
Yhappy = mul(Xhappy, W)
Ysad = mul(Xsad, W)
print("PCA Projected matrices computed")

#Because all means of rows (happy, sad, total) are zero by virtue of the PCA procedure, we don't need to subtract anything from the rows of the Y's

#Function to find the covariance matrix of a list of vectors
def covarianceMatrix(l):
  m = meanVector(l)
  X = [np.subtract(i,m).tolist() for i in l]
  Xt = np.transpose(X).tolist()
  S = mul(Xt, X)
  S = [[j/len(l) for j in i] for i in S]
  return S

#Finding Sw and m2-m1
Sw = np.add(covarianceMatrix(Yhappy), covarianceMatrix(Ysad)).tolist()

newM1 = meanVector(Yhappy)
newM2 = meanVector(Ysad)
mdiff = [np.subtract(newM2,newM1).tolist()]
print("Sw and (m2-m1) computed for LDA")

#From 4.30 in the book (Section 4.1.4), we know that the projection vector w is given by inv(Sw)(m2-m1) where m2-m1 is a column vector. So in row vector form here, we see that w = (m2-m1)inv(Sw)

w = mul(mdiff, np.linalg.inv(Sw).tolist())
w = [[i/np.linalg.norm(w) for i in w[0]]]

happyList = mul(w, np.transpose(Yhappy).tolist())[0]
sadList = mul(w, np.transpose(Ysad).tolist())[0]
totalList = happyList + sadList

#Plotting projected training values (happy+sad)
"""plt.scatter(totalList, np.zeros_like(totalList).tolist(), c = ([1]*len(happyList) + [0]*len(sadList)), cmap = "hot_r", vmin = -2)

plt.yticks([])
plt.show()"""

#As mentioned in class (lecture 7), we will use a simple non-statistical threshold for this assignment given by (m1+m2)/2. 

#Under the assumptions of identical distributiond for both happy and sad pictures, this also matches the MAP condition. From looking at the projected values (see the picture for the scatterplot for Q4), it is fairly clear that both seem to be similarly distributed in that they are both concentrated very close to their respective means - about -61 and 50 respectively.

happyMean = sum(happyList)/len(happyList)
sadMean = sum(sadList)/len(sadList)
threshold = (happyMean+sadMean)/2
print("LDA projections and threshold computed")

def classifier(filepath):
  imageList = flatten(np.asarray(Image.open(filepath).convert()).tolist())
  imagePCA = mul([imageList], W)
  imageLDA = mul(w, np.transpose(imagePCA).tolist())
  if((imageLDA[0][0]- threshold)*(happyMean - threshold) >0):
    return 1
  else:
    return 0

#Checking accuracy on training data
def checkData(folderPath, dataType):
  happyCorrect = 0
  happyWrong =  0
  sadCorrect = 0
  sadWrong = 0
  listOfImagesToCheck = os.listdir(folderPath)
  for i in listOfImagesToCheck:
    if ('happy' in i):
      if(classifier(folderPath + '/' + i) == 1):
        happyCorrect += 1
      else:
        happyWrong +=1
    else:
      if(classifier(folderPath + '/' + i) == 0):
        sadCorrect +=1
      else:
        sadWrong +=1
  print("\n", dataType, " data:\nHappy: ", happyCorrect, " correct, ", happyWrong, "wrong")
  print("Sad: ", sadCorrect, " correct, ",sadWrong, "wrong")
  print("Total: ", happyCorrect+sadCorrect, " correct, ",happyWrong+ sadWrong, "wrong")
  print("Performance: ", (happyCorrect+sadCorrect)/(happyCorrect+sadCorrect+happyWrong+sadWrong)*100, "%")
  return (happyCorrect+sadCorrect)/(happyCorrect+sadCorrect+happyWrong+sadWrong)



#Running checks
checkData('Assignment_1/Fischer_faces_data/Training', 'Training') #Checking accuracy on training data
checkData('Assignment_1/Fischer_faces_data/test', 'Test') #Checking accuracy on test data

#Notice that the final accuracy for test data is 90%

#K = 20 is the K with the most number of linearly independent eigenvectors (since XXt is a 20x20 matrix) and gives maximum separability. Although since the 20th eigenvalue is of the order e-10, K=19 should also have a similar performance (and it does, both give an accurcay of 90% with test data).