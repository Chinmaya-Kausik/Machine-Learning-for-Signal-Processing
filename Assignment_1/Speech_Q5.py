from scipy.io import wavfile
import np
import math

import scipy
import matplotlib.pyplot as plt

#Loading data from the two files
sr, cleanFile = wavfile.read('Assignment_1/speechFiles/clean.wav')
sr, noisyFile = wavfile.read('Assignment_1/speechFiles/noisy.wav') 

#Function to determine the whitening transform
def whiteningTransform(spectrogram):
    
    covarianceMatrix = np.cov(spectrogram)
    [eigVal, eigVect] = np.linalg.eigh(covarianceMatrix)
    correctIndices = np.argsort(-eigVal)
    eigVal = eigVal[correctIndices]
    eigVect = eigVect[:, correctIndices]
    Lsqrtinv = np.linalg.inv(np.power(np.diag(eigVal), 0.5))
    return Lsqrtinv@(eigVect.T)

#Function to extract the spectrogram
def spec(windowSize, stepSize, input, fftMag, sr):
    
    windowSize = int(windowSize)
    stepSize = int(stepSize)
    numberOfSteps = int(math.floor((len(input) - windowSize)/stepSize) + 1)
    spectrogram = np.zeros((int(fftMag/2), numberOfSteps))
    timeArray = np.zeros((numberOfSteps))
    for i in range(numberOfSteps):
        currentWindow = input[stepSize*i:(stepSize*i)+windowSize]
        transformedWindow = scipy.fft.fft(currentWindow, n = fftMag)
        truncatedWindow = transformedWindow[:int(fftMag/2)]
        spectrogram[:, i] = np.log(np.abs(truncatedWindow) + 1)
        timeArray[i] = (stepSize*i + (stepSize*i+windowSize))/(2 * sr)
    return spectrogram, timeArray

def recenter(targetArray, sourceArray):
  mean = np.zeros(targetArray.shape[0])
  for i in np.transpose(sourceArray):
    mean +=i
  mean = mean/sourceArray.shape[1]
  a = np.zeros(targetArray.shape)
  for i in range(targetArray.shape[1]):
    a[:,i] = targetArray[:,i]-mean
  return a


#Computing both spectrograms
[specClean, timeClean] = spec(0.025*sr, 0.01*sr, cleanFile, 256, sr)
[specNoisy, timeNoisy] = spec(0.025*sr, 0.01*sr, noisyFile, 256, sr)

#Computing the whitening transform for the clean matrix and whitening both the matrices with the same transform (hence also using the mean for clean to recenter both)
WTClean = whiteningTransform(specClean)
YCleanFromClean = WTClean @ recenter(specClean, specClean)
YNoisyFromClean = WTClean @ recenter(specNoisy, specClean)


#Computing the whitening transform for the noisy matrix and whitening both the matrices with the same transform (hence also using the mean for noisy to recenter both)
WTNoisy = whiteningTransform(specNoisy)
YNoisyFromNoisy = WTNoisy @ recenter(specNoisy, specNoisy)
YCleanFromNoisy = WTNoisy @ recenter(specClean, specNoisy)

#Function to compute the mean of absolute values of non-diagonal elements
def meanNonDiag(whitenedMatrix):
    
    whitenedCov = np.abs(np.cov(whitenedMatrix))
    sumNonDiag = np.sum(whitenedCov) - np.trace(whitenedCov)
    n = whitenedCov.shape[0]
    return sumNonDiag/(n*n-n)


meanNonDiagCleanFromClean = meanNonDiag(YCleanFromClean)
meanNonDiagNoisyFromClean = meanNonDiag(YNoisyFromClean)
meanNonDiagNoisyFromNoisy = meanNonDiag(YNoisyFromNoisy)
meanNonDiagCleanFromNoisy = meanNonDiag(YCleanFromNoisy)

freq = np.arange(1,129)

"""
# Spectrograms for (a)
figure, axes = plt.subplots(2, 2)

axes[0,0].pcolor(timeClean, freq, specClean)
axes[0,0].set(xlabel='Time', ylabel='Frequency')
axes[0,0].set_title('Original Clean Spectrogram')

axes[1,0].pcolor(timeClean, freq, YCleanFromClean)
axes[1,0].set(xlabel='Time', ylabel='Frequency')
axes[1,0].set_title('Clean Whitened using Clean')

axes[0,1].pcolor(timeNoisy, freq, specNoisy)
axes[0,1].set(xlabel='Time', ylabel='Frequency')
axes[0,1].set_title('Original Noisy Spectrogram')

axes[1,1].pcolor(timeNoisy, freq, YNoisyFromClean)
axes[1,1].set(xlabel='Time', ylabel='Frequency')
axes[1,1].set_title('Noisy Whitened using Clean')

figure.suptitle('Whitening using the Clean File')
figure.tight_layout()
plt.show()

#Spectrograms for (b)
figure1, axes1 = plt.subplots(2, 2)

axes1[0,0].pcolor(timeClean, freq, specClean)
axes1[0,0].set(xlabel='Time', ylabel='Frequency')
axes1[0,0].set_title('Original Clean Spectrogram')

axes1[1,0].pcolor(timeClean, freq, YCleanFromNoisy)
axes1[1,0].set(xlabel='Time', ylabel='Frequency')
axes1[1,0].set_title('Clean Whitened using Noisy')

axes1[0,1].pcolor(timeNoisy, freq, specNoisy)
axes1[0,1].set(xlabel='Time', ylabel='Frequency')
axes1[0,1].set_title('Original Noisy Spectrogram')

axes1[1,1].pcolor(timeNoisy, freq, YNoisyFromNoisy)
axes1[1,1].set(xlabel='Time', ylabel='Frequency')
axes1[1,1].set_title('Noisy Whitened From Noisy')

figure1.suptitle('Whitening using the Noisy File')
figure1.tight_layout()
plt.show()
"""

#You can find the figures generated using the code above in approp files 
#Printing the means of the absolute values for non-diagonal elements

print('\nWhitening using the Clean File:\n')
print('Mean value of non-diagonal elements in the covariance matrix of whitened Clean input %f\n' %meanNonDiagCleanFromClean)
#0.0 
print('Mean value of non-diagonal elements in the covariance matrix of whitened Noisy input %f\n' %meanNonDiagNoisyFromClean)
#About 0.213

print('\n\nWhitening using the Noisy file:\n')
print('Mean value of non-diagonal elements in the covariance matrix of whitened Noisy input %f\n' %meanNonDiagNoisyFromNoisy)
#0.0
print('Mean value of non-diagonal elements in the covariance matrix of whitened Clean input %f\n' %meanNonDiagCleanFromNoisy)
#About 0.12

#COMMENTS ON WHITENING WHEN THE DATA DISTRIBUTION CHANGES: We can see that the mean of the absolute values of non-diagonal entries is zero if we use the whitening transform generated by the same data distribution as the one it is being applied to. It does increase to a non-zero value when the distribution changes to something else, since then we are not using the eigenvectors of the covariance matrix of the data in question. So the rotation does not necessarily decorrelate the features (or even recenter them appropriately) and we get non-zero non-diagonal terms in the covariance matrix fo this "whitened" data. However, since in our example the two data distributions were just clean and noisy versions of the same file, the difference between the distributions is not huge on the mean and the amount of correlation remaining in features when using the whitening transform for one of the files on the other is still not tremendous. It is of the order of 0.1.
