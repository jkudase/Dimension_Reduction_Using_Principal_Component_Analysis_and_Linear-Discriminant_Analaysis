import numpy as np
import os
import pdb
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


#dataset directory location
datasets_dir =""

def one_hot(x, n):
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h
    

def mnist(noTrSamples=1000, noTsSamples=100, \
                        digit_range=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], \
                        noTrPerClass=100, noTsPerClass=10):
    assert noTrSamples==noTrPerClass*len(digit_range), 'noTrSamples and noTrPerClass mismatch'
    assert noTsSamples==noTsPerClass*len(digit_range), 'noTrSamples and noTrPerClass mismatch'
    data_dir = os.path.join(datasets_dir, 'mnist')
    fd = open(os.path.join(data_dir, 'train-images.idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trData = loaded[16:].reshape((60000, 28*28)).astype(float)

    fd = open(os.path.join(data_dir, 'train-labels.idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trLabels = loaded[8:].reshape((60000)).astype(float)

    fd = open(os.path.join(data_dir, 't10k-images.idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    tsData = loaded[16:].reshape((10000, 28*28)).astype(float)

    fd = open(os.path.join(data_dir, 't10k-labels.idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    tsLabels = loaded[8:].reshape((10000)).astype(float)

    trData = trData/255.
    tsData = tsData/255.

    tsX = np.zeros((noTsSamples, 28*28))
    trX = np.zeros((noTrSamples, 28*28))
    tsY = np.zeros(noTsSamples)
    trY = np.zeros(noTrSamples)

    count = 0
    for ll in digit_range:
        # Train data
        idl = np.where(trLabels == ll)
        idl = idl[0][: noTrPerClass]
        idx = list(range(count*noTrPerClass, (count+1)*noTrPerClass))
        trX[idx, :] = trData[idl, :]
        trY[idx] = trLabels[idl]
        # Test data
        idl = np.where(tsLabels == ll)
        idl = idl[0][: noTsPerClass]
        idx = list(range(count*noTsPerClass, (count+1)*noTsPerClass))
        tsX[idx, :] = tsData[idl, :]
        tsY[idx] = tsLabels[idl]
        count += 1
    
    np.random.seed(1)
    test_idx = np.random.permutation(tsX.shape[0])
    tsX = tsX[test_idx,:]
    tsY = tsY[test_idx]

    trX = trX.T
    tsX = tsX.T
    trY = trY.reshape(1, -1)
    tsY = tsY.reshape(1, -1)
    return trX, trY, tsX, tsY
    
    
def principalComponentAnalysis(trX, tsX):
    
    trainingX=trX.T
    testX=tsX.T
    
    #Scaling
    scaler=StandardScaler()
    scaler.fit(trainingX)
    scaledTrainingX=scaler.transform(trainingX)
    scaledTestX=scaler.transform(testX)
    
    #Pricipal Component Analysis
    pca=PCA(n_components=10)
    pca_trainingX = pca.fit_transform(scaledTrainingX)
    pca_testX = pca.transform(scaledTestX)
    
    #Covariance Matrix
    dimension=pca_trainingX.T
    covarianceMatrix=np.cov(dimension)
    print(covarianceMatrix)
    plt.matshow(covarianceMatrix)
    
    #Reconstruction
    approximation = pca.inverse_transform(pca_trainingX)
 
    #plotting the reconstruction
    fig, axs = plt.subplots(4, 5)
    fig.subplots_adjust(hspace=0.5, wspace=0.7)
    
    #Plotting Original images followed by reconstructed images for digit 5
    for i in range(0,5):
        axs[0,i].imshow(trX[:,40*i+i].reshape(28,-1));
        axs[1,i].imshow(np.real(approximation.T[:,40*i+i]).reshape(28,-1));
    #Plotting Original images followed by reconstructed images for digit 8
    for i in range(0,5):
        axs[2,i].imshow(trX[:,40*i-i+200].reshape(28,-1));
        axs[3,i].imshow(np.real(approximation.T[:,40*i-i+200]).reshape(28,-1));
    
    return pca_trainingX, pca_testX
    
    
def linearDiscriminantAnalysis(pca_trainingX,pca_testX,trY,tsY):
    
    #lda train and test data
    lda_trainingX=pca_trainingX.T
    lda_testingX=pca_testX.T
    
    #lda target data for train and test set
    lda_trainingY=trY
    lda_testingY=tsY
    
    #Sperating the data in two classes - Class for digit 5  and class for digit 8
    trainDataClass5=lda_trainingX[:,0:200]
    trainDataClass8=lda_trainingX[:,200:400]

    #Calculations for digit 5
    m1=np.mean(trainDataClass5,axis=1)
    trainDataClass5_t=trainDataClass5-m1[:,None]
    covar_trainDataClass5_t=trainDataClass5_t.dot(trainDataClass5_t.T)
    
    
    #Calculations for digit 8
    m2=np.mean(trainDataClass8,axis=1)
    trainDataClass8_t=trainDataClass8-m2[:,None]
    covar_trainDataClass8_t=trainDataClass8_t.dot(trainDataClass8_t.T)
    
    #Total covariance (Within class scatter)-Sw
    Sw=(covar_trainDataClass5_t+covar_trainDataClass8_t)
    
    #calculating direction (m1-m2)
    m=m1-m2

    #vector along which the data points need to be projected- w
    SwInv=np.linalg.inv(Sw)
    w=SwInv.dot(m)
    
    Wt=w.T
    
    #Projecting the training,testing,mean1,mean2 data onto the new axis 
    projected_trainData=Wt.dot(lda_trainingX)
    Projected_testData=Wt.dot(lda_testingX)
    projected_mean1=Wt.dot(m1)
    projected_mean2=Wt.dot(m2)
    
    #Calculating the threshold
    threshold=(projected_mean1+projected_mean2)/2;

    #Calculating the training accuracy
    err=0
    z=lda_trainingY[0]
    for i,j in zip(projected_trainData,z):
        if((i<=threshold) and (j==5.)):
            err+=1
        else:
            if((i>threshold) and (j==8.)):
                err+=1
               
    trainingAccuracy=(float(400-err)/400)*100
    print("The training accuracy is %s%%" %trainingAccuracy)
    
    #Calculating the test accuracy
    original_labels=lda_testingY[0]   
    error=0
    for i,j in zip(Projected_testData,original_labels):
        if((i<=threshold) and (j==5.)):
            error+=1
        else:
            if((i>threshold) and (j==8.)):
                error+=1
    testAccuracy=(float(100-error)/100)*100
    print("The test accuracy is %s%%" %testAccuracy) 
    
    
    
def main():
    trX, trY, tsX, tsY = mnist(noTrSamples=400,
                               noTsSamples=100, digit_range=[5, 8],
                               noTrPerClass=200, noTsPerClass=50)
    
    
    #Principal Component Analysis
    pca_trainingX,pca_testX = principalComponentAnalysis(trX, tsX)
    
    #Fisher's Linear Discriminant Analysis
    linearDiscriminantAnalysis(pca_trainingX,pca_testX,trY,tsY)
    
    
    
if __name__ == "__main__":
    main()
