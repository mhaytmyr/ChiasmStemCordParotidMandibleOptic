import time
import cv2, numpy as np

from config import *
from helper_prepare_data import *

def plot_generator(myGen):
    while True:
        X,Y = next(myGen)
        #print(np.unique(Y))    

        #create label
        idx = np.random.randint(0,X.shape[0]-1);
        label = (Y[idx].squeeze()-Y[idx].min())/Y[idx].max();    
        #create image
        img = (X[idx].squeeze()-X[idx].min())/X[idx].max();        

        cv2.imshow("T",img);
        k = cv2.waitKey(0);
        if k==27: break

if __name__=="__main__":
    fileName = "TRAIN.h5"
    BATCH = 7
    myGen = data_generator_stratified(fileName,batchSize=BATCH);
    plot_generator(myGen)
