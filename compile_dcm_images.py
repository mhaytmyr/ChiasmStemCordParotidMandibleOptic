import os, sys, glob
import numpy as np
import pydicom, cv2
import h5py, dask.array as da

import matplotlib.pyplot as plt
from skimage import exposure
from skimage.draw import polygon
from collections import defaultdict

from config import *
from helper_prepare_data import body_bounding_box, crop_body_roi

def struct_name_map(roiName):
    #standarize name
    roiName = roiName.lower();
    #look for string matching
    if "body" in roiName: return 1 
    elif "chiasm" in roiName: return 2 
    elif "brainstem" in roiName and "+" not in roiName and "_" not in roiName: return 3 
    elif "cord" in roiName and "+" not in roiName and "_" not in roiName: return 4
    elif "parotid" in roiName and "ptv" not in roiName: return 5
    elif "mandible" in roiName: return 6 
    elif "on_l" in roiName or "on_r" in roiName or "optic n" in roiName: return 7 #optical nerve LR
    else: return 0; 


def read_structures(structure):

    #create key pairs with key corresponding to z component
    contours = defaultdict(dict);
    for i in range(len(structure.ROIContourSequence)):

        #create contour data
        roiName = structure.StructureSetROISequence[i].ROIName;
        roiNumber = struct_name_map(roiName);
        print("Extracting contour for ",roiName);

        try:
            #get collection of contours for this label
            for s in structure.ROIContourSequence[i].ContourSequence:
                node = np.array(s.ContourData).reshape(-1,3);
                zIndex = float("{0:.3f}".format(node[0,2])); #convert to float for string
                if roiNumber in contours[zIndex]:
                    contours[zIndex][roiNumber].append(node[:,:2]);
                else:
                    contours[zIndex][roiNumber] = [node[:,:2]];
            print(contours[zIndex].keys());
        except Exception as e:
            print(e)

    #contours is dictionary of dictionaries
    return contours

def post_process_mask(bodyMask,labelMask):
    """
    Convert anything that is not labels to something else
    """
    newIndex = 1;
    #no organ has been labeled, then label everything inside body as body
    if labelMask.max()==0:
        labelMask[bodyMask==1] = newIndex;
    else:
        #there has been some organs labeled, label anything that is inside body and not organ as other, 
        undefinedMask = (labelMask==0);
        mask = bodyMask * undefinedMask;
        labelMask[mask.astype('bool')] = newIndex;
    # return labelMask;
  

def create_contour_mask(contours,slices):

    imgData = np.stack([standardize_slice(s) for s in slices], axis=-1);
    imgData = np.transpose(imgData,[2,0,1]);
    zSlices = [float('{0:.3f}'.format(s.ImagePositionPatient[2])) for s in slices];
    zStruct = sorted(contours.keys());

    #information about patient positioning
    posR = slices[0].ImagePositionPatient[1]
    spacingR = slices[0].PixelSpacing[1]
    posC = slices[0].ImagePositionPatient[0]
    spacingC = slices[0].PixelSpacing[0]

    #imaging size
    imgRows = slices[0].Rows;
    imgColumns = slices[0].Columns;
    imgBits = slices[0].BitsStored;

    #create image mask to ztore data
    ax = None;
    organSlices = np.zeros((len(zSlices)),dtype=np.bool);

    imgLabel = np.zeros((len(zSlices),imgRows,imgColumns),dtype=np.uint8);
    imgBody = np.zeros((len(zSlices),imgRows,imgColumns),dtype=np.uint8);
    for zIndex, zDistance in enumerate(zSlices):
        print("Masking contours ",list(contours[zDistance].keys()));
        #access contours in given slice
        for organ in contours[zDistance]:
            for contour in contours[zDistance][organ]:
                r = (contour[:, 1] - posR) / spacingR
                c = (contour[:, 0] - posC) / spacingC
                rr, cc = polygon(r, c)

                if organ!=1:
                    imgLabel[zIndex, rr, cc] = organ;
                else:
                    imgBody[zIndex, rr, cc] = 1;

        #post process current mask
        post_process_mask(imgBody[zIndex,...],imgLabel[zIndex,...]);

        #ax = debug_image_slice(imgLabel[zIndex,...],imgData[zIndex,...],ax);
        #if slice contains only body, increment start idx    
        if np.max(imgLabel[zIndex,...])>1:
            organSlices[zIndex] = True;

    return imgData[organSlices,...],imgLabel[organSlices,...]
    #return imgData,imgLabel,imgBody


def save_image_mask(features, labels, fileName):

    #create a hdf file to store organ
    if os.path.exists(fileName): #open file to modify
        print("Appending to dataset ... ")

        #create dataset to store images and masks
        with h5py.File(fileName, "r+") as organFile:

            #get original dataset size
            nFeatures = organFile["features"].shape[0];
            nLabels = organFile["labels"].shape[0];
            
            #resize dataset size
            organFile["features"].resize(nFeatures+features.shape[0],axis=0);
            organFile["labels"].resize(nLabels+labels.shape[0],axis=0);

            #now append new data to the end of hdf file
            organFile["features"][-features.shape[0]:] = features;
            organFile["labels"][-labels.shape[0]:] = labels;

            assert(organFile["features"].shape==organFile["labels"].shape);

    else: #create new file
        print("Creating new dataset...");

        #create dataset to store images and masks
        with h5py.File(fileName, "w") as organFile:
            organFile.create_dataset("features", data=features, maxshape=(None,)+features.shape[1:], dtype=np.uint16);
            organFile.create_dataset("labels", data=labels, maxshape=(None,)+labels.shape[1:], dtype=np.uint8);
    
    return fileName



def debug_image_slice(label,image,ax):
    if ax is None:
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 8));
        ax = axes.ravel();

    print(set(label.ravel()));
    ax[0].imshow(label,cmap="gray");
    ax[1].imshow(image,cmap="gray");
    #plt.axis('off')
    plt.tight_layout();
    plt.pause(0.1);
    return ax

def normalize_image(img):
    #get 0th and 99th percentile
    p0, p99 = np.percentile(img.ravel(), (0, 99.9));
    #normalize image 
    imgRescaled = exposure.rescale_intensity(img, in_range=(p0, p99));
    return imgRescaled;

def standardize_slice(imgSlice,minHU=-1000, maxHU=3000):

    #1. convert pixel data to HU
    slope = imgSlice.RescaleSlope; 
    intercept = imgSlice.RescaleIntercept;
    sliceHU = imgSlice.pixel_array*slope+intercept;

    #print("Before clipping ",sliceHU.max(), sliceHU.min(), sliceHU.dtype);
    #2 clip HU between [-1000, 3000]
    sliceHU = (sliceHU.clip(minHU,maxHU)+1000).astype('uint16');
    #3. normalize image between 0, and 1
    #print("After clipping ",sliceHU.max(), sliceHU.min(), sliceHU.dtype);

    return sliceHU


def test_random_images(fileName):

    with h5py.File(fileName,"r") as organFile:
        data = organFile["features"];
        labels = organFile["labels"];

        idx, n = 0, labels.shape[0];

        while True:

            print(labels[idx%n].max(), labels[idx%n].min())
            #labels
            image = data[idx%n,...].astype("float32")/data[idx%n,...].max();
            label = labels[idx%n,...].astype("float32")/labels[idx%n,...].max();

            imgStack = np.hstack([image,label]);
            cv2.imshow("Test",imgStack);
            k=cv2.waitKey(0);

            if k==ord("d"): idx+=1;
            elif k==ord("a"): idx-=1;
            elif k==27: break   


def compile_dataset(trainDataPath = "../TrainData/",fileName = "TRAIN.h5"):

    trainPatients = [os.path.join(trainDataPath, name) 
	   for name in os.listdir(trainDataPath) if os.path.isdir(os.path.join(trainDataPath, name))];

    # extract subsample
    trainPatients = [patient for patient in trainPatients if 'HEADNECK' in patient];

    print(trainPatients)

    for patient in trainPatients:
        print(patient)
        for subdir, dirs, files in os.walk(patient):
            dcms = glob.glob(os.path.join(subdir, "*.dcm"))
            if len(dcms) == 1:
                structure = pydicom.read_file(os.path.join(subdir, files[0]))
                contours = read_structures(structure)
            elif len(dcms) > 1:
                slices = [pydicom.read_file(dcm) for dcm in dcms]
                slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))

        #declare new original spacing
        #new_spacing = np.array([slices[0].PixelSpacing[0], slices[0].PixelSpacing[1], slices[0].PixelSpacing[2]])                

        features,labels = create_contour_mask(contours,slices);

        #remove if patient doesn't contain all labels
        if len(np.unique(labels))==NUMCLASSES+1:        
            save_image_mask(features, labels, fileName)
        else:
            print("Skipping to save...",patient)


def save_train_stats(trainFileName):

    fp = h5py.File(trainFileName, "r");
    data = fp["features"];
    data_crop = np.zeros((data.shape[0],W,H));
    batch = 32;
    #apply bounding box preprocesing for each image before stats
    for idx in range(0,data.shape[0],batch):
        data_crop[idx:idx+batch] = crop_body_roi(data[idx:idx+batch],None);
        if idx%256==0:
            print("Processing batch ",idx)
    print("Completed cropping!")
    #create dask array for further processing
    data_da = da.from_array(data_crop,chunks=(4,W,H)) #parallelize

    #label = fp["labels"][:];
    #label_da = da.from_array(label,chunks=(4,512,512)) 
    
    #compute class weights as follows: 
	#   first, calculate how many pixels are in each image and then average over all slices
    #classFreq = {idx:(label_da==idx).sum(axis=(1,2)).mean().compute() for idx in range(NUMCLASSES)}
    #maxValue = max(classFreq.values())
    #classWeights = {key:maxValue/classFreq[key] for key in classFreq}
    #print(classWeights)
    

    #Mask out zero measurements, air and recompute mean and variance
    nonZero = da.ma.masked_equal(data_da,0);
    mean = nonZero.mean(axis=0,keepdims=True);
    variance = nonZero.std(axis=0,keepdims=True);
    print("Computed masked array!");

    #now store them in new file
    with h5py.File(fileName.replace(".h5","_STAT.h5"), "w") as newFile:
        newFile.create_dataset("means", data=mean.compute(), dtype=np.float32);
        newFile.create_dataset("variance", data=variance.compute(), dtype=np.float32);
    print("Computed variance and mean array!");

import time
def save_label_idx_map(trainFileName):
    #get labels
    labels = h5py.File(trainFileName,"r")["labels"];
    labels_da = da.from_array(labels,chunks=(4,512,512));
    
    label_idx_map = {}    
    #count number of occurences for each label
    for idx in range(1,NUMCLASSES):
        start = time.time()
        X,Y,Z = da.where(labels_da==idx)
        label_idx_map[idx] = da.unique(X).compute(); 
        print("Finished label {0} in {1:.3f} s".format(idx,time.time()-start));    

    with h5py.File(fileName.replace(".h5","_IDX_MAP.h5"),"w") as newFile:
        for idx in range(1,NUMCLASSES):
            newFile.create_dataset(str(idx), data=label_idx_map[idx], dtype=np.int16);
    
    
if __name__=="__main__":
    #trainDataPath = "../TestData/";
    #fileName = "TEST.h5";
    
    #trainDataPath = "../TrainData/";
    fileName = "TRAIN.h5";

    #compile_dataset(trainDataPath,fileName)
    save_train_stats(fileName)
    #test_random_images(fileName)
    #save_label_idx_map(fileName)

