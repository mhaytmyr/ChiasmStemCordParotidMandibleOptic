import numpy as np, sys
import cv2, h5py, dask
import dask.array as da
import dask
from scipy import ndimage
from config import *
from keras.utils.np_utils import to_categorical
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates

def crop_image_roi(img):
    """
    Crop image with hard-coded ROI
    """
    if len(img.shape)==2:
        return img[ROW:ROW+W,COL:COL+H]
    elif len(img.shape)==3:
        return img[:,ROW:ROW+W,COL:COL+H]
    elif len(img.shape)==4:
        return img[:,ROW:ROW+W,COL:COL+H,:]
    else:
        sys.exit("compile_dcm_images.crop_image_roi: \n Img size must be 2,3 or 4")
        

def body_bounding_box(img):
    """
    Finds bounding box of a given image, need to apply this for tensor
    """
    #define kernel
    SIZE1 = 11; SIZE2 = 27;    
    kernel1 = np.ones((SIZE1,SIZE1),np.uint8);
    kernel2 = np.ones((SIZE2,SIZE2),np.uint8);

    #calculate center points
    row_mean = img.mean(axis=1);
    rows = np.arange(img.shape[0]);
    row_center = np.int((rows*row_mean).sum()//row_mean.sum());
    col_mean = img.mean(axis=0);
    cols = np.arange(img.shape[1]);
    col_center = np.int((cols*col_mean).sum()//col_mean.sum());

    #create label img
    label = np.zeros((W0,H0),dtype=np.uint8);
    body = label.copy()

    #first binarize image
    label[img>200] = 1;

    #add white spot at the center
    img[row_center-5:row_center+5,col_center-5:col_center+5] = img.max();

    #apply morphology
    body[:row_center] = cv2.morphologyEx(label[:row_center], cv2.MORPH_OPEN, kernel1);
    body[row_center:] = cv2.morphologyEx(label[row_center:], cv2.MORPH_ERODE, kernel2);

    #compute non zeros
    rows = np.any(body, axis=1)
    cols = np.any(body, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]

    #apply margin
    ymin, ymax = max(0,ymin-SIZE1//2), ymax+SIZE2//2;
    xmin, xmax = max(0,xmin-SIZE1//2), xmax+SIZE1//2; 
    
    #return crop positions
    return (ymin,ymax),(xmin,xmax)


def crop_body_roi(imgInput):
    """
    Function to dynamically crop ROI of image or label
    """
    #remove axis
    img = imgInput.squeeze();

    #first check dimensions of image
    if len(img.shape)==2:
        rows, cols = body_bounding_box(imgInput);
        imgCrop = imgInput[rows[0]:rows[1],cols[0]:cols[1]];
        imgZoom = cv2.resize(imgCrop,(H,W));
        return imgZoom

    elif len(img.shape)==3:
        N = img.shape[0];
        cropImg = np.zeros((N,W,H));

        for idx in range(N):
            rows, cols = body_bounding_box(imgInput[idx]);
            imgCrop = imgInput[idx,rows[0]:rows[1],cols[0]:cols[1]];
            imgZoom = cv2.resize(imgCrop,(H,W));
            cropImg[idx] = imgZoom;
        return cropImg
    else:
        sys.exit("compile_dcm_images.crop_body_roi: \n Img size must be (H,W) or (N,H,W)")


def pre_process_img(imgInput,normalize,removeAir=True):
    """
    Normalizes image in ROI
    """
    #crop body first
    #imgInput = crop_body_roi(imgInput);

    if removeAir:
        nonZero = np.ma.masked_equal(imgInput,0);
        normalized = ((nonZero-normalize["means"])/normalize["vars"]).data;
    else:
        normalized = (imgInput-normalize["means"])/normalize["vars"];

    #crop image
    normalized = crop_image_roi(normalized);

    return normalized.astype("float32");

from compile_dcm_images import remove_body_mask
def pre_process_label(label,organToSegment=None):
    #crop image for roi
    #label = crop_image_roi(label);
    label = crop_body_roi(label)

    #remove body mask
    remove_body_mask(label)

    #Add preprocessing for labels, applies on training stage
    if organToSegment:
        label[label!=organToSegment] = 0;
        label[label==organToSegment] = 1;

    return label.astype("float32");
    

# Define function to draw a grid
def draw_grid(im, grid_size):
    # Draw grid lines
    for i in range(0, im.shape[1], grid_size):
        cv2.line(im, (i, 0), (i, im.shape[0]), color=(1,))
    for j in range(0, im.shape[0], grid_size):
        cv2.line(im, (0, j), (im.shape[1], j), color=(1,))


def elastic_transform(image, alpha, sigma, alpha_affine, random_state=None):
    """
     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))

    return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)

def augment_data(img,organ):
    """
    Augment data using horizontla flip, elastic deformation and random zooming
    """

    #copy image to memory, hdf5 doesn't allow inplace
    #image are already loaded to memory using dask.array
    imgNew = img;
    organNew = organ;

    #get image info
    n,row,col = img.shape;

    for idx in range(n):	
        choice = np.random.choice(['flip','nothing','deform','zoom','zoom','nothing']);

        if choice=='flip':
            img[idx,...] = imgNew[idx,:,::-1];
            organ[idx,...] = organNew[idx,:,::-1];
        elif choice=='rotate':
            img[idx,...] = imgNew[idx,::-1,:];
            organ[idx,...] = organNew[idx,::-1,:];
        elif choice=='zoom':
            zoomfactor = np.random.randint(11,21)/10;
            dx = np.random.randint(-20,20);
            dy = np.random.randint(-20,20);
            M_zoom = cv2.getRotationMatrix2D((row/2+dx,col/2+dy), 0, zoomfactor)
        
            img[idx,...] = cv2.warpAffine(imgNew[idx,...], M_zoom,(col,row))
            organ[idx,...] = cv2.warpAffine(organNew[idx,...], M_zoom,(col,row))

        elif choice=='deform':
            #draw_grid(imgNew[idx,...], 50)
            #draw_grid(organNew[idx,...], 50)
            
            #combine two images
            merged = np.dstack([imgNew[idx,...], organNew[idx,...]]);
            #apply transformation
            mergedTrans = elastic_transform(merged, merged.shape[1] * 3, merged.shape[1] * 0.08, merged.shape[1] * 0.08)
            #now put images back
            img[idx,...] = mergedTrans[...,0];
            organ[idx,...] = mergedTrans[...,1:];

    return img,organ

from queue import Queue
import time
def data_generator_stratified(hdfFileName,batchSize=50,augment=True,normalize=None):

    #create place holder for image and label batch
    img_batch = np.zeros((batchSize,H0,W0),dtype=np.float32);
    label_batch = np.zeros((batchSize,H0,W0),dtype=np.float32);
    
    #get pointer to features and labels
    hdfFile = h5py.File(hdfFileName,"r");
    features = hdfFile["features"];        
    labels = hdfFile["labels"];

    #create dask array for efficienct access    
    daskFeatures = dask.array.from_array(features,chunks=(4,H0,W0));
    daskLabels = dask.array.from_array(labels,chunks=(4,H0,W0));

    #create queue for keys
    label_queue = Queue();
        
    #create dictionary to store queue indices
    label_idx_map = {}
    #(no need to shuffle data?), add each index to queue
    with h5py.File(hdfFileName.replace(".h5","_IDX_MAP.h5"),"r") as fp:
        for key in fp.keys():
            label_queue.put(key)
            label_idx_map[key] = Queue();
            for item in fp[key]:
                label_idx_map[key].put(item);

    #yield batches
    while True:
        #start = time.time()
        for n in range(batchSize):
            #get key from keys queue
            key = label_queue.get();
            #get corresponding index
            index = label_idx_map[key].get();            
            #append them to img_batch and label_batch
            img_batch[n] = daskFeatures[index].compute();
            label_batch[n] = daskLabels[index].compute();

            #circulate queue
            label_queue.put(key);
            label_idx_map[key].put(index);

        #debug queue
        #print("{0:.3f} msec took to generate {1} batch".format((time.time()-start)*1000,batchSize))
        #print(label_idx_map["2"].queue);

        #apply pre-processing operations
        feature = pre_process_img(img_batch,normalize);
        organ = pre_process_label(label_batch);

        #convert to one-hot encoded
        organ = to_categorical(organ,num_classes=NUMCLASSES).reshape((-1,W,H,NUMCLASSES));          

        #augment data
        if augment:
            feature,organ = augment_data(feature,organ);

        #create generator
        #yield (feature[...,np.newaxis],{'organ_output':organ});

        #yield data 
        yield (feature[...,np.newaxis], {'organ_output':organ})


def data_generator(hdfFileName,batchSize=50,augment=True,shuffle=True,normalize=None):

    #yield data with or w/o augmentation
    with h5py.File(hdfFileName,"r") as hdfFile:

        #initialize pointer
        idx,n = 0, hdfFile["features"].shape[0];
        indices = np.arange(n);
        #shuffle indices
        if shuffle:
            np.random.shuffle(indices);

        while True:
            start = idx;
            end = (idx+batchSize);
        
            if idx>=n:
                #shuffle indices after each epoch
                if shuffle: 
                    np.random.shuffle(indices);

                slice = np.arange(start,end);
                subIndex = sorted(indices[slice%n]);
                idx = end%n;

                #get data    
                feature = pre_process_img(hdfFile["features"][subIndex,...],normalize);
                organ = pre_process_label(hdfFile["labels"][subIndex,...]);
            else:
                #increment counter
                idx+=batchSize;

                if shuffle:
                    subIndex = sorted(indices[start:end]);
                    feature = pre_process_img(hdfFile["features"][subIndex,...],normalize);
                    organ = pre_process_label(hdfFile["labels"][subIndex,...]);
                else:
                    feature = pre_process_img(hdfFile["features"][start:end,...],normalize);
                    organ = pre_process_label(hdfFile["labels"][start:end,...]);

            #convert to one-hot encoded
            organ = to_categorical(organ,num_classes=NUMCLASSES).reshape((-1,W,H,NUMCLASSES));          

            #augment data
            if augment:
                feature,organ = augment_data(feature,organ);

            #create generator
            yield (feature[...,np.newaxis],{'organ_output':organ});
            
