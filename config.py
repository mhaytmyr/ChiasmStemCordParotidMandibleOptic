import numpy as np

########################
#### CONFIG ResNet #####
FILTER1 = 16;#24
FILTER2 = 32;#24
FILTER3 = 64;#32
FILTER4 = 128;#64
FILTER5 = 256;#128
FILTER6 = 512
FILTER7 = 1024;
BATCHSIZE = 8;
NUMEPOCHS = 100;
NUMCLASSES = 7;
L2PENALTY = 0.0001;
LEARNRATE = 0.0001#0.0001
TRAINSIZE, VALSIZE = 7380, 1677;
STEPPEREPOCHS = int(TRAINSIZE/BATCHSIZE); 
VALSTEPS = int(VALSIZE/BATCHSIZE); 
DECAYRATE = 1/(STEPPEREPOCHS*NUMEPOCHS);
#CLASSWEIGHTS = {'bck': 1.005679488718510, 'chiasm': 33265.33, 'bsteam': 1047.53, 'scord': 1125.72, 
#		'parotid': 609.43, 'mandible':478.34, 'onerve':23286.27} # (total_for_all_categories/total_for_category)
CLASSWEIGHTS = np.array([1, 10.41, 6.954, 7.03, 6.413, 6.17, 10.056]); #logarithm of above numbers

#image crop indeces
ROW, COL = 115,54;
H,W,C = 384,256,1
H0,W0,C0 = 512,512,1


#using regular model with loss function
#modelName = "1x256x384_UNet_3D16_3D32_3D64_3D128_3D256_3D512_1C1024"

#training batch is equilized; each batch contains at least one class
#modelName = "1x256x384_Stratified_3D16_3D32_3D64_3D128_3D256_3D512_1C1024"

#training batch is equilized; each batch contains at least one class
modelName = "1x256x384_Strat_WeightedLoss_3D16_3D32_3D64_3D128_3D256_3D512_1C1024"


########################

