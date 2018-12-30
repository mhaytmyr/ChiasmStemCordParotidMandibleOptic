
import keras as K
import tensorflow as tf
#from keras import backend as K
from keras.engine.topology import Layer
from keras.callbacks import Callback
#from keras.backend import tf as ktf
from keras.utils import conv_utils
from keras.engine import InputSpec


class Metrics(Callback):
    def __init__(self,generator, steps, batch):
        self.generator = generator;
        self.steps = steps;
        self.batch = batch;

    def on_train_begin(self,logs={}):
        n = self.params["metrics"].__len__();
        self.val_score_names = self.params["metrics"][n//2:];
        self.train_score_names = self.params["metrics"][:n//2];
        
        self.train_metrics = {item:0 for item in self.train_score_names if 'loss' not in item};
        self.batch_accumilator = {item:0 for item in self.train_score_names if 'loss' not in item};

        print(self.val_score_names)
        print(self.train_score_names)

    def on_epoch_begin(self,epoch,logs={}):
        #reset values to zero
        for key in self.train_metrics:
            self.train_metrics[key] = 0;
            self.batch_accumilator[key] = 0;

    def on_batch_end(self,batch,logs={}):

        #now perform weighets sum
        for key in self.train_metrics:
            #don't update results that are zero
            if logs[key]>0:
                #update batch accumilator
                self.batch_accumilator[key] += logs["size"]; 
                self.train_metrics[key] += logs["size"]*logs[key];
            
    def on_epoch_end(self,epoch,logs={}):

        print("end of epochs");
        print("original scores ....")
        #print(logs)
        self.train_scores = {key:logs[key] for key in self.train_score_names if 'loss' not in key};
        print(self.train_scores)
        
        print("manual calculation ....")        
        self.train_scores = {key:self.train_metrics[key]/max(1,self.batch_accumilator[key]) for key in self.train_metrics};
        print(self.train_scores)    

        #assign updated scores to log
        for key in self.train_scores:
            logs[key] = self.train_scores[key]

class LRN(Layer):
    def __init__(self, alpha=0.0001,k=1,beta=0.75,n=5, **kwargs):
        self.alpha = alpha
        self.k = k
        self.beta = beta
        self.n = n
        super(LRN, self).__init__(**kwargs)

    def call(self, x, mask=None):
        b, ch, r, c = x.shape
        half_n = self.n // 2 # half the local region
        input_sqr = T.sqr(x) # square the input
        extra_channels = T.alloc(0., b, ch + 2*half_n, r, c) # make an empty tensor with zero pads along channel dimension
        input_sqr = T.set_subtensor(extra_channels[:, half_n:half_n+ch, :, :],input_sqr) # set the center to be the squared input
        scale = self.k # offset for the scale
        norm_alpha = self.alpha / self.n # normalized alpha
        for i in range(self.n):
            scale += norm_alpha * input_sqr[:, i:i+ch, :, :]
        scale = scale ** self.beta
        x = x / scale
        return x

    def get_config(self):
        config = {"alpha": self.alpha,"k": self.k,"beta": self.beta,"n": self.n}
        base_config = super(LRN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class BilinearInterpolation(Layer):
    """Performs bilinear interpolation as a keras layer
    References
    ----------
    [1]  Spatial Transformer Networks, Max Jaderberg, et al.
    [2]  https://github.com/skaae/transformer_network
    [3]  https://github.com/EderSantana/seya
    """

    def __init__(self, output_size, invert_matrix, **kwargs):
        self.output_size = output_size
        self.invert = invert_matrix;
        super(BilinearInterpolation, self).__init__(**kwargs)

    def compute_output_shape(self, input_shapes):
        height, width = self.output_size
        num_channels = input_shapes[0][-1]
        return (None, height, width, num_channels)

    def call(self, tensors, mask=None):
        X, transformation = tensors
        output = self._transform(X, transformation, self.output_size)
        return output

    def get_config(self):
        config = {'output_size': self.output_size,'invert_matrix':self.invert}
        base_config = super(BilinearInterpolation, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def _interpolate(self, image, sampled_grids, output_size):

        batch_size = K.backend.shape(image)[0]
        height = K.backend.shape(image)[1]
        width = K.backend.shape(image)[2]
        num_channels = K.backend.shape(image)[3]

        x = K.backend.cast(K.backend.flatten(sampled_grids[:, 0:1, :]), dtype='float32')
        y = K.backend.cast(K.backend.flatten(sampled_grids[:, 1:2, :]), dtype='float32')

        x = .5 * (x + 1.0) * K.backend.cast(height, dtype='float32')
        y = .5 * (y + 1.0) * K.backend.cast(width, dtype='float32')

        x0 = K.backend.cast(x, 'int32')
        x1 = x0 + 1
        y0 = K.backend.cast(y, 'int32')
        y1 = y0 + 1

        max_x = int(K.backend.int_shape(image)[1] - 1)
        max_y = int(K.backend.int_shape(image)[2] - 1)

        x0 = K.backend.clip(x0, 0, max_x)
        x1 = K.backend.clip(x1, 0, max_x)
        y0 = K.backend.clip(y0, 0, max_y)
        y1 = K.backend.clip(y1, 0, max_y)

        pixels_batch = K.backend.arange(0, batch_size) * (height * width)
        pixels_batch = K.backend.expand_dims(pixels_batch, axis=-1)
        flat_output_size = output_size[0] * output_size[1]
        base = K.backend.repeat_elements(pixels_batch, flat_output_size, axis=1)
        base = K.backend.flatten(base)

        # base_y0 = base + (y0 * width)
        base_y0 = y0 * width
        base_y0 = base + base_y0
        # base_y1 = base + (y1 * width)
        base_y1 = y1 * width
        base_y1 = base_y1 + base

        indices_a = base_y0 + x0
        indices_b = base_y1 + x0
        indices_c = base_y0 + x1
        indices_d = base_y1 + x1

        flat_image = K.backend.reshape(image, shape=(-1, num_channels))
        flat_image = K.backend.cast(flat_image, dtype='float32')
        pixel_values_a = K.backend.gather(flat_image, indices_a)
        pixel_values_b = K.backend.gather(flat_image, indices_b)
        pixel_values_c = K.backend.gather(flat_image, indices_c)
        pixel_values_d = K.backend.gather(flat_image, indices_d)

        x0 = K.backend.cast(x0, 'float32')
        x1 = K.backend.cast(x1, 'float32')
        y0 = K.backend.cast(y0, 'float32')
        y1 = K.backend.cast(y1, 'float32')

        area_a = K.backend.expand_dims(((x1 - x) * (y1 - y)), 1)
        area_b = K.backend.expand_dims(((x1 - x) * (y - y0)), 1)
        area_c = K.backend.expand_dims(((x - x0) * (y1 - y)), 1)
        area_d = K.backend.expand_dims(((x - x0) * (y - y0)), 1)

        values_a = area_a * pixel_values_a
        values_b = area_b * pixel_values_b
        values_c = area_c * pixel_values_c
        values_d = area_d * pixel_values_d
        return values_a + values_b + values_c + values_d

    def _make_regular_grids(self, batch_size, height, width):
        # making a single regular grid
        x_linspace = K_linspace(-1., 1., width)
        y_linspace = K_linspace(-1., 1., height)
        x_coordinates, y_coordinates = K_meshgrid(x_linspace, y_linspace)
        x_coordinates = K.backend.flatten(x_coordinates)
        y_coordinates = K.backend.flatten(y_coordinates)
        ones = K.backend.ones_like(x_coordinates)
        grid =K.backend.concatenate([x_coordinates, y_coordinates, ones], 0)

        # repeating grids for each batch
        grid = K.backend.flatten(grid)
        grids = K.backend.tile(grid, K.backend.stack([batch_size]))
        return K.backend.reshape(grids, (batch_size, 3, height * width))

    def _invert_transformation(self,M):
        """
        Inverts affine tranfroamtion invert_matrix
        Args:
            M: tranfroamtion matrix with (BATCHSIZE, 2, 3)

        returns:
            inverted tranformation with same dimensions
        """
        det = M[:,0,0]*M[:,1,1]-M[:,0,1]*M[:,1,0];
        b1 = M[:,0,1]*M[:,1,2]-M[:,1,1]*M[:,0,2];
        b2 = M[:,1,0]*M[:,0,2]-M[:,0,0]*M[:,1,2];

        affine = tf.linalg.inv(M[:,:2,:2]);
        trans = tf.stack([K.backend.transpose(b1/det), K.backend.transpose(b2/det)],axis=1);
        trans = tf.expand_dims(trans,axis=-1);
        return K.backend.concatenate([affine, trans]);

    def _transform(self, X, affine_transformation, output_size):
        batch_size, num_channels = K.backend.shape(X)[0], K.backend.shape(X)[3]
        transformations = K.backend.reshape(affine_transformation,
                                    shape=(batch_size, 2, 3))

        if self.invert==1:
            print("Affine inverted")
            transformations = self._invert_transformation(transformations);
        else:
            print("Affine not invert")

        # transformations = K.cast(affine_transformation[:, 0:2, :], 'float32')
        regular_grids = self._make_regular_grids(batch_size, *output_size)
        sampled_grids = K.backend.batch_dot(transformations, regular_grids)
        interpolated_image = self._interpolate(X, sampled_grids, output_size)
        new_shape = (batch_size, output_size[0], output_size[1], num_channels)
        interpolated_image = K.backend.reshape(interpolated_image, new_shape)

        return interpolated_image


class MorpholocialOperations(Layer):
    def __init__(self,output_size, morph_opt, **kwargs):
        self.output_size = output_size;
        self.morph_opt = morph_opt; #can be erosion, dilation, opening or closing
        super(MorpholocialOperations,self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        pass
    def call(self, inputs):
        #apply morphology here
        pass

    def get_config(self):
        config = {'morph_opt': self.morph_opt,
                'output_size': self.output_size}

        base_config = super(MorpholocialOperations, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class BilinearUpsampling(Layer):
    def __init__(self, upsampling=(2, 2), output_size=None, data_format=None, **kwargs):
        super(BilinearUpsampling, self).__init__(**kwargs)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.input_spec = InputSpec(ndim=4)
        if output_size:
            self.output_size = conv_utils.normalize_tuple(output_size, 2, 'output_size')
            self.upsampling = None
        else:
            self.output_size = None
            self.upsampling = conv_utils.normalize_tuple(upsampling, 2, 'upsampling')

    def compute_output_shape(self, input_shape):
        if self.upsampling:
            height = self.upsampling[0] * input_shape[1] if input_shape[1] is not None else None
            width = self.upsampling[1] * input_shape[2] if input_shape[2] is not None else None
        else:
            height = self.output_size[0]
            width = self.output_size[1]

        return (input_shape[0],height,width,input_shape[3])

    def call(self, inputs):
        if self.upsampling:
            return tf.image.resize_bilinear(inputs,
                                (inputs.shape[1] * self.upsampling[0],inputs.shape[2] * self.upsampling[1]),
                                align_corners=True)
        else:
            return tf.image.resize_bilinear(inputs,
                                (self.output_size[0],self.output_size[1]),
                                align_corners=True)

    def get_config(self):
        config = {'upsampling': self.upsampling,
                'output_size': self.output_size,
                'data_format': self.data_format}

        base_config = super(BilinearUpsampling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def K_meshgrid(x, y):
    return tf.meshgrid(x, y)

def K_linspace(start, stop, num):
    return tf.linspace(start, stop, num)

def relu6(x):
    return K.activations.relu(x, max_value=6)
