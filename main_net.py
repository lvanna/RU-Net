from __future__ import print_function

import os
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
import keras
import argparse
import matplotlib
import sys
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import pickle
from keras.models import Model, load_model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Dense, add, Activation, GlobalAveragePooling2D, Dropout
from keras import regularizers
from keras.optimizers import Adam
from keras.callbacks import  LearningRateScheduler, TensorBoard, ModelCheckpoint
from keras import backend as K
from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
# from keras.layers.normalization import BatchNormalization
from tensorflow.keras.layers import BatchNormalization
from sklearn.model_selection import train_test_split
from keras.initializers import Constant

from data import load_train_data, load_test_data
# from lr_finder import LRFinder
from torch_lr_finder import LRFinder
# from newlayer import MaxPoolingWithArgmax2D, MaxUnpooling2D
from keras.layers import Layer
import PyQt5

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

img_rows            = 256
img_cols            = 256
channels            = 1
smooth              = 1.
stack_n             = np.array([1,1,1,1,1,1,1,1,1,1])
weight_decay        = 1e-4
num_classes         = 1
batch_size          = 3
# epochs              = 100
epochs              = 3
val_split           = 0.2
initial_filter      = 32*2
seed                = 7
expand_times        = 10

if len(sys.argv) > 1:
    input_type = sys.argv[1] 
else:
    input_type = "infarct" 
    
test                = 55
filename            = '{:s}'.format(input_type)
weight_name         = 'weight_{:s}_Final_test.h5'.format(input_type)


class MaxPoolingWithArgmax2D(Layer):
    def __init__(self, pool_size=(2, 2), strides=(2, 2), padding='same', **kwargs):
        super(MaxPoolingWithArgmax2D, self).__init__(**kwargs)
        self.padding = padding
        self.pool_size = pool_size
        self.strides = strides

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            "pool_size": self.pool_size,
            "strides": self.strides,
            "padding": self.padding,
        })
        return config

    def call(self, inputs, **kwargs):
        padding = self.padding
        pool_size = self.pool_size
        strides = self.strides
        if K.backend() == 'tensorflow':
            ksize = [1, pool_size[0], pool_size[1], 1]
            padding = padding.upper()
            strides = [1, strides[0], strides[1], 1]
            output, argmax = K.tf.nn.max_pool_with_argmax(inputs, ksize=ksize, strides=strides, padding=padding)
        else:
            errmsg = '{} backend is not supported for layer {}'.format(K.backend(), type(self).__name__)
            raise NotImplementedError(errmsg)
        argmax = K.cast(argmax, K.floatx())
        return [output, argmax]

    def compute_output_shape(self, input_shape):
        ratio = (1, 2, 2, 1)
        output_shape = [dim // ratio[idx] if dim is not None else None for idx, dim in enumerate(input_shape)]
        output_shape = tuple(output_shape)
        return [output_shape, output_shape]

    def compute_mask(self, inputs, mask=None):
        return 2 * [None]

class MaxUnpooling2D(Layer):
    def __init__(self, size=(2, 2), **kwargs):
        super(MaxUnpooling2D, self).__init__(**kwargs)
        self.size = size

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            "size": self.size,
        })
        return config

    def call(self, inputs, output_shape=None):
        updates, mask = inputs[0], inputs[1]
        # with K.tf.variable_scope(self.name):
        with K.tf.compat.v1.variable_scope(self.name):
            mask = K.cast(mask, 'int32')
            input_shape = K.tf.shape(updates, out_type='int32')
            #  calculation new shape
            if output_shape is None:
                output_shape = (input_shape[0], input_shape[1] * self.size[0], input_shape[2] * self.size[1], input_shape[3])
                self.output_shape1 = output_shape

        # calculation indices for batch, height, width and feature maps
        one_like_mask = K.ones_like(mask, dtype='int32')
        batch_shape = K.concatenate([[input_shape[0]], [1], [1], [1]], axis=0)
        batch_range = K.reshape(K.tf.range(output_shape[0], dtype='int32'), shape=batch_shape)
        b = one_like_mask * batch_range
        y = mask // (output_shape[2] * output_shape[3])
        x = (mask // output_shape[3]) % output_shape[2]
        feature_range = K.tf.range(output_shape[3], dtype='int32')
        f = one_like_mask * feature_range

        # transpose indices & reshape update values to one dimension
        updates_size = K.tf.size(updates)
        indices = K.transpose(K.reshape(K.stack([b, y, x, f]), [4, updates_size]))
        values = K.reshape(updates, [updates_size])
        ret = K.tf.scatter_nd(indices, values, output_shape)
        return ret

    def compute_output_shape(self, input_shape):
        mask_shape = input_shape[1]
        return mask_shape[0], mask_shape[1] * self.size[0], mask_shape[2] * self.size[1], mask_shape[3]

def history_show(logs):
    # dice_coef, val_loss, loss, val_dice_coef
    # Accuracy
    accy = logs.history['dice_coef']
    accy_val = logs.history['val_dice_coef']
    plt.plot(accy)
    plt.plot(accy_val)
    plt.title('model accuracy(batch:{:d})'.format(batch_size))
    plt.ylabel('dice')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig('data/proposed/{:s}/acc_log_{:s}.png'.format(input_type,filename), bbox_inches='tight')
    # plt.show()
    plt.close()

    # Loss
    loss = logs.history['loss']
    loss_val = logs.history['val_loss']
    plt.plot(loss)
    plt.plot(loss_val)
    plt.title('model loss(batch:{:d})'.format(batch_size))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    print('heeeeee1')
    plt.savefig('data/proposed/{:s}/loss_log_{:s}.png'.format(input_type,filename), bbox_inches='tight')
    # plt.show()
    print('heeeeee2')
    plt.close()
    print('heeeeee3')

    # save acc in text    
    first_row = 'epoch,dice_coef,val_dice_coef,loss,val_loss'
    file_path = 'data/proposed/{:s}/acc_loss_{:s}.csv'.format(input_type,filename)
    print('heeeeee4')

    with open(file_path, 'w+') as f:
        f.write(first_row + '\n')
        for i in range(len(accy)):
            s = str(i+1) + ',' + str(accy[i]) + ',' + str(accy_val[i]) + ',' + str(loss[i]) + ',' + str(loss_val[i])
            f.write(s + '\n')

def scheduler(epoch):
    if epoch < 5: 
        return 3e-4
    if epoch < 35:
        return 2e-4
    return 1e-4

''' loss funtion '''
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

def upsample_filt(size):
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)

def bilinear_upsample_weights(factor, number_of_classes):
    filter_size = factor*2 - factor%2
    weights = np.zeros((filter_size, filter_size, number_of_classes, number_of_classes),
                       dtype=np.float32)
    upsample_kernel = upsample_filt(filter_size)
    for i in range(number_of_classes):
        weights[:, :, i, i] = upsample_kernel
    return weights

def BN_ReLU(x):
    return Activation('relu')(BatchNormalization(momentum=0.9, epsilon=1e-5)(x))
    
def ConvD(x,o_filters,kernel,stride):
    return Conv2D(o_filters,kernel_size=kernel,strides=stride,padding='same',
                    kernel_initializer="he_normal"
                    )(x)
# kernel_regularizer=regularizers.l2(weight_decay)
def residual_block_basic(x,o_filters,increase=False,decrease=False):
    stride = (1,1)
    if increase:
        stride = (2,2)
    o1      = BN_ReLU(x)
    conv1   = ConvD(o1,o_filters,(3,3),stride)
    output  = ConvD(BN_ReLU(conv1),o_filters,(3,3),(1,1))

    if increase:
        projection = Conv2D(o_filters,kernel_size=(1,1),strides=(2,2),padding='same',
                            kernel_initializer="he_normal"
                            )(o1)
        block = add([output, projection])
    elif decrease:
        projection = Conv2D(int(o_filters),kernel_size=(1,1),strides=(1,1),padding='same',
                            kernel_initializer="he_normal"
                            )(o1)
        block = add([output, projection])
    else:
        block = add([output, x])
    return block

def up_block_basic(o1, o2, o3, o_filters,stride, n):
    conv    = MaxUnpooling2D((2,2))([o1, o2])
    marge   = concatenate([conv, o3], axis=3)
    # conv1 = ConvD(BN_ReLU(o1),o_filters,(3,3),(1,1))
    # conv2 = CoConvDnv(BN_ReLU(conv1),o_filters,(3,3),(1,1))
    # marge    = concatenate([Conv2DTranspose(o_filters, (2, 2), strides=stride, padding='same')(o1), o2], axis=3)
    conv = ConvD(BN_ReLU(marge),o_filters,(3,3),(1,1))
    # conv     = residual_block_basic(conv, o_filters, False, True)
    for _ in range(0,stack_n[n]):
        conv = residual_block_basic(conv, o_filters, False, False)

    # conv = ConvD(BN_ReLU(conv),o_filters,(3,3),(1,1))
    return conv

def residual_block_bottleneck(x,o_filters,increase=False,decrease=False):
    stride = (1,1)
    if increase:
        stride = (2,2)
    o1 = BN_ReLU(x)
    conv    = ConvD(o1,int(o_filters/4),(1,1),stride)
    conv    = ConvD(BN_ReLU(conv),int(o_filters/4),(3,3),(1,1))
    output  = ConvD(BN_ReLU(conv),o_filters,(1,1),(1,1))

    if increase:
        projection = Conv2D(o_filters,kernel_size=(1,1),strides=(2,2),padding='same',
                            kernel_initializer="he_normal"
                            )(o1)
        block = add([output, projection])
    elif decrease:
        projection = Conv2D(int(o_filters),kernel_size=(1,1),strides=(1,1),padding='same',
                            kernel_initializer="he_normal"
                            )(o1)
        block = add([output, projection])
    else:
        block = add([output, x])
    return block

def up_block_bottleneck(o1, o2, o3, o_filters,stride, n):
    conv    = MaxUnpooling2D((2,2))([o1, o2])
    marge   = concatenate([conv, o3], axis=3)
    # conv1 = ConvD(BN_ReLU(o1),o_filters,(3,3),(1,1))
    # conv2 = ConvD(BN_ReLU(conv1),o_filters,(3,3),(1,1))
    # marge    = concatenate([Conv2DTranspose(o_filters, (2, 2), strides=stride, padding='same')(o1), o2], axis=3)
    conv = ConvD(BN_ReLU(marge),o_filters,(3,3),(1,1))
    # conv     = residual_block_bottleneck(conv, o_filters, False, True)
    for _ in range(0,stack_n[n]):
        conv = residual_block_bottleneck(conv, o_filters, False, False)

    # conv = ConvD(BN_ReLU(conv),o_filters,(3,3),(1,1))
    return conv

def get_net(inputs, num_classes):
    conv1   =  ConvD(inputs,initial_filter,(3,3),(1,1))

    # conv1   = ConvD(BN_ReLU(conv1),initial_filter,(3,3),(1,1))
    # conv1 = residual_block(conv1, initial_filter,False,True)
    for _ in range(0,stack_n[0]):
        conv1 = residual_block_basic(conv1, initial_filter,False,False)
    
    ''' change pooling to stride=2 '''
    # conv1   = ConvD(BN_ReLU(conv1),initial_filter,(3,3),(1,1))
    # pool1   = MaxPooling2D(pool_size=(2, 2))(conv1)
    pool1, mask1 = MaxPoolingWithArgmax2D((2, 2))(conv1)

    # conv2   = ConvD(BN_ReLU(pool1),initial_filter,(3,3),(1,1))
    # conv2   = residual_block(pool1,initial_filter*2,False,True)
    for _ in range(0,stack_n[1]):
        conv2 = residual_block_basic(pool1,initial_filter,False,False)
    # conv2   = ConvD(BN_ReLU(conv2),initial_filter*2,(3,3),(1,1))
    # pool2   = MaxPooling2D(pool_size=(2, 2))(conv2)
    pool2, mask2 = MaxPoolingWithArgmax2D((2, 2))(conv2)

    # conv3   = ConvD(BN_ReLU(pool2),initial_filter,(3,3),(1,1))
    # conv3   = residual_block(pool2,initial_filter*2**2,False,True)
    for _ in range(0,stack_n[2]):
        conv3 = residual_block_bottleneck(pool2,initial_filter,False,False)
    # conv3   = ConvD(BN_ReLU(pool2),initial_filter*2**2,(3,3),(1,1))
    # pool3   = MaxPooling2D(pool_size=(2, 2))(conv3)
    pool3, mask3 = MaxPoolingWithArgmax2D((2, 2))(conv3)

    # conv4   = ConvD(BN_ReLU(pool3),initial_filter,(3,3),(1,1))
    # conv4   = residual_block(pool3,initial_filter*2**3,False,True)
    for _ in range(0,stack_n[3]):
        conv4 = residual_block_bottleneck(pool3,initial_filter,False,False)
    # conv4   = ConvD(BN_ReLU(conv4),initial_filter*2**3,(3,3),(1,1))
    # pool4   = MaxPooling2D(pool_size=(2, 2))(conv4)
    pool4, mask4 = MaxPoolingWithArgmax2D((2, 2))(conv4)

    # conv5   = ConvD(BN_ReLU(pool4),initial_filter,(3,3),(1,1))
    # conv5   = residual_block(pool4,initial_filter*2**4,False,True)
    for _ in range(0,stack_n[4]):
        conv5 = residual_block_bottleneck(pool4,initial_filter,False,False)
    # conv5   = ConvD(BN_ReLU(conv5),initial_filter*2**4,(3,3),(1,1))
    # pool5   = MaxPooling2D(pool_size=(2, 2))(conv5)
    pool5, mask5 = MaxPoolingWithArgmax2D((2, 2))(conv5)
    
    up6     = up_block_bottleneck(pool5,mask5,conv5,initial_filter,(2,2),5)

    up7     = up_block_bottleneck(up6,mask4,conv4,initial_filter,(2,2),6)

    up8     = up_block_bottleneck(up7,mask3,conv3,initial_filter,(2,2),7)

    up9     = up_block_basic(up8,mask2,conv2,initial_filter,(2,2),8)

    up10    = up_block_basic(up9,mask1,conv1,initial_filter,(2,2),9)

    conv10  = BN_ReLU(up10)
    # conv10  = Dropout(0.5)(conv10)
    conv10  = Conv2D(1, (1, 1), activation='sigmoid')(conv10)

    # conv10 = GlobalAveragePooling2D()(conv10)
    
    ''' input: 64 output: 1 '''
    # conv10 = Dense(num_classes,activation='sigmoid',kernel_initializer="he_normal",
            #   kernel_regularizer=regularizers.l2(weight_decay))(conv10)
              
    return conv10


def preprocess(imgs):
    imgs_p = np.ndarray((imgs.shape[0], img_rows, img_cols), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i] = resize(imgs[i], (img_cols, img_rows), preserve_range=True, mode="constant")

    imgs_p = imgs_p[..., np.newaxis]
    return imgs_p
# image acquction
def ImageDGenerator():
    img_generator = ImageDataGenerator()
    if expand_times >1:
        img_generator = ImageDataGenerator(
                shear_range = 0.3,
                rotation_range = 30,
                zoom_range = 0.2,
                horizontal_flip=True,
                fill_mode='constant'
            )
    return img_generator

def train_and_predict():
    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)
    imgs_train, imgs_mask_train = load_train_data(input_type)

    imgs_train = preprocess(imgs_train)
    imgs_mask_train = preprocess(imgs_mask_train)

    imgs_train = imgs_train.astype('float32')
    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization

    imgs_train -= mean
    imgs_train /= std

    imgs_mask_train = imgs_mask_train.astype('float32')
    imgs_mask_train /= 255.  # scale masks to [0, 1]
    # imgs_mask_train = imgs_mask_train - 1
    # imgs_mask_train = keras.utils.to_categorical(imgs_mask_train, num_classes=num_classes)

    ''' random split 20% train data to val. data '''
    x_train, x_val, y_train, y_val = train_test_split(imgs_train, imgs_mask_train, test_size=val_split, random_state=seed)

    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)

    inputs = Input(shape=(img_rows, img_cols, channels))
    output = get_net(inputs,num_classes)
    model = Model(inputs, output)


    model.compile(optimizer=Adam(lr=1e-4), loss=dice_coef_loss, metrics=[dice_coef])
    
    if not os.path.exists('data/proposed/'):
        os.mkdir('data/proposed/')
        
    plot_model(model, to_file='data/proposed/model.png', show_shapes=True, show_layer_names=True)
    # print(model.summary())

    
    print('-'*30)
    print('Fitting model...')
    print('-'*30)

    img_datagen = ImageDGenerator()
    mask_datagen = ImageDGenerator()
    img_datagen.fit(x_train, augment=True, seed=seed)
    mask_datagen.fit(y_train, augment=True, seed=seed)

    img_generator = img_datagen.flow(x_train, batch_size=batch_size,seed=seed)
    mask_generator = mask_datagen.flow(y_train, batch_size=batch_size,seed=seed)

    generator = zip(img_generator, mask_generator)

    steps_per_epoch = int((len(x_train)*expand_times)/batch_size)

    # lr_finder = LRFinder(model)
    # lr_finder.find(x_train, y_train, start_lr=1e-5, end_lr=1e-1, batch_size=batch_size, epochs=epochs)
    # lr_finder.plot_loss(n_skip_beginning=5, n_skip_end=5)
    # lr_finder.plot_loss_change(sma=20, n_skip_beginning=5, n_skip_end=5, y_lim=(-0.02, 0.01))

    ''' find lr  / must with fit_generator'''
    
    # lr_finder = LRFinder(min_lr=1e-5, 
    #                              max_lr=1e-1, 
    #                              step_size=steps_per_epoch, 
    #                              beta=0.98)

    ''' earlystop '''
    # early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=1, mode='auto')
    # tb_cb     = TensorBoard(log_dir='data/proposed/model{:d}_{:d}/log'.format(layers,test), histogram_freq=0, write_graph=True, batch_size=batch_size)
    change_lr = LearningRateScheduler(scheduler)
    ckpt      = ModelCheckpoint(weight_name, monitor='val_loss', save_best_only=True)
    callbacks = [change_lr,ckpt]
    # model_checkpoint = [ModelCheckpoint('weights.h5', monitor='val_loss', save_best_only=True)]

    ''' extend dataset '''
    print('hee1')
    training_log = model.fit_generator(generator,
                                        validation_data=(x_val, y_val),
                                        steps_per_epoch=steps_per_epoch,
                                        epochs=epochs, verbose=1,
                                        callbacks=callbacks)
    
    # lr_finder.plot_loss()
    # lr_finder.plot_avg_loss()
    
    # training_log = model.fit(imgs_train, imgs_mask_train, batch_size=batch_size, epochs=epochs, verbose=1, shuffle=True,
    #                         validation_split=val_split,
    #                         callbacks=callbacks)

    # save models
    model.save('models.h5')

    # models = load_model('models.h5', {'dice_coef_loss':dice_coef_loss, 'dice_coef':dice_coef}) 
    
    ''' save history '''
    with open('data/proposed/trainHistoryDict', 'wb') as file_pi:
        pickle.dump(training_log.history, file_pi)
    history_show(training_log)

    print('-'*30)
    print('Loading and preprocessing test data...')
    print('-'*30)
    imgs_test, imgs_id_test = load_test_data(input_type)
    imgs_test = preprocess(imgs_test)

    imgs_test = imgs_test.astype('float32')
    imgs_test -= mean
    imgs_test /= std

    print('-'*30)
    print('Loading saved weights...')
    print('-'*30)
    inputs = Input(shape=(img_rows, img_cols, channels))
    output = get_net(inputs,num_classes)
    model = Model(inputs, output)
    model.load_weights(weight_name)
    

    print('-'*30)
    print('Predicting masks on test data...')
    print('-'*30)
    imgs_mask_test = model.predict(imgs_test, verbose=1)
    np.save('{:s}_imgs_mask_test.npy'.format(input_type), imgs_mask_test)

    print('-' * 30)
    print('Saving predicted masks to files...')
    print('-' * 30)
    pred_dir = '{:s}_preds'.format(input_type)
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    for image, image_id in zip(imgs_mask_test, imgs_id_test):
        image = (image[:, :, 0] * 255.).astype(np.uint8)
        imsave(os.path.join(pred_dir, '{:d}_pred.png'.format(image_id)), image)
    
    del model
    del training_log
    K.clear_session()


def predict():
    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)
    imgs_train, imgs_mask_train = load_train_data(input_type)

    imgs_train = preprocess(imgs_train)

    imgs_train = imgs_train.astype('float32')
    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization

    imgs_train -= mean
    imgs_train /= std


    print('start predict')
    model = load_model('models.h5', {'dice_coef_loss':dice_coef_loss, 'dice_coef':dice_coef, 'MaxPoolingWithArgmax2D':MaxPoolingWithArgmax2D, 'MaxUnpooling2D':MaxUnpooling2D}) 

    print('-'*30)
    print('Loading and preprocessing test data...')
    print('-'*30)
    imgs_test, imgs_id_test = load_test_data(input_type)
    imgs_test = preprocess(imgs_test)

    imgs_test = imgs_test.astype('float32')
    imgs_test -= mean
    imgs_test /= std

    print('-'*30)
    print('Loading saved weights...')
    print('-'*30)
    inputs = Input(shape=(img_rows, img_cols, channels))
    output = get_net(inputs,num_classes)
    model = Model(inputs, output)
    model.load_weights(weight_name)
    
    print('-'*30)
    print('Predicting masks on test data...')
    print('-'*30)
    imgs_mask_test = model.predict(imgs_test, verbose=1)
    np.save('{:s}_imgs_mask_test.npy'.format(input_type), imgs_mask_test)

    print('-' * 30)
    print('Saving predicted masks to files...')
    print('-' * 30)
    pred_dir = '{:s}_preds'.format(input_type)
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    for image, image_id in zip(imgs_mask_test, imgs_id_test):
        image = (image[:, :, 0] * 255.).astype(np.uint8)
        imsave(os.path.join(pred_dir, '{:d}_pred.png'.format(image_id)), image)
    
    del model
    K.clear_session()


if __name__ == '__main__':
    # predict()
    train_and_predict()
