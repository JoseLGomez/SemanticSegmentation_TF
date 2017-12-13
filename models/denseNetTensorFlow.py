"""DenseNet models for Keras.
# Reference
- [Densely Connected Convolutional Networks](https://arxiv.org/pdf/1608.06993.pdf)
- [The One Hundred Layers Tiramisu: Fully Convolutional DenseNets for Semantic Segmentation](https://arxiv.org/pdf/1611.09326.pdf)
"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import warnings
import tensorflow as tf

# Custom layers import
from layers.ourlayers import (CropLayer2D, NdSoftmax)

def DenseNetFCN(img_input, nb_dense_block=5, growth_rate=16, nb_layers_per_block=4,
                reduction=0.0, dropout_rate=0.0, weight_decay=1E-4, init_conv_filters=48,
                include_top=True, weights=None, input_tensor=None, classes=1, activation='softmax',
                upsampling_conv=128, upsampling_type='upsampling', batchsize=None, freeze_layers_from=None, freezeUpsampling=False,
                is_training=True):
    input_shape = img_input.get_shape().as_list()[1:]
    if weights not in {None}:
        raise ValueError('The `weights` argument should be '
                         '`None` (random initialization) as no '
                         'model weights are provided.')

    upsampling_type = upsampling_type.lower()

    if upsampling_type not in ['upsampling', 'deconv', 'atrous', 'subpixel']:
        raise ValueError('Parameter "upsampling_type" must be one of "upsampling", '
                         '"deconv", "atrous" or "subpixel".')

    '''if upsampling_type == 'deconv' and batchsize is None:
        raise ValueError('If "upsampling_type" is deconvoloution, then a fixed '
                         'batch size must be provided in batchsize parameter.')'''

    if input_shape is None:
        raise ValueError('For fully convolutional models, input shape must be supplied.')

    if type(nb_layers_per_block) is not list and nb_dense_block < 1:
        raise ValueError('Number of dense layers per block must be greater than 1. Argument '
                         'value was %d.' % (nb_layers_per_block))

    if upsampling_type == 'atrous':
        warnings.warn(
            'Atrous Convolution upsampling does not correctly work (see https://github.com/fchollet/keras/issues/4018).\n'
            'Switching to `upsampling` type upscaling.')
        upsampling_type = 'upsampling'

    if activation not in ['softmax', 'sigmoid']:
        raise ValueError('activation must be one of "softmax" or "sigmoid"')

    if activation == 'sigmoid' and classes != 1:
        raise ValueError('sigmoid activation can only be used when classes = 1')

    # Determine proper input shape
    min_size = 2 ** nb_dense_block

    '''if input_shape is not None:
        if ((input_shape[0] is not None and input_shape[0] < min_size) or
                (input_shape[1] is not None and input_shape[1] < min_size)):
            raise ValueError('Input size must be at least ' +
                             str(min_size) + 'x' + str(min_size) + ', got '
                                                                   '`input_shape=' + str(input_shape) + '`')
    else:
        input_shape = (None, None, classes)'''

    #img_input = tf.layers.Input(shape=input_shape)
    x = __create_fcn_dense_net(classes, img_input, include_top, nb_dense_block,
                               growth_rate, reduction, dropout_rate, weight_decay,
                               nb_layers_per_block, upsampling_conv, upsampling_type,
                               batchsize, init_conv_filters, input_shape, is_training)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    '''if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:'''
    inputs = img_input
    # Create model.
    #model = Model(inputs, x, name='fcn-densenet')
    
    # Freeze some layers
    if freeze_layers_from is not None:
        freeze_layers(model, freeze_layers_from, nb_dense_block, nb_layers_per_block, freezeUpsampling)

    return x


def __conv_block(ip, nb_filter, bottleneck=False, dropout_rate=None, weight_decay=1E-4, block='', is_training=True):
    ''' Apply BatchNorm, Relu, 3x3 Conv2D, optional bottleneck block and dropout
    Args:
        ip: Input keras tensor
        nb_filter: number of filters
        bottleneck: add bottleneck block
        dropout_rate: dropout rate
        weight_decay: weight decay factor
    Returns: keras tensor with batch_norm, relu and convolution2d added (optional bottleneck)
    '''

    concat_axis = -1

    '''x = BatchNormalization(axis=concat_axis, gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay), name='bn1_'+block)(ip)'''
    x = tf.contrib.layers.batch_norm(ip, scale=True, param_regularizers = {'beta':tf.contrib.slim.l2_regularizer(weight_decay),
                                      'gamma':tf.contrib.slim.l2_regularizer(weight_decay)} , scope='bn1_'+block, is_training=is_training)
    
    #x = Activation('relu',name='relu1_'+block)(x)
    x = tf.nn.relu(x, name='relu1_'+block)

    if bottleneck:
        inter_channel = nb_filter * 4  # Obtained from https://github.com/liuzhuang13/DenseNet/blob/master/densenet.lua

        '''x = Convolution2D(inter_channel, (1, 1), kernel_initializer='he_uniform', padding='same',
                          use_bias=False, kernel_regularizer=l2(weight_decay),name='convbneck_'+block)(x)'''
        x = tf.contrib.slim.conv2d(x, inter_channel, [1,1], padding='SAME',weights_regularizer=tf.contrib.slim.l2_regularizer(weight_decay), 
                                   weights_initializer=tf.contrib.layers.xavier_initializer(uniform=True), scope='convbneck_'+block)       
        #x = tf.nn.conv2(x,filter=,strides=[1,1,1,1],padding='SAME',name='convbneck_'+block)

        if dropout_rate:
            #x = Dropout(dropout_rate, name='dropbneck_'+block)(x)
            x = tf.nn.dropout(x, dropout_rate, name='dropbneck_'+block)

        '''x = BatchNormalization(axis=concat_axis, gamma_regularizer=l2(weight_decay),
                               beta_regularizer=l2(weight_decay),name='bnbneck_'+block)(x)'''
        x = tf.contrib.layers.batch_norm(x, scale=True, param_regularizers = {'beta':tf.contrib.slim.l2_regularizer(weight_decay),
                                          'gamma':tf.contrib.slim.l2_regularizer(weight_decay)} , scope='bnbneck_'+block,is_training=is_training)
        
        #x = Activation('relu',name='relubneck_'+block)(x)
        x = tf.nn.relu(x, name='relubneck_'+block)
        
    '''x = Convolution2D(nb_filter, (3, 3), kernel_initializer="he_uniform", padding="same", use_bias=False,
                      kernel_regularizer=l2(weight_decay),name='conv1_'+block)(x)'''
    x = tf.contrib.slim.conv2d(x, nb_filter, [3,3], padding='SAME',weights_regularizer=tf.contrib.slim.l2_regularizer(weight_decay), 
                                   weights_initializer=tf.contrib.layers.xavier_initializer(uniform=True), scope='conv1_'+block)                  
    if dropout_rate:
        #x = Dropout(dropout_rate, name='drop1_'+block)(x)
        x = tf.nn.dropout(x, dropout_rate, name='drop1_'+block)
    return x


def __transition_block(ip, nb_filter, compression=1.0, dropout_rate=None, weight_decay=1E-4, block='', is_training=True):
    ''' Apply BatchNorm, Relu 1x1, Conv2D, optional compression, dropout and Maxpooling2D
    Args:
        ip: keras tensor
        nb_filter: number of filters
        compression: calculated as 1 - reduction. Reduces the number of feature maps
                    in the transition block.
        dropout_rate: dropout rate
        weight_decay: weight decay factor
    Returns: keras tensor, after applying batch_norm, relu-conv, dropout, maxpool
    '''

    concat_axis = -1

    '''x = BatchNormalization(axis=concat_axis, gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay),name='bn2_'+block)(ip)'''
    x = tf.contrib.layers.batch_norm(ip, scale=True, param_regularizers = {'beta':tf.contrib.slim.l2_regularizer(weight_decay),
                                      'gamma':tf.contrib.slim.l2_regularizer(weight_decay)} , scope='bn2_'+block, is_training=is_training)                       
    
    #x = Activation('relu',name='relu2_'+block)(x)
    x = tf.nn.relu(x, name='relu2_'+block)
    
    '''x = Convolution2D(int(nb_filter * compression), (1, 1), kernel_initializer="he_uniform", padding="same",
                      use_bias=False, kernel_regularizer=l2(weight_decay),name='conv2_'+block)(x)'''
    x = tf.contrib.slim.conv2d(x, int(nb_filter * compression), [1,1], padding='SAME',weights_regularizer=tf.contrib.slim.l2_regularizer(weight_decay), 
                                   weights_initializer=tf.contrib.layers.xavier_initializer(uniform=True), scope='conv2_'+block)                  
    
    if dropout_rate:
        #x = Dropout(dropout_rate, name='drop2_'+block)(x)
        x = tf.nn.dropout(x, dropout_rate, name='drop2_'+block)
        
    #x = AveragePooling2D((2, 2), strides=(2, 2),name='pool_'+block)(x)
    x = tf.nn.avg_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME',name='pool_'+block)

    return x


def __dense_block(x, nb_layers, nb_filter, growth_rate, bottleneck=False, dropout_rate=None, weight_decay=1E-4,
                  grow_nb_filters=True, return_concat_list=False, block='', is_training=True):
    ''' Build a dense_block where the output of each conv_block is fed to subsequent ones
    Args:
        x: keras tensor
        nb_layers: the number of layers of conv_block to append to the model.
        nb_filter: number of filters
        growth_rate: growth rate
        bottleneck: bottleneck block
        dropout_rate: dropout rate
        weight_decay: weight decay factor
        grow_nb_filters: flag to decide to allow number of filters to grow
        return_concat_list: return the list of feature maps along with the actual output
    Returns: keras tensor with nb_layers of conv_block appended
    '''

    concat_axis = -1

    x_list = [x]

    for i in range(nb_layers):
        x = __conv_block(x, growth_rate, bottleneck, dropout_rate, weight_decay, 'layer' + str(i+1) + '_' + block, is_training=is_training)
        x_list.append(x)
        #x = concatenate(x_list, axis=concat_axis, name='concat_layer' + str(i+1) + '_' + block)
        x = tf.concat(x_list, axis=concat_axis, name='concat_layer' + str(i+1) + '_' + block)
        if grow_nb_filters:
            nb_filter += growth_rate

    if return_concat_list:
        return x, nb_filter, x_list
    else:
        return x, nb_filter


def __transition_up_block(ip, nb_filters, type='upsampling', weight_decay=1E-4, block=''):
    ''' SubpixelConvolutional Upscaling (factor = 2)
    Args:
        ip: keras tensor
        nb_filters: number of layers
        type: can be 'upsampling', 'subpixel', 'deconv', or 'atrous'. Determines type of upsampling performed
        output_shape: required if type = 'deconv'. Output shape of tensor
        weight_decay: weight decay factor
    Returns: keras tensor, after applying upsampling operation.
    '''

    #x = ZeroPadding2D(padding=(1, 1), name='pad_'+block)(ip)
    x = tf.pad(ip, tf.constant([[0, 0], [1, 1],[1, 1], [0, 0]]), "CONSTANT", name='pad_'+block)
    
    '''x = Deconvolution2D(nb_filters, (3, 3), activation='relu', padding='same',
                            strides=(2, 2), kernel_initializer='he_uniform',name='deconv_'+block)(x)'''
    x = tf.contrib.slim.layers.conv2d_transpose(x,x.get_shape()[3], 3, stride=2, scope='deconv_'+block, padding='SAME',
                                        weights_initializer=tf.contrib.layers.xavier_initializer(uniform=True))

    return x


def __create_fcn_dense_net(nb_classes, img_input, include_top, nb_dense_block=5, growth_rate=12,
                           reduction=0.0, dropout_rate=None, weight_decay=1E-4,
                           nb_layers_per_block=4, nb_upsampling_conv=128, upsampling_type='upsampling',
                           batchsize=None, init_conv_filters=48, input_shape=None, activation='softmax', is_training=True):

    concat_axis = -1
    rows, cols, _ = input_shape

    if reduction != 0.0:
        assert reduction <= 1.0 and reduction > 0.0, "reduction value must lie between 0.0 and 1.0"

    # check if upsampling_conv has minimum number of filters
    # minimum is set to 12, as at least 3 color channels are needed for correct upsampling
    assert nb_upsampling_conv > 12 and nb_upsampling_conv % 4 == 0, "Parameter `upsampling_conv` number of channels must " \
                                                                    "be a positive number divisible by 4 and greater " \
                                                                    "than 12"

    # layers in each dense block
    if type(nb_layers_per_block) is list or type(nb_layers_per_block) is tuple:
        nb_layers = list(nb_layers_per_block)  # Convert tuple to list

        assert len(nb_layers) == (nb_dense_block + 1), "If list, nb_layer is used as provided. " \
                                                       "Note that list size must be (nb_dense_block + 1)"

        bottleneck_nb_layers = nb_layers[-1]
        rev_layers = nb_layers[::-1]
        nb_layers.extend(rev_layers[1:])
    else:
        bottleneck_nb_layers = nb_layers_per_block
        nb_layers = [nb_layers_per_block] * (2 * nb_dense_block + 1)

    # compute compression factor
    compression = 1.0 - reduction

    # Initial convolution
    '''x = Convolution2D(init_conv_filters, (3, 3), kernel_initializer="he_uniform",
                      padding="same",name="initial_conv2D", use_bias=False,
                      kernel_regularizer=l2(weight_decay))(img_input)'''
    x = tf.contrib.slim.conv2d(img_input, init_conv_filters, [3,3], padding='SAME',weights_regularizer=tf.contrib.slim.l2_regularizer(weight_decay), 
                                   weights_initializer=tf.contrib.layers.xavier_initializer(uniform=True), scope='initial_conv2D')
                                   
    nb_filter = init_conv_filters

    skip_list = []

    # Add dense blocks and transition down block
    for block_idx in range(nb_dense_block):
        x, nb_filter = __dense_block(x, nb_layers[block_idx], nb_filter, growth_rate,
                                     dropout_rate=dropout_rate, weight_decay=weight_decay, 
                                     block='block_down' + str(block_idx+1), is_training=is_training)

        # Skip connection
        skip_list.append(x)

        # add transition_block
        x = __transition_block(x, nb_filter, compression=compression, dropout_rate=dropout_rate,
                               weight_decay=weight_decay, block='block_down' + str(block_idx+1), is_training=is_training)

        nb_filter = int(nb_filter * compression)  # this is calculated inside transition_down_block

    # The last dense_block does not have a transition_down_block
    # return the concatenated feature maps without the concatenation of the input
    _, nb_filter, concat_list = __dense_block(x, bottleneck_nb_layers, nb_filter, growth_rate,
                                              dropout_rate=dropout_rate, weight_decay=weight_decay,
                                              return_concat_list=True,block='block_down' + str(nb_dense_block+1), is_training=is_training)

    skip_list = skip_list[::-1]  # reverse the skip list
    #out_shape = [batchsize, rows // 16, cols // 16, nb_filter]

    # Add dense blocks and transition up block
    for block_idx in range(nb_dense_block):
        n_filters_keep = growth_rate * nb_layers[nb_dense_block + block_idx]

        #out_shape[3] = n_filters_keep

        # upsampling block must upsample only the feature maps (concat_list[1:]),
        # not the concatenation of the input with the feature maps (concat_list[0].
        #l = concatenate(concat_list[1:], axis=concat_axis, name='concat1_block_up' + str(block_idx+1))
        l = tf.concat(concat_list[1:], axis=concat_axis, name='concat1_block_up' + str(block_idx+1))
        t = __transition_up_block(l, nb_filters=n_filters_keep, type=upsampling_type, block='block_up' + str(block_idx+1))
        t = CropLayer2D(skip_list[block_idx],name='crop_block_up' + str(block_idx+1))(t)
        # concatenate the skip connection with the transition block
        #x = concatenate([t, skip_list[block_idx]], axis=concat_axis, name='concat2_block_up' + str(block_idx+1))
        x = tf.concat([t, skip_list[block_idx]], axis=concat_axis, name='concat2_block_up' + str(block_idx+1))

        '''out_shape[1] *= 2
        out_shape[2] *= 2'''

        # Dont allow the feature map size to grow in upsampling dense blocks
        _, nb_filter, concat_list = __dense_block(x, nb_layers[nb_dense_block + block_idx + 1],
                                                  nb_filter=growth_rate,
                                                  growth_rate=growth_rate, dropout_rate=dropout_rate,
                                                  weight_decay=weight_decay,
                                                  return_concat_list=True, grow_nb_filters=False,
                                                  block='block_up' + str(block_idx+1))

    if include_top:
        '''x = Convolution2D(nb_classes, (1, 1), activation='linear', padding='same',
                          kernel_regularizer=l2(weight_decay), use_bias=False, name='convTop')(x)'''
        x = tf.contrib.slim.conv2d(x, nb_classes, [1,1], padding='SAME',weights_regularizer=tf.contrib.slim.l2_regularizer(weight_decay), 
                                   activation_fn=tf.nn.relu, scope='convTop')
        x = CropLayer2D(img_input, name='score')(x)
        #x = NdSoftmax(x)
        x = tf.nn.softmax(x)
    return x

# Freeze layers for finetunning
'''def freeze_layers(model, freeze_layers_from, nb_dense_block, nb_layers_per_block, freezeUpsampling):
    # Freeze the downsampling part only including the middle block
    if freeze_layers_from == 'base_model':
        print ('   Freezing base model layers')
        freeze_layers_from = 1+(nb_layers_per_block*4)*(nb_dense_block+1)

    # Show layers (Debug pruposes)
    for i, layer in enumerate(model.layers):
        print(i, layer.name)
    print ('   Freezing from layer 0 to ' + str(freeze_layers_from))

    # Freeze layers
    for layer in model.layers[:freeze_layers_from]:
        layer.trainable = False
    for layer in model.layers[freeze_layers_from:]:
        layer.trainable = True
    if freezeUpsampling:
        #Only for 5 denseblocks blocks
        print ('   Freezing from layer 117 to 204')    
        for layer in model.layers[117:]:
            layer.trainable = False    '''

if __name__ == '__main__':
    model = DenseNetFCN((32, 32, 1), nb_dense_block=5, growth_rate=16,
                        nb_layers_per_block=4, upsampling_type='upsampling', classes=1, activation='sigmoid')
    model.summary()
