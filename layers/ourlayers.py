import tensorflow as tf

class CropLayer2D(tf.layers.Layer):
    def __init__(self, img_in, *args, **kwargs):
        self.img_in = img_in
        super(CropLayer2D, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        self.crop_size = self.img_in.get_shape().as_list()[1:3]
        super(CropLayer2D, self).build(input_shape)

    def call(self, x, mask=False):
        input_shape = tf.shape(x)
        cs = tf.shape(self.img_in)
        input_shape = input_shape[1:3]
        cs = cs[1:3]
        dif = (input_shape - cs)/2
        if tf.rank(x) == 5:
            return x[:, :, dif[0]:dif[0]+cs[0], dif[1]:dif[1]+cs[1], :]
        return x[:, dif[0]:dif[0]+cs[0], dif[1]:dif[1]+cs[1], :]

    def compute_output_shape(self, input_shape):
        return ((input_shape[:1]) + (self.crop_size[0], self.crop_size[1]) + (input_shape[-1], ))

class NdSoftmax(tf.layers.Layer):
    '''N-dimensional Softmax
    Will compute the Softmax on channel_idx and return a tensor of the
    same shape as the input'''
    def __init__(self, data_format='default', *args, **kwargs):
        self.channel_index = 3
        super(NdSoftmax, self).__init__(*args, **kwargs)

    def compute_output_shape(self, input_shape):
        return input_shape

    def NdSoftmax( x, mask=None):
        ch_idx = 3
        l_idx = tf.rank(x) - 1  # last index
        x = tf.transpose(
            x, tuple(i for i in range(tf.rank(x)) if i != ch_idx) + (ch_idx,))
        sh = tf.shape(x)
        x = tf.reshape(x, (-1, sh[-1]))
        x = tf.nn.softmax(x)
        x = tf.reshape(x, sh)
        x = tf.transpose(
            x, tuple(range(ch_idx) + [l_idx] + range(ch_idx, l_idx)))
        return x