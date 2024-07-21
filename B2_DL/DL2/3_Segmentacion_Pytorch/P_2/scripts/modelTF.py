import tensorflow as tf

class Stack2Convs(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels, batch_norm=True, input_shape=None):
        super(Stack2Convs, self).__init__()
        self.batch_norm = batch_norm
        if input_shape:
            self.conv2d_a = tf.keras.layers.Conv2D(out_channels, 3, strides=1, padding='same', use_bias=~self.batch_norm, input_shape=input_shape)
        else:
            self.conv2d_a = tf.keras.layers.Conv2D(out_channels, 3, strides=1, padding='same', use_bias=~self.batch_norm)
            
        self.conv2d_b = tf.keras.layers.Conv2D(out_channels, 3, strides=1, padding='same', use_bias=~self.batch_norm)
        if self.batch_norm:
            self.batchnorm_a = tf.keras.layers.BatchNormalization()
            self.batchnorm_b = tf.keras.layers.BatchNormalization()

    def call(self, x):
        x = self.conv2d_a(x)
        if self.batch_norm:
            x = self.batchnorm_a(x)
        x = tf.nn.relu(x)
        x = self.conv2d_b(x)
        if self.batch_norm:
            x = self.batchnorm_b(x)
        x = tf.nn.relu(x)
        return x

class UNETtf(tf.keras.models.Model):
    def __init__(self, input_shape, in_channels=3, out_channels=1, feats_per_block=[64, 128, 256, 512], batch_norm=True, crop=True, logits=True):
        super(UNETtf, self).__init__()

        self.n_blocks = len(feats_per_block)
        self.crop = crop
        self.logits = logits
        self.contraction_path = []
        self.expansion_path_convs = []
        self.expansion_path_upsamplers = []

        # Build contraction path layers
        for feats in feats_per_block:
            self.contraction_path.append(Stack2Convs(in_channels, feats, batch_norm=batch_norm, input_shape=input_shape))
            in_channels = feats

        # Build expansion path layers
        for feats in feats_per_block[::-1]:
            self.expansion_path_convs.append(Stack2Convs(feats*2, feats, batch_norm=batch_norm))
            self.expansion_path_upsamplers.append(tf.keras.layers.Conv2DTranspose(feats, 2, strides=2))
        # Bottleneck
        self.bottleneck = Stack2Convs(feats_per_block[-1], feats_per_block[-1]*2, batch_norm=batch_norm)
        # Point-wise convolution as final layer
        self.pointwise_conv = tf.keras.layers.Conv2D(out_channels, 1)
        
    def call(self, x):

        skips = []
        # Contraction path
        for block in self.contraction_path:
            x = block(x)
            skips.append(x)
            x = tf.keras.layers.MaxPooling2D(2, strides=2)(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Expansion path
        for i in range(self.n_blocks):
            x = self.expansion_path_upsamplers[i](x)
            skip = skips[-(i + 1)]

            valid_height = tf.shape(x)[1]
            valid_width = tf.shape(x)[2]
            height_diff = tf.shape(skip)[1] - valid_height
            width_diff = tf.shape(skip)[2] - valid_width

            pad_left = width_diff // 2
            pad_right = width_diff - pad_left
            pad_top = height_diff // 2
            pad_bottom = height_diff - pad_top

            to_concat = skip
            # Convert to boolean for Python conditional
            if tf.logical_or(tf.math.greater(height_diff, 0), tf.math.greater(width_diff, 0)):
                if self.crop:
                    to_concat = skip[:, height_diff:height_diff + valid_height, width_diff:width_diff + valid_width, :]
                else:
                    x = tf.pad(x, 
                               paddings=[[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], 
                               mode='CONSTANT', 
                               constant_values=0
                               )

            x = tf.concat([to_concat, x], axis=-1)
            x = self.expansion_path_convs[i](x)

        # Final layer
        x = self.pointwise_conv(x)
        if not self.logits:
            x = tf.nn.sigmoid(x)
        return x