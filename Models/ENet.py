# coding=utf-8
"""
    # 相比原始的Enet
    # 去掉第一部分重复的两层bottleneck
    # 去掉第三部分的除了空洞卷积的bottleneck
    # val0.85 mpa0.8787 train0.83 mpa0.86
    # By LiangBo
"""
#
from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import Conv2D, Conv2DTranspose, UpSampling2D, ZeroPadding2D
from keras.layers.core import SpatialDropout2D, Permute, Activation, Reshape
from keras.layers.merge import add, concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.engine.topology import Input
from keras.models import Model


#inp为输入图像尺寸
def initial_block(inp, nb_filter=13, nb_row=3, nb_col=3, strides=(2, 2)):           #初始模块：步长为2的3*3卷积(输出通道13)，maxpooling输出通道为3，融合后为16
    conv = Conv2D(nb_filter, (nb_row, nb_col), padding='same', strides=strides)(inp)#padding边界处理，same代表保留边界处的卷积结果，通常会导致输出shape与输入shape相同
    max_pool = MaxPooling2D()(inp)#池化
    merged = concatenate([conv, max_pool], axis=3)                                   #拼接第三轴
    return merged

def bottleneck(inp, output, internal_scale=4, asymmetric=0, dilated=0, downsample=False, dropout_rate=0.1):
    # main branch
    internal = output // internal_scale
    encoder = inp
    # 1x1
    input_stride = 2 if downsample else 1                                            #如果bottleneck为下采样时，第一个1*1卷积被为2*2卷积替换
    encoder = Conv2D(internal, (input_stride, input_stride),
                     # padding='same',
                     strides=(input_stride, input_stride), use_bias=False)(encoder)
    # Batch normalization + PReLU
    encoder = BatchNormalization(momentum=0.1)(encoder)                               # enet uses momentum of 0.1, keras default is 0.99
    encoder = PReLU(shared_axes=[1, 2])(encoder)

    # conv层的选择
    if not asymmetric and not dilated:                                               #如果不是asymmetric和dilated层，conv为3*3卷积
        encoder = Conv2D(internal, (3, 3), padding='same')(encoder)
    elif asymmetric:                                                                  #如果是asymmetric层，根据asymmetric值(5)为1*5,5*1卷积
        encoder = Conv2D(internal, (1, asymmetric), padding='same', use_bias=False)(encoder)
        encoder = Conv2D(internal, (asymmetric, 1), padding='same')(encoder)
    elif dilated:                                                                     #如果是dilated层，则为3*3的膨胀卷积卷积率为dilated值卷积
        encoder = Conv2D(internal, (3, 3), dilation_rate=(dilated, dilated), padding='same')(encoder)
    else:
        raise (Exception('You shouldn\'t be here'))

    encoder = BatchNormalization(momentum=0.1)(encoder)                               # enet uses momentum of 0.1, keras default is 0.99
    encoder = PReLU(shared_axes=[1, 2])(encoder)

    # 1x1
    encoder = Conv2D(output, (1, 1), use_bias=False)(encoder)

    encoder = BatchNormalization(momentum=0.1)(encoder)                               # enet uses momentum of 0.1, keras default is 0.99
    encoder = SpatialDropout2D(dropout_rate)(encoder)

    other = inp
    # 旁分支增加maxpooling，padding与主分支add
    if downsample:
        other = MaxPooling2D()(other)

        other = Permute((1, 3, 2))(other)
        pad_feature_maps = output - inp.get_shape().as_list()[3]
        tb_pad = (0, 0)
        lr_pad = (0, pad_feature_maps)
        other = ZeroPadding2D(padding=(tb_pad, lr_pad))(other)
        other = Permute((1, 3, 2))(other)

    encoder = add([encoder, other])
    encoder = PReLU(shared_axes=[1, 2])(encoder)

    return encoder

def en_build(inp, dropout_rate=0.01):                                                 #编码部分，dropout_rate在2.0前是0.1，之后为0.01
    enet = initial_block(inp)                                                         #输出（16*256*256）
    enet = BatchNormalization(momentum=0.1)(enet)                                     # BN层enet_unpooling uses momentum of 0.1, keras default is 0.99
    enet = PReLU(shared_axes=[1, 2])(enet)                                            #Prelu激活函数
    enet = bottleneck(enet, 64, downsample=True, dropout_rate=dropout_rate)           #bottleneck 1.0
    # 去掉两个bottleneck层
    for _ in range(2):
        enet = bottleneck(enet, 64, dropout_rate=dropout_rate)                         # bottleneck 1.i

    enet = bottleneck(enet, 128, downsample=True)                                      # bottleneck 2.0
    # 去掉重复的第三部分
    # bottleneck 2.x and 3.x
    enet = bottleneck(enet, 128)                                                       # bottleneck 2.1
    enet = bottleneck(enet, 128, dilated=2)                                            # bottleneck 2.2
    enet = bottleneck(enet, 128, asymmetric=5)                                         # bottleneck 2.3
    enet = bottleneck(enet, 128, dilated=4)                                            # bottleneck 2.4
    enet = bottleneck(enet, 128)                                                       # bottleneck 2.5
    enet = bottleneck(enet, 128, dilated=8)                                            # bottleneck 2.6
    enet = bottleneck(enet, 128, asymmetric=5)                                         # bottleneck 2.7
    enet = bottleneck(enet, 128, dilated=16)                                           # bottleneck 2.8
    return enet

# decoder
def de_bottleneck(encoder, output, upsample=False, reverse_module=False):
    internal = output // 4

    x = Conv2D(internal, (1, 1), use_bias=False)(encoder)
    x = BatchNormalization(momentum=0.1)(x)
    x = Activation('relu')(x)
    if not upsample:
        x = Conv2D(internal, (3, 3), padding='same', use_bias=True)(x)
    else:
        x = Conv2DTranspose(filters=internal, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = BatchNormalization(momentum=0.1)(x)
    x = Activation('relu')(x)

    x = Conv2D(output, (1, 1), padding='same', use_bias=False)(x)

    other = encoder
    if encoder.get_shape()[-1] != output or upsample:
        other = Conv2D(output, (1, 1), padding='same', use_bias=False)(other)
        other = BatchNormalization(momentum=0.1)(other)
        if upsample and reverse_module is not False:
            other = UpSampling2D(size=(2, 2))(other)

    if upsample and reverse_module is False:
        decoder = x
    else:
        x = BatchNormalization(momentum=0.1)(x)
        decoder = add([x, other])
        decoder = Activation('relu')(decoder)

    return decoder

def de_build(encoder, nc):
    enet = de_bottleneck(encoder, 64, upsample=True, reverse_module=True)           # bottleneck 4.0
    enet = de_bottleneck(enet, 64)                                                   # bottleneck 4.1
    enet = de_bottleneck(enet, 64)                                                   # bottleneck 4.2
    enet = de_bottleneck(enet, 16, upsample=True, reverse_module=True)              # bottleneck 5.0
    enet = de_bottleneck(enet, 16)                                                   # bottleneck 5.1

    enet = Conv2DTranspose(filters=nc, kernel_size=(2, 2), strides=(2, 2), padding='same')(enet)#反卷积
    return enet

def ENet(n_classes, input_height=512, input_width=512): #512 512
    assert input_height % 32 == 0
    assert input_width % 32 == 0
    img_input = Input(shape=(input_height, input_width, 3))
    enet = en_build(img_input)#编码输入(512,512,3)
    enet = de_build(enet, n_classes)
    o_shape = Model(img_input, enet).output_shape
    outputHeight = o_shape[1]
    outputWidth = o_shape[2]
    enet = (Reshape((outputHeight*outputWidth, n_classes)))(enet)
    enet = Activation('softmax')(enet)#softmax
    model = Model(img_input, enet)
    print(outputHeight)
    print(outputWidth)
    model.outputWidth = outputWidth
    model.outputHeight = outputHeight

    return model



