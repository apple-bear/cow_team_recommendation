import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import *
import tensorflow.keras.backend as K
import os, sys
# import matplotlib.pyplot as plt

# 在vscode中运行时需要加上下面这个切换目录的操作，切换到调用python解释器脚本所在目录
os.chdir(sys.path[0])

# xDeepFM中，从模型结构图可以看出，FM 和 DNN共用embedding层的结果，然后FM部分负责低阶特征组合（二阶），而DNN负责高阶特征组合，然后将低阶和高阶部分合在一起训练
# 有了隐式特征交互为什么还要做显示特征交叉的参考理由：https://cloud.tencent.com/developer/article/1759397
# embedding的意义是对于高维、稀疏的id类特征，通过将单个id映射成一个稠密向量，变id特征的精确匹配为embedding向量的模糊查找。
# embedding之后的向量不是整形，是实数。会不会有重复有待验证
# 在xDeepFM中数值型特征只会进入线性部分，类别型特征会进入线性部分，CIN，DNN？？？
# 字符串的类别型特征要怎么处理，需要labelencoder还是怎样处理？？？
# ID类特征和类别型特征都做labelencoder么？？？
# 对于类别特征来说，如果后面要做embedding，那么之前是做one-hot或者labelencoder都可以

def data_load(data_path):
    user_file_path = '../dataset/dataset1/user.csv'
    movie_file_path = '../dataset/dataset1/movie.csv'
    # user_file_path = u"E:\\Guoping Tan\\Training-Course\\xiaoxiang training\\team\\dataset\\dataset1\\user.csv"
    # movie_file_path = u"E:\\Guoping Tan\\Training-Course\\xiaoxiang training\\team\\dataset\\dataset1\\movie.csv"
    user_data = pd.read_csv(user_file_path, usecols=['评分', '用户ID', '电影名'])
    movie_data = pd.read_csv(movie_file_path, usecols=['评分', '类型', '特色', '电影名', '导演', '主演'])
    data = pd.merge(user_data, movie_data, on='电影名', how='left', suffixes=('_x', '_y'))

    # 数值型特征
    dense_feats = ['评分_x', '评分_y']
    # 类别型特征
    sparse_feats = ['用户ID', '电影名', '类型', '特色', '导演', '主演']

    return data, dense_feats, sparse_feats


def feature_construction(data, dense_feats, sparse_feats):
    def process_dense_feats(data, feats):
        d = data.copy()
        d = d[feats].fillna(0.0)
        for f in feats:
            d[f] = d[f].apply(lambda x: np.log(x + 1) if x > -1 else -1)
        return d

    # 每个sparse特征都要做labelencoder，否则会出错
    def process_sparse_feats(data, feats):
        d = data.copy()
        d = d[feats].fillna("-1")
        for f in feats:
            label_encoder = LabelEncoder()
            d[f] = label_encoder.fit_transform(d[f])
        return d

    data_dense = process_dense_feats(data, dense_feats)
    data_sparse = process_sparse_feats(data, sparse_feats)
    total_data = pd.concat([data_dense, data_sparse], axis=1)
    total_data['label'] = np.random.randint(0, 2, 808835)

    # 构造每个dense特征的输入
    dense_inputs = []
    for f in dense_feats:
        _input = Input([1], name=f)
        dense_inputs.append(_input)


    # 单独对每一个sparse特征构造输入，目的是方便后面构造交叉特征
    sparse_inputs = []
    for f in sparse_feats:
        _input = Input([1], name=f)
        sparse_inputs.append(_input)
    # sparse_inputs现在只是一个结构吧，没有数据？？？
    sparse_1d_embed = []
    for i, _input in enumerate(sparse_inputs):
        f = sparse_feats[i]
        # 获取特征列f的不重复元素个数
        voc_size = total_data[f].nunique()
        # 使用l2正则化防止过拟合
        reg = tf.keras.regularizers.l2(0.5)
        # 线性部分为什么要做embedding，查看一下这个embedding的结果？？？
        # 把sparse特征embedding到1维通过embedding lookup方式找到对应的wi，
        # 可能是先进行了labelencoder然后embeeding到1维之后就可以计算特征值之间距离了，可能是linear部分输入所以embeeding到1维？？？
        # 下面的_input是实际数据么还是只是输入数据的结构？？？
        _embed = Embedding(voc_size, 1, embeddings_regularizer=reg)(_input)
        # 由于Embedding的结果是二维的，因此如果需要在Embedding之后加入Dense层，需要先连接上Flatten层
        _embed = Flatten()(_embed)
        sparse_1d_embed.append(_embed)
    return total_data, dense_inputs, sparse_inputs, sparse_1d_embed


# 将dense特征与sparse特征加权求和结果相加，完成模型最左侧Linear部分
def linear(dense_inputs, sparse_1d_embed):
    # 将dense输入拼接到一起
    concat_dense_inputs = Concatenate(axis=1)(dense_inputs)
    # Dense:全连接层
    # 该层实现了outputs=activation(inputs*kernel+bias),其中激活函数是作为参数提供的，kernel是该层创建的权重矩阵，
    # 而bias是该层创建的偏置向量(仅当use_bias为True)。
    # 对dense特征加权求和，加权体现在哪里
    fst_order_dense_layer = Dense(1)(concat_dense_inputs)
    # 对sparse特征加权求和，为什么dense加权求和是Dense，而sparse加权求和是Add？？？可能是sparse_1d_embed本身会做类似全连接的操作，所以只需要Add就可以了
    # Dense全连接是一种结构，我们可以用使用FC进行Embedding的功能，但不是必须用这种结构。所以这里做embedding时候应该是使用了类似Dense的结构，所以只做了Add操作
    fst_order_sparse_layer = Add()(sparse_1d_embed)
    linear_part = Add()([fst_order_dense_layer, fst_order_sparse_layer])
    return linear_part


# CIN
# 有没有可能引入更高阶的vector-wise的交叉特征，同时又能控制模型的复杂度，避免产生过多的无效交叉特征呢？让我们先来思考一个问题。二阶交叉特征通过穷举所有的原始特征对得到，那么通过穷举的方法得到更高阶的交叉特征，必然会产生组合爆炸的维数灾难，导致网络参数过于庞大而无法学习，同时也会产生很多的无效交叉特征。让我们把这个问题称之为维数灾难挑战。
# 解决维数灾难挑战不可避免的就是要引入某种"压缩"机制，就是要把高阶的组合特征向量的维数降到一个合理的范围，同时在这个过程中尽量多的保留有效的交叉特征，去除无效的交叉特征。让我们谨记，所有构建高阶交叉特征的模型必然要引入特定的"压缩"机制，在学习建模高阶交叉特征的模型时我们脑中要始终绷紧一根弦，那就是这种压缩机制是如何实现的？这种压缩机制的效率和效果如何？
# CIN:最终学习出的特征交互的阶数是由网络的层数决定的，每一层隐层都通过一个池化操作连接到输出层，从而保证了输出单元可以见到不同阶数的特征交互模式。
# CIN的输入来自embedding层，假设有m个field，每个field的embedding维度为D，在进入CIN网络之前，需要先将sparse特征进行embedding并构建X0
def get_sparse_kd_embed(sparse_feats, sparse_inputs, D=8):
    # D:embedding size
    
    # 下面做的操作应该是把每个特征embedding成8维的向量，然后把embedding之后的每个特征拼接起来
    sparse_kd_embed = []
    for i, _input in enumerate(sparse_inputs):
        # f:特征名称
        f = sparse_feats[i]
        # voc_size:f特征不重复元素个数
        voc_size = total_data[f].nunique()
        reg = tf.keras.regularizers.l2(0.7)
        # voc_size为什么要加1？？？
        _embed = Embedding(voc_size+1, D, embeddings_regularizer=reg)(_input)
        sparse_kd_embed.append(_embed)
    return sparse_kd_embed


def compressed_interaction_net(x0, xl, D, n_filters):
    """

    :param x0: 原始输入
    :param xl: 第l层的输入
    :param D: embedding的维度
    :param n_filters: 压缩网络filter的数量
    :return:
    """
    # m个特征(field)，每个特征D维，X0的第i个特征D维与Xk第j个特征的D维进行
    # 在这里假设x0中有m个特征，xl中有h个特征
    # 1. 将x0与xl按照k所在的维度(-1)进行拆分，每个都可以拆成D列。应该是在embedding维度D上拆分，得到D个向量。
    x0_cols = tf.split(x0, D, axis=-1)  # ?, m, D
    xl_cols = tf.split(xl, D, axis=-1)  # ?, h, D
    assert len(x0_cols) == len(xl_cols), print('error shape!')

    # 2. 遍历D列，对于x0与xl所在的第i列进行外积计算，存在feature_maps中
    feature_maps = []
    for i in range(D):
        # transpose_b=True将x0_cols[i]转置
        feature_map = tf.matmul(xl_cols[i], x0_cols[i], transpose_b=True)  # 外积 ?,h,m
        feature_map = tf.expand_dims(feature_map, axis=-1)  # ?,h,m,1
        feature_maps.append(feature_map)

    # 3. 得到h × m × D 的三维tensor，查看axis设置的意义？？？
    feature_maps = Concatenate(axis=-1)(feature_maps)  # ?,h,m,D

    # 4. 压缩网络
    x0_n_feats = x0.get_shape()[1]  # m
    xl_n_feats = xl.get_shape()[1]  # h
    reshaped_feature_maps = Reshape(target_shape=(x0_n_feats*xl_n_feats, D))(feature_maps)  # ?,h*m,D
    transposed_feature_maps = tf.transpose(reshaped_feature_maps, [0, 2, 1])  # ?,D,h*m
    # Conv1D：使用n_filters个 形状为 1*(h*m)的卷积核以1为步长
    # 按嵌入维度D的方向进行卷积，最终得到形状为 ?,D, n_filters的输出,kernel_size？？？
    new_feature_maps = Conv1D(n_filters, kernel_size=1, strides=1)(transposed_feature_maps)  # ?,D,n_filters
    # 为了保持输出结果最后一维为嵌入维度D，需要进行转置操作，为什么要保持这样的shape？？？
    new_feature_maps = tf.transpose(new_feature_maps, [0, 2, 1])  # ?, n_filters, D
    return new_feature_maps


# 这里 n_filters 是经过该 CIN 层后，输出的 feature map 的个数，也就是说最终生成了由 n_filters 个 D 维向量组成的输出矩阵。
# 有了单层的CIN实现，下面可以实现多层CIN网络，注意CIN网络的输出是每一层的feature maps进行sum pooling，然后concat起来
# feature maps是沿着embedding维度进行卷积后的向量，最后会得到Hk+1个featuremap
# 一层CIN包含了从x0到xl，多层CIN包括了多个x0到xl
def build_cin(x0, D=8, n_layers=3, n_filters=12):
    """
    构建多层CIN网络
    :param x0: 原始输入的feature maps：?,m,D
    :param D: 特征embedding的维度
    :param n_layers: CIN网络层数
    :param n_filters: 每层CIN网络输出的feature_maps的个数
    :return:
    """
    # 存储每一层cin sum pooling的结果
    pooling_layers = []
    xl = x0  
    for layer in range(n_layers):
        xl = compressed_interaction_net(x0, xl, D, n_filters)
        # sum pooling
        pooling = Lambda(lambda x: K.sum(x, axis=-1))(xl)
        pooling_layers.append(pooling)

    # 将所有层的pooling结果concat起来
    output = Concatenate(axis=-1)(pooling_layers)
    return output

# bit-wise和vector-wise的区别应该是：bit-wise在隐向量的每一个维度对应一个权重，而vector-wise在隐向量整体对应一个权重参数
# DNN有能力学习出任意的函数，却没有理论结论，而且无法知道特征交叉到了什么程度。但是学习到的特征交互都是隐式的，并且都是基于bit-wise level， bit-wise特征交叉就是神经网络中节点之间的交互。隐式我们不知道模型所做的特征交叉实在多哪些特征的交叉
# vector-wise特征交互是每个特征embedding后的整个embedding vector作为一个域，然后对域和域之间进行特征交叉
# DNN部分
def dnn(sparse_kd_embed):
    # 输入DNN部分的sparse_kd_embed连接方式与输入CIN的sparse_kd_embed连接方式不一样，axis不一样
    embed_inputs = Flatten()(Concatenate(axis=-1)(sparse_kd_embed))

    fc_layer = Dropout(0.5)(Dense(128, activation='relu')(embed_inputs))
    fc_layer = Dropout(0.3)(Dense(128, activation='relu')(fc_layer))
    fc_layer_output = Dropout(0.1)(Dense(128, activation='relu')(fc_layer))
    return fc_layer_output


def output_and_model(linear_part, cin_layer, fc_layer_output):

    # 输出部分
    concat_layer = Concatenate()([linear_part, cin_layer, fc_layer_output])
    output_layer = Dense(1, activation='sigmoid')(concat_layer)
    # 完善模型
    model = Model(dense_inputs+sparse_inputs, output_layer)
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['binary_crossentropy', tf.keras.metrics.AUC(name='auc')])
    return model


if __name__ == "__main__":
    # 两个数据源(用其中一个就可以)，criteo_sampled_data是参考代码中提供的数据，user和movie是小象数据
    data_path = '../dataset/criteo_sampled_data/criteo_sampled_data.csv'
    data, dense_feats, sparse_feats = data_load(data_path)
    total_data, dense_inputs, sparse_inputs, sparse_1d_embed = feature_construction(data, dense_feats, sparse_feats)
    # Linear
    linear_part = linear(dense_inputs, sparse_1d_embed)
    # CIN
    sparse_kd_embed = get_sparse_kd_embed(sparse_feats, sparse_inputs, 8)
    # 构建feature_map X0
    input_feature_map = Concatenate(axis=1)(sparse_kd_embed)
    # 生成CIN, input_feature_map来自sparse特征
    # 经过3层CIN网络，每一层filter个数是12，意味着会产生出3个12*D的feature maps，再经过sum-pooling和concat后，就得到3*12=36维的向量
    cin_layer = build_cin(input_feature_map)
    # DNN
    fc_layer_output = dnn(sparse_kd_embed)
    model = output_and_model(linear_part, cin_layer, fc_layer_output)

    # 训练模型
    train_data = total_data.loc[:800000-1]
    valid_data = total_data.loc[800000:]

    # 下面这是什么类型，列表中包含
    train_dense_x = [train_data[f].values for f in dense_feats]
    train_sparse_x = [train_data[f].values for f in sparse_feats]
    train_label = [train_data['label'].values]

    val_dense_x = [valid_data[f].values for f in dense_feats]
    val_sparse_x = [valid_data[f].values for f in sparse_feats]
    val_label = [valid_data['label'].values]

    model.fit(train_dense_x+train_sparse_x, train_label, epochs=5, batch_size=128,
              validation_data=(val_dense_x+val_sparse_x, val_label))
    print('finish')

# model.有evaluate，predict等方法









