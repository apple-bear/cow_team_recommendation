import pandas as pd
import numpy as np
import tensorflow as tf


def train_test_split(df, holdout_num):
    """
    对于每个user，在所有打分的movie中留出holdout_num个movie rating作为测试集
    :param df:
    :param holdout_num: number of items to be held out per user as testing items
    :return:
    """
    df = df.sort_values(['user_id', 'timestamp'], ascending=[True, False])
    df_train = df.copy(deep=True)
    df_test = df.copy(deep=True)

    df_test = df_test.groupby(['user_id']).head(holdout_num).reset_index()
    df_train = df_train.merge(df_test[['user_id', 'movie_id', 'rating', 'timestamp']].assign(remove=1), how='left').\
        query('remove != 1').drop('remove', 1).reset_index(drop=True)
    assert len(df) == len(df_train) + len(df_test)
    return df_train, df_test


# 第一种：根据rating的值设定一个threshold，大于等于threshold就是正向标签，小于threshold就是负向标签。(这版代码我目前采用>=5分认为是喜欢，<5分是不喜欢)
# 第二种：我们正在进行的是一项二分类任务，而且能够通过正向标签判断用户喜爱哪些条目，（相反的问题是如何定义不喜欢呢？）
# 这里我们可以对用户未评分的影片进行随机抽样，并将其视为负向标签。这种方法被称为负采样。以下函数可用于实现负采样过程：
# 这两种确定负向标签的方法，哪种更合适呢？
def negative_sampling(user_ids, movie_ids, items, n_neg):
    """
    没有rating的(user, movie)作为负向标签
    :param user_ids: list of uesr ids
    :param movie_ids: list of movie ids
    :param items: unique list of movie ids
    :param n_neg: number of negative labels to sample
    :return: negative sample dataframe
    """
    neg = []
    ui_pairs = zip(user_ids, movie_ids)
    records = set(user_ids)
    # 针对每一个正向样本，都会生成n_neg个负向样本
    for (u, i) in records:
        for _ in range(n_neg):
            j = np.random.choice(items)
            # resample if the movie already exists for that user
            while (u, j) in records:
                j = np.random.choice(items)
            neg.append([u, j, 0])

    df_neg = pd.DataFrame(neg, columns=['user_id', 'movie_id', 'rating'])
    return df_neg


# 定义GMF和MLP的嵌入层
def _get_user_embedding_layers(inputs, emb_dim):
    # 生成user embeddings
    user_gmf_emb = tf.keras.layers.Dense(emb_dim, activation='relu')(inputs)
    user_mlp_emb = tf.keras.layers.Dense(emb_dim, activation='relu')(inputs)
    return user_gmf_emb, user_mlp_emb


def _get_item_embedding_layers(inputs, emb_dim):
    item_gmf_emb = tf.keras.layers.Dense(emb_dim, activation='relu')(inputs)
    item_mlp_emb = tf.keras.layers.Dense(emb_dim, activation='relu')(inputs)
    return item_gmf_emb, item_mlp_emb
# 上面这两个生成embedding的函数应该可以合为一个吧？？？


# 实现GMF，将user与item嵌入相乘
def _gmf(user_emb, item_emb):
    # Multiply层接收一个列表的同shape张量，返回它们的逐元素积的张量，shape不变
    gmf_mat = tf.keras.layers.Multiply()([user_emb, item_emb])
    return gmf_mat


# Layer这个类，我们可以让一个对象当函数用，因此将下面代码拆成后两行的样子是等价的
# x = Dense(64, activation='relu')(x)
# 等价于
# fc = Dense(64, activation='relu')
# x = fc(x)


# 实现MLP，神经协作过滤的作者们已经表示，根据多轮不同实验的验证，拥有64维用户与条目隐性因子的四层MLP具备最好的性能表现，
# 因此我们也将在示例中使用同样的多层感知器结果：
def _mlp(user_emb, item_emb, dropout_rate):
    def add_layer(dim, input_layer, dropout_rate):
        hidden_layer = tf.keras.layers.Dense(dim, activation='relu')(input_layer)
        if dropout_rate:
            dropout_layer = tf.keras.layers.Dropout(dropout_rate)(hidden_layer)
            return dropout_layer
        return hidden_layer

    concat_layer = tf.keras.layers.Concatenate()([user_emb, item_emb])
    dropout_l1 = tf.keras.layers.Dropout(dropout_rate)(concat_layer)
    dense_layer_1 = add_layer(64, dropout_l1, dropout_rate)
    dense_layer_2 = add_layer(32, dense_layer_1, dropout_rate)
    # 为什么最后两个隐层不用dropout了呢？？？
    dense_layer_3 = add_layer(16, dense_layer_2, None)
    dense_layer_4 = add_layer(8, dense_layer_3, None)
    return dense_layer_4


# 为了产生最终预测，将GMF与MLP的输出结果合并如下：
def _neuCF(gmf, mlp, dropout_rate):
    concat_layer = tf.keras.layers.Concatenate()([gmf, mlp])
    output_layer = tf.keras.layers.Dense(1, activation='sigmoid')(concat_layer)
    return output_layer


# 为了分步构建这套完整的解决方案，下面创建一项图构建函数
def build_graph(user_dim, item_dim, dropout_rate=0.25):
    """
    Neural Collaborative Filtering model
    :param use_dim: one hot encoded user dimension
    :param item_dim: ont hot encoded item dimension
    :param dropout_rate:
    :return: Neural Collaborative Filtering model graph
    """
    user_input = tf.keras.Input(shape=(user_dim))
    item_input = tf.keras.Input(shape=(item_dim))

    # 生成embedding层
    user_gmf_emb, user_mlp_emb = _get_user_embedding_layers(user_input, 32)
    item_gmf_emb, item_mlp_emb = _get_item_embedding_layers(item_input, 32)

    # General Matrix Factorization
    gmf = _gmf(user_gmf_emb, item_gmf_emb)
    # Multi Layer Perceptron
    mlp = _mlp(user_mlp_emb, item_mlp_emb, dropout_rate)
    # output
    output = _neuCF(gmf, mlp, dropout_rate)
    # create the model
    model = tf.keras.Model(inputs=[user_input, item_input], outputs=output)
    return model

# # 使用keras的plot_model验证上面构建的网络架构是否正确
# # build graph
# # one hot encoded user dimension就是有多少不同的user_id么？？？
# n_user = user_item_rating['user_id'].nunique()
# n_item = user_item_rating['movie_id'].nunique()
# ncf_model = build_graph(n_user, n_item)
# tf.keras.utils.plot_model(ncf_model, to_file='neural_collaborative_filtering_model.png')


def train_eval_pred(model, x_train, y_train, x_val, y_val, x_test, y_test):
    model.compile(optimizer=tf.keras.optimizers.RMSprop(),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    print('start fit')

    history = model.fit(x_train, y_train, batch_size=64, epochs=3, validation_data=(x_val, y_val))
    print('history dict:\n', history.history)
    eval_result = model.evaluate(x_test, y_test, batch_size=128)
    print('test loss, test acc:', eval_result)
    predictions = model.predict(x_test[:3])
    return history, eval_result, predictions

if __name__ == "__main__":
    # 读入user数据
    user_file_path = '../dataset/dataset1/user.csv'
    user_data = pd.read_csv(user_file_path, usecols=['评分', '评论时间', '用户ID', '电影名'])

    # 读入movie数据
    movie_file_path = '../dataset/dataset1/movie.csv'
    movie_data = pd.read_csv(movie_file_path)

    # movie表中不重复的电影名列表
    movie_list = movie_data['电影名'].unique().tolist()
    movie_data.loc[:, 'movie_id'] = movie_data['电影名'].apply(lambda x: movie_list.index(x))
    #
    # 把movie表中的movie_id映射到user表中
    # 需要对movie_data按照电影名和movie_id进行去重，不然后面进行merge时候会出现重复值
    movie_name_id = movie_data[['电影名', 'movie_id']].drop_duplicates()
    user_data = user_data.merge(movie_name_id, on=['电影名'], how='left')

    user_item_rating = user_data[['用户ID', 'movie_id', '评分', '评论时间']]
    user_item_rating.rename(columns={'用户ID': 'user_id', '评分': 'rating', '评论时间': 'timestamp'}, inplace=True)
    # 根据rating生成label，rating>=5,label=1(喜欢)，rating<5,label=0(不喜欢)
    user_item_rating.loc[:, 'label'] = pd.cut(user_item_rating.rating, bins=[0, 5, 10], include_lowest=True, labels=[0, 1]).astype(int)
    # 划分训练集和测试集，整体数据根据timestamp排序，把时间靠后的N个作为测试集
    df_train, df_test = train_test_split(user_item_rating, 3)
    # 除了Numpy arrays和TensorFlow Datasets，还可以使用Pandas dataframes或从生成批处理的Python generators中训练Keras模型。
    # 通常，如果您的数据较小且适合内存，则建议您使用Numpy输入数据，否则，建议使用数据集。
    # 把dataframe转换成array
    array_train = df_train[['user_id', 'movie_id', 'label']].values
    array_test = df_test[['user_id', 'movie_id', 'label']].values
    # array_train数据的前两列作为x_train
    x_train = array_train[:, [0, 1]].reshape((-1, 2))
    y_train = array_train[:, 2].reshape((-1,1))
    # 从train中取1000个样本作为验证集
    x_val = x_train[-1000:]
    y_val = y_train[-1000:]
    x_train = x_train[:-1000]
    y_train = y_train[:-1000]
    x_test = array_test[:, [0, 1]].reshape((-1, 2))
    y_test = array_test[:, 2]
    # 13545是不重复的user_id的数量，23031是不重复的movie_id的数量
    model = build_graph(13545, 23031)
    train_eval_pred(model, x_train, y_train, x_val, y_val, x_test, y_test)
