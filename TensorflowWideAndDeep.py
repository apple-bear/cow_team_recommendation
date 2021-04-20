import pandas
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


# Training samples path, change to your local path
# samples_file_path = tf.keras.utils.get_file("user.csv","/Users/qizhenpeng/学习/机器学习/推荐系统/练习项目资料/dataset/dataset1/user.csv")


# load sample as tf dataset
def get_dataset(file_path):
    dataset = tf.data.experimental.make_csv_dataset(
        file_path,
        batch_size=12,
        label_name='label',
        na_value="0",
        num_epochs=1,
        ignore_errors=True)
    return dataset

# 用于创建一个特征列
# 并转换一批次数据的一个实用程序方法
def demo(feature_column, batch=10):
    feature_layer = tf.keras.layers.DenseFeatures(feature_column)
    print(feature_layer(batch).numpy())

# 一种从 Pandas Dataframe 创建 tf.data 数据集的实用程序方法（utility method）
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    dataframe = dataframe.copy()
    labels = dataframe.pop('target')
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    return ds

if __name__ == '__main__':
    user_df = pandas.read_csv("../../data/dataset1/user.csv")
    print(user_df.head(5))

    enc = LabelEncoder()
    user_df['movieId'] = enc.fit_transform(user_df['电影名'])
    user_df['target'] = user_df['评分'].apply(lambda x: 1 if x > 3 else 0)
    user_df.rename(columns={"用户ID": "userId", "类型": "movieType"}, inplace=True)

    maxMovieId = user_df['movieId'].max()
    maxUserId = user_df['userId'].max()
    print(maxMovieId)
    print(maxUserId)

    train, test = train_test_split(user_df, test_size=0.5)

    test_movie_count = len(test['movieId'].unique())
    test_user_count = len(test['userId'].unique())

    print(user_df.head(5))

    batch_size = 50  # 小批量大小用于演示
    train_ds = df_to_dataset(train, batch_size=batch_size)
    test_ds = df_to_dataset(test, batch_size=batch_size)
    # genre features vocabulary
    genre_vocab = user_df['movieType'].unique()

    GENRE_FEATURES = {
        'movieType': genre_vocab,
    }

    # all categorical features
    categorical_columns = []
    for feature, vocab in GENRE_FEATURES.items():
        cat_col = tf.feature_column.categorical_column_with_vocabulary_list(
            key=feature, vocabulary_list=vocab)
        emb_col = tf.feature_column.embedding_column(cat_col, 10)
        categorical_columns.append(emb_col)
    # movie id embedding feature
    movie_col = tf.feature_column.categorical_column_with_identity(key='movieId', num_buckets=maxMovieId + 1)
    movie_emb_col = tf.feature_column.embedding_column(movie_col, 10)
    categorical_columns.append(movie_emb_col)

    # user id embedding feature
    user_col = tf.feature_column.categorical_column_with_identity(key='userId', num_buckets=maxUserId + 1)
    user_emb_col = tf.feature_column.embedding_column(user_col, 10)
    categorical_columns.append(user_emb_col)

    # define input for keras model
    inputs = {
        'movieId': tf.keras.layers.Input(name='movieId', shape=(), dtype='int32'),
        'userId': tf.keras.layers.Input(name='userId', shape=(), dtype='int32'),
        'movieType': tf.keras.layers.Input(name='movieType', shape=(), dtype='string'),
    }

    # wide and deep model architecture
    # deep part for all input features
    deep = tf.keras.layers.DenseFeatures(categorical_columns)(inputs)
    deep = tf.keras.layers.Dense(1024, activation='relu')(deep)
    deep = tf.keras.layers.Dense(1024, activation='relu')(deep)
    # wide part for cross feature
    wide = tf.keras.layers.DenseFeatures(categorical_columns)(inputs)
    both = tf.keras.layers.concatenate([deep, wide])
    output_layer = tf.keras.layers.Dense(1, activation='sigmoid')(both)
    model = tf.keras.Model(inputs, output_layer)

    # compile the model, set loss function, optimizer and evaluation metrics
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy', tf.keras.metrics.AUC(curve='ROC'), tf.keras.metrics.AUC(curve='PR')])

    # train the model
    model.fit(train_ds, epochs=5)
    # 获取用户embedding
    user_weights = model.get_weights()[2]
    model.summary()
    # evaluate the model
    test_loss, test_accuracy, test_roc_auc, test_pr_auc = model.evaluate(test_ds)
    print('\n\nTest Loss {}, Test Accuracy {}, Test ROC AUC {}, Test PR AUC {}'.format(test_loss, test_accuracy,
                                                                                       test_roc_auc, test_pr_auc))
    # print some predict results
    predictions = model.predict(test_ds)
    for prediction, goodRating in zip(predictions[:20], list(test_ds)[0][1][:20]):
        print("Predicted good rating: {:.2%}".format(prediction[0]),
              " | Actual rating label: ",
              ("Good Rating" if bool(goodRating) else "Bad Rating"))



