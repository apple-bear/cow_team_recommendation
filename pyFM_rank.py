import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from pyfm import pylibfm
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from functools import reduce
import pickle
import os


def save_variable(v, filename):
    f = open(filename, 'wb')
    pickle.dump(v, f)
    f.close()
    return filename


def load_variable(filename):
    f = open(filename, 'rb')
    r = pickle.load(f)
    f.close()
    return r


def train_test_split(df, holdout_num):
    """
    对于每个user，在所有打分的movie中留出holdout_num个movie rating作为测试集
    :param df:
    :param holdout_num: number of items to be held out per user as testing items
    :return:
    """
    # 按照时间进行降序，所以head就是后出现的电影，方便作为测试集
    df = df.sort_values(['user_id', 'timestamp'], ascending=[True, False])
    df_train = df.copy(deep=True)
    df_test = df.copy(deep=True)

    df_test = df_test.groupby(['user_id']).head(holdout_num).reset_index()
    # df_train = df_train.merge(df_test[['user_id', 'movie_id', 'rating', 'timestamp']].assign(remove=1), how='left'). \
    #     query('remove != 1').drop('remove', 1).reset_index(drop=True)
    df_train = df_train.merge(df_test.assign(remove=1), how='left').\
        query('remove != 1').drop('remove', 1).reset_index(drop=True)
    assert len(df) == len(df_train) + len(df_test)
    return df_train, df_test
# def loadData(filename, path='../dataset/dataset1/'):
#     data = []
#     y = []
#     users = set()
#     items = set()
#     with open(path+filename) as f:
#         for line in f:
#             (user, movieid, rating, ts) = line.split(',')
#             data.append({"user_id": str(user), "movie_id": str(movieid)})
#             y.append(float(rating))
#             users.add(user)
#             items.add(movieid)
#     return data, np.array(y), users, items

# 导入数据，并转换格式

# train_data, y_train, train_users, train_items = loadData('user.csv')
# test_data, y_test, test_users, test_items = loadData('user.csv')
# 将类型、导演、演员、特色这样的离散特征列进行拆分统计，分别形成dict
def get_dim_dict(df, dim_name):
    # 把dim_name列的每一行转换成一个list
    type_list = list(map(lambda x: x.split('|'), df[dim_name]))

    # 把上一步的嵌套列表转换成一个大的list
    type_list = [x for l in type_list for x in l]

    def reduce_func(x, y):
        # 把第2项不断加入第1项，
        # 并且判断如果第2个单内容列表在第1项多内容列表中出现过，那么就把第1个列表中的相应内容删掉，然后在第1个列表中添加相应内容出现次数加1后的tuple
        # 如果不等于那就只是把第2项加入第1项中
        for i in x:
            if i[0] == y[0][0]:
                x.remove(i)
                x.append((i[0], i[1]+1))
                return x
        x.append(y[0])
        return x
    # 把每个出现的dim_name中的项带上数字1组成一个tuple的list
    l = filter(lambda x: x!=None, map(lambda x: [(x, 1)], type_list))
    type_zip = reduce(reduce_func, list(l))

    # 把tuple组成的列表转换成dict
    type_dict = {}
    for i in type_zip:
        type_dict[i[0]] = i[1]
    return type_dict


def data_process(user_file_path, movie_file_path):
    """
    把电影名映射成movie_id，合并user数据和movie数据
    :param user_file_path:
    :param movie_file_path:
    :return:
    """
    # 注意user表中的评分和movie表中的评分意义不一样
    # 读入user数据
    user_data = pd.read_csv(user_file_path, usecols=['评分', '评论时间', '用户ID', '电影名'])
    # 选择部分用户数据，减少运行时间
    user_data = user_data.iloc[:100, :]
    # 读入movie数据
    movie_data = pd.read_csv(movie_file_path, usecols=['电影名', '类型', '主演', '地区', '导演', '特色'])
    # 选择部分电影数据，减少运行时间
    movie_data = movie_data
    # movie表中不重复的电影名列表
    movie_list = movie_data['电影名'].unique().tolist()
    movie_data.loc[:, 'movie_id'] = movie_data['电影名'].apply(lambda x: movie_list.index(x))
    # 把movie表中的movie_id映射到user表中
    # 需要对movie_data按照电影名和movie_id进行去重，不然后面进行merge时候会出现重复值
    movie_name_id = movie_data[['电影名', 'movie_id']].drop_duplicates()
    # user_data = user_data.merge(movie_name_id, on=['电影名'], how='left')
    print('head1:\n', user_data.head())
    print('head2:\n', movie_data.head())
    user_data = user_data.merge(movie_data, on=['电影名'], how='left')
    print('head3:\n', user_data.head())

    user_item_rating = user_data[['用户ID', 'movie_id', '评分', '评论时间', '类型', '主演', '地区', '导演', '特色']]
    user_item_rating.rename(columns={'用户ID': 'user_id', '评分': 'rating', '评论时间': 'timestamp', '类型': 'type',
                                     '主演': 'actors', '地区': 'region', '导演': 'director', '特色': 'trait'}, inplace=True)
    # # 根据rating生成label，rating>=5,label=1(喜欢)，rating<5,label=0(不喜欢)
    # user_item_rating.loc[:, 'label'] = pd.cut(user_item_rating.rating, bins=[0, 5, 10], include_lowest=True,
    #                                           labels=[0, 1]).astype(int)
    return user_item_rating


# user_id和item_id每个id作为一列，导演或者演员就会冷门归为一列。最后我们用于训练的数据就是
# user_id embedding+item_id embedding+其他特征+其他特征的embedding，是这样吧？？？
# 还有就是既然做embedding了，演员和导演多一些有啥影响么，为啥要做这种冷门处理
# 反正都要降维的，多一些列再做embedding容易过拟合？
# 上面说法有问题，FM输入数据没有做embedding，只是做了one-hot，FM这个算法可以起到embedding的作用
# and 连接两个表达式，前面不满足后面就不计算了，&前面不满足后面还要计算
def user_and_movie_2_dict(df, actors_dict, director_dict):
    # 'user_id', 'movie_id',
    df[['type', 'actors', 'region', 'trait', 'director']] = df[['type', 'actors', 'region', 'trait', 'director']].astype(str)
    movie_dict_list = []
    # movie.csv中的每一行转换成一个dict，最后把所有行的dict组成一个list
    for i in df.index:
        movie_dict = {}
        movie_dict['user_id'] = df.loc[i, 'user_id']
        movie_dict['movie_id'] = df.loc[i, 'movie_id']
        for s_type in df.loc[i, 'type'].split('|'):
            movie_dict[s_type] = 1
            # if s_type != 'nan':
            #     movie_dict[s_type] = 1
        for actor in df.loc[i, 'actors'].split('|'):
            # if actor != 'nan' and actors_dict[actor] < 2:
            #     movie_dict['other_actor'] = 1
            if actors_dict[actor] < 2:
                movie_dict['other_actor'] = 1
            else:
                movie_dict['actor'] = 1
        movie_dict[df.loc[i, 'region']] = 1
        for director in df.loc[i, 'director'].split('|'):
            # if director != 'nan' and director_dict[director] < 2:
            #     movie_dict['other_director'] = 1
            if director_dict[director] < 2:
                movie_dict['other_director'] = 1
            else:
                movie_dict[director] = 1
        for trait in df.loc[i, 'trait'].split('|'):
            movie_dict[trait] = 1
            # if trait != 'nan':
            #     movie_dict[trait] = 1
        movie_dict_list.append(movie_dict)
    return movie_dict_list


if __name__ == "__main__":

    user_file_path = '../dataset/dataset1/user.csv'
    movie_file_path = '../dataset/dataset1/movie.csv'
    actors_dict_path = '../dataset/dataset1/dict/actors_dict.txt'
    director_dict_path = '../dataset/dataset1/dict/director_dict.txt'
    x_train_dict_path = '../dataset/dataset1/dict/x_train_dict.txt'
    x_test_dict_path = '../dataset/dataset1/dict/x_test_dict.txt'

    user_item_rating = data_process(user_file_path, movie_file_path)
    df_train, df_test = train_test_split(user_item_rating, 1)

    feature_col = ['user_id', 'movie_id', 'rating', 'type', 'actors', 'region', 'director', 'trait']

    train = df_train.loc[:, feature_col]
    test = df_test.loc[:, feature_col]

    # 为什么要变成double，变成double就不能one-hot了吧？？？
    # df_train = df_train[['user_id', 'movie_id', 'rating']].astype('double')
    # df_test = df_test[['user_id', 'movie_id', 'rating']].astype('double')

    # 为了后面做DictVectorizer，把特征列从df转换成dict
    if os.path.exists(actors_dict_path) and os.path.exists(director_dict_path):
        actors_dict = load_variable(actors_dict_path)
        director_dict = load_variable(director_dict_path)
    else:
        usecols = ['类型', '主演', '地区', '导演', '特色']
        movie_data = pd.read_csv(movie_file_path, usecols=usecols)
        movie_data.rename(columns={'类型': 'type', '主演': 'actors', '地区': 'region', '导演': 'director', '特色': 'trait'},
                          inplace=True)
        # 选择部分电影数据，减少运行时间
        movie_data = movie_data

        actors_dict = get_dim_dict(movie_data, 'actors')
        director_dict = get_dim_dict(movie_data, 'director')

        save_variable(actors_dict, actors_dict_path)
        save_variable(director_dict, director_dict_path)

    # train.dropna(axis=0, how='any', inplace=True)

    # 把训练集中的样本(用户+电影)数据转换成字典，（不确定。再检查一遍数据类型，id类特征数值类型是字符串，其他特征时转成one-hot形式后的数值型
    if os.path.exists(x_train_dict_path):
        x_train_dict = load_variable(x_train_dict_path)
    else:
        x_train_dict = user_and_movie_2_dict(train, actors_dict, director_dict)

    y_train = df_train['rating'].values

    if os.path.exists(x_test_dict_path):
        x_test_dict = load_variable(x_test_dict_path)
    else:
        x_test_dict = user_and_movie_2_dict(test, actors_dict, director_dict)
    y_test = df_test['rating'].values

    v = DictVectorizer()
    x_train = v.fit_transform(x_train_dict)
    x_test = v.transform(x_test_dict)

    scalar = preprocessing.MaxAbsScaler()
    scalar.fit(x_train)
    x_train_scaling = scalar.transform(x_train)
    x_test_scaling = scalar.transform(x_test)

    # 训练模型并测试
    fm = pylibfm.FM(num_factors=2, num_iter=2, verbose=True, task='regression', initial_learning_rate=0.001,
                    learning_rate_schedule='optimal')
    print('step 1')

    # y值类型必须是double，否则报错
    y_train = (y_train[train.index.values]).astype(np.double)

    fm.fit(x_train_scaling, y_train)
    print('step 2')
    # 预测结果打印误差
    pred = fm.predict(x_test_scaling)
    print('pred:', pred)
    print('step 3')
    print('FM MSE: %.4f' % mean_squared_error(y_test, pred))
    print('step 4')



    print('type1:', type(x_train_dict[0]['user_id']))
    print('type2:', type(x_train_dict[0]['movie_id']))
    # print('type3:', type(x_train_dict[0]['type']))
    # print('type4:', type(x_train_dict[0]['actors']))
    # print('type5:', type(x_train_dict[0]['region']))
    print('x_train shape:', x_train.toarray().shape)
    key_list = []
    for i in x_train_dict:
        key_list.append(i.keys()['Value'])
    key_set = set(key_list)
    print('key_set:\n', key_set)
    print('len key_set:', len(key_set))



