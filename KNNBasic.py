import pandas as pd
import numpy as np
from surprise import Dataset
from surprise import Reader
from surprise import KNNBasic
from collections import defaultdict
from surprise.model_selection import train_test_split

from surprise.model_selection import GridSearchCV

user_data = pd.read_csv(r'./dataset/dataset1/user.csv')
movie_data = pd.read_csv(r'./dataset/dataset1/movie.csv')

movie_list = movie_data['电影名'].unique().tolist()
movie_data['movie_id'] = movie_data['电影名'].apply(lambda x: movie_list.index(x))
movie_name_id = movie_data[['电影名', 'movie_id']].drop_duplicates()

user_data.drop(columns=['用户名','评论时间','类型'], inplace=True)

user_item_rating = pd.merge(user_data, movie_name_id, on='电影名')
user_item_rating.rename(columns = {'评分':'rating', '用户ID':'userID', '电影名':'itemID'}, inplace=True)

# 传入的列必须对应着 userID，itemID 和 rating（严格按此顺序）
reader = Reader(rating_scale = (2, 10))
data = Dataset.load_from_df(user_item_rating[['userID', 'itemID', 'rating']], reader)

# test set is made of 25% of the ratings.
#trainset, testset = train_test_split(data, test_size=.25)
trainset = data.build_full_trainset()

#网格搜索最佳参数,KNNBasic不需要网格搜索吗
# param_grid = {'n_epochs': [5, 10], 'lr_all': [0.002, 0.005],
#               'reg_all': [0.4, 0.6]}
# gs = GridSearchCV(KNNBasic, param_grid, measures=['rmse', 'mae'], cv=3)
# gs.fit(data)
# print(gs.best_score['rmse'])
# print(gs.best_params['rmse'])

sim_options = {'name': 'cosine',
               'user_based': False  # 计算物品间的相似度
               }
algo = KNNBasic(sim_options=sim_options)
algo.fit(trainset)

predictions = algo.test(trainset.build_testset())

def get_top_n(predictions, n=10):
    '从一个预测集中为每个用户返回top-N个推荐'
    '''
    参数：
        predictions(list of Prediction objects): 预测对象列表，由某个用于预测的算法返回.
            —————————————————————————————————————————————————————————————————
        n(int):为每个用户进行的推荐的数量。 默认值为10.
            —————————————————————————————————————————————————————————————————

    返回：
        一个字典，字典的键是用户（原始）ID，字典对应的值是为这个用户推荐的n个元组的列表：
        [(物品1原始id, 评分预测1), ...,(物品n原始id, 评分预测n)]
    '''

    # 首先将预测值映射至每个用户
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n

top_n = get_top_n(predictions, n=10)
