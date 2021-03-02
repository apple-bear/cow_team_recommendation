#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os
from surprise import SVD
from surprise import Reader, Dataset
from surprise.model_selection import GridSearchCV
# from surprise import evaluate, print_perf


# 指定文件路径（包含‘user-item-rating’的数据）
# 协同过滤召回阶段不需要timestamp列吧？？？
file_path = "./dataset/dataset1/user.csv"
df = pd.read_csv(file_path, usecols=['用户ID', '电影名', '评分'])
df = df.rename(columns={'评分':'rating', '用户ID':'user', '电影名':'item'})

# 从数据中看评分的值有5种：2,4,6,8,10，范围是(2,10)
# rating_scale, tuple:The minimum and maximal rating of the rating scale.
reader = Reader(rating_scale=(2,10))
data = Dataset.load_from_df(df[['user', 'item', 'rating']], reader)

# 定义需要优选的参数网格
param_grid = {'n_epochs': [5, 10], 'lr_all': [0.002, 0.005],
              'reg_all': [0.4, 0.6]}
gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae', 'fcp'], cv=3)
gs.fit(data)

# 最佳 RMSE 得分
print('最佳RMSE得分\n', gs.best_score['rmse'])
print('最佳FCP得分\n', gs.best_score['fcp'])

# 能达到最佳 RMSE 得分的参数组合
print('最佳RMSE对应的参数组合\n', gs.best_params['rmse'])
print('最佳FCP对应的参数组合\n', gs.best_params['fcp'])


# In[29]:


# 算法效果对比
from surprise import NormalPredictor # 假设评分数据来自一个正态分布
from surprise import BaselineOnly    #
from surprise import KNNWithMeans    # 在基础的CF算法上，去除了平均的均值
from surprise import KNNBaseline     # 在KNNWithMeans基础上，用baseline的值替换均值
from surprise import SVD             # 矩阵分解算法svd(biasSVD与funkSVD)
from surprise import SVDpp           # 考虑了隐反馈
from surprise import NMF             # 非负矩阵分解
from surprise.model_selection import cross_validate

algo = NormalPredictor()
perf = cross_validate(algo, data, measures=['rmse', 'mae', 'fcp'], cv=3)
print('NormalPredictor perf:\n', perf)

algo = BaselineOnly()
perf = cross_validate(algo, data, measures=['rmse', 'mae', 'fcp'], cv=3)
print('BaselineOnly perf:\n', perf)

algo = KNNWithMeans()
perf = cross_validate(algo, data, measures=['rmse', 'mae', 'fcp'], cv=3)
print('KNNWithMeans perf:\n', perf)

algo = KNNBaseline()
perf = cross_validate(algo, data, measures=['rmse', 'mae', 'fcp'], cv=3)
print('KNNBaseline perf:\n', perf)

algo = SVD()
perf = cross_validate(algo, data, measures=['rmse', 'mae', 'fcp'], cv=3)
print('SVD perf:\n', perf)

algo = SVDpp()
perf = cross_validate(algo, data, measures=['rmse', 'mae', 'fcp'], cv=3)
print('SVDpp perf:\n', perf)

algo = NMF()
perf = cross_validate(algo, data, measures=['rmse', 'mae', 'fcp'], cv=3)
print('NMF perf:\n', perf)


# In[3]:


# 算法选定，调参完成，进行预测

trainset = data.build_full_trainset()
# sim_options = {'name': 'pearson_baseline', 'user_based':False}
# # 基线估计(Baselines estimates)配置
# bsl_options = {'method':'als',
#               'reg_u':12,
#               'reg_i':5,
#               'n_epochs':5,}

# # bsl_options = {'method':'sgd',
# #               'learning_rate':.00005,
# #               'n_epochs':10}
# algo = KNNBaseline(k=20, min_k=1, sim_options=sim_options, bsl_options=bsl_options)
# algo.fit(trainset)

# 假设给定一个用户列表，给每一个用户推荐topN
user_inner_id_list = trainset.all_users()
for user_inner_id in user_inner_id_list:
    


# In[99]:


# 方法一：基于KNNBASE的推荐，找到用户评分最高的电影，推荐N个
from surprise import KNNBasic
movie = pd.read_csv(r".\dataset\dataset1\movie.csv")
rating = pd.read_csv(r".\dataset\dataset1\user.csv")
movie["movie_id"] = movie.index+1

rating_new = rating.merge(movie[["电影名","movie_id"]],how="left", left_on = "电影名", right_on ="电影名")
rating_new = rating_new[["用户ID",'movie_id','评分']]


# In[100]:


# 训练推荐模型 步骤:1
def getSimModel():
    # A reader is still needed but only the rating_scale param is requiered.
    reader = Reader(rating_scale=(2, 10))
    # The columns must correspond to user id, item id and ratings (in that order).
    data = Dataset.load_from_df(rating_new, reader)
    
    trainset = data.build_full_trainset()
    
    #sim_options = {'name': 'pearson_baseline', 'user_based': False}
    ##使用KNNBaseline算法
    #algo = KNNBaseline(sim_options=sim_options)
    algo = KNNBasic()

    #训练模型
    algo.fit(trainset)
    return algo


# In[101]:


# 获取id到name的互相映射  步骤:2
def read_item_names():
    """
    获取电影名到电影id 和 电影id到电影名的映射
    """
    rid_to_name = {}
    name_to_rid = {}
    for i in range(0,len(movie)):
#     print(movie.loc[i,"电影名"],movie.loc[i,"movie_id"])
        rid_to_name[movie.loc[i,"movie_id"]] = movie.loc[i,"电影名"]
        name_to_rid[movie.loc[i,"电影名"]] = movie.loc[i,"movie_id"]
    return rid_to_name, name_to_rid


# In[102]:


# 基于之前训练的模型 进行相关电影的推荐  步骤：3
def showSimilarMovies(algo, rid_to_name, name_to_rid, movie_name, k):
    # 获得电影Toy Story (1995)的raw_id
    toy_story_raw_id = name_to_rid[movie_name]
    #把电影的raw_id转换为模型的内部id
    toy_story_inner_id = algo.trainset.to_inner_iid(toy_story_raw_id)
    #通过模型获取推荐电影 这里设置的是10部
    toy_story_neighbors = algo.get_neighbors(toy_story_inner_id, k)
    #模型内部id转换为实际电影id
    neighbors_raw_ids = [algo.trainset.to_raw_iid(inner_id) for inner_id in toy_story_neighbors]
    #通过电影id列表 或得电影推荐列表
    neighbors_movies = [rid_to_name[raw_id] for raw_id in neighbors_raw_ids]
    print("The 10 nearest neighbors of "+ movie_name+" are:")
    for movie in neighbors_movies:
        print(movie) 


# In[103]:


def getmovieid(userid):
    rating_new_user = rating_new[rating_new["用户ID"]==int(userid)]
    result = rating_new_user.sort_values(by="评分",ascending=False).reset_index(drop = False)
    return list(result.loc[0,["movie_id"]])[0]


# In[104]:


# 获取id到name的互相映射
rid_to_name, name_to_rid = read_item_names()


# In[105]:


# 训练推荐模型
algo = getSimModel()


# In[106]:


userid = input("input userid")


# In[108]:


movieid = getmovieid(userid)


# In[109]:


moviename = rid_to_name[movieid]


# In[110]:


##显示相关电影
showSimilarMovies(algo, rid_to_name, name_to_rid, moviename,10)


# In[111]:


# 方法二：基于svd的推荐
def get_top_n(predictions, n=10):
    """Return the top-N recommendation for each user from a set of predictions.

    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.

    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    """

    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n


# In[112]:


# First train an SVD algorithm on the douban dataset.
trainset = data.build_full_trainset()
algo = SVD()
algo.fit(trainset)


# In[113]:


# Than predict ratings for all pairs (u, i) that are NOT in the training set.
testset = trainset.build_anti_testset()
predictions = algo.test(testset)


# In[114]:


top_n = get_top_n(predictions, n=10)

# Print the recommended items for each user
for uid, user_ratings in top_n.items():
    print(uid, [iid for (iid, _) in user_ratings])


# In[ ]:




