from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.mllib.recommendation import ALS
from pyspark.sql.functions import lit
from pyspark.ml.feature import StringIndexer


# 创建SparkContext和SparkSession
def create_spark_context():
    spark_conf = SparkConf().setAppName('movie_recommend').setMaster('local[*]')
    sc = SparkContext(conf=spark_conf)
    sc.setLogLevel("WARN")
    spark = SparkSession.builder.config(conf=spark_conf).getOrCreate()
    return sc, spark

# 这个方法是要把movie表变成一个dict，推荐电影时候可以根据movie_id(或者电影名)带出其他电影信息
# 两种方法：1是给movie这个表添加一个新列movie_id，2是movie表与user表根据电影名进行join，获得user表中的movie_id，
# 因为user表中有电影名StringIndexer之后的movie_id
# def get_movie_dict(spark):
#     movie_info_df = spark.read.csv('../dataset/dataset1/movie.csv', header=True)
#     movie_info_df = movie_info_df.select('类型', '主演', '导演')
#     movie_info_df = movie_info_df.withColumnRenamed("类型", 'movie_type').withColumnRenamed("主演", 'actors').withColumnRenamed("导演", 'director')
#     movie_info_dict = map(lambda row: {row[0]:row[1],row[2]})
#     return movie_info_dict


def recommend_movie_by_userid(model, user_id, num=5):
    """给定user_id，向其推荐top N电影"""
    # user_id需要传入的是int类型的值
    result = model.recommendProducts(user_id, num)
    return [(r.user, r.product, r.rating) for r in result]


def recommend_user_by_movieid(model, movie_id, num=5):
    """给定movie_id，推荐对该电影感兴趣的top N个用户"""
    # movie_id需要传入的是int类型的值
    result = model.recommendUsers(movie_id, num)
    return [(r.user, r.product, r.rating) for r in result]


if __name__ == "__main__":
    spark_context, spark = create_spark_context()
    # 读取user表的数据
    user_rdd = spark.read.csv('../dataset/dataset1/user.csv', header=True)
    # 增加一个新列movie_id
    # user_rdd.withColumn('movie_id', lit(0)).show()
    # 把电影名映射成数字，按照出现次数降序分配序号
    indexer = StringIndexer(inputCol="电影名", outputCol="movie_id").fit(user_rdd)
    movie_name_to_id = indexer.transform(user_rdd)
    movie_name_to_id.show()
    # 选择列
    user_item_rating = movie_name_to_id.select('用户ID', 'movie_id', '评分')
    print(user_item_rating.take(5))
    # 修改列名
    user_item_rating = user_item_rating.withColumnRenamed("评分", 'rating').withColumnRenamed("用户ID", 'user_id')
    print(user_item_rating.take(5))
    # 训练模型
    model = ALS.train(user_item_rating, 10, 10, 0.01)
    # 模型保存
    result1 = recommend_movie_by_userid(model, 1)
    print(result1)
    model.save(spark_context, '../code/model/spark_als_model2')
    result2 = recommend_user_by_movieid(model, 1)
    print(result2)








