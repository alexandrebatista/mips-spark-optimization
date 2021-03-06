from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row

import pandas as pd

def generateUsersItensVectors(inputFilePath):
    sc = SparkContext('local')
    spark = SparkSession(sc)

    lines = spark.read.text(inputFilePath).rdd
    parts = lines.map(lambda row: row.value.split("::"))
    ratingsRDD = parts.map(lambda p: Row(userId=int(p[0]), movieId=int(p[1]),
                                        rating=float(p[2])))
    ratings = spark.createDataFrame(ratingsRDD)
    (training, test) = ratings.randomSplit([0.8, 0.2])

    # Build the recommendation model using ALS on the training data
    # Note we set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics
    als = ALS(maxIter=5, regParam=0.01, userCol="userId", itemCol="movieId", ratingCol="rating",
            coldStartStrategy="drop")
    model = als.fit(training)

    return model.userFactors, model.itemFactors

# Evaluate the model by computing the RMSE on the test data
#predictions = model.transform(test)
#evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
#                                predictionCol="prediction")
#rmse = evaluator.evaluate(predictions)
#print("Root-mean-square error = " + str(rmse))
#
## Generate top 10 movie recommendations for each user
#userRecs = model.recommendForAllUsers(10)
## Generate top 10 user recommendations for each movie
#movieRecs = model.recommendForAllItems(10)
#
##als = ALS(maxIter=5, regParam=0.01, implicitPrefs=True,
##          userCol="userId", itemCol="movieId", ratingCol="rating")