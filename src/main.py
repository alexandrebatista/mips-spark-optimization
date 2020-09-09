# import some modules to be used in py-spark SQL
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql import HiveContext

from tools import getAllUsers 
# Initialize Saprk Session
import os

DATA_PATH = '../data/'
RESULTS_PATH = '../results/'

USERS_FILE_NAME = 'users.txt'

fileNames = [DATA_PATH + 'combined_data_1.txt', DATA_PATH + 'combined_data_2.txt', DATA_PATH + 'combined_data_3.txt', DATA_PATH + 'combined_data_4.txt']

if not os.path.isfile(RESULTS_PATH + USERS_FILE_NAME):
    users = getAllUsers(fileNames)

    f = open(USERS_FILE_NAME, "w")
    for user in users:
        f.write(str(user) + ',')
    f.close()

#print(users)
#spark=SparkSession.builder.appName("PySpark_Testing").getOrCreate()
#sc = spark.sparkContext
#sqlContext = HiveContext(sc)