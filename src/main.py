# import some modules to be used in py-spark SQL
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql import HiveContext

from tools import buildFiles 
from test import generateUsersItensVectors
from mips import process

# Initialize Saprk Session
import os

DATA_PATH = '../data/'
RESULTS_PATH = '../results/'

INPUT_FILE_NAME = 'inputData-top10million.txt'

fileNames = [DATA_PATH + 'combined_data_1.txt', DATA_PATH + 'combined_data_2.txt', DATA_PATH + 'combined_data_3.txt', DATA_PATH + 'combined_data_4.txt']

if not os.path.isfile(RESULTS_PATH + INPUT_FILE_NAME):
    buildFiles(fileNames, RESULTS_PATH + INPUT_FILE_NAME)

usersFactors, itensFactors = generateUsersItensVectors(RESULTS_PATH + INPUT_FILE_NAME)

usersDataframe = usersFactors.toPandas()
itensDataframe = itensFactors.toPandas()

process(usersFactors, itensFactors)

print(usersDataframe)
print(itensDataframe)
    #f = open(USERS_FILE_NAME, "w")
    #for user in users:
    #    f.write(str(user) + ',')
    #f.close()

#print(users)
#spark=SparkSession.builder.appName("PySpark_Testing").getOrCreate()
#sc = spark.sparkContext
#sqlContext = HiveContext(sc)