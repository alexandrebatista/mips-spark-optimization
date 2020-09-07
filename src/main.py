# import some modules to be used in py-spark SQL
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql import HiveContext
# Initialize Saprk Session
spark=SparkSession.builder.appName("PySpark_Testing").getOrCreate()
sc = spark.sparkContext
sqlContext = HiveContext(sc)