from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
import pyspark.sql.functions as F


def f(iterator, centers):
    biggest = 0
    c = []
    for x in iterator:
        id = x.__getitem__("id")
        latentFactorsVector = x.__getitem__("features")
        partition = x.__getitem__("prediction")
        centroid = centers[partition]
        if id > biggest:
            biggest = id
            c = centroid
    print(biggest)
    print(c)

def process(usersfactors, itensfactors):
    kmeans = KMeans(k=4, seed=1)  # 4 clusters here
    model = kmeans.fit(usersfactors.select('features'))

    transformed = model.transform(usersfactors)
    transformed.show() 

    # Trains a k-means model.
    #kmeans = KMeans().setK(2).setSeed(1)
    #model = kmeans.fit(dataset)

    # Make predictions
    #predictions = model.transform(dataset)

    # Shows the result.
    centers = model.clusterCenters()
    print("Cluster Centers: ")
    for center in centers:
        print(center)

    #new_df = old_df.withColumn('col_n', old_df.col_1 - old_df.col_2)
    #transformed = transformed.withColumn('centroid', F.col('prediction'))
    #transformed = transformed.withColumn('centroid', F.col('prediction'))

    transformed = transformed.repartition('prediction')
    transformed.show()

    #transformed.select('id').write.csv('numbers')

    #transformed.foreachPartition(f)
    transformed.foreachPartition(lambda iterator: f(iterator, centers))