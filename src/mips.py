from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

def process(usersfactors, itensfactors):

    kmeans = KMeans(k=2, seed=1)  # 2 clusters here
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