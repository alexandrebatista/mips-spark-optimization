from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
import pyspark.sql.functions as F
import numpy as np
from heapq import heapify, heappush, heappop 

def unit_vector(vector):
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def CBound(vec_c, vec_i, ang_tic, ang_tb):
	norma_i = np.linalg.norm(vec_i)
	test = norma_i * np.cos(ang_tic - ang_tb)
	return test if ang_tb < ang_tic else norma_i


def queryIndex():
    list_li = list(li)
    for i in L[K:]:
        if CBound(vec_c, vec_i, ang_tic, ang_tb) < min_heap:
            break
        elif (vec_u.T * vec_i) > min_heap:
            list_li.append(vec_u.T * i) # add i to H with weight u.T*i
        print(list_li)

def f(iterator, centers, itensDataframe):
    biggest = 0
    theta = -1000
    L = []
    k = 10
    
    # Construct Index
    for user in iterator:
        id = user.__getitem__("id")
        userLatentFactors = user.__getitem__("features")
        partition = user.__getitem__("prediction")
        centroid = centers[partition]

        angle = angle_between(userLatentFactors, centroid)
        if angle > theta:
            theta = angle

    if theta != -1000:
        for index, row in itensDataframe.iterrows():
            itemLatentFactors = row[1]
            angle = angle_between(itemLatentFactors, centroid)
            CB = CBound(centroid, itemLatentFactors, angle, theta)
            LItem = (row[0], CB, itemLatentFactors)
            L.append(LItem)

        L = sorted(L, reverse = True, key = lambda x: x[1])
        
        # Query Index
        for user in iterator:
            id = user.__getitem__("id")
            userLatentFactors = user.__getitem__("features")

            for itens in L[:K]:
                itemLatentFactors = itens[2]
                weight = np.dot(userLatentFactors, itemLatentFactors)
                #(weight, itens[0])
                #heapq.heapify(li)

         
        min_heap = heapq.heappop(li)

    if len(L) > 0:
        print(L)

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

    transformed = transformed.repartition('prediction')
    transformed.show()

    #transformed.select('features').write.csv('numbers')

    #transformed.foreachPartition(f)
    itensDataframe = itensfactors.toPandas()
    transformed.foreachPartition(lambda iterator: f(iterator, centers, itensDataframe))