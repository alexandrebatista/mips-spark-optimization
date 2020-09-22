from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
import pyspark.sql.functions as F
import numpy as np
from heapq import heapify, heappush, heappop, nlargest
import threading
import logging
import time

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

def ParalelMaximus(iterator, centers, itensDataframe):
    theta = -1000
    L = []
    K = 10
    users = []

    # Construct Index
    for user in iterator:
        id = user.__getitem__("id")
        userLatentFactors = user.__getitem__("features")
        partition = user.__getitem__("prediction")
        centroid = centers[partition]

        users.append((id, userLatentFactors))

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
        for user in users:
            id = user[0]
            userLatentFactors = user[1]
            heap = []

            for itens in L[:K]:
                itemLatentFactors = itens[2]
                weight = np.dot(userLatentFactors, itemLatentFactors)
                heappush(heap, (weight, itens[0]))

            for itens in L[K:]:
                if itens[1] < min(heap):
                    break
                else:
                    itemLatentFactors = itens[2]
                    weight = np.dot(userLatentFactors, itemLatentFactors)
                    if weight > min(heap):
                        heappush(heap, (weight, itens[0]))
            #print(nlargest(K, heap))

def SequencialMaximus(threadId, usersDataframe, itensDataframe, centers):
    threadCluster = usersDataframe[usersDataframe.prediction == threadId]

    theta = -1000
    L = []
    K = 10
    users = []

    # Construct Index
    for index, row in threadCluster.iterrows():
        id = row["id"]
        userLatentFactors = row["features"]
        partition = row["prediction"]
        centroid = centers[partition]

        users.append((id, userLatentFactors))

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
        for user in users:
            id = user[0]
            userLatentFactors = user[1]
            heap = []

            for itens in L[:K]:
                itemLatentFactors = itens[2]
                weight = np.dot(userLatentFactors, itemLatentFactors)
                heappush(heap, (weight, itens[0]))

            for itens in L[K:]:
                if itens[1] < min(heap):
                    break
                else:
                    itemLatentFactors = itens[2]
                    weight = np.dot(userLatentFactors, itemLatentFactors)
                    if weight > min(heap):
                        heappush(heap, (weight, itens[0]))
            #print(nlargest(K, heap))

def process(usersfactors, itensfactors):
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO,
                        datefmt="%H:%M:%S")
    numberOfClusters = 4
    kmeans = KMeans(k = numberOfClusters, seed=1)  # 4 clusters here
    model = kmeans.fit(usersfactors.select('features'))

    transformed = model.transform(usersfactors)

    # Trains a k-means model.
    #kmeans = KMeans().setK(2).setSeed(1)
    #model = kmeans.fit(dataset)

    # Make predictions
    #predictions = model.transform(dataset)

    # Shows the result.
    centers = model.clusterCenters()

    transformed = transformed.repartition('prediction')

    #transformed.select('features').write.csv('numbers')

    usersDataframe = transformed.toPandas()
    itensDataframe = itensfactors.toPandas()

    # Sequencial version
    threads = list()
    logging.info("Multithread Execution: Started!")
    startingTimeMultithreading = time.time()
    for threadId in range(numberOfClusters):
        logging.info("Multithread Execution: create and start thread " + str(threadId))
        x = threading.Thread(target=SequencialMaximus, args=(threadId, usersDataframe, itensDataframe, centers))
        threads.append(x)
        x.start()

    for threadId, thread in enumerate(threads):
        logging.info("Multithread Execution: before joining thread " + str(threadId))
        thread.join()
        logging.info("Multithread Execution: thread " + str(threadId) +" done")
    logging.info("Multithread Execution: Finished!")
    endingTimeMultithreading = time.time()

    # Parallel version
    logging.info("Spark Execution: Started!")
    startingTimeSpark = time.time()
    transformed.foreachPartition(lambda iterator: ParalelMaximus(iterator, centers, itensDataframe))
    logging.info("Spark Execution: Finished!")
    endingTimeSpark = time.time()

    print("Time elapsed")
    print("Multithreading Execution: " + str(endingTimeMultithreading - startingTimeMultithreading) + "s")
    print("Spark Execution: " + str(endingTimeSpark - startingTimeSpark) + "s")