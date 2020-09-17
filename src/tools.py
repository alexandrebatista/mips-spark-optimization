from sortedcontainers import SortedList

def getUserIdAndRating(line):
    results = line.split(',')
    userId = results[0]
    rating = results[1]
    return userId, rating

def getAllUsers(fileNames):
    users = SortedList()

    for fileName in fileNames:
        users = enumerateUsers(users, fileName)

    return users

def enumerateUsers(users, fileName):
    with open(fileName) as file:
        for line in file:
            if ':' not in line:
                user, rating = getUserIdAndRating(line)
                if int(user) not in users:
                    users.add(int(user))

    return users

def buildFiles(fileNames, resultFileName):
    resultFile = open(resultFileName, "w")
    count = 0
    limit = 10000
    for fileName in fileNames:
        with open(fileName) as file:
            for line in file:
                if ':' not in line:
                    user, rating = getUserIdAndRating(line)
                    resultFile.write(str(user) + '::' + str(movie) + '::' + str(rating) + '\n')
                    count +=1
                else:
                    movie = line.split(':')[0]

                if count >= limit:
                    break

        if count >= limit:
            break