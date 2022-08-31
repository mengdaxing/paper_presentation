import math

import numpy as np
from sklearn import datasets  # 导入库
from sklearn.metrics import log_loss


# 数据处理
def loadData(fileName):
    dataList = []; labelList = []
    fr = open(fileName, 'r')
    for line in fr.readlines():
        curLine = line.strip().split(',')
        labelList.append(float(curLine[2]))
        dataList.append([float(num) for num in curLine[0:1]])
    return dataList, labelList

# LR预测
def predict(w, x):
    wx = np.dot(w, x)
    P1 = 1 / (1 + np.exp(-wx))
    if P1 >= 0.5:
        return 1
    return 0

# LR预测
def predict2(w, data):
    wx = np.dot(data, w)
    P1 = 1 / (1 + np.exp(-wx))
    return P1
    # return (P1 >= 0.5).astype(int)

def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma

# 梯度下降训练
def GD(trainDataList, trainLabelList, iter=30, batchsize=16):
    for i in range(len(trainDataList)):
        trainDataList[i].append(1)
    trainDataList = np.array(trainDataList)
    w = np.zeros(trainDataList.shape[1])
    alpha = 0.01
    batchCount = int(len(trainDataList)/batchsize)
    for i in range(iter):
        for j in range(batchCount):
                start = batchsize*j
                end = batchsize*(j+1)
                data = trainDataList[start:end]
                wx = np.dot(data, w)
                yb = trainLabelList[start:end]
                xb = trainDataList[start:end]

                temp = (yb - 1 / (1 + np.exp(-wx))).reshape(-1, 1)


                w += alpha * (1/batchsize) * np.sum(temp * xb, axis = 0)

        y_hat_probability = predict2(w, trainDataList)
        y_hat = (y_hat_probability >= 0.5).astype(int)
        y =                       np.array(trainLabelList)

        loss1 = log_loss(
            y_hat, y
        )
        loss2 = - 1/len(y) * np.sum(
                    y * np.log(y_hat_probability) +
                    (1 - y) * np.log(1 - y_hat_probability)
                )
        print(loss1, loss2)
    return w

# 测试
def test(testDataList, testLabelList, w):
    for i in range(len(testDataList)):
        testDataList[i].append(1)
    errorCnt = 0
    for i in range(len(testDataList)):
        if testLabelList[i] != predict(w, testDataList[i]):
            errorCnt += 1
    return 1 - errorCnt / len(testDataList)

# 打印准确率
if __name__ == '__main__':
    cancer = datasets.load_breast_cancer()  # 导入乳腺癌数据

    data = cancer['data']
    data = standardization(data).tolist()
    target = cancer['target'].tolist()
    n = len(data)
    print(n)
    trainSize  = int(n*0.7)
    trainData, trainLabel = data[:trainSize], target[:trainSize]
    testData, testLabel = data[trainSize:], target[trainSize:]

    # trainData, trainLabel = loadData('../data/train.txt')
    # testData, testLabel = loadData('../data/test.txt')
    w = GD(trainData, trainLabel, 100)
    accuracy = test(testData, testLabel, w)
    print('the test accuracy is:', accuracy)