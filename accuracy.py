import cv2 as cv
import numpy as np
import os
import imp
from sklearn.metrics import accuracy_score

def traindata(dir):
    count = 1

    listFiles = os.listdir("/home/tuannguyen/PycharmProjects/FaceRecoginze/data_train/" + str(dir))

    path = "/home/tuannguyen/PycharmProjects/FaceRecoginze/data_train/" + str(dir) + "/%d.jpg"

    img = cv.imread(path % count)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    sift = cv.xfeatures2d.SIFT_create()

    kp, des = sift.detectAndCompute(gray, None)

    while (count < len(listFiles)):
        count = count + 1

        img1 = cv.imread(path % count)

        gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)

        kp1, des1 = sift.detectAndCompute(gray1, None)

        des = np.concatenate((des, des1))

    return des


# get train data
listDirs = os.listdir("/home/tuannguyen/PycharmProjects/FaceRecoginze/data_train")

for i in range(0, len(listDirs)):
    des = traindata(i)

    if i == 0:
        trainData = des
        ketqua = np.full((len(des), 1), i)
    else:
        trainData = np.concatenate((trainData, des), axis=0).astype(np.float32)
        ketqua = np.concatenate((ketqua, np.full((len(des), 1), i)), axis=0).astype(np.float32)

# y_test
listDirsTest = os.listdir("/home/tuannguyen/PycharmProjects/FaceRecoginze/ImageTest")

for i in range(0, len(listDirsTest)):
    list = os.listdir("/home/tuannguyen/PycharmProjects/FaceRecoginze/ImageTest/" + str(i))

    if i == 0:
        y_test = np.full((len(list), 1), i)
    else:
        y_test = np.concatenate((y_test, np.full((len(list), 1), i)), axis=0).astype(np.float32)

def knn(k):
    for i in range(0, len(listDirsTest)):
        list = os.listdir("/home/tuannguyen/PycharmProjects/FaceRecoginze/ImageTest/" + str(i))

        for j in range(0, len(list)):
            # get des, key image input
            path = "/home/tuannguyen/PycharmProjects/FaceRecoginze/ImageTest/" + str(i) + "/%d.jpg"

            imgtest = cv.imread(path % (j + 1))

            graytest = cv.cvtColor(imgtest, cv.COLOR_BGR2GRAY)

            sift = cv.xfeatures2d.SIFT_create()

            kptest, destest = sift.detectAndCompute(graytest, None)

            knn = cv.ml.KNearest_create()
            knn.train(trainData, 0, ketqua)
            temp, result, nearest, distance = knn.findNearest(destest, k)

            # predict
            counts = np.bincount(np.array(result).reshape(-1).astype(int))

            lablePred = np.argmax(counts)
            # y_pred
            if i == 0 and j == 0:
                y_pred = np.full((1, 1), lablePred)
            else:
                y_pred = np.concatenate((y_pred, np.full((1, 1), lablePred)), axis=0).astype(np.float32)

    print("Accuracy " + str(k) +"NN : %.2f %%" % (100 * accuracy_score(y_test, y_pred)))

for i in range(3,20):
    knn(i)