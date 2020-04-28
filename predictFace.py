import cv2 as cv
import numpy as np
import os

def traindata(dir):
    count = 1

    listFiles = os.listdir("data_train/" + str(dir))

    path = "data_train/" + str(dir) + "/%d.jpg"

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


# method detect face and save face
def detectFace():
    img = cv.imread("input.jpg")
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    face_cascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray)

    for (x, y, w, h) in faces:
        faceCrop = img[y:y + h, x:x + w]
        croppedImg = cv.resize(faceCrop, (60, 60))  # resize to 32x32 px

    cv.imwrite("face.jpg", croppedImg)


# get des, key image input
detectFace()
imgtest = cv.imread("face.jpg")

graytest = cv.cvtColor(imgtest, cv.COLOR_BGR2GRAY)

sift = cv.xfeatures2d.SIFT_create()

kptest, destest = sift.detectAndCompute(graytest, None)

# get train data
listDirs = os.listdir("data_train")

for i in range(0, len(listDirs)):
    des = traindata(i)

    if i == 0:
        trainData = des
        ketqua = np.full((len(des), 1), i)
    else:
        trainData = np.concatenate((trainData, des), axis=0).astype(np.float32)
        ketqua = np.concatenate((ketqua, np.full((len(des), 1), i)), axis=0).astype(np.float32)

knn = cv.ml.KNearest_create()
knn.train(trainData, 0, ketqua)
temp, result, nearest, distance = knn.findNearest(destest, 1)

# print("Result: {}\n".format(result))
# print("Nearest: {}\n".format(nearest))
# print("Distance: {}\n".format(distance))

# predict
counts = np.bincount(np.array(result).reshape(-1).astype(int))
# print(np.argmax(counts))
print("Predict image's information")
if np.argmax(counts) == 0:
	print("Cristiano Ronaldo")
if np.argmax(counts) == 1:
	print("Donal Trump")
if np.argmax(counts) == 2:
	print("Fernando Torres")
if np.argmax(counts) == 3:
	print("Karim Benzema")
if np.argmax(counts) == 4:
	print("Lionel Messi")
if np.argmax(counts) == 5:
	print("Lukaku")
if np.argmax(counts) == 6:
	print("Nguyen Xuan Truong")
if np.argmax(counts) == 7:
	print("Neymar Jr")
if np.argmax(counts) == 8:
	print("Nguyen Cong Phuong")
if np.argmax(counts) == 9:
	print("Nguyen Quang Hai")
if np.argmax(counts) == 10:
	print("Son Tung MTP")
if np.argmax(counts) == 11:
	print("Sergio Ramos")
if np.argmax(counts) == 12:
	print("Nguyen Van Tuan")
