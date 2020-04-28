# face-recogize-knn

                                                Face recogize using k-nn algorithm
 - IDE : PyCharm Community Edition , Python 3.7.0
 - Library : OpenCV, haarcascade_frontalface_default.xml
 - Author : Tuan Nguyen Van - https://www.facebook.com/tuanelnino9
 
                                   **********************************************
 - pip3 install opencv-python==3.4.1 opencv-contrib-python==3.4.1
 - python3 accuracy.py : Getting accuracy of algoritm
 - python3 predictFace.py : Predicting image's information.
      + Input : input.jpg
      + Output : Image's information which has in data train
 
                                    **********************************************
 #trainning
 
- Step 1 : From all image in folder name is "data_train", its will be detected face and cropped image size 60x60
- Step 2 : Using Sift algorithm to extract feature image : keypoint, descriptor. Only one descriptor is vector 128x1
- Step 3 : Save all descriptor
 
 #predict
 
- Step 1 : Input : image need to predict information, it will be detected face and cropped image size 60x60
- Step 2 : Using Sift algorithm to extract feature image => getting all descriptor of image ( m descriptor )
- Step 3 : Classification descriptor :
            + Each descriptor :
              - Using Euclid distance to find k descriptor (step 2 tranning) which similar as descriptor are under review.                   Descriptor will belong to class which has maximum descriptor similar as descriptor are under review
- Step 4 : Loop step 3 ( m times )
- Step 5 : Collecting result classification descriptor. If class has maximum descriptor then image will belong to class
- Step 6 : Response image's information
 
