<h1 align="center"> Armed People Detectors to use in each video frame (APD4F)</h1> 

<p align="center">
  <img width="950" height="220" src="https://github.com/user-attachments/assets/9acd9692-2079-4073-b71f-568cde7cf21a/">
</p>


![Static Badge](https://img.shields.io/badge/YOLO-Link-blue?labelColor=blue&color=red&link=https%3A%2F%2Fgithub.com%2FAlexeyAB%2Fdarknet)
![Static Badge](https://img.shields.io/badge/LabelImg-Link-red?labelColor=blue&color=yellow&link=https%3A%2F%2Fgithub.com%2Fheartexlabs%2FlabelImg)
![Static Badge](https://img.shields.io/badge/Release%20date-Jun-blue?labelColor=blue&color=green)

This research aims to improve the automatic identification of armed people in surveillance videos. We focus on people armed with pistols and revolvers. Furthermore, we use the YOLOv4 to detect people and weapons in each video frame. We developed a series of algorithms to create a dataset with the information extracted from the bounding boxes generated by YOLOv4 in real time. Thereby, we initially developed six-armed people detectors (APD) based on six machine learning models: Random Forest Classifier (RFC-APD), Multilayer Perceptron (MLP-APD), Support Vector Machine (SVM-APD), Logistic Regression (LR-APD), Naive Bayes (NB-APD), and Gradient Boosting Classifier (GBC-APD). These models use 20 predictors to make their predictions. These predictors are computed from the bounding box coordinates of the detected people and weapons, their distances, and areas of intersection. Based on our results, the RFC-APD was the best-performing detector, with an accuracy of 95.59%, a recall of 94.51%, and an F1-score of 95.65%. In this work, we propose to create selectors for deciding which APD to use in each video frame (APD4F) to improve the detection results. We implemented two types of APD4Fs, one based on a Random Forest Classifier (RFC-APD4F) and another in a Multilayer Perceptron (MLP-APD4F). We developed 43 APD4Fs combining subsets of the six APDs. Both APD4F types outperformed most of the independent use of all six APDs. A multilayer perceptron-based APD4F, which combines an MLP-APD, a NB-APD, and a LR-APD, presented the best performance, achieving an accuracy of 95.84%, a recall of 99.285% and an F1 score of 96.078%.

## Table of Contents

* [APD4F](APD4F_scripts)

* [APD](models/APD)

* [Dataset](dataset)

* [Videos](videos)

* [Yolo](yolo)

## Armed People Detectors to use in each video frame
The main goal of APD4Fs is to identify and apply the best APDs to a specific situation presented in the video. APD4Fs are designed by combining two or three APDs. We developed 43 APD4Fs using two ML models, Random Forest and Multilayer Perceptron. The APDs are RFC-APD, MLP-APD, SVM-APD, LR-APD, NB-APD, and GBC-APD. The APD dataset is developed taking as ground truth the APD with the highest probability of correctly predicting the armed or unarmed person in each case of the training video

The two versions of the algorithm, with two or three APDs can be downloaded from [APD4F_scripts](APD4F_scripts). The file are saved in the APD4F_scripts folder with the name ams_2_modelos and ams_3_modelos. These algorithms work together with YOLOv4 so the [darknet module](https://github.com/AlexeyAB/darknet) must be imported. The APD4Fs, APDs and face detection models have been trained in [Jupyter Notebook](notebooks) and imported into these algorithms through the use of the pickle library. These trained models are shared in the folder named [models/APD4F](models/APD4F), [models/APD](models/APD) and [models/ml_faces_armed_people_detection](models/ml_faces_armed_people_detection).

## Armed People Detectors 
APDs are generated using six ML models, RFC-APD, MLP-APD, SVM-APD, LR-APD, GBC-APD, and NB-APD. The APDs receive 20 measurements about the bounding boxes generated by YOLO in real-time. The information is relative to the three classes, handguns, people, and faces. APDs process the information and identify the armed people involved in each video frame. Then, the information related to people, and faces is fed into the face ML model to recognize the faces of the armed people. The trained models can be downloaded at [models/APD4F](models/APD4F).

The dataset used for the MLP-APD and SVM-APD training process was standardized before training using the function StandardScaler from the Scikit-Learn library (Jupyter Notebook). However, we import the models into our general system to receive the input data from YOLO’s live stream. It implies that the input data must be in the same conditions as the training process. Consequently, it was mandatory to standardize the input data in real-time, so we have applied the mathematical formula used by the StandardScaler function according to z = (x − u)/s, where x represents the input data to be standardized, u stands for the mean, and s is the standard deviation of the training samples. Below is the mean and standard deviation used to normalize the data received by YOLO in real time.

```
Line 13:
    To normalize the MLP-APD, SVM-APD data (All predictors-training dataset - 28 predictors):
    
    u = np.array([9.09106623e+02, 1.05738223e+00, 4.01556492e+02, 4.15881999e+02,
                  3.15145035e+02, 3.00285205e+02, 4.87967948e+02, 5.31478794e+02,
                  3.87211508e-01, 3.15303209e+02, 3.59825385e+02, 2.92086914e+02,
                  3.39647453e+02, 3.38519504e+02, 3.80003317e+02, 3.83417641e-01,
                  9.68300700e+02, 2.15281262e+03, 2.31289485e+02, 1.56180841e-01,
                  6.82105596e-02, 5.53272210e-04, 1.58077774e-04, 1.58631046e-01,
                  5.61650332e-01, 2.06291495e-02, 5.29560544e-03, 2.86911160e-02])

Line 265:   
  s = np.array([4.43693600e+02, 9.50245842e-01, 2.16075512e+02, 4.92883235e+01,
                2.16727143e+02, 8.71466305e+01, 2.25161619e+02, 2.14580602e+01,
                6.72153476e-01, 1.68249783e+02, 7.25382729e+01, 1.65957845e+02,
                7.71079606e+01, 1.71079386e+02, 6.88302982e+01, 4.86218627e-01,
                1.90893059e+03, 2.14921043e+03, 1.47367328e+02, 3.63026701e-01,
                2.52106880e-01, 2.35152312e-02, 1.25719046e-02, 3.65331682e-01,
                4.96184680e-01, 1.42139325e-01, 7.25779719e-02, 1.66936922e-01])
```
```
On line 189, we can unlock the normalization of the data:
Line 676 and 805 (ADP4F with 2 APDs):
    #predictors_per = ((predictors_per - u)) / s # Only for mlp and svm. <--------------
    predictors_per = np.array(predictors_per).reshape(1,-1)
    print(predictors_per)
    ypredic_per = loaded_model_per.predict(predictors_per)
    print(f"Prediction_person {nper}: {ypredic_per}")
```

```
On line 322, we can find the code to select the models to use both to detect people and their faces:
Line 938(ADP4F with 2 APDs):
    #People detection model
    #loaded_model_per = pickle.load(open('./modelos/hg/rfc_t4.sav', 'rb'))
    loaded_model_per = pickle.load(open('./modelos/hg/rfc11.sav', 'rb'))
    #loaded_model_per = pickle.load(open('./modelos/hg/mlp_t10.sav', 'rb'))
    #loaded_model_per = pickle.load(open('./modelos/hg/mlp17.sav', 'rb'))
    #loaded_model_per = pickle.load(open('./modelos/hg/bag_model.sav', 'rb'))
    
    #Faces detection model
    #loaded_model_fac = pickle.load(open('./modelos/fac/rfc_t.sav', 'rb'))
    loaded_model_fac = pickle.load(open('./modelos/fac/rfc.sav', 'rb'))
    #loaded_model_fac = pickle.load(open('./modelos/fac/mlp_t.sav', 'rb'))
    #loaded_model_fac = pickle.load(open('./modelos/fac/mlp.sav', 'rb'))
```
On line 993 (ADP4F with 2 APDs), we can unlock the code that allows us to make detections through the web camera in real-time. On line 994 (ADP4F with 2 APDs), we can specify the path where the video we want to work with is.
```
Line 993 (ADP4F with 2 APDs):
    #cap = cv2.VideoCapture(0)

Line 994 (ADP4F with 2 APDs):
    cap = cv2.VideoCapture("./videos_entrada/trasera.mp4")   # <----- Replace with your video directory
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    new_height, new_width = frame_height // 2, frame_width // 2
```
On line 93, we can modify the position where the file with the results of the processed video will be saved. This shows us that people were armed and unarmed. In lines 202 and 208 we can modify the predictors that we want to be shown in the results.
```
Line 566 (ADP4F with 2 APDs):
    archivo = open("./label/results" + "_video_test" + ".txt", "a")
```
```
Lines 924 (ADP4F with 2 APDs):
    archivo.write(f"{currentframe},{nper},{per_xmid},{ypredic_per},{prob0},{prob1},{prediction}\n")
```
On line 916 (ADP4F with 2 APDs), you can modify the route where the faces of armed people are stored.
```
Line 916 (ADP4F with 2 APDs):
    cv2.imwrite('./cropped_faces/frame' + str(currentframe) + '.jpg', cropped_image)
```

## Dataset
The dataset was created by processing each video frame. We extracted data from each frame related to combining all people with all weapons. Thus, the records are made up by grouping the data of the first person with the data corresponding to the first weapon. Then, the first person with the second weapon, and so on, taking all the people and handguns. Consequently, the number of records per frame depends on the number of people and guns. The ground truth used for APDs indicates whether the person is armed or unarmed. Meanwhile, the applied ground truth for the APD4F is represented by the APDs with the highest probability of correctly predicting the record. The datasets used for both
techniques, APD and APD4F, have the same predictors. They only vary in the ground truth and the amount of data used for training and testing.The datasets can be downloaded at [dataset](dataset). The distribution of the datasets is detailed in the table shown below.

<p align="center">
  <img width="500" height="180" src="https://github.com/user-attachments/assets/25f25202-da6e-455a-8c1a-1ab66e1b79d9">
</p>

## YOLO
We used our dataset to train the YOLOv4 object detector. We trained it from scratch to recognize faces, handguns, and people in the video. We randomly divided our [dataset](dataset/YOLO) into 4,000 images for training and 1,000 for testing.  Afterward, we downloaded YOLOv4 from [Alexey Bochkovskiy’s GitHub](https://github.com/AlexeyAB/darknet) (YOLOv4 creator’s GitHub repository). This repository explains in detail how to configure YOLO. In the folder that contains the YOLO dataset we can find four files, which detail the classes, the location of the training and test images. These files are necessary for its operation. Furthermore, we trained YOLOv4 for 6,000 iterations. The [YOLO folder](yolo) includes two files, one is the settings used in YOLO, and the other contains the training weights. These files will allow you to apply YOLO to detect the three classes: weapons, faces, and people.

## Run the program
APD4Fs and APDs should be saved in the ...\Yolo_v4\darknet\build\darknet\x64 folder. Then, run the following command in your terminal or command prompt:

Run for APD4F with two APDs:
```
python ams_2_modelos.py
```
Run for APD4F with three APDs:
```
ams_3_modelos.py
```
