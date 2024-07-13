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

