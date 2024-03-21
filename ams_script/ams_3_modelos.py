import math
import os
import time
import pickle
import cv2
import darknet
import numpy as np



#To normalize the mlp data (All predictors-training dataset - 28 predictors)
#APDM
u = np.array([9.09106623e+02, 1.05738223e+00, 4.01556492e+02, 4.15881999e+02,
              3.15145035e+02, 3.00285205e+02, 4.87967948e+02, 5.31478794e+02,
              3.87211508e-01, 3.15303209e+02, 3.59825385e+02, 2.92086914e+02,
              3.39647453e+02, 3.38519504e+02, 3.80003317e+02, 3.83417641e-01,
              9.68300700e+02, 2.15281262e+03, 2.31289485e+02, 1.56180841e-01,
              6.82105596e-02, 5.53272210e-04, 1.58077774e-04, 1.58631046e-01,
              5.61650332e-01, 2.06291495e-02, 5.29560544e-03, 2.86911160e-02])

#MLP22_ams (GBC=71.67%+LR=75.11% + MLP =79.18%)
"""u2 = np.array([8.58048650e+02, 8.79589633e-01, 3.92806059e+02, 3.82440691e+02,
             3.05539236e+02, 2.46107454e+02, 4.80072882e+02, 5.18773927e+02,
             2.98218143e-01, 3.10902093e+02, 3.12445589e+02, 2.87521764e+02,
             2.92895493e+02, 3.34282422e+02, 3.31995685e+02, 4.19492441e-01,
             9.30794571e+02, 2.02757651e+03, 2.31759339e+02, 1.23164147e-01,
             7.43520518e-02, 3.23974082e-04, 1.07991361e-04, 2.24406048e-01,
             5.40172786e-01, 1.41468683e-02, 3.77969762e-03, 1.95464363e-02])"""

#MLP23_ams (MLP+LR+SVM)
"""u2 = np.array([8.58912743e+02, 8.79439909e-01, 3.92910451e+02, 3.82398852e+02,
             3.05665487e+02, 2.46086410e+02, 4.80155414e+02, 5.18711294e+02,
             2.98480835e-01, 3.10712804e+02, 3.12374381e+02, 2.87345702e+02,
             2.92834884e+02, 3.34079906e+02, 3.31913878e+02, 4.19365302e-01,
             9.28490131e+02, 2.02421631e+03, 2.31870596e+02, 1.23101043e-01,
             7.42823161e-02, 2.70314105e-04, 1.08125642e-04, 2.25009461e-01,
             5.40411959e-01, 1.41103963e-02, 3.78439747e-03, 1.89219873e-02])"""

#MLP24_ams (MLP+GBC+RFC)
"""u2 = np.array([8.59563202e+02, 8.79807692e-01, 3.93086265e+02, 3.82350806e+02,
             3.05761672e+02, 2.46019627e+02, 4.80410857e+02, 5.18681985e+02,
             2.98347018e-01, 3.10991858e+02, 3.12347984e+02, 2.87607708e+02,
             2.92795352e+02, 3.34376009e+02, 3.31900615e+02, 4.20429991e-01,
             9.31724565e+02, 2.02814656e+03, 2.31838166e+02, 1.23217373e-01,
             7.31957649e-02, 3.24114088e-04, 1.08038029e-04, 2.25259291e-01,
             5.40406223e-01, 1.41529818e-02, 3.78133103e-03, 1.95548833e-02])"""

#MLP25_ams (MLP+GBC+SVM)
"""u2 = np.array([8.59563202e+02, 8.79807692e-01, 3.93086265e+02, 3.82350806e+02,
             3.05761672e+02, 2.46019627e+02, 4.80410857e+02, 5.18681985e+02,
             2.98347018e-01, 3.10991858e+02, 3.12347984e+02, 2.87607708e+02,
             2.92795352e+02, 3.34376009e+02, 3.31900615e+02, 4.20429991e-01,
             9.31724565e+02, 2.02814656e+03, 2.31838166e+02, 1.23217373e-01,
             7.31957649e-02, 3.24114088e-04, 1.08038029e-04, 2.25259291e-01,
             5.40406223e-01, 1.41529818e-02, 3.78133103e-03, 1.95548833e-02])"""

#MLP26_ams (MLP+RFC+SVM)
"""u2 = np.array([8.59853112e+02, 8.80246980e-01, 3.93099740e+02, 3.82412786e+02,
             3.05841375e+02, 2.46141187e+02, 4.80358105e+02, 5.18684385e+02,
             2.99030493e-01, 3.10622080e+02, 3.12410656e+02, 2.87259574e+02,
             2.92872541e+02, 3.33984585e+02, 3.31948771e+02, 4.19758436e-01,
             9.27397412e+02, 2.02297112e+03, 2.32075899e+02, 1.23273574e-01,
             7.30108866e-02, 2.70811894e-04, 1.08324758e-04, 2.25315496e-01,
             5.41407139e-01, 1.41905432e-02, 3.79136652e-03, 1.86318583e-02])"""

#MLP27_ams (MLP+NB+SVM)
"""u2 = np.array([8.59156794e+02, 8.79580586e-01, 3.92872480e+02, 3.82384260e+02,
             3.05658233e+02, 2.46085701e+02, 4.80086727e+02, 5.18682818e+02,
             2.98400173e-01, 3.10665370e+02, 3.12348658e+02, 2.87309904e+02,
             2.92817722e+02, 3.34020837e+02, 3.31879595e+02, 4.18873635e-01,
             9.26579969e+02, 2.02154523e+03, 2.31828991e+02, 1.23283969e-01,
             7.45324830e-02, 2.70241055e-04, 1.08096422e-04, 2.24840558e-01,
             5.40265917e-01, 1.41606313e-02, 3.78337477e-03, 1.87547292e-02])"""

#MLP28_ams (LR+NB+GBC)
"""u2 = np.array([8.60354662e+02, 8.78678061e-01, 3.92909894e+02, 3.82447146e+02,
             3.05549552e+02, 2.46079439e+02, 4.80270237e+02, 5.18814854e+02,
             2.98734314e-01, 3.10721446e+02, 3.12482648e+02, 2.87328701e+02,
             2.92918688e+02, 3.34114192e+02, 3.32046607e+02, 4.18163133e-01,
             9.32062846e+02, 2.02996039e+03, 2.31990043e+02, 1.23377326e-01,
             7.46430117e-02, 3.24534833e-04, 1.08178278e-04, 2.25281264e-01,
             5.40945478e-01, 1.41713544e-02, 3.78623972e-03, 1.73626136e-02])"""

#MLP29_ams (MLP+NB+LR)
u2 = np.array([8.59146407e+02, 8.79254457e-01, 3.92879951e+02, 3.82356581e+02,
             3.05643294e+02, 2.46028462e+02, 4.80116609e+02, 5.18684701e+02,
             2.98271205e-01, 3.10756034e+02, 3.12311208e+02, 2.87390757e+02,
             2.92772122e+02, 3.34121312e+02, 3.31850295e+02, 4.19124797e-01,
             9.28342651e+02, 2.02399560e+03, 2.31797811e+02, 1.23230686e-01,
             7.45002701e-02, 2.70124257e-04, 1.08049703e-04, 2.24851432e-01,
             5.40032415e-01, 1.41545111e-02, 3.78173960e-03, 1.90707726e-02])


#APDM
s = np.array([4.43693600e+02, 9.50245842e-01, 2.16075512e+02, 4.92883235e+01,
              2.16727143e+02, 8.71466305e+01, 2.25161619e+02, 2.14580602e+01,
              6.72153476e-01, 1.68249783e+02, 7.25382729e+01, 1.65957845e+02,
              7.71079606e+01, 1.71079386e+02, 6.88302982e+01, 4.86218627e-01,
              1.90893059e+03, 2.14921043e+03, 1.47367328e+02, 3.63026701e-01,
              2.52106880e-01, 2.35152312e-02, 1.25719046e-02, 3.65331682e-01,
              4.96184680e-01, 1.42139325e-01, 7.25779719e-02, 1.66936922e-01])


#MLP22_ams (GBC=71.67%+LR=75.11% + MLP =79.18%)
"""s2 = np.array([4.60619751e+02, 8.74139036e-01, 2.12100424e+02, 6.87713103e+01,
             2.19849845e+02, 1.11882718e+02, 2.12574870e+02, 3.91572907e+01,
             6.00483121e-01, 1.53033612e+02, 9.71997711e+01, 1.52301205e+02,
             9.91538035e+01, 1.54258535e+02, 9.58178561e+01, 4.93475970e-01,
             1.66851299e+03, 1.81753387e+03, 1.37437601e+02, 3.28625531e-01,
             2.62342952e-01, 1.79963642e-02, 1.03913280e-02, 4.17190572e-01,
             4.98383534e-01, 1.18096293e-01, 6.13629490e-02, 1.38435447e-01])"""

#MLP23_ams (MLP+LR+SVM)
"""s2 = np.array([4.60991833e+02, 8.73991854e-01, 2.12174858e+02, 6.88271261e+01,
             2.19903311e+02, 1.11897384e+02, 2.12648552e+02, 3.92326331e+01,
             6.00727934e-01, 1.52892379e+02, 9.72615204e+01, 1.52180777e+02,
             9.92032562e+01, 1.54094354e+02, 9.58892688e+01, 4.93455211e-01,
             1.66401144e+03, 1.81114275e+03, 1.37472411e+02, 3.28553156e-01,
             2.62229772e-01, 1.64390096e-02, 1.03977859e-02, 4.17588558e-01,
             4.98364198e-01, 1.17946144e-01, 6.14009430e-02, 1.36249572e-01])"""

#MLP24_ams (MLP+GBC+RFC)
"""s2 = np.array([4.60617210e+02, 8.74207016e-01, 2.12100162e+02, 6.88626293e+01,
             2.19863285e+02, 1.11973285e+02, 2.12564143e+02, 3.92423666e+01,
             6.00580847e-01, 1.53082048e+02, 9.73115103e+01, 1.52349871e+02,
             9.92629631e+01, 1.54306673e+02, 9.59317416e+01, 4.93628011e-01,
             1.66892407e+03, 1.81777378e+03, 1.37424040e+02, 3.28686555e-01,
             2.60457568e-01, 1.80002511e-02, 1.03935729e-02, 4.17752969e-01,
             4.98364663e-01, 1.18121442e-01, 6.13761563e-02, 1.38464760e-01])"""

#MLP25_ams (MLP+GBC+SVM)
"""s2 = np.array([4.60617210e+02, 8.74207016e-01, 2.12100162e+02, 6.88626293e+01,
             2.19863285e+02, 1.11973285e+02, 2.12564143e+02, 3.92423666e+01,
             6.00580847e-01, 1.53082048e+02, 9.73115103e+01, 1.52349871e+02,
             9.92629631e+01, 1.54306673e+02, 9.59317416e+01, 4.93628011e-01,
             1.66892407e+03, 1.81777378e+03, 1.37424040e+02, 3.28686555e-01,
             2.60457568e-01, 1.80002511e-02, 1.03935729e-02, 4.17752969e-01,
             4.98364663e-01, 1.18121442e-01, 6.13761563e-02, 1.38464760e-01])"""

#MLP26_ams (MLP+RFC+SVM)
"""s2 = np.array([4.60662088e+02, 8.74086243e-01, 2.12286845e+02, 6.88579007e+01,
             2.20039580e+02, 1.11927398e+02, 2.12729975e+02, 3.92582504e+01,
             6.01144112e-01, 1.52927569e+02, 9.72948205e+01, 1.52227439e+02,
             9.92282433e+01, 1.54116716e+02, 9.59297255e+01, 4.93519292e-01,
             1.66284209e+03, 1.80722287e+03, 1.37507191e+02, 3.28750970e-01,
             2.60154372e-01, 1.64541349e-02, 1.04073543e-02, 4.17789927e-01,
             4.98282499e-01, 1.18275829e-01, 6.14572376e-02, 1.35220975e-01])"""

#MLP27_ams (MLP+NB+SVM)
"""s2 = np.array([4.61200009e+02, 8.73829051e-01, 2.12180070e+02, 6.88305187e+01,
             2.19896115e+02, 1.11868318e+02, 2.12656612e+02, 3.92568978e+01,
             6.00666794e-01, 1.52828715e+02, 9.72628259e+01, 1.52131404e+02,
             9.91937777e+01, 1.54014725e+02, 9.58999926e+01, 4.93374617e-01,
             1.66123687e+03, 1.80600545e+03, 1.37471575e+02, 3.28762881e-01,
             2.62635474e-01, 1.64367888e-02, 1.03963810e-02, 4.17477282e-01,
             4.98376019e-01, 1.18152900e-01, 6.13926775e-02, 1.35657618e-01])"""

#MLP28_ams (LP+NB+GBC)
"""s2 = np.array([4.60946500e+02, 8.74443474e-01, 2.12287940e+02, 6.88123628e+01,
             2.20038135e+02, 1.11946757e+02, 2.12763532e+02, 3.91830351e+01,
             6.00874272e-01, 1.53079874e+02, 9.72046840e+01, 1.52336839e+02,
             9.91681113e+01, 1.54315404e+02, 9.58134475e+01, 4.93257263e-01,
             1.67000976e+03, 1.81867324e+03, 1.37445609e+02, 3.28869824e-01,
             2.62814445e-01, 1.80119269e-02, 1.04003161e-02, 4.17767418e-01,
             4.98320648e-01, 1.18196984e-01, 6.14158295e-02, 1.30618350e-01])"""

#MLP29_ams (MLP+NB+LR)
s2 = np.array([4.61179530e+02, 8.73808016e-01, 2.12136004e+02, 6.88346052e+01,
             2.19857274e+02, 1.11890935e+02, 2.12616318e+02, 3.92503565e+01,
             6.00569003e-01, 1.52860732e+02, 9.72650246e+01, 1.52149684e+02,
             9.92044112e+01, 1.54062068e+02, 9.58952900e+01, 4.93415851e-01,
             1.66353699e+03, 1.81090961e+03, 1.37452383e+02, 3.28701816e-01,
             2.62583282e-01, 1.64332374e-02, 1.03941343e-02, 4.17484449e-01,
             4.98394829e-01, 1.18127731e-01, 6.13794595e-02, 1.36773821e-01])


def convertBack(x, y, w, h):
#Converts center coordinates to rectangle coordinates
    xmin = x - (w / 2)
    xmax = x + (w / 2)
    ymin = y - (h / 2)
    ymax = y + (h / 2)
    return xmin, ymin, xmax, ymax



def cvDrawBoxes(detections, img):
    global currentframe
    currentframe += 1
    if len(detections) > 0:
        persons = dict()
        handguns = dict()
        faces = dict()
        perId, hgId, facId = 0,0,0
        for detections in detections:
            name_tag = detections[0]


            if name_tag == 'Person':
                xmid,ymid,w,h = detections[2][0], \
                        detections[2][1], \
                        detections[2][2], \
                        detections[2][3],
                xmin, ymin, xmax, ymax = convertBack(float(xmid),float(ymid),float(w),float(h))
                persons[perId] = (xmid, ymid, xmin, ymin, xmax, ymax)
                perId += 1

            elif name_tag == 'Handgun':
                xmid,ymid,w,h = detections[2][0], \
                        detections[2][1], \
                        detections[2][2], \
                        detections[2][3],
                xmin, ymin, xmax, ymax = convertBack(float(xmid),float(ymid),float(w),float(h))
                handguns[hgId] = (float(xmid), float(ymid), xmin, ymin, xmax, ymax)
                hgId += 1

            elif name_tag == 'Face':
                xmid,ymid,w,h = detections[2][0], \
                        detections[2][1], \
                        detections[2][2], \
                        detections[2][3],
                xmin, ymin, xmax, ymax = convertBack(float(xmid),float(ymid),float(w),float(h))
                faces[facId] = (float(xmid), float(ymid), xmin, ymin, xmax, ymax)
                facId += 1


        archivo = open("./label/results" + "_video_test" + ".txt", "a")

        for nper, per in enumerate(persons.values()):
            per_xmid = per[0]
            per_ymid = per[1]
            per_xmin = per[2]
            per_ymin = per[3]
            per_xmax = per[4]
            per_ymax = per[5]
            for nhg, hg in enumerate(handguns.values()):
                Intersection_Up_center, Intersection_Up_left, Intersection_Up_right, Intersection_Down_left, Intersection_Down_center, \
                Intersection_Down_right, Intersection_Center_right, Intersection_Center_left, Intersection_Center_right, Intersection_Center_left, \
                Intersection_Inside = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                hg_xmid = hg[0]
                hg_ymid = hg[1]
                hg_xmin = hg[2]
                hg_ymin = hg[3]
                hg_xmax = hg[4]
                hg_ymax = hg[5]
                a2 = hg_xmax - hg_xmin
                b2 = hg_ymax - hg_ymin
                areah = a2 * b2
                areai, ai, bi = 0, 0, 0
                p1 = hg_xmid - per_xmid
                p2 = hg_ymid - per_ymid
                dist = math.sqrt(p1 ** 2 + p2 ** 2)


                if per_xmin < hg_xmid and hg_xmid < per_xmax and per_ymin < hg_ymid and hg_ymid < per_ymax:
                    included_center = 1
                else:
                    included_center = 0

                if hg_xmax < per_xmin or hg_ymax < per_ymin or hg_xmin > per_xmax or hg_ymin > per_ymax:
                    Intersection_No_intersection = 1
                else:
                    Intersection_No_intersection = 0

                    if hg_xmin < per_xmin: #Left side
                        if hg_ymin < per_ymin:          # SI
                            ai = hg_xmax - per_xmin
                            bi = hg_ymax - per_ymin
                            Intersection_Up_left = 1
                        elif hg_ymax > per_ymax:        # II
                            ai = hg_xmax - per_xmin
                            bi = per_ymax - hg_ymin
                            Intersection_Down_left = 1
                        else:                           # CI
                            ai = hg_xmax - per_xmin
                            bi = hg_ymax - hg_ymin
                            Intersection_Center_left = 1


                    elif hg_xmax > per_xmax: #Rigth side
                        if hg_ymin < per_ymin:          # SD
                            ai = per_xmax - hg_xmin
                            bi = hg_ymax - per_ymin
                            Intersection_Up_right = 1
                        elif hg_ymax > per_ymax:        # ID
                            ai = per_xmax - hg_xmin
                            bi = per_ymax - hg_ymin
                            Intersection_Down_rigth = 1
                        else:                           # CD
                            ai = per_xmax - hg_xmin
                            bi = hg_ymax - hg_ymin
                            Intersection_Center_right = 1


                    elif hg_xmin > per_xmin and hg_xmax < per_xmax: #center
                        if hg_ymin < per_ymin:          # SC
                            ai = hg_xmax - hg_xmin
                            bi = hg_ymax - per_ymin
                            Intersection_Up_center = 1
                        elif hg_ymax > per_ymax:        # IC
                            ai = hg_xmax - hg_xmin
                            bi = per_ymax - hg_ymin
                            Intersection_Down_center = 1
                        else:                           # Hg_in_per
                            ai = hg_xmax - hg_xmin
                            bi = hg_ymax - hg_ymin
                            Intersection_Inside = 1

                areai = ai * bi


                predictors_per = [[currentframe, nper, per_xmid, per_ymid, per_xmin, per_ymin, per_xmax, per_ymax, nhg, hg_xmid, hg_ymid, hg_xmin, hg_ymin,
                   hg_xmax, hg_ymax, included_center, areai, areah, dist, Intersection_Center_left, Intersection_Center_right,
                   Intersection_Down_center, Intersection_Down_left, Intersection_Inside, Intersection_No_intersection, Intersection_Up_center,
                   Intersection_Up_left, Intersection_Up_right]]


                print(f'predictor_per {predictors_per}')
                print(f'ci={Intersection_Center_left}, cd={Intersection_Center_right}, ic={Intersection_Down_center}, ii={Intersection_Down_left},'
                      f' hginper={Intersection_Inside}, ni={Intersection_No_intersection}, sc={Intersection_Up_center}, si={Intersection_Up_left},'
                      f' sd={Intersection_Up_right}')


                #For MLP AMS
                #predictors_per2 = ((predictors_per - u2)) / s2  # Only for mlp
                #predictors_model = np.array(predictors_per2).reshape(1,-1)
                #ypredic_model = loaded_model_selection.predict(predictors_model)


                #For RFC AMS
                predictors_model = np.array(predictors_per).reshape(1,-1)
                ypredic_model = loaded_model_selection.predict(predictors_model)


                if ypredic_model == 0:
                    modelo = "MLP"
                    predictors_per = ((predictors_per - u)) / s #Only for mlp, knn and SVM
                    predictors_per = np.array(predictors_per).reshape(1, -1)
                    ypredic_per = loaded_model_per0.predict(predictors_per)
                    print(f"Prediction_person {nper}: {ypredic_per}")
                    probability = loaded_model_per0.predict_proba(predictors_per)
                    prob1 = probability[0, 1]
                    prob0 = probability[0, 0]
                    if ypredic_per == 1:
                        prediction = 1
                        cv2.rectangle(img, (int(per_xmin), int(per_ymin)), (int(per_xmax), int(per_ymax)), (255, 0, 0), 2)
                        for nfac, fac in enumerate(faces.values()):
                            fac_xmid = fac[0]
                            fac_ymid = fac[1]
                            fac_xmin = fac[2]
                            fac_ymin = fac[3]
                            fac_xmax = fac[4]
                            fac_ymax = fac[5]
                            a2 = fac_xmax - fac_xmin
                            b2 = fac_ymax - fac_ymin
                            areaf = a2 * b2
                            areai, ai, bi = 0, 0, 0
                            p1 = fac_xmid - per_xmid
                            p2 = fac_ymid - per_ymid
                            dist = math.sqrt(p1 ** 2 + p2 ** 2)

                            Intersection_Up_center, Intersection_Up_left, Intersection_Up_right, Intersection_Down_left, Intersection_Down_center, \
                            Intersection_Down_right, Intersection_Center_right, Intersection_Center_left, Intersection_Center_right, Intersection_Center_left, \
                            Intersection_Inside = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

                            if (
                                    per_xmin < fac_xmid and fac_xmid < per_xmax and per_ymin < fac_ymid and fac_ymid < per_ymax):
                                included_center = 1
                            else:
                                included_center = 0

                            if fac_xmax < per_xmin or fac_ymax < per_ymin or fac_xmin > per_xmax or fac_ymin > per_ymax:
                                Intersection_No_intersection = 1
                                continue
                            else:
                                Intersection_No_intersection = 0

                            if fac_xmin >= per_xmin and fac_xmax <= per_xmax and fac_ymin <= per_ymin and fac_ymax >= per_ymin:  # SC
                                ai = fac_xmax - fac_xmin
                                bi = fac_ymax - per_ymin
                                Intersection_Up_center = 1

                            elif fac_xmax >= per_xmin and fac_xmin <= per_xmin and fac_ymax >= per_ymin and fac_ymin <= per_ymin:  # SI
                                ai = fac_xmax - per_xmin
                                bi = fac_ymax - per_ymin
                                Intersection_Up_left = 1

                            elif fac_xmax >= per_xmax and fac_xmin <= per_xmax and fac_ymax >= per_ymin and fac_ymin <= per_ymin:  # SD
                                ai = per_xmax - fac_xmin
                                bi = fac_ymax - per_ymin
                                Intersection_Up_right = 1

                            elif fac_xmax >= per_xmin and fac_xmin <= per_xmin and fac_ymax >= per_ymax and fac_ymin <= per_ymax:  # II
                                ai = fac_xmax - per_xmin
                                bi = per_ymax - fac_ymin
                                Intersection_Down_left = 1

                            elif fac_xmax >= per_xmin and fac_xmin >= per_xmin and fac_ymax >= per_ymax and fac_ymin <= per_ymax:  # IC
                                ai = fac_xmax - fac_xmin
                                bi = per_ymax - fac_ymin
                                Intersection_Down_center = 1

                            elif fac_xmax >= per_xmax and fac_xmin <= per_xmax and fac_ymax >= per_ymax and fac_ymin <= per_ymax:  # ID
                                ai = per_xmax - fac_xmin
                                bi = per_ymax - fac_ymin
                                Intersection_Down_right = 1

                            elif fac_xmax >= per_xmax and fac_xmin <= per_xmax and fac_ymax <= per_ymax and fac_ymin >= per_ymin:  # CD
                                ai = per_xmax - fac_xmin
                                bi = fac_ymax - fac_ymin
                                Intersection_Center_right = 1

                            elif fac_xmax >= per_xmin and fac_xmin <= per_xmin and fac_ymax <= per_ymax and fac_ymin >= per_ymin:  # CI
                                ai = fac_xmax - per_xmin
                                bi = fac_ymax - fac_ymin
                                Intersection_Center_left = 1

                            elif per_xmin < fac_xmid and fac_xmid < per_xmax and per_ymin < fac_ymid and fac_ymid < per_ymax:  # Hg_in_per
                                ai = fac_xmax - fac_xmin
                                bi = fac_ymax - fac_ymin
                                Intersection_Inside = 1

                            areai = ai * bi

                            predictors_fac = [currentframe, per_xmid, per_ymid, per_xmin, per_ymin, per_xmax, per_ymax,
                                              fac_xmid, fac_ymid, fac_xmin, fac_ymin,
                                              fac_xmax, fac_ymax, included_center, areai, areaf, dist,
                                              Intersection_Center_left, Intersection_Center_right,
                                              Intersection_Inside, Intersection_No_intersection, Intersection_Up_center,
                                              Intersection_Up_left,
                                              Intersection_Up_right]

                            #predictors_fac = ((predictors_fac - u)) / s  # Only for mlp

                            predictors_fac = np.array(predictors_fac).reshape(1, -1)  # una fila y el resto de datos indistinto.

                            ypredic_fac = loaded_model_fac.predict(predictors_fac)

                            print(f"Prediction_face {nfac}: {ypredic_fac}")
                            if ypredic_fac == 1:
                                cv2.rectangle(img, (int(fac_xmin), int(fac_ymin)), (int(fac_xmax), int(fac_ymax)),
                                              (255, 0, 0), 2)

                                x = int(fac_xmin)
                                y = int(fac_ymin)
                                h = (int(fac_ymax) - int(fac_ymin))
                                w = (int(fac_xmax) - int(fac_xmin))
                                # print(h,w)
                                cropped_image = img[y:y + h, x:x + w]
                                try:
                                    cv2.imshow("Cropped Image", cropped_image)  # Muestra la img cortada.
                                    cv2.imwrite('./cropped_faces/frame' + str(currentframe) + '.jpg', cropped_image)
                                except:
                                    print(f"Error en frame {currentframe}")

                    else:
                        prediction = 0

                elif ypredic_model == 1:
                    modelo = "NB"
                    predictors_per = [[currentframe, nper, per_xmid, per_ymid, per_xmin, per_ymin, per_xmax, per_ymax, nhg, hg_xmid, hg_ymid, hg_xmin, hg_ymin,
                   hg_xmax, hg_ymax, included_center, areai, areah, dist, Intersection_Center_left, Intersection_Center_right,
                   Intersection_Down_center, Intersection_Down_left, Intersection_Inside, Intersection_No_intersection, Intersection_Up_center,
                   Intersection_Up_left, Intersection_Up_right]]

                    #predictors_per = ((predictors_per - u)) / s  # # Only for mlp, knn and SVM
                    predictors_per = np.array(predictors_per).reshape(1, -1)
                    ypredic_per = loaded_model_per1.predict(predictors_per)
                    print(f"Prediction_person {nper}: {ypredic_per}")
                    probability = loaded_model_per1.predict_proba(predictors_per)
                    prob1 = probability[0, 1]
                    prob0 = probability[0, 0]
                    if ypredic_per == 1:
                        prediction = 1
                        cv2.rectangle(img, (int(per_xmin), int(per_ymin)), (int(per_xmax), int(per_ymax)), (255, 0, 0), 2)
                        for nfac, fac in enumerate(faces.values()):
                            fac_xmid = fac[0]
                            fac_ymid = fac[1]
                            fac_xmin = fac[2]
                            fac_ymin = fac[3]
                            fac_xmax = fac[4]
                            fac_ymax = fac[5]
                            a2 = fac_xmax - fac_xmin
                            b2 = fac_ymax - fac_ymin
                            areaf = a2 * b2
                            areai, ai, bi = 0, 0, 0
                            p1 = fac_xmid - per_xmid
                            p2 = fac_ymid - per_ymid
                            dist = math.sqrt(p1 ** 2 + p2 ** 2)

                            Intersection_Up_center, Intersection_Up_left, Intersection_Up_right, Intersection_Down_left, Intersection_Down_center, \
                            Intersection_Down_right, Intersection_Center_right, Intersection_Center_left, Intersection_Center_right, Intersection_Center_left, \
                            Intersection_Inside = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

                            if (per_xmin < fac_xmid and fac_xmid < per_xmax and per_ymin < fac_ymid and fac_ymid < per_ymax):
                                included_center = 1
                            else:
                                included_center = 0

                            if fac_xmax < per_xmin or fac_ymax < per_ymin or fac_xmin > per_xmax or fac_ymin > per_ymax:
                                Intersection_No_intersection = 1
                                continue
                            else:
                                Intersection_No_intersection = 0

                            if  fac_xmin >= per_xmin and fac_xmax <= per_xmax and fac_ymin <= per_ymin and fac_ymax >= per_ymin:  # SC
                                ai = fac_xmax - fac_xmin
                                bi = fac_ymax - per_ymin
                                Intersection_Up_center = 1

                            elif fac_xmax >= per_xmin and fac_xmin <= per_xmin and fac_ymax >= per_ymin and fac_ymin <= per_ymin:  # SI
                                ai = fac_xmax - per_xmin
                                bi = fac_ymax - per_ymin
                                Intersection_Up_left = 1

                            elif fac_xmax >= per_xmax and fac_xmin <= per_xmax and fac_ymax >= per_ymin and fac_ymin <= per_ymin:  # SD
                                ai = per_xmax - fac_xmin
                                bi = fac_ymax - per_ymin
                                Intersection_Up_right = 1

                            elif fac_xmax >= per_xmin and fac_xmin <= per_xmin and fac_ymax >= per_ymax and fac_ymin <= per_ymax:  # II
                                ai = fac_xmax - per_xmin
                                bi = per_ymax - fac_ymin
                                Intersection_Down_left = 1

                            elif fac_xmax >= per_xmin and fac_xmin >= per_xmin and fac_ymax >= per_ymax and fac_ymin <= per_ymax:  # IC
                                ai = fac_xmax - fac_xmin
                                bi = per_ymax - fac_ymin
                                Intersection_Down_center = 1

                            elif fac_xmax >= per_xmax and fac_xmin <= per_xmax and fac_ymax >= per_ymax and fac_ymin <= per_ymax:  # ID
                                ai = per_xmax - fac_xmin
                                bi = per_ymax - fac_ymin
                                Intersection_Down_right = 1

                            elif fac_xmax >= per_xmax and fac_xmin <= per_xmax and fac_ymax <= per_ymax and fac_ymin >= per_ymin:  # CD
                                ai = per_xmax - fac_xmin
                                bi = fac_ymax - fac_ymin
                                Intersection_Center_right = 1

                            elif fac_xmax >= per_xmin and fac_xmin <= per_xmin and fac_ymax <= per_ymax and fac_ymin >= per_ymin:  # CI
                                ai = fac_xmax - per_xmin
                                bi = fac_ymax - fac_ymin
                                Intersection_Center_left = 1

                            elif per_xmin < fac_xmid and fac_xmid < per_xmax and per_ymin < fac_ymid and fac_ymid < per_ymax:  # Hg_in_per
                                ai = fac_xmax - fac_xmin
                                bi = fac_ymax - fac_ymin
                                Intersection_Inside = 1

                            areai = ai * bi


                            predictors_fac = [currentframe, per_xmid, per_ymid, per_xmin, per_ymin, per_xmax, per_ymax, fac_xmid, fac_ymid, fac_xmin, fac_ymin,
                                              fac_xmax, fac_ymax, included_center, areai, areaf, dist, Intersection_Center_left, Intersection_Center_right,
                                              Intersection_Inside, Intersection_No_intersection, Intersection_Up_center, Intersection_Up_left,
                                              Intersection_Up_right]

                            #predictors_fac = ((predictors_fac - u)) / s  # Only for mlp

                            predictors_fac = np.array(predictors_fac).reshape(1,-1) # una fila y el resto de datos indistinto.

                            ypredic_fac = loaded_model_fac.predict(predictors_fac)

                            print(f"Prediction_face {nfac}: {ypredic_fac}")
                            if ypredic_fac == 1:
                                cv2.rectangle(img, (int(fac_xmin), int(fac_ymin)), (int(fac_xmax), int(fac_ymax)), (255, 0, 0), 2)

                                x = int(fac_xmin)
                                y = int(fac_ymin)
                                h = (int(fac_ymax) - int(fac_ymin))
                                w = (int(fac_xmax) - int(fac_xmin))
                                #print(h,w)
                                cropped_image = img[y:y + h, x:x + w]
                                try:
                                    cv2.imshow("Cropped Image", cropped_image)  # Muestra la img cortada.
                                    cv2.imwrite('./cropped_faces/frame' + str(currentframe) + '.jpg', cropped_image)
                                except:
                                    print(f"Error en frame {currentframe}")

                    else:
                        prediction = 0

                elif ypredic_model == 2:
                    modelo = "LR"
                    predictors_per = [[currentframe, nper, per_xmid, per_ymid, per_xmin, per_ymin, per_xmax, per_ymax, nhg, hg_xmid, hg_ymid, hg_xmin, hg_ymin,
                   hg_xmax, hg_ymax, included_center, areai, areah, dist, Intersection_Center_left, Intersection_Center_right,
                   Intersection_Down_center, Intersection_Down_left, Intersection_Inside, Intersection_No_intersection, Intersection_Up_center,
                   Intersection_Up_left, Intersection_Up_right]]

                    #predictors_per = ((predictors_per - u)) / s  # # Only for mlp, knn and SVM
                    predictors_per = np.array(predictors_per).reshape(1, -1)
                    ypredic_per = loaded_model_per2.predict(predictors_per)
                    print(f"Prediction_person {nper}: {ypredic_per}")
                    probability = loaded_model_per2.predict_proba(predictors_per)
                    prob1 = probability[0, 1]
                    prob0 = probability[0, 0]
                    if ypredic_per == 1:
                        prediction = 1
                        cv2.rectangle(img, (int(per_xmin), int(per_ymin)), (int(per_xmax), int(per_ymax)), (255, 0, 0), 2)
                        for nfac, fac in enumerate(faces.values()):
                            fac_xmid = fac[0]
                            fac_ymid = fac[1]
                            fac_xmin = fac[2]
                            fac_ymin = fac[3]
                            fac_xmax = fac[4]
                            fac_ymax = fac[5]
                            a2 = fac_xmax - fac_xmin
                            b2 = fac_ymax - fac_ymin
                            areaf = a2 * b2
                            areai, ai, bi = 0, 0, 0
                            p1 = fac_xmid - per_xmid
                            p2 = fac_ymid - per_ymid
                            dist = math.sqrt(p1 ** 2 + p2 ** 2)

                            Intersection_Up_center, Intersection_Up_left, Intersection_Up_right, Intersection_Down_left, Intersection_Down_center, \
                            Intersection_Down_right, Intersection_Center_right, Intersection_Center_left, Intersection_Center_right, Intersection_Center_left, \
                            Intersection_Inside = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

                            if (per_xmin < fac_xmid and fac_xmid < per_xmax and per_ymin < fac_ymid and fac_ymid < per_ymax):
                                included_center = 1
                            else:
                                included_center = 0

                            if fac_xmax < per_xmin or fac_ymax < per_ymin or fac_xmin > per_xmax or fac_ymin > per_ymax:
                                Intersection_No_intersection = 1
                                continue
                            else:
                                Intersection_No_intersection = 0

                            if  fac_xmin >= per_xmin and fac_xmax <= per_xmax and fac_ymin <= per_ymin and fac_ymax >= per_ymin:  # SC
                                ai = fac_xmax - fac_xmin
                                bi = fac_ymax - per_ymin
                                Intersection_Up_center = 1

                            elif fac_xmax >= per_xmin and fac_xmin <= per_xmin and fac_ymax >= per_ymin and fac_ymin <= per_ymin:  # SI
                                ai = fac_xmax - per_xmin
                                bi = fac_ymax - per_ymin
                                Intersection_Up_left = 1

                            elif fac_xmax >= per_xmax and fac_xmin <= per_xmax and fac_ymax >= per_ymin and fac_ymin <= per_ymin:  # SD
                                ai = per_xmax - fac_xmin
                                bi = fac_ymax - per_ymin
                                Intersection_Up_right = 1

                            elif fac_xmax >= per_xmin and fac_xmin <= per_xmin and fac_ymax >= per_ymax and fac_ymin <= per_ymax:  # II
                                ai = fac_xmax - per_xmin
                                bi = per_ymax - fac_ymin
                                Intersection_Down_left = 1

                            elif fac_xmax >= per_xmin and fac_xmin >= per_xmin and fac_ymax >= per_ymax and fac_ymin <= per_ymax:  # IC
                                ai = fac_xmax - fac_xmin
                                bi = per_ymax - fac_ymin
                                Intersection_Down_center = 1

                            elif fac_xmax >= per_xmax and fac_xmin <= per_xmax and fac_ymax >= per_ymax and fac_ymin <= per_ymax:  # ID
                                ai = per_xmax - fac_xmin
                                bi = per_ymax - fac_ymin
                                Intersection_Down_right = 1

                            elif fac_xmax >= per_xmax and fac_xmin <= per_xmax and fac_ymax <= per_ymax and fac_ymin >= per_ymin:  # CD
                                ai = per_xmax - fac_xmin
                                bi = fac_ymax - fac_ymin
                                Intersection_Center_right = 1

                            elif fac_xmax >= per_xmin and fac_xmin <= per_xmin and fac_ymax <= per_ymax and fac_ymin >= per_ymin:  # CI
                                ai = fac_xmax - per_xmin
                                bi = fac_ymax - fac_ymin
                                Intersection_Center_left = 1

                            elif per_xmin < fac_xmid and fac_xmid < per_xmax and per_ymin < fac_ymid and fac_ymid < per_ymax:  # Hg_in_per
                                ai = fac_xmax - fac_xmin
                                bi = fac_ymax - fac_ymin
                                Intersection_Inside = 1

                            areai = ai * bi


                            predictors_fac = [currentframe, per_xmid, per_ymid, per_xmin, per_ymin, per_xmax, per_ymax, fac_xmid, fac_ymid, fac_xmin, fac_ymin,
                                              fac_xmax, fac_ymax, included_center, areai, areaf, dist, Intersection_Center_left, Intersection_Center_right,
                                              Intersection_Inside, Intersection_No_intersection, Intersection_Up_center, Intersection_Up_left,
                                              Intersection_Up_right]

                            #predictors_fac = ((predictors_fac - u)) / s  # Only for mlp

                            predictors_fac = np.array(predictors_fac).reshape(1,-1) # una fila y el resto de datos indistinto.

                            ypredic_fac = loaded_model_fac.predict(predictors_fac)

                            print(f"Prediction_face {nfac}: {ypredic_fac}")
                            if ypredic_fac == 1:
                                cv2.rectangle(img, (int(fac_xmin), int(fac_ymin)), (int(fac_xmax), int(fac_ymax)), (255, 0, 0), 2)

                                x = int(fac_xmin)
                                y = int(fac_ymin)
                                h = (int(fac_ymax) - int(fac_ymin))
                                w = (int(fac_xmax) - int(fac_xmin))
                                #print(h,w)
                                cropped_image = img[y:y + h, x:x + w]
                                try:
                                    cv2.imshow("Cropped Image", cropped_image)  # Muestra la img cortada.
                                    cv2.imwrite('./cropped_faces/frame' + str(currentframe) + '.jpg', cropped_image)
                                except:
                                    print(f"Error en frame {currentframe}")

                    else:
                        prediction = 0

                archivo.write(f"{currentframe},{nper},{per_xmid},{ypredic_per},{prob0},{prob1},{prediction},{modelo}\n")
        archivo.close()

    return img

netMain = None
metaMain = None
altNames = None
currentframe=-1 #Creamos una variable global q usamos en el contador de cropped faces.

#Automatic model selection
loaded_model_selection = pickle.load(open('./modelos/ams/29/rfc_model_intersection.sav', 'rb'))
#loaded_model_selection = pickle.load(open('./modelos/ams/10/rfc_model_intersection.sav','rb'))

#People detection model
loaded_model_per0 = pickle.load(open('./modelos/hg/modelos_finales/mlp10.sav', 'rb'))
loaded_model_per1 = pickle.load(open('./modelos/hg/modelos_finales/gnb.sav', 'rb'))
loaded_model_per2 = pickle.load(open('./modelos/hg/modelos_finales/logreg1.sav', 'rb'))
#loaded_model_per3 = pickle.load(open('./modelos/hg/modelos_finales/mlp.sav', 'rb'))
#loaded_model_per4 = pickle.load(open('./modelos/hg/modelos_finales/logreg.sav', 'rb'))

#Faces detection model
#loaded_model_fac = pickle.load(open('./modelos/fac/rfc_t.sav', 'rb'))
loaded_model_fac = pickle.load(open('./modelos/fac/rfc.sav', 'rb'))
#loaded_model_fac = pickle.load(open('./modelos/fac/mlp_t.sav', 'rb'))
#loaded_model_fac = pickle.load(open('./modelos/fac/mlp.sav', 'rb'))

def YOLO():
    """
    Perform Object detection
    """
    global metaMain, netMain, altNames
    configPath = "./cfg/yolov4_2.cfg"
    weightPath = "./backup/yolov4_2_best2.weights"
    metaPath = "./data/custom/piford.data"
    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `" +
                         os.path.abspath(configPath)+"`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weightPath)+"`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(metaPath)+"`")
    if netMain is None:
        netMain = darknet.load_net_custom(configPath.encode(
            "ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
    if metaMain is None:
        metaMain = darknet.load_meta(metaPath.encode("ascii"))
    if altNames is None:
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents,
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass
    #cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture("./videos_entrada/Test4.mp4")   # <----- Replace with your video directory
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    new_height, new_width = frame_height // 2, frame_width // 2
    # print("Video Resolution: ",(width, height))

    out = cv2.VideoWriter(
            "./videos_salida/prueba.avi", cv2.VideoWriter_fourcc(*"MJPG"), 10.0, # <----- Replace with your output directory
            (new_width, new_height))
    
    # print("Starting the YOLO loop...")

    # Create an image we reuse for each detect
    darknet_image = darknet.make_image(new_width, new_height, 3)

    frame = 0
    while True:
        frame += 1
        prev_time = time.time()
        ret, frame_read = cap.read()
        # Check if frame present :: 'ret' returns True if frame present, otherwise break the loop.
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb,
                                   (new_width, new_height),
                                   interpolation=cv2.INTER_LINEAR)

        darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes())

        detections = darknet.detect_image(netMain, namesList, darknet_image, thresh=0.25)
        image = cvDrawBoxes(detections, frame_resized)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print("\n")
        print(f"FRAME: {frame}")
        print("FPS: " + str(1/(time.time()-prev_time)))
        cv2.imshow('Demo', image)
        cv2.waitKey(3)
        if cv2.waitKey(1) == ord("q"):
            break
        out.write(image)

    cap.release()
    out.release()
    print(":::Video Write Completed")

if __name__ == "__main__":
    YOLO()