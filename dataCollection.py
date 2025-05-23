import cv2
import cvzone
from cvzone.FaceDetectionModule import FaceDetector
from time import time

#########################
classID = 0 # 0 is fake and 1 is real
outPutFolderPath = 'Dataset/DataCollect'
confidence = 0.8
save = True
blurThreshold = 40

offsetpercentageW = 10
offsetPercentageH = 20
camwidht, camheight = 640,480
floatingPoint = 6
########################

cap = cv2.VideoCapture(0)
detector = FaceDetector(minDetectionCon=0.5, modelSelection=0)
cap.set(3, camwidht)
cap.set(4, camheight)

while True:
    success, img = cap.read()
    imgOut = img.copy()
    img, bboxs = detector.findFaces(img, draw=False)

    listBlur = [] #True False values indicating if the faces are blur or not
    listInfo = [] #The normalized values and the class name for the label txt file

    if bboxs:
        for bbox in bboxs:

            # ---- Get Data  ---- #
            x, y, w, h = bbox['bbox']
            score = bbox['score'][0]

            if score > confidence:
                #----- Adding an offset to the face Detected ------#
                offsetW = (offsetPercentageW / 100) * w
                x = int(x - offsetW)
                w = int(w + offsetW * 2)

                offsetH = (offsetPercentageH / 100 ) * h
                y = int(y - offsetH * 3)
                h = int(h + offsetH * 3.5)

                #-------- Find Blurriness ----------#
                imgFace = img[y:y + h, x:x + w]
                if imgFace.size > 0:
                    blurValue = int(cv2.Laplacian(imgFace, cv2.CV_64F).var())
                    if blurValue > blurThreshold:
                        listBlur.append(True)
                    else:
                        listBlur.append(False)

                #------- Normalize Values ---------#
                # ih → Image height, iw → Image width, _ → Ignoring the third value (number of color channels, usually 3 for RGB)
                ih, iw, _ = img.shape
                # xc = x + w / 2 → Center X-coordinate, yc = y + h / 2 → Center Y-coordinate
                xc, yc = x+w/2, y+h/2
                # xcn = xc / iw → Normalized X center (relative to image width).
                # ycn = yc / ih → Normalized Y center (relative to image height).
                xcn, ycn = round(xc/iw, floatingPoint), round(yc/ih, floatingPoint)
                # wn = w / iw → Normalized width (relative to image width).
                # hn = h / ih → Normalized height (relative to image height).
                wn, hn = round(w/iw, floatingPoint), round(h/ih, floatingPoint)
                # print(xcn,ycn, wn, hn)

                #------ Avoid Value Above 1 -------#
                if xcn > 1: xcn = 1
                if ycn > 1: ycn = 1
                if wn > 1: wn = 1
                if hn > 1: hn = 1

                listInfo.append(f"{classID} {xcn} {ycn} {wn} {hn}\n")

                # ---- Draw Data  ---- #
                cv2.rectangle(imgOut, (x, y, w, h), (255,0,0), 3)
                cvzone.putTextRect(imgOut,f'Score: {int(score*100)}% Blur: {blurValue}', (x,y-20),scale=2,thickness=3)

        if save:
            if all(listBlur) and listBlur != []:
                #----- Save Image ------#
                timeNow = time()
                timeNow = str(timeNow).split('.')
                timeNow = timeNow[0]+timeNow[1]
                cv2.imwrite(f"{outPutFolderPath}/{timeNow}.jpg", img)

                #------ Save Label Text File -------#
                for info in listInfo:
                    f = open(f"{outPutFolderPath}/{timeNow}.txt", 'a')
                    f.write(info)
                    f.close()

    # Display the image in a window named 'Image'
    cv2.imshow("Image", imgOut)
    # Wait for 1 millisecond, and keep the window open
    cv2.waitKey(1)


