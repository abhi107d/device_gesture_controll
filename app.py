import cv2
import sys
import mediapipe as mp
import numpy as np

#variables for camera
cam_source2=0

modelConfiguration = 'cfg/yolov3-tiny.cfg'
modelWeights = 'weights/yolov3-tiny.weights'
net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)

with open('data/coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

confidence_threshold = 0.5
nms_threshold = 0.7




#initlizing 
hand_dect=mp.solutions.hands
draw=mp.solutions.drawing_utils
cam=cv2.VideoCapture(cam_source2)



if not cam.isOpened():
    print("failed to open camera ")
    sys.exit()


def detect(flipped_frame):
    #detection XD
    rgb_img=cv2.cvtColor(flipped_frame,cv2.COLOR_BGR2RGB)
    rgb_img.flags.writeable=False
    rslt=hands.process(rgb_img)   #model result
    return rslt

#draw the trace
def draw_hand(rslt,flipped_frame):
   
    for num,hand in enumerate(rslt.multi_hand_landmarks):
        draw.draw_landmarks(flipped_frame,hand,hand_dect.HAND_CONNECTIONS,
                            draw.DrawingSpec(color=(255, 0, 255),thickness=2,circle_radius=4),
                            draw.DrawingSpec(color=(255,176,18),thickness=2,circle_radius=4))
    #print(rslt.multi_hand_landmarks[0].landmark[0])
                    

#getting servo angles
def On_off_dectect(landmarks):
    flag=[]


    #finger tips
    finger_tips=[]
    for i in range(4,21,8):
        finger_tips.append(np.array([landmarks.landmark[i].x,landmarks.landmark[i].y]))


    #clamp
    dist1=np.linalg.norm(finger_tips[0]-finger_tips[2])
    dist2=np.linalg.norm(finger_tips[0]-finger_tips[1])
    if dist1 and dist2 <=0.1:
        flag.append("ON")
    else :
        flag.append("OFF")

    return flag

    






if __name__=="__main__":
    
    with hand_dect.Hands(min_detection_confidence=0.8,min_tracking_confidence=0.4,max_num_hands=1) as hands:
        while True:
            #capture frame
            ret,frame=cam.read()

            if not ret:
                print("failed to capture frame")
                sys.exit()
 


            frame=cv2.flip(frame,1)

            rslt=detect(frame)
            (H, W) = frame.shape[:2]

            blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
            net.setInput(blob)
            layerOutputs = net.forward(net.getUnconnectedOutLayersNames())

            boxes = []
            confidences = []
            classIDs = []

            for output in layerOutputs:
                for detection in output:
                    scores = detection[5:]
                    classID = np.argmax(scores)
                    confidence = scores[classID]

                    if confidence > confidence_threshold:
                        box = detection[0:4] * np.array([W, H, W, H])
                        (centerX, centerY, width, height) = box.astype("int")

                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))

                        boxes.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))
                        classIDs.append(classID)

            indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)
            colors = np.random.uniform(0, 255, size=(len(classes), 3))

            if len(indices) > 0:
                for i in indices.flatten():
                    (x, y, w, h) = boxes[i]
                    confidence = confidences[i]

                    color = [255,2,200] 
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    text = f"{classes[classIDs[i]]}: {confidence:.2f}"
                    cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
##########################################################
            if rslt.multi_hand_landmarks:
                #draw_hand(rslt,flipped_frame)
                on_or_off=On_off_dectect(rslt.multi_hand_landmarks[0])
                cv2.putText(frame, str(on_or_off), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            
            #showing image
            cv2.imshow("hand_detection",frame)


            #exit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break


    cam.release()
    cv2.destroyAllWindows()

