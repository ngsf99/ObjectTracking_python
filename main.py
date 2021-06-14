from tracker import *
import cv2 

#create tracker object
tracker = EuclideanDistTracker() #takes all the bounding boxes of the object

#load video
cap = cv2.VideoCapture("highway.mp4") 

#Object detection from stable camera
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40) #the longer the history the higher the precise(but hardly adapt if the camera is moving), #varThreshold, the lower the value the more FP , so higher the value mean lesser detection , so lower FP  

#start a loop , because its a video so we need extract each frame, one after another
while True:
    ret, frame = cap.read()
    height,weight,_ = frame.shape

    #print(height,weight) # 720,1280   
    
    #Extract region of interest
    roi = frame[340:720,500:800]

    #1.Object Detection
    mask = object_detector.apply(roi)
    _, mask = cv2.threshold(mask,254,255,cv2.THRESH_BINARY) # 0-255 , 0 is completely dark , 255 is white
    #mask = object_detector.apply(frame) #make everything in "frame" black, and moving object into white
    contours , _ = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    detections = [] 
    #trace boxes
    for cnt in contours:
        #Calculate area and remove small elements
        area = cv2.contourArea(cnt)
        if area > 100:
            #cv2.drawContours(roi,[cnt],-1,(0,255,0),2)
            x,y,w,h = cv2.boundingRect(cnt)
            
            detections.append([x,y,w,h]) 

    #2.Object Tracking
    boxes_ids = tracker.update(detections)
    for box_id in boxes_ids:
        x,y,w,h,id = box_id
        cv2.putText(roi,str(id),(x,y-15),cv2.FONT_HERSHEY_PLAIN,1,(0,0,255),2)
        cv2.rectangle(roi, (x,y), (x+w,y+h), (255,0,0),3)

    #show realtime
    cv2.imshow("Frame",frame) 
    cv2.imshow("Mask",mask)
    cv2.imshow("Roi",roi)

    key = cv2.waitKey(30)
    if key == 27:   # s key 
        break 

cap.release()
cv2.destroyAllWindows()
