import cv2

#img_file = 'car2.jpeg'
video=cv2.VideoCapture('Pedestrians Compilation .mp4')

classifier_file='Cars.xml'
pedestrian_tracker_file='pedestrain.xml'

car_tracker=cv2.CascadeClassifier(classifier_file)
pedestrian_tracker = cv2.CascadeClassifier(pedestrian_tracker_file)

while True:
    (read_succesful,frame)=video.read()
    if not read_succesful:
        break
    if read_succesful:
        grayscaled_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    else:
        break
    cars=car_tracker.detectMultiScale(grayscaled_frame)
    pedestrians=pedestrian_tracker.detectMultiScale(grayscaled_frame)
    for (x,y,w,z) in cars:
        cv2.rectangle(frame,(x,y),(x+w,y+z),(0,0,255),4)
    for (x,y,w,z) in pedestrians:
        cv2.rectangle(frame,(x,y),(x+w,y+z),(0,255,255),4)

    cv2.imshow('Car and Pedestrian Tracker',frame)
    key=cv2.waitKey(1)
    if key==81 or key==113:
        break
video.release()

"""
for images
img=cv2.imread(img_file)

black_n_white=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

car_tracker=cv2.CascadeClassifier(classifier_file)

cars=car_tracker.detectMultiScale(black_n_white)

print(cars)

for (x,y,w,z) in cars:
    cv2.rectangle(img,(x,y),(x+w,y+z),(0,0,255),4)



cv2.imshow("Car Detector",img)
cv2.waitKey()
"""



print("Code Completed")
