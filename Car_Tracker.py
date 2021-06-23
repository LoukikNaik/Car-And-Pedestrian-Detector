import cv2

#img_file = 'car2.jpeg'
video=cv2.VideoCapture('Tesla Dashcam.mp4')

classifier_file='Cars.xml'
car_tracker=cv2.CascadeClassifier(classifier_file)

while True:
    (read_succesful,frame)=video.read()
    if read_succesful:
        grayscaled_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    else:
        break
    cars=car_tracker.detectMultiScale(grayscaled_frame)
    for (x,y,w,z) in cars:
        cv2.rectangle(frame,(x,y),(x+w,y+z),(0,0,255),4)

    cv2.imshow('Car Tracker',frame)
    cv2.waitKey(2)



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