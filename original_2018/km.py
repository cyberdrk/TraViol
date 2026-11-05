import cv2


backsub = cv2.bgsegm.createBackgroundSubtractorMOG() #background subtraction to isolate moving cars
capture = cv2.VideoCapture("short.mp4") #change to destination on your pc 
w = 400
h = 400
i = 0

minArea=75000

x = 0 
y = 0
cv2.namedWindow('TRACK', cv2.WINDOW_NORMAL) # Creating a resizable image window

flag = 0

while True:
    ret, frame = capture.read()
    
    fgmask = backsub.apply(frame, None, 0.01)
    
    erode=cv2.erode(fgmask,None,iterations=3)     #erosion to erase unwanted small contours
    erode=cv2.dilate(fgmask,None,iterations=5)     #erosion to erase unwanted small contours
    moments=cv2.moments(erode,True)               #moments method applied
    
    area=moments['m00']    
    print area

    print flag

    if flag == 7:
        flag = 0
    
    if flag == 0:
        if moments['m00'] >=minArea:
            x=int(moments['m10']/moments['m00'])
            y=int (moments['m01']/moments['m00'])

            i = i + 1

            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 3)
            cv2.imwrite('C:\\TraViol\\cars\\{}.png'.format(i), frame)
    
    flag = flag + 1


    '''
        if x>40 and x<55 and y>50 and y<65:       #range of line coordinates for values on left lane
            i=i+1
            print(i)
    #    print(x,y)
    '''
    

    

    cv2.putText(frame,'COUNT: %r' %i, (10,30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 0, 0), 2)
    cv2.imshow("Track", frame)
    cv2.imshow("background sub", fgmask)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break