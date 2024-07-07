import cv2

cam = cv2.VideoCapture(0)

while True:
    value, frame = cam.read()

    
    #frame2 = cv2.imread(frame, cv2.IMREAD_GRAYSCALE)

    frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    GRID = (5, 5)

    img_captured_corners = None


    found, corners = cv2.findChessboardCorners(frame2, GRID, None)

    print("found: ", found, " corners: ", corners)

    cv2.imshow("Camera View", frame)

    
    if found:
        img_captured_corners = cv2.drawChessboardCorners(frame, GRID, corners, found)
        cv2.imshow("Camera View", img_captured_corners)

    # else:
    #     cv2.imshow("camera view", frame2)

    #cv2.imshow("img_cap", img_captured_corners)


    
    '''
    cv2.imshow("Camera View", frame2)

    

    found, corners = cv2.findChessboardCorners(frame2, GRID, cv2.CALIB_CB_ADAPTIVE_THRESH)

    img_captured_corners = cv2.drawChessboardCorners(frame2, GRID, corners, found)

    cv2.imshow("img_captured_corners", img_captured_corners)
    
    '''
    


    #cv2.imshow("Camera View", frame)

    #cv2.findChessboardCorners(frame, patternSize= cv::Size(8,8), flags=CALIB_CB_FAST_CHECK)

    if cv2.waitKey(1) == ord('q'):
        break




