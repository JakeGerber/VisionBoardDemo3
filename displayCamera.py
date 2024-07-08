import cv2
import cvzone
import numpy as np

capture = cv2.VideoCapture(0)

chessboardOverlay = cv2.imread("bop.png", cv2.IMREAD_UNCHANGED)
#chessboardOverlay2 = cv2.resize(chessboardOverlay, (0, 0), None, 0.5, 0.5)


if not capture.isOpened():
    print("Error: Can't Open Camera")
    exit()

while True:
    success, frame = capture.read()
    
    if not success:
        print("Error: Can't Get Frame")
        break
    
    # Check if the image has an alpha channel
    if chessboardOverlay.shape[2] == 3:
        # Create an alpha channel (fully opaque)
        alpha_channel = np.ones((chessboardOverlay.shape[0], chessboardOverlay.shape[1]), dtype=chessboardOverlay.dtype) * 255
        # Add the alpha channel to the image
        chessboardOverlay = cv2.merge((chessboardOverlay, alpha_channel))

    

    alpha_value = 128
    chessboardOverlay[:, :, 3] = alpha_value

    #Note: Some of the photos don't have an alpha channel (which controls transparency)


    # Now you can use cvzone.overlayPNG without issues
    overlayResult = cvzone.overlayPNG(frame, chessboardOverlay, [0, 0])
    

    #cv2.imshow('Camera View', overlayResult)
    cv2.imshow("bo", overlayResult)
    #cv2.imshow("bo2", chessboardOverlay2)
    
    # Break the loop on 'q' key press
    if cv2.waitKey(1) == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()

