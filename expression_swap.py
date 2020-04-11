import sys
import dlib
import cv2
import numpy as np
import time
import math
import constants
from swap import Swap

### FUNCTIONS ###

# Returns two images from one separated in the middle of two rectangles
# Is used to separate 2 faces in two images to treat them separatly
def splitBetweenFaces(img, rect_face_A, rect_face_B):
    if rect_face_A.right() < rect_face_B.left():
        middle = int((rect_face_B.left() + rect_face_A.right()) / 2)
    elif rect_face_B.right() < rect_face_A.left():
        middle = int((rect_face_A.left() + rect_face_B.right()) / 2)
        # Img_A will be the left one
        rect_face_temp = rect_face_A
        rect_face_A = rect_face_B
        rect_face_B = rect_face_temp
    else:
        return [], rect_face_A, [], rect_face_B

    img_A = img[0:img.shape[0], 0:middle]
    img_B = img[0:img.shape[0], middle:img.shape[1]]
    rect_face_B = dlib.rectangle(rect_face_B.left() - middle + 1, rect_face_B.top(), rect_face_B.right() - middle + 1, rect_face_B.bottom())
    return img_A, rect_face_A, img_B, rect_face_B

# This function takes a picture from the camera with user help
# To take the picture you need two faces in front of the camera, separated by the line
def getInitFaces():
    errorNotEnoughFaces = False
    errorFacesPlacement = False
    errorTimer = 0
    displayInitImages = False
    src_img = None
    dst_img = None
    src_rect_face = None
    dst_rect_face = None
    while True:
        # Read from camera
        _, cam_img = cam.read()
        saved_cam_img = cam_img.copy()
        cam_img_gray = cv2.cvtColor(cam_img, cv2.COLOR_BGR2GRAY)
        faces = face_detector(cam_img_gray)
        if not displayInitImages:
            if len(faces) > 0:
                src_rect_face = faces[0]
            else:
                src_rect_face = None

            if len(faces) > 1:
                dst_rect_face = faces[1]
                for i in range(0, 10):
                    cv2.rectangle(cam_img, (int(cam_img.shape[1]/2 - 2), int(cam_img.shape[0] * i/10)), (int(cam_img.shape[1]/2 + 2), int(cam_img.shape[0] * i/10) + 30), (0, 0, 255), cv2.FILLED)
            else:
                dst_rect_face = None

            if errorNotEnoughFaces:
                cv2.putText(cam_img, "No faces found on this frame. Please retry.", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255))
                if time.perf_counter() - errorTimer >= constants._DISPLAY_ERROR_TIME_:
                    errorNotEnoughFaces = False

            if errorFacesPlacement:
                cv2.putText(cam_img, "Please leave a gap between your faces and retry.", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255))
                if time.perf_counter() - errorTimer >= constants._DISPLAY_ERROR_TIME_:
                    errorFacesPlacement = False

        if displayInitImages:
            cv2.putText(cam_img, "Alright ? (y/n)", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255))
            cv2.imshow("Source", src_img)
            cv2.imshow("Destination", dst_img)

        if src_rect_face != None:
            # Draw rect around face
            cv2.rectangle(cam_img, (src_rect_face.left(), src_rect_face.top()), (src_rect_face.right(), src_rect_face.bottom()), (255, 255, 255), 1)
        
        if dst_rect_face != None:
            # Draw rect around face
            cv2.rectangle(cam_img, (dst_rect_face.left(), dst_rect_face.top()), (dst_rect_face.right(), dst_rect_face.bottom()), (255, 255, 255), 1)
        
        # Show img
        cv2.imshow("Initialization", cam_img)

        key = cv2.waitKey(1)
        if not displayInitImages and key == 32: # 32 == Space
            if len(faces) < 2:
                errorNotEnoughFaces = True
                errorFacesPlacement = False
                errorTimer = time.perf_counter()
            else:
                src_img, src_rect_face, dst_img, dst_rect_face = splitBetweenFaces(saved_cam_img, src_rect_face, dst_rect_face)
                if len(src_img) == 0:
                    errorFacesPlacement = True
                    errorNotEnoughFaces = False
                    errorTimer = time.perf_counter()
                else:
                    errorFacesPlacement = False
                    errorNotEnoughFaces = False
                    displayInitImages = True
        elif displayInitImages and key == 121: # 121 == 'y'
            if constants._DEBUG_:
                cv2.destroyWindow("Initialization")
            cv2.destroyWindow("Source")
            cv2.destroyWindow("Destination")
            return src_img, src_rect_face, dst_img, dst_rect_face
        elif displayInitImages and key == 110: # 110 == 'n'
            cv2.destroyWindow("Source")
            cv2.destroyWindow("Destination")
            displayInitImages = False
        elif key == 27: # 27 == Escape
            exit()

### INITIALIZATION ###

if len(sys.argv) != 2:
    print("python expression_swap.py predictor.dat")
    exit()

# Set detector / predictor
shape_predictor = dlib.shape_predictor(sys.argv[1])
face_detector = dlib.get_frontal_face_detector()

## SOURCE
# Camera initialization
cam = cv2.VideoCapture(0)

# Get init src img
src_img_init, src_rect_face_init, dst_img_init, dst_rect_face_init = getInitFaces()

# Create Swap object
right_swap = Swap(src_img_init, src_rect_face_init, dst_img_init, dst_rect_face_init, shape_predictor)

### EXPRESSION SWAP ###
while True:
    if constants._DEBUG_:
        initial_time = time.perf_counter()

    # Get camera image (1 frame)
    _, cam_img = cam.read()
    cam_img_gray = cv2.cvtColor(cam_img, cv2.COLOR_BGR2GRAY)
    img_final = cam_img.copy()
    if constants._DEBUG_:
        debug_img = cam_img_gray.copy()

    # Get Faces
    faces = face_detector(cam_img_gray)
    if len(faces) > 1:
        # Get Faces
        left_rect_face = faces[0]
        right_rect_face = faces[1]
        
        # Split the image in two
        left_img, left_rect_face, right_img, right_rect_face = splitBetweenFaces(cam_img, left_rect_face, right_rect_face)
        if len(left_img) != 0:
            right_img = right_swap.getFrame(left_img, left_rect_face, right_img, right_rect_face)

            # Put the dst part in the whole img
            img_final[0:cam_img.shape[0], 0:left_img.shape[1]] = left_img
            img_final[0:cam_img.shape[0], left_img.shape[1]:cam_img.shape[1]] = right_img

            if constants._DEBUG_:
                left_debug_img, right_debug_img = right_swap.getDebugFrame()
                debug_img[0:debug_img.shape[0], 0:left_debug_img.shape[1]] = left_debug_img
                debug_img[0:debug_img.shape[0], left_debug_img.shape[1]:debug_img.shape[1]] = right_debug_img

    ## Display images
    cv2.imshow("Final", img_final)

    if constants._DEBUG_:
        # Display FPS
        fps = str(int(1 / (time.perf_counter() - initial_time))) + " FPS"
        cv2.putText(debug_img, fps, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
        cv2.imshow("DEBUG", debug_img)

    # Get key
    key = cv2.waitKey(1)
    if key == 27: # 27 == Escape
        exit()