import sys
import dlib
import cv2
import numpy as np
import time
import math

_DEBUG_ = True
_DISPLAY_ERROR_TIME_ = 5
_HEIGHT_IMG_ = 800

### FUNCTIONS ###

def getFace(img):
    # Detect faces
    faces_rect = face_detector(img, 0)
    if len(faces_rect) > 0:
        return faces_rect[0]
    else:
        return None


def getLandmarks(img, face_rect):
    # Get landmarks and convert it to array
    landmarks = shape_predictor(img, face_rect)
    landmarks_points = []
    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        landmarks_points.append((x, y))

        # Display landmarks
        if _DEBUG_:
            cv2.circle(img, (x, y), 2, (0, 0, 255), 2)

    return landmarks_points

def getInitFace():
    displayError = False
    displayErrorTimer = 0
    displayInitImage = False
    src_img_init = None
    while True:
        # Read from camera
        _, src_img = cam.read()
        src_img_gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
        src_rect_face = getFace(src_img_gray)

        if displayError:
            cv2.putText(src_img, "No faces found on this frame. Please retry.", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255))
            if time.perf_counter() - displayErrorTimer >= _DISPLAY_ERROR_TIME_:
                displayError = False

        if displayInitImage:
            cv2.putText(src_img, "Alright ? (y/n)", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255))
            cv2.imshow("Initialization Source", src_img_init)

        if src_rect_face != None:
            # Draw rect around face
            cv2.rectangle(src_img, (src_rect_face.left(), src_rect_face.top()), (src_rect_face.right(), src_rect_face.bottom()), (255, 255, 255), 1)
        # Show img
        cv2.imshow("Source", src_img)

        key = cv2.waitKey(1)
        if key == 32: # 32 == Space
            if src_rect_face == None:
                displayError = True
                displayErrorTimer = time.perf_counter()
            else:
                _, src_img_init = cam.read()
                displayInitImage = True
        elif displayInitImage and key == 121: # 121 == 'y'
            if _DEBUG_:
                cv2.destroyWindow("Source")
            cv2.destroyWindow("Initialization Source")
            return src_img_init, src_rect_face
        elif displayInitImage and key == 110: # 110 == 'n'
            cv2.destroyWindow("Initialization Source")
            displayInitImage = False
        elif key == 27: # 27 == Escape
            exit()

def getConvexRect(points):
    # Find convex hull and the bounding rect of it - This permit some optimization for the subdivision
    convexhull = cv2.convexHull(points)
    convex_rect = cv2.boundingRect(convexhull)
    return convex_rect

def getPointsFromRect(points, rect):
    new_points = []
    for i in range(0, len(points)):
        new_points.append((points[i][0] - rect[0], points[i][1] - rect[1]))
    return new_points

def getTriangles(landmarks, img):
    # Convert points into numpy array
    points = np.array(landmarks, np.int32)

    # Get convex rect
    convex_rect = getConvexRect(points)

    # Delaunay subdivision
    subdiv = cv2.Subdiv2D(convex_rect)
    subdiv.insert(landmarks)
    triangles = np.array(subdiv.getTriangleList(), dtype=np.int32)

    # Get indexes of all the points in each triangle
    indexes_triangles = []
    for t in triangles:
        # Get the 3 points of the triangle
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        # Find the corresponding points and get their indexes
        index_pt1 = np.where((points == pt1).all(axis=1))[0][0]
        index_pt2 = np.where((points == pt2).all(axis=1))[0][0]
        index_pt3 = np.where((points == pt3).all(axis=1))[0][0]

        if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
            # Add it in the array as a triangle
            triangle = [index_pt1, index_pt2, index_pt3]
            indexes_triangles.append(triangle)

            if _DEBUG_:
                # Display triangle
                cv2.line(img, pt1, pt2, (255, 255, 255), 1)
                cv2.line(img, pt2, pt3, (255, 255, 255), 1)
                cv2.line(img, pt3, pt1, (255, 255, 255), 1)
    return indexes_triangles, convex_rect

### INITIALIZATION ###

if len(sys.argv) != 2:
    print("python expression_swap.py predictor.dat")
    exit()

# Set detector / predictor
shape_predictor = dlib.shape_predictor(sys.argv[1])
face_detector = dlib.get_frontal_face_detector()

## INIT SOURCE
# Camera initialization
cam = cv2.VideoCapture(0)

# Get init src img
src_img_init, src_rect_face = getInitFace()
src_img_gray_init = cv2.cvtColor(src_img_init, cv2.COLOR_BGR2GRAY)

# Get init src landmarks
src_landmarks_init = getLandmarks(src_img_gray_init, src_rect_face)
# Take all the eyebrows
for i in range(17, 26):
    src_landmarks_init[i] = (src_landmarks_init[i][0], src_landmarks_init[i][1] - 20)

# Get landmarks coordinates from face coords
src_convex_rect_init = getConvexRect(np.array(src_landmarks_init, np.int32))
src_points_init = getPointsFromRect(src_landmarks_init, src_convex_rect_init)


### EXPRESSION SWAP ###
while True:
    if _DEBUG_:
        initial_time = time.perf_counter()

    # Get src img
    _, cam_img = cam.read()
    cam_img_gray = cv2.cvtColor(cam_img, cv2.COLOR_BGR2GRAY)

    # Get Face
    faces = face_detector(cam_img_gray)
    if len(faces) > 1:
        ## DESTINATION
        #Get DST Face
        dst_rect_face = faces[1]

        # Get dst landmarks
        dst_landmarks = getLandmarks(cam_img_gray, dst_rect_face)
        # Take all the eyebrows
        for i in range(17, 26):
            dst_landmarks[i] = (dst_landmarks[i][0], dst_landmarks[i][1] - 20)

        # Get triangles from init src
        dst_triangles, dst_convex_rect = getTriangles(dst_landmarks, cam_img_gray)

        # Get landmarks coordinates from face coords
        dst_points = getPointsFromRect(dst_landmarks, dst_convex_rect)

        ## SOURCE
        #Get SRC face
        src_rect_face = faces[0]

        # Take beyond the face to prevent bugs when openning mouth
        # src_rect_face = dlib.rectangle(src_rect_face.left(), src_rect_face.top(), src_rect_face.right(), src_rect_face.bottom() + 20)

        # Get src landmarks
        src_landmarks = getLandmarks(cam_img_gray, src_rect_face)
        # Take all the eyebrows
        for i in range(17, 26):
            src_landmarks[i] = (src_landmarks[i][0], src_landmarks[i][1] - 20)

        ## Face rotation issues
        # Get coordinates from temple
        src_points_temple = []
        for i in range(0, len(src_landmarks)):
            src_points_temple.append((src_landmarks[i][0] - src_landmarks[0][0], src_landmarks[i][1] - src_landmarks[0][1]))

        # Z-axis
        z_rotation_angle = -math.atan2(src_points_temple[16][1], src_points_temple[16][0])
        z_trigo = (math.cos(z_rotation_angle), math.sin(z_rotation_angle))
        for i in range(1, len(src_points_temple)):
            src_points_temple[i] = (int(src_points_temple[i][0] * z_trigo[0] - src_points_temple[i][1] * z_trigo[1]),
                                    int(src_points_temple[i][0] * z_trigo[1] + src_points_temple[i][1] * z_trigo[0]))

        # Get coordinates back in landmarks
        for i in range(0, len(src_landmarks)):
            src_landmarks[i] = (src_points_temple[i][0] + src_landmarks[0][0], src_points_temple[i][1] + src_landmarks[0][1])

        # Get convex rect after fixing rotation issues
        src_convex_rect = getConvexRect(np.array(src_landmarks, np.int32))

        # Eyebrows issue
        src_eyebrows_movement = src_landmarks[0][1] - src_convex_rect[1] - src_points_init[0][1]
        src_convex_rect = (src_convex_rect[0], src_convex_rect[1] + src_eyebrows_movement,
                            src_convex_rect[2], src_convex_rect[3] - src_eyebrows_movement)

        # Get points from rect
        src_points = getPointsFromRect(src_landmarks, src_convex_rect)

        # Calculate ratio
        src_rect_ratio = src_convex_rect_init[2] / src_convex_rect[2]

        # Get new dst landmarks
        dst_landmarks_new = []
        dst_points_new = []
        for i in range(0, len(src_points)):
            # Apply ration on src landmarks
            src_point_normalized = (int(src_points[i][0] * src_rect_ratio), int(src_points[i][1] * src_rect_ratio))

            # Make subtraction of init and current src points
            substract_result = (src_points_init[i][0] - src_point_normalized[0], src_points_init[i][1] - src_point_normalized[1])
            # Apply the substraction on dst img
            dst_point_new = (dst_points[i][0] - substract_result[0], dst_points[i][1] - substract_result[1])

            # Save the new point
            dst_points_new.append(dst_point_new)

            # Get point from the whole image
            dst_landmark_new = (dst_point_new[0] + dst_convex_rect[0], dst_point_new[1] + dst_convex_rect[1])

            dst_landmarks_new.append(dst_landmark_new)

            # Check if landmark in dst img to provide crashes
            arePointsInImage = True
            if dst_landmark_new[0] < 0 or dst_landmark_new[1] < 0 or dst_landmark_new[0] >= cam_img.shape[1] or dst_landmark_new[1] >= cam_img.shape[0]:
                arePointsInImage = False
                break
            
            if _DEBUG_:
                # Display landmarks
                cv2.circle(cam_img_gray, dst_landmark_new, 2, (0, 0, 255), 2)

        if not arePointsInImage:
                continue

        # Creating new image
        img_new = np.zeros(cam_img.shape, np.uint8)

        for dst_triangle in dst_triangles:
            # Get old triangle
            pt1 = dst_landmarks[dst_triangle[0]]
            pt2 = dst_landmarks[dst_triangle[1]]
            pt3 = dst_landmarks[dst_triangle[2]]
            old_triangle = np.array([pt1, pt2, pt3], np.int32)

            # Get rect of triangle
            (x, y, w, h) = cv2.boundingRect(old_triangle)

            # Crop the image into the rect of the triangle
            dst_img_triangle_rect = cam_img[y: y + h, x: x + w]

            # Substract rect position to triangle's
            old_triangle = np.array([[pt1[0] - x, pt1[1] - y],
                                    [pt2[0] - x, pt2[1] - y],
                                    [pt3[0] - x, pt3[1] - y]], np.int32)

            # Get new triangle
            pt1 = dst_landmarks_new[dst_triangle[0]]
            pt2 = dst_landmarks_new[dst_triangle[1]]
            pt3 = dst_landmarks_new[dst_triangle[2]]
            new_triangle = np.array([pt1, pt2, pt3], np.int32)

            # Get rect of triangle
            (x, y, w, h) = cv2.boundingRect(new_triangle)

            # Substract rect position to triangle's
            new_triangle = np.array([[pt1[0] - x, pt1[1] - y],
                                    [pt2[0] - x, pt2[1] - y],
                                    [pt3[0] - x, pt3[1] - y]], np.int32)

            # Create a mask to cut the rect into a real triangle
            new_triangle_mask = np.zeros((h, w), np.uint8)
            cv2.fillConvexPoly(new_triangle_mask, new_triangle, 255)

            # Warping
            transformationMatrix = cv2.getAffineTransform(np.float32(old_triangle), np.float32(new_triangle))
            warped_triangle = cv2.warpAffine(dst_img_triangle_rect, transformationMatrix, (w, h))
            warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=new_triangle_mask)

            img_new_triangle_rect = img_new[y: y + h, x: x + w]
            
            img_new_triangle_rect_gray = cv2.cvtColor(img_new_triangle_rect, cv2.COLOR_BGR2GRAY)
            _, mask_triangle_border = cv2.threshold(img_new_triangle_rect_gray, 1, 255, cv2.THRESH_BINARY_INV)
            warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=mask_triangle_border)

            img_new_triangle_rect = cv2.add(img_new_triangle_rect, warped_triangle)
            img_new[y: y + h, x: x + w] = img_new_triangle_rect
        
        # Draw new face on old image
        dst_img_new = np.zeros_like(cam_img_gray)
        dst_convex_new = cv2.convexHull(np.array(dst_landmarks_new, np.int32))
        dst_face_mask = cv2.bitwise_not(cv2.fillConvexPoly(dst_img_new, dst_convex_new, 255))

        # Blur around new face



        # Adding alpha channel
        
        
        dst_head_noface = cv2.bitwise_and(cam_img, cam_img, mask=dst_face_mask)
        img_new = cv2.add(dst_head_noface, img_new)

    # Display images
    if _DEBUG_:
        # Display FPS
        fps = str(int(1 / (time.perf_counter() - initial_time))) + " FPS"
        cv2.putText(cam_img_gray, fps, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))

        if len(faces) > 1:
            # Display rect around face
            cv2.rectangle(cam_img_gray, (src_convex_rect[0], src_convex_rect[1]),
                        (src_convex_rect[0] + src_convex_rect[2], src_convex_rect[1] + src_convex_rect[3]), (255, 255, 255), 2)

        if len(faces) > 1:
            cv2.imshow("Final", img_new)
        else:
            cv2.imshow("Final", cam_img)
        cv2.imshow("DEBUG", cam_img_gray)
    else:
        if len(faces) > 1:
            cv2.imshow("Final", img_new)
        else:
            cv2.imshow("Final", cam_img)

    # Get key
    key = cv2.waitKey(1)
    if key == 27: # 27 == Escape
        exit()