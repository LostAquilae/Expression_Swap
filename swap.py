import dlib
import cv2
import numpy as np
import math
import constants

class Swap:
    """
    This class contains all the variables and the functions needed for a facial swap.
    There are attributes :
    - shape_predictor : function that detects landmarks on a face
    - dst_img_init : destination image where the subject hasn't any facial expressions (initial image)
    - src_convex_rect_init : rectangle made by landmarks around the face on the source initial image
    - src_points_init : landmarks of source initial image but coordinates depends on the convex rect
    - dst_landmarks_init : landmarks on the destination init face
    - dst_triangles_init : a list of tuple that contains the index for every point of each triangles. The indexes refer to dst_landmarks_init.
    """

    # Constructor
    def __init__(self, src_img_init, src_rect_face_init, dst_img_init, dst_rect_face_init, shape_predictor):
        self.dst_img_init = dst_img_init
        self.shape_predictor = shape_predictor

        src_img_gray_init = cv2.cvtColor(src_img_init, cv2.COLOR_BGR2GRAY)
        dst_img_gray_init = cv2.cvtColor(dst_img_init, cv2.COLOR_BGR2GRAY)

        ## Source
        # Get init src landmarks
        src_landmarks_init = self.getLandmarks(src_img_gray_init, src_rect_face_init)
        # Get landmarks coordinates from face coords
        self.src_convex_rect_init = self.getConvexRect(np.array(src_landmarks_init, np.int32))
        self.src_points_init = self.getPointsFromRect(src_landmarks_init, self.src_convex_rect_init)


        ## Destination
        # Get init dst landmarks
        self.dst_landmarks_init = self.getLandmarks(dst_img_gray_init, dst_rect_face_init)
        # Get triangles from init src
        self.dst_triangles_init, self.dst_convex_rect_init = self.getTriangles(self.dst_landmarks_init)
        self.dst_points_init = self.getPointsFromRect(self.dst_landmarks_init, self.dst_convex_rect_init)

    # Get landmarks and convert it to array
    def getLandmarks(self, img, face_rect):
        landmarks = self.shape_predictor(img, face_rect)
        landmarks_points = []
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            landmarks_points.append((x, y))

            # Display landmarks
            if constants._DEBUG_:
                cv2.circle(img, (x, y), 2, (0, 0, 255), 2)

        return landmarks_points

    # Get rectangle made around a numpy array of points
    def getConvexRect(self, points):
        # Find convex hull and the bounding rect of it - This permit some optimization for the subdivision
        convexhull = cv2.convexHull(points)
        convex_rect = cv2.boundingRect(convexhull)
        return convex_rect

    # Get points and change their coordinates depending on the rectangle's coordinates
    def getPointsFromRect(self, points, rect):
        new_points = []
        for i in range(0, len(points)):
            new_points.append((points[i][0] - rect[0], points[i][1] - rect[1]))
        return new_points

    # Get triangles from an array of points thanks to Delauney Subdivision
    def getTriangles(self, landmarks):
        # Convert points into numpy array
        points = np.array(landmarks, np.int32)

        # Get convex rect
        convex_rect = self.getConvexRect(points)

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
                
        return indexes_triangles, convex_rect

    # Returns destination image after facial swapping
    # This function is meant to be called every frame (so in a while loop for exemple)
    def getFrame(self, src_img, src_rect_face, dst_img, dst_rect_face):
        ## DESTINATION
        self.dst_img_gray = cv2.cvtColor(dst_img, cv2.COLOR_BGR2GRAY)
        # Get dst landmarks
        dst_landmarks = self.getLandmarks(self.dst_img_gray, dst_rect_face)

        if constants._DEBUG_:
            # Get ConvexRect from dst
            dst_convex_rect = self.getConvexRect(np.array(dst_landmarks, np.int32))
            # Display rect around face
            cv2.rectangle(self.dst_img_gray, (dst_convex_rect[0], dst_convex_rect[1]),
                    (dst_convex_rect[0] + dst_convex_rect[2], dst_convex_rect[1] + dst_convex_rect[3]), (255, 255, 255), 2)

        ## SOURCE
        self.src_img_gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)

        # Get src landmarks
        src_landmarks = self.getLandmarks(self.src_img_gray, src_rect_face)

        ## Face rotation issues
        # Get coordinates from temple
        src_points_temple = []
        for i in range(0, len(src_landmarks)):
            src_points_temple.append((src_landmarks[i][0] - src_landmarks[0][0], src_landmarks[i][1] - src_landmarks[0][1]))

        # Z-axis
        dst_landmark_temple = (dst_landmarks[16][0] - dst_landmarks[0][0], dst_landmarks[16][1] - dst_landmarks[0][1])
        dst_z_rotation_angle = -math.atan2(dst_landmark_temple[1], dst_landmark_temple[0])
        src_z_rotation_angle = -math.atan2(src_points_temple[16][1], src_points_temple[16][0])
        z_rotation_angle = src_z_rotation_angle - dst_z_rotation_angle
        z_trigo = (math.cos(z_rotation_angle), math.sin(z_rotation_angle))
        for i in range(1, len(src_points_temple)):
            src_points_temple[i] = (int(src_points_temple[i][0] * z_trigo[0] - src_points_temple[i][1] * z_trigo[1]),
                                    int(src_points_temple[i][0] * z_trigo[1] + src_points_temple[i][1] * z_trigo[0]))

        # Get coordinates back in landmarks
        for i in range(0, len(src_landmarks)):
            src_landmarks[i] = (src_points_temple[i][0] + src_landmarks[0][0], src_points_temple[i][1] + src_landmarks[0][1])

        # Get convex rect after fixing rotation issues
        src_convex_rect = self.getConvexRect(np.array(src_landmarks, np.int32))

        # Eyebrows issue
        src_eyebrows_movement = src_landmarks[0][1] - src_convex_rect[1] - self.src_points_init[0][1]
        src_convex_rect = (src_convex_rect[0], src_convex_rect[1] + src_eyebrows_movement,
                            src_convex_rect[2], src_convex_rect[3] - src_eyebrows_movement)

        if constants._DEBUG_:
            # Display rect around face
            cv2.rectangle(self.src_img_gray, (src_convex_rect[0], src_convex_rect[1]),
                    (src_convex_rect[0] + src_convex_rect[2], src_convex_rect[1] + src_convex_rect[3]), (255, 255, 255), 2)

        # Get points from rect
        src_points = self.getPointsFromRect(src_landmarks, src_convex_rect)

        # Calculate ratio
        src_rect_ratio = self.src_convex_rect_init[2] / src_convex_rect[2]

        ## New landmarks calcul
        dst_landmarks_new = []
        dst_points_new = []
        distance_first_landmark = (0, 0)
        arePointsInImage = True

        # Calculate ratio
        dst_rect_ratio = dst_convex_rect[2] / self.dst_convex_rect_init[2]
        for i in range(0, len(src_points)):
            ## Face calibration
            if i < 17:
                dst_landmark_new = dst_landmarks[i]
                if i == 0:
                    distance_first_landmark = (dst_landmarks[i][0] - self.dst_landmarks_init[i][0], dst_landmarks[i][1] - self.dst_landmarks_init[i][1])
            else:
                # Apply ratio on src landmarks
                src_point_normalized = (int(src_points[i][0] * src_rect_ratio), int(src_points[i][1] * src_rect_ratio))

                ## Create new points
                # Make subtraction of init and current src points
                substract_result = (self.src_points_init[i][0] - src_point_normalized[0], self.src_points_init[i][1] - src_point_normalized[1])
                # Apply the substraction on dst
                dst_point_new = (self.dst_points_init[i][0] - substract_result[0], self.dst_points_init[i][1] - substract_result[1])
                # Apply ratio on dst points
                dst_point_new = (int(dst_point_new[0] * dst_rect_ratio), int(dst_point_new[1] * dst_rect_ratio))
                # Get point from the whole image
                dst_landmark_new = (dst_point_new[0] + self.dst_convex_rect_init[0], dst_point_new[1] + self.dst_convex_rect_init[1])
                # Move landmark to dst face
                dst_landmark_new = (dst_landmark_new[0] + distance_first_landmark[0], dst_landmark_new[1] + distance_first_landmark[1])

            # Check if landmark in dst img to provide crashes
            if dst_landmark_new[0] < 0 or dst_landmark_new[1] < 0 or dst_landmark_new[0] >= dst_img.shape[1] or dst_landmark_new[1] >= dst_img.shape[0]:
                arePointsInImage = False
                return dst_img

            # Add point into list
            dst_landmarks_new.append(dst_landmark_new)

        if constants._DEBUG_:
            # Display new landmarks
            cv2.circle(self.dst_img_gray, dst_landmark_new, 2, (0, 0, 0), 2)

        ## Make Triangles
        img_final = dst_img.copy()
        
        if arePointsInImage:
            # Creating new image
            img_new = np.zeros(dst_img.shape, np.uint8)

            for dst_triangle_init in self.dst_triangles_init:
                # Get old triangle
                pt1 = self.dst_landmarks_init[dst_triangle_init[0]]
                pt2 = self.dst_landmarks_init[dst_triangle_init[1]]
                pt3 = self.dst_landmarks_init[dst_triangle_init[2]]
                old_triangle = np.array([pt1, pt2, pt3], np.int32)

                # Get rect of triangle
                (x, y, w, h) = cv2.boundingRect(old_triangle)

                # Crop the image into the rect of the triangle
                dst_img_triangle_rect = self.dst_img_init[y: y + h, x: x + w]

                # Substract rect position to triangle's
                old_triangle = np.array([[pt1[0] - x, pt1[1] - y],
                                        [pt2[0] - x, pt2[1] - y],
                                        [pt3[0] - x, pt3[1] - y]], np.int32)

                # Get new triangle
                pt1 = dst_landmarks_new[dst_triangle_init[0]]
                pt2 = dst_landmarks_new[dst_triangle_init[1]]
                pt3 = dst_landmarks_new[dst_triangle_init[2]]
                new_triangle = np.array([pt1, pt2, pt3], np.int32)

                if constants._DEBUG_:
                    # Display triangle
                    cv2.line(self.dst_img_gray, pt1, pt2, (255, 255, 255), 1)
                    cv2.line(self.dst_img_gray, pt2, pt3, (255, 255, 255), 1)
                    cv2.line(self.dst_img_gray, pt3, pt1, (255, 255, 255), 1)
                
                # Get rect of triangle
                (x, y, w, h) = cv2.boundingRect(new_triangle)

                # Substract rect position to triangle's
                new_triangle = np.array([[pt1[0] - x, pt1[1] - y],
                                        [pt2[0] - x, pt2[1] - y],
                                        [pt3[0] - x, pt3[1] - y]], np.int32)

                # Create a mask to cut the rect into a real triangle
                new_triangle_mask = np.zeros((h, w), np.uint8)
                cv2.fillConvexPoly(new_triangle_mask, new_triangle, 255)

                ## Warping
                # Get transformation matrix
                transformationMatrix = cv2.getAffineTransform(np.float32(old_triangle), np.float32(new_triangle))
                # Apply triangle matrix on triangle
                warped_triangle = cv2.warpAffine(dst_img_triangle_rect, transformationMatrix, (w, h))
                # Apply mask to have only a triangle in the image
                warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=new_triangle_mask)

                img_new_triangle_rect = img_new[y: y + h, x: x + w]
                
                # Remove the lines made by the triangles
                img_new_triangle_rect_gray = cv2.cvtColor(img_new_triangle_rect, cv2.COLOR_BGR2GRAY)
                _, mask_triangle_border = cv2.threshold(img_new_triangle_rect_gray, 1, 255, cv2.THRESH_BINARY_INV)
                warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=mask_triangle_border)

                # Add triangles together in a new image
                img_new_triangle_rect = cv2.add(img_new_triangle_rect, warped_triangle)
                img_new[y: y + h, x: x + w] = img_new_triangle_rect
            
            # Draw new face on old image
            dst_img_new = np.zeros_like(self.dst_img_gray)
            dst_convex_new = cv2.convexHull(np.array(dst_landmarks_new, np.int32))
            dst_face_mask = cv2.bitwise_not(cv2.fillConvexPoly(dst_img_new, dst_convex_new, 255))

            dst_head_noface = cv2.bitwise_and(dst_img, dst_img, mask=dst_face_mask)
            dst_img_new = cv2.add(dst_head_noface, img_new)

            return dst_img_new
        else:
            return dst_img

    # Returns debug src_img and dst_img
    # You must call this function after a getFrame() call
    def getDebugFrame(self):
        if hasattr(self, 'src_img_gray') and hasattr(self, 'dst_img_gray'):
            return self.src_img_gray, self.dst_img_gray
        return [], []