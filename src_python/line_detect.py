"""
line_detect_rev7.cpp
>> good straight driving (When all 2 lines are detected) 
""" 
### about ros
import rospy
from geometry_msgs.msg import Twist
import cv2
import numpy as np
import math
import time #for FPS

mode_left = 0
mode_right = 1

## reading images from video
cap_right = cv2.VideoCapture(2) # 0: default camera
cap_left = cv2.VideoCapture(4)

# cap_right = cv2.VideoCapture("/home/miseon/KOSOMO/Real/right_line.avi")
# cap_left = cv2.VideoCapture("/home/miseon/KOSOMO/Real/left_line.avi")

if cap_right.isOpened()==False or cap_left.isOpened()==False:
    print("Can\'t open the Video")
    exit()

fourcc = cv2.VideoWriter_fourcc(*'XVID')

width_right = int(cap_right.get(cv2.CAP_PROP_FRAME_WIDTH))
height_right = int(cap_right.get(cv2.CAP_PROP_FRAME_HEIGHT))
width_left = int(cap_left.get(cv2.CAP_PROP_FRAME_WIDTH))
height_left = int(cap_left.get(cv2.CAP_PROP_FRAME_HEIGHT))

# print("right(w, h)", width_right, height_right) #640
# print("left(w, h)", width_left, height_left)    #480

width = width_left*2
height = height_left

# save as color
writer = cv2.VideoWriter('output.avi', fourcc, 30.0, (width, height_right))
writer2 = cv2.VideoWriter('output_preprocessed.avi', fourcc, 30.0, (width, height_right))

prevTime = 0

#################### PARAMETERS #####################
channel_count = 3

# ROI_vertices = [
#     (0,height/2),
#     (width, height/2),
#     (width, height),
#     (0,height)
# ]
ROI_vertices_left = [
    (0,height/3),
    (int(width/7), height/3),
    (int(width/7), height),
    (0,height)
]
ROI_vertices_right = [
    (width,height/3),
    (int((width*6)/7), height/2),
    (int((width*6)/7), height),
    (width,height)
]

min_y = int(height*(1/3))
max_y = int(height)

coefficient_mid_0 = []
coefficient_mid_1 = []
right_curve_err = []
coefficient_queSize = 20

line_height_min = height/3
line_height_max = height*(2/3)
####################################################

def ROI(img, vertices, channel_count):
    mask = np.zeros_like(img)
    match_mask_color = (255,)*channel_count
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def draw_lines(img, lines, color=[0,255,0], thickness=3):
    if lines is None:
        return
    
    img = np.copy(img)
    line_img = np.zeros(
        (
            img.shape[0],
            img.shape[1],
            3
        ),
        dtype=np.uint8
    )

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)
    
    img = cv2.addWeighted(img, 0.9, line_img, 1.0, 1.0)

    return img

def preprocessing(img) :
    gray_img = grayscale(img)

    kernel_size = 5
    blur_gray_img = cv2.GaussianBlur(gray_img, (kernel_size, kernel_size), 0)
    # ret, blur_gray_img = cv2.threshold(blur_gray_img, 220, 255, cv2.THRESH_BINARY)

    low_threshold = 300
    high_threshold = 350
    edges_img = cv2.Canny(blur_gray_img, low_threshold, high_threshold)

    cropped_img_left = ROI(edges_img, np.array([ROI_vertices_left], np.int32), channel_count)
    cropped_img_right = ROI(edges_img, np.array([ROI_vertices_right], np.int32), channel_count)
    cropped_img = cropped_img_left + cropped_img_right
    return cropped_img


def get_lines(img) :
    lines_raw = cv2.HoughLinesP( img,
        rho=6,
        theta=np.pi/120,
        threshold=160,
        lines=np.array([]),
        minLineLength=40,
        maxLineGap=25
    )    
    return lines_raw

def straightDetect(img):

    preprocessed_img = preprocessing(img)

    raw_line = get_lines(preprocessed_img)

    left_line_x = []
    left_line_y = []
    right_line_x = []
    right_line_y = []

    if raw_line is None:
        err = 0
        return err, img, preprocessed_img
    else:       
        for line in raw_line:
            for x1, y1, x2, y2 in line:
                slope = (y2-y1)/(x2-x1+0.000000001)
                if math.fabs(slope) < 0.5: # only consider extreme slope
                    continue
                if slope <=0:
                    left_line_x.extend([x1, x2])
                    left_line_y.extend([y1, y2])
                else:
                    right_line_x.extend([x1, x2])
                    right_line_y.extend([y1, y2])

        # get polynomial of the left line
        if len(left_line_x) :
            coefficient_left = np.polyfit(
                left_line_y,
                left_line_x,
                deg=1
            )
            coefficient_mid_0.append(coefficient_left[0])
            coefficient_mid_1.append(coefficient_left[1])
            poly_left = np.poly1d(coefficient_left)
            left_x_start = int(poly_left(max_y))
            left_x_end = int(poly_left(min_y))
        else :
            left_x_start = 0

        # get polynomial of the right line   
        if len(right_line_x) :
            coefficient_right = np.polyfit(
                right_line_y,
                right_line_x,
                deg=1
            )
            coefficient_mid_0.append(coefficient_right[0])
            coefficient_mid_1.append(coefficient_right[1])
            poly_right = np.poly1d(coefficient_right)
            right_x_start = int(poly_right(max_y))
            right_x_end = int(poly_right(min_y))
        else :
            right_x_start = 0

        # get polynomial of the mid line
        if(len(left_line_x) and len(right_line_x)):
            if(len(coefficient_mid_0)==coefficient_queSize):
                coefficient_0_avg = sum(coefficient_mid_0,0.0)/len(coefficient_mid_0)
                coefficient_1_avg = sum(coefficient_mid_1,0.0)/len(coefficient_mid_1)
                coefficient_mid_0.pop(0)
                coefficient_mid_0.pop(0)
                coefficient_mid_1.pop(0)
                coefficient_mid_1.pop(0)
            else:
                coefficient_0_avg = sum(coefficient_mid_0,0.0)/len(coefficient_mid_0)
                coefficient_1_avg = sum(coefficient_mid_1,0.0)/len(coefficient_mid_1)
            coefficient_mid = [coefficient_0_avg, coefficient_1_avg]
            poly_mid = np.poly1d(coefficient_mid)
            mid_err = (poly_mid(0)+poly_mid(max_y))/2
            mid_start = int(poly_mid(max_y))
            mid_end = int(poly_mid(min_y))

        # draw Right, Left Lines
        if left_x_start and right_x_start:
            # mid_line_image = draw_lines(img,
            #     [[
            #         [mid_start, max_y, mid_end, min_y]
            #     ]],
            #     color=[255,0,0],
            #     thickness=5
            # )
            # mid_err_list = [int(round(mid_err)), int(height/3), int(round(mid_err)), int(height/6)]
            # result_image = draw_lines(mid_line_image,
            #     [[mid_err_list]], color=[0,255,0], thickness=3
            # )
            # fixed line
            result_image = cv2.line(img, (int(width/2), int(height/3)), (int(width/2), int(height/6)), color=[0,0,255], thickness=3)
            # error line
            # result_image = cv2.line(result_image, (int(width/2), int(height/4)), (int(round(mid_err)), int(height/4)), color=[0,255,0], thickness=3)
            err = width/2 - mid_err

        else:
            err = 0
            return err, img, preprocessed_img

        return err, result_image, preprocessed_img

def curveDetect(img):
    print("This is curve")
    gray_img = grayscale(img)

    kernel_size = 5
    blur_gray_img = cv2.GaussianBlur(gray_img, (kernel_size, kernel_size), 0)

    ret, blur_gray_img = cv2.threshold(blur_gray_img, 220, 255, cv2.THRESH_BINARY)

    low_threshold = 150
    high_threshold = 200
    edges_img = cv2.Canny(blur_gray_img, low_threshold, high_threshold)   
    
    cropped_img = ROI(edges_img, np.array([ROI_vertices_right], np.int32), channel_count)
    
    # get Right, Left Lines
    lines_raw = cv2.HoughLinesP( cropped_img,
        rho=6,
        theta=np.pi/120,
        threshold=160,
        lines=np.array([]),
        minLineLength=40,
        maxLineGap=25
    )

    right_line_x = []
    right_line_y = []
    
    if lines_raw is None :
        err = 0
        return err, img
    else:       
        for line in lines_raw:
            for x1, y1, x2, y2 in line:
                slope = (y2-y1)/(x2-x1)
                if math.fabs(slope) < 0.5: # only consider extreme slope
                    continue
                if slope > 0:
                    right_line_x.extend([x1, x2])
                    right_line_y.extend([y1, y2])

        # get polynomial of the right line   
        if len(right_line_x) :
            coefficient_right = np.polyfit(
                right_line_y,
                right_line_x,
                deg=2
            )
            # poly_right = np.poly1d(coefficient_right)
            poly_right_1 = np.poly1d([coefficient_right[1], coefficient_right[2]]) # 0.5276 x + 1507 (the first straight line)
            right_curve_err.append(0.53-coefficient_right[1])
            # print('\npoly_right_1 : ', poly_right_1)
            right_x_start = int(poly_right_1(max_y))
            right_x_end = int(poly_right_1(min_y))
        else :
            right_x_start = 0

        # get polynomial of the mid line
        if len(right_line_x):
            if(len(right_curve_err)==coefficient_queSize):
                right_curve_err_avg = sum(right_curve_err,0.0)/len(right_curve_err)
                right_curve_err.pop(0)
            else:
                right_curve_err_avg = sum(right_curve_err,0.0)/len(right_curve_err)
            err = right_curve_err_avg

            # fixed line
            result_image = cv2.line(img, (int(width/2), int(line_height_min)), (int(width/2), int(line_height_max)), color=[0,0,255], thickness=3)
            # error line
            result_image = cv2.line(img, (int(width/2), int((line_height_min+line_height_max)/2)), 
                                    (int((width/2)-(err*50)), int((line_height_min+line_height_max)/2)), color=[0,255,0], thickness=3)
            result_image = cv2.line(img, (int((width/2)-(err*50)), int(line_height_min)), 
                                (int((width/2)-(err*50)), int(line_height_max)), color=[0,255,0], thickness=3)

        # # draw Right, Left Lines
        # if right_x_start:
        #     right_line_image = draw_lines(img,
        #         [[
        #             [right_x_start, max_y, right_x_end, min_y]
        #         ]],
        #         color=[255,0,0],
        #         thickness=7
        #     )
        #     # mid_err_list = [round(mid_err), int(line_height_min), round(mid_err), int(line_height_max)]
        #     # result_image = draw_lines(mid_line_image,
        #     #     [[mid_err_list]], color=[0,255,0], thickness=3
        #     # )
        #     ## fixed line
        #     # result_image = cv2.line(result_image, (int(width/2), int(line_height_min)), (int(width/2), int(line_height_max)), color=[0,0,255], thickness=3)
        #     ## error line
        #     # result_image = cv2.line(result_image, (int(width/2), int((line_height_min+line_height_max)/2)), 
        #     #                         (round(mid_err), int((line_height_min+line_height_max)/2)), color=[0,255,0], thickness=3)
        #     # err = width/2 - mid_err

        else:
            err = 0
            return err, img

        return err, result_image


######## ros init #########
twist = Twist()
pub = rospy.Publisher('/control', Twist, queue_size=20)
rospy.init_node('control')

before_err = 0
# main
while(1):
    success_right, frame_right = cap_right.read()
    success_left, frame_left = cap_left.read()

    if success_right==False or success_left==False :
        print("theres no video")
        break
    
    frame_left = cv2.flip(frame_left, -1)
    # frame_right = cv2.flip(frame_right, 1)

    
    new_frame = np.concatenate((frame_left, frame_right), axis=1)
    err, new_frame, preprocessed_img = straightDetect(new_frame)
    if abs(err - before_err) > 30 :
        err = before_err
#    mid_err_list = [int((width/2)+err, int(height/3), int((width/2)+err, int(height/6)]
    # new_frame = cv2.line(new_frame,
    #     ((int((width/2)+err), int(height/3)), (int((width/2)+err, int(height/6))), color=[0,255,0], thickness=3
    # )
    new_frame = cv2.line(new_frame, (int((width/2)+err), int(height/3)), (int((width/2)+err), int(height/6)), color=[0,255,0], thickness=3)
    new_frame = cv2.line(new_frame, (int(width/2), int(height/4)), (int((width/2)+err), int(height/4)), color=[0,255,0], thickness=3)
    before_err = err


    # if err_right>100 or err_right==0:
    #     err_right, new_frame_right = curveDetect(frame_right)
    #     err_left = 0
    #     new_frame_left = frame_left

    ##### about ros
    if(not rospy.is_shutdown()):
        twist.angular.z = err/1000
        pub.publish(twist)

    # for FPS
    # curTime = time.time()
    # sec = curTime - prevTime
    # prevTime = curTime
    # fps = 1/sec
    fps = cap_left.get(cv2.CAP_PROP_FPS)

    str = "FPS : {0:0.1f}, ERROR : {1:0.2f}".format(fps, err)
    cv2.putText(new_frame, str, (0,100), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255))

    cv2.imshow('OUTPUT', new_frame)
    cv2.imshow('preprocessed', preprocessed_img)
    writer.write(new_frame)
    writer2.write(preprocessed_img)

    # exit when press the ESC
    if cv2.waitKey(1)&0xFF == 27:
        break

cap_left.release()
cap_right.release()
writer.release()
writer2.release()
cv2.destroyAllWindows()