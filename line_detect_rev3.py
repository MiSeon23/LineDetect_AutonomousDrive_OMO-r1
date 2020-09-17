"""
line_detect_rev3.cpp
>> In order to reduce oscillation, the average value was taken.
>> change the value of 'coefficient_queSize' parameter.
>> removed the width, height, channel_count from pipeline function.
>> the width, height is gotten from VideoCapture(Line 24,25)
>> Originally, the channel_count shoud be set automatically,
   but it is set as global variable to simplify the code.
>> Removed the right and left line from the output video.
""" 

import cv2
import numpy as np
import math
import time #for FPS

## reading images from video
# cap = cv2.VideoCapture(0) # 0: default camera
cap = cv2.VideoCapture("/home/miseon/KOSOMO/test1.mp4")
if cap.isOpened() == False:
    print("Can\'t open the Video")
    exit()

fourcc = cv2.VideoWriter_fourcc(*'XVID')

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# save as color
writer = cv2.VideoWriter('output.avi', fourcc, 30.0, (width, height))

prevTime = 0

#################### PARAMETERS #####################
channel_count = 3

ROI_vertices = [
    (0, 0),
    (width, 0),
    (width, height*(1/2)),
    (0, height*(1/2))
]

min_y = 0
max_y = int(height*(1/2))

coefficient_mid_0 = []
coefficient_mid_1 = []
coefficient_queSize = 20
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

def pipeline(img):
    gray_img = grayscale(img)

    kernel_size = 5
    blur_gray_img = cv2.GaussianBlur(gray_img, (kernel_size, kernel_size), 0)

    low_threshold = 150
    high_threshold = 200
    edges_img = cv2.Canny(blur_gray_img, low_threshold, high_threshold)

    cropped_img = ROI(edges_img, np.array([ROI_vertices], np.int32), channel_count)

    # get Right, Left Lines
    lines_raw = cv2.HoughLinesP( cropped_img,
        rho=6,
        theta=np.pi/120,
        threshold=160,
        lines=np.array([]),
        minLineLength=40,
        maxLineGap=25
    )

    left_line_x = []
    left_line_y = []
    right_line_x = []
    right_line_y = []

    if lines_raw is None :
        return img
    else:       
        for line in lines_raw:
            for x1, y1, x2, y2 in line:
                slope = (y2-y1)/(x2-x1)
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
            mid_start = int(poly_mid(max_y))
            mid_end = int(poly_mid(min_y))

        # draw Right, Left Lines
        if left_x_start and right_x_start:
            mid_line_image = draw_lines(img,
                [[
                    [mid_start, max_y, mid_end, min_y]
                ]],
                color=[255,255,0],
                thickness=5
            )
        else:
            return img

        return mid_line_image

# main
while(1):
    success, frame = cap.read()

    if success == False:
        print("theres no video")
        break
    
    frame = cv2.flip(frame, 0)
    new_frame = pipeline(frame)

    # for FPS
    # curTime = time.time()
    # sec = curTime - prevTime
    # prevTime = curTime
    # fps = 1/sec
    fps = cap.get(cv2.CAP_PROP_FPS)

    str = "FPS : %0.1f" % fps
    cv2.putText(new_frame, str, (0,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0))

    cv2.imshow('Camera Window', new_frame)
    writer.write(new_frame)

    # exit when press the ESC
    if cv2.waitKey(1)&0xFF == 27:
        break

cap.release()
writer.release()
cv2.destroyAllWindows()