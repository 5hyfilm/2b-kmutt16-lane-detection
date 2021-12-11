import cv2
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
from tkinter import *

try:
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt

    def make_coordinates(image, line_parameters):
        slope, intercept = line_parameters
        #print(image.shape)
        #set starting point of y from the buttom (width)
        y1 = image.shape[0]
        #set endding point of y as 3/5*width of the image
        y2 = int(y1*(3/5))
        #calculate values of x (x=(y-b)/m)
        x1 = int((y1 - intercept)/slope)
        x2 = int((y2 - intercept)/slope)
        return np.array([x1, y1, x2, y2]) #(x1,y1) is the starting points and (x2,y2) is the endding points

    def average_slope_intercept(image, lines):
        left_fit = [] #contain coordinates of the left line
        right_fit = [] #contain coordinates of the right line
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            parameters = np.polyfit((x1, x2), (y1, y2), 1) #fit polynomial degree 1 = fit points to y = mx+b
            #print(parameters)
            #find m and b of each line
            slope = parameters[0] # m
            intercept = parameters[1] # b
            if slope < 0:
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))
            #print(left_fit)
            #print(right_fit)

        left_fit_average = np.average(left_fit, axis = 0) #average slope(m) and intercept(b) to get only one value for left line
        right_fit_average = np.average(right_fit, axis = 0) #average slope(m) and intercept(b) to get only one value for right line
        #print(left_fit_average, 'left') #left_fit_average contains of m and b for the left line
        #print(right_fit_average, 'right') #right_fit_average contains of m and b for the right line

        #calculate intersection points between two lines
        m1, b1 = left_fit_average
        m2, b2 = right_fit_average
        xVal = int(((b2 - b1) / (m1 - m2)))
        yVal = int((m1 * xVal) + b1)
        print (xVal,'x intersection')
        print (yVal,'y intersection')

        #find (x1,y1) is the starting points and (x2,y2) is the endding points =>left_line
        left_line = make_coordinates(image, left_fit_average) #find coordinates of the left line
        #find (x1,y1) is the starting points and (x2,y2) is the endding points =>right_line
        right_line = make_coordinates(image, right_fit_average) #find coordinates of the right line
        point = Point(xVal, yVal)
        coords = [(570, 445), (570, 295), (660,295), (660, 445)]
        area = Polygon(coords)
        print('INTERSECTION IS ', point)
        print(area)
        point.within(area)
        return np.array([left_line, right_line])

    def canny(image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        canny = cv2.Canny(blur, 50, 150)
        return canny

    #display lines on image
    def diaplay_lines(image, lines):
        line_image = np.zeros_like(image)
        if lines is not None:
            for x1, y1, x2, y2 in lines:
                #x1,y1 is the starting point of each line
                #x2,y2 is the ending point of each line
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
            #for line in lines:
                #x1, y1, x2, y2 = line.reshape(4)
                #cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
        return line_image

    def region_of_interest(image):
        height = image.shape[0]
        #polygons = np.array([
        #[(200, height), (1100, height), (550, 250)]
        #])
        polygons = np.array([
        [(200, 650), (1100, 650), (615, 400)]
        ])
        mask = np.zeros_like(image)
        cv2.fillPoly(mask, polygons, 255)
        masked_image = cv2.bitwise_and(image, mask)
        return masked_image

    #image = cv2.imread('test_image.jpg')
    image = cv2.imread('road_images/straight_2.jpg')
    lane_image = np.copy(image)
    canny_image = canny(lane_image)
    cropped_image = region_of_interest(canny_image)

    #return values is starting(x1,y1) and ending(x2,y2) points of each line
    #return many lines
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength = 40, maxLineGap = 5)

    averaged_lines = average_slope_intercept(lane_image, lines)
    line_image = diaplay_lines(lane_image, averaged_lines)
    #line_image = diaplay_lines(lane_image, lines)
    combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
    #point_check = check_point(xVal, yVal)
    cv2.imshow("result", combo_image)
    print('STRAIGHT ROAD')
    #plt.imshow(cropped_image)
    cv2.waitKey(0)
    #plt.show()

except TypeError:
    print('CURVED ROAD')
