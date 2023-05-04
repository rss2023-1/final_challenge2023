import cv2
import numpy as np
import pdb

#################### X-Y CONVENTIONS #########################
# 0,0  X  > > > > >
#
#  Y
#
#  v  This is the image. Y increases downwards, X increases rightwards
#  v  Please return bounding boxes as ((xmin, ymin), (xmax, ymax))
#  v
#  v
#  v
###############################################################

def image_print(img):
	"""
	Helper function to print out images, for debugging. Pass them in as a list.
	Press any key to continue.
	"""
	cv2.imshow("image", img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def merge_lines(lines, slope_filter):
	"""
	Filters lines using slope_filter condition returns the slope 
	and intercept that is the average of those lines.

	NOTE: slopes are flipped because y is flipped. So positive slope -> right side, negative -> left.
	Returns None if slope range is empty.
	"""
	sum_slope = 0
	sum_intercept = 0
	num_lines = 0
	for line in lines:
		for x1,y1,x2,y2 in line:
			line_slope = (y2 - y1) / (x2 - x1)
			if (slope_filter(line_slope)):
				num_lines += 1
				intercept = y1/line_slope - x1
				sum_slope += line_slope
				sum_intercept += intercept
	print(num_lines)
	
	if (num_lines > 0):
		return ((sum_slope/num_lines), (sum_intercept/num_lines))
	print("No lines in slope range")
	return None

def line_in_image(slope, intercept, xmax, ymax):
    ymin = 0
    xmin = 0
    # calculate x-coordinate of entry point
    entry_x = xmin
    entry_y = slope * entry_x + intercept

    # check if entry point is out of bounds
    if entry_y < ymin:
        entry_y = ymin
        entry_x = (entry_y - intercept) / slope
    elif entry_y > ymax:
        entry_y = ymax
        entry_x = (entry_y - intercept) / slope

    # calculate x-coordinate of exit point
    exit_x_top = (ymax - intercept) / slope
    exit_x_bottom = (ymin - intercept) / slope

    if ymin <= exit_x_top <= ymax:
        exit_x = exit_x_top
        exit_y = ymax
    elif ymin <= exit_x_bottom <= ymax:
        exit_x = exit_x_bottom
        exit_y = ymin
    else:
        exit_y = slope * xmax + intercept
        if exit_y < ymin:
            exit_y = ymin
            exit_x = (exit_y - intercept) / slope
        elif exit_y > ymax:
            exit_y = ymax
            exit_x = (exit_y - intercept) / slope
        else:
            exit_x = xmax

    # check if exit point is out of bounds in x-direction
    if exit_x < xmin:
        exit_x = xmin
        exit_y = slope * exit_x + intercept
    elif exit_x > xmax:
        exit_x = xmax
        exit_y = slope * exit_x + intercept

    return (int(entry_x), int(entry_y), int(exit_x), int(exit_y))

	#  # calculate x-coordinate of entry point
    # entry_x = 0
    # entry_y = slope * entry_x + intercept

    # # check if entry point is out of bounds
    # if entry_y < 0:
    #     entry_y = 0
    #     entry_x = (entry_y - intercept) / slope
    # # calculate x-coordinate of exit point
    # exit_x = x_max
    # exit_y = slope * exit_x + intercept
    # # check if exit point is out of bounds
    # if exit_y > y_max:
    #     exit_y = y_max
    #     exit_x = (exit_y - intercept) / slope
    # return (int(entry_x), int(entry_y), int(exit_x), int(exit_y))
def select_longest(lines, slope_filter):
	"""
	Finds longest line that matches slope_filter condition.
	Returns 'None' if no line found.
	"""
	max_line = [None]
	max = 0
	for line in lines:
		for x1,y1,x2,y2 in line:
			line_slope = (y2 - y1) / (x2 - x1)
			if (slope_filter(line_slope)):
				length = ((x2-x1)**2 + (y2-y1)**2)**(1/2)
				if (length > max):
					max = length
					max_line = line[0]
	return max_line

def cd_color_segmentation(img, template):
	"""
	Implement the cone detection using color segmentation algorithm
	Input:
		img: np.3darray; the input image with a cone to be detected. BGR.
		template_file_path; Not required, but can optionally be used to automate setting hue filter values.
	Return:
		bbox: ((x1, y1), (x2, y2)); the bounding box of the cone, unit in px
				(x1, y1) is the top left of the bbox and (x2, y2) is the bottom right of the bbox
	"""
	LANE_SLOPE_MIN = 0.3
	LANE_SLOPE_MAX = 1
	LANE_Y_THRESHOLD = 220
	########## YOUR CODE STARTS HERE ##########
	hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)
	#image_print(hsv_img)
	# orange ranges in hues from 30-45
	# settings for good performance on testing: (0, 220, 90), (45, 255, 255)
	mask = cv2.inRange(img, (170, 170, 170), (255, 255, 255))
	#image_print(mask)
	kernel = np.ones((3,3), np.uint8)
	eroded = cv2.erode(mask, kernel)
	dilated = cv2.dilate(eroded, kernel)
	#image_print(eroded)
	#image_print(dilated)
	# gray
	#gray = cv2.cvtColor(mask ,cv2.COLOR_BGR2GRAY)
	# blur
	kernel_size = 5
	blur_gray = cv2.GaussianBlur(mask,(kernel_size, kernel_size),0)
	# canny edge detection
	low_threshold = 50
	high_threshold = 150
	edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
	# hough for lines
	rho = 1  # distance resolution in pixels of the Hough grid
	theta = np.pi / 180  # angular resolution in radians of the Hough grid
	threshold = 70  # minimum number of votes (intersections in Hough grid cell)
	min_line_length = 70 # minimum number of pixels making up a line
	max_line_gap = 20  # maximum gap in pixels between connectable line segments
	line_image = np.copy(img) * 0  # creating a blank to draw lines on

	# Run Hough on edge detected image
	# Output "lines" is an array containing endpoints of detected line segments
	lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
						min_line_length, max_line_gap)
	# print(lines.shape)
	# slopes = (lines[:,:, 3] - lines[:,:, 2] / lines[:,:, 1] - lines[:,:, 0])
	# THRESHOLD = 0.1
	# dup_ids = set()
	# for i in range(len(slopes)):
	# 	for j in range(i+1, len(slopes)):
	# 		if abs(slopes[i] - slopes[j]) < THRESHOLD:
	# 			dup_ids.add(j)
	# print(dup_ids)
	# singleton_ids = [i for i in range(len(slopes)) if i not in dup_ids]
	# lines = lines[singleton_ids]
	
	# filter out lines that are not in correct slope range (too flat)
	# 	or that are too high in the image (y too low)
	filtered_lines = []
	if lines is not None:
		for line in lines:
			for x1,y1,x2,y2 in line:
				line_slope = abs((y2 - y1) / (x2 - x1))
				if (line_slope > LANE_SLOPE_MIN and max(y2, y1) > LANE_Y_THRESHOLD):
					#print(line)
					filtered_lines.append(line)
					cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)
	if (len(filtered_lines) == 0):
		print('no good lines found')
		return
	filtered_lines = np.array(filtered_lines)
	# right_line = merge_lines(lines, lambda slope: slope > LANE_SLOPE_MIN)
	# left_line = merge_lines(filtered_lines, lambda slope: slope < -1 * LANE_SLOPE_MIN)
	# print(left_line)
	# image_height = img.shape[0]
	# image_width = img.shape[1]
	# #RIGHT LINE
	# right_line_frame = line_in_image(right_line[0], right_line[1], image_width, image_height)
	# left_line_frame = line_in_image(left_line[0], left_line[1], image_width, image_height)

	# cv2.line(line_image,(right_line_frame[0],right_line_frame[1]),(right_line_frame[2],right_line_frame[3]),(0,255,0),5)
	# cv2.line(line_image,(left_line_frame[0],left_line_frame[1]),(left_line_frame[2],left_line_frame[3]),(0,255,0),5)
	# #cv2.line(line_image, (0, 0), (10, 1000), 5)

	longest_right = select_longest(filtered_lines, lambda slope: slope > LANE_SLOPE_MIN)
	longest_left = select_longest(filtered_lines, lambda slope: slope < -1 * LANE_SLOPE_MIN)

	# Draw the lines on the  image
	if(longest_right[0] == None):
		print("no longest right found")
	else:
		cv2.line(line_image,(longest_right[0],longest_right[1]),(longest_right[2],longest_right[3]),(0,0,255),5)
	if(longest_left[0] == None):
		print("no longest left found")
	else:
		cv2.line(line_image,(longest_left[0],longest_left[1]),(longest_left[2],longest_left[3]),(0,0,255),5)
	lines_edges = cv2.addWeighted(img, 0.8, line_image, 1, 0)

	cv2.imshow('test',lines_edges)
	# waits for user to press any key
	# (this is necessary to avoid Python kernel form crashing)
	cv2.waitKey(0) 
	# closing all open windows
	cv2.destroyAllWindows()

def target_pixel(lookahead,n,x11,x21,slope1,slope2):
	"""
	Finds the target midpoint pixel between lines 1 (left) and 2 (right) at a given lookahead distance
	Input:
		lookahead: int; the desired lookahead distance in pixels, measured from the bottom of the frame.
		n: int; frame height in pixels
		x11: int; 1st pixel index of 1st line, assumes pixel at bottom left.
		x21: int; 1st pixel index of 2nd line, assumes pixel at bottom right.
		slope1: float; slope of line 1.
		slope2: float; slope of line 2.
	Return:
		target_pixel: np.array([x, y]); the target pixel.
	"""
	midy = n - lookahead
	midx1 = x11 + lookahead//slope1
	midx2 = x21 + lookahead//slope2
	midx = (midx1+midx2)//2
	return np.array([midx, midy])	


### status checks/safety controller

slope_nomimal = np.tan(24.5*np.pi/180)
arg_line1 = np.arctan(slope1)
arg_line2 = np.arctan(slope2)
arg_sum = (arg_line1 + arg_line2)
kp = 1
if abs(arg_sum) > (45*np.pi/180)
	drive.steering_angle = kp*arg_sum # assuming positive steering is right, check this

if __name__ == "__main__":
	for i in range(1, 60):
		filename = "src/final_challenge2023/testtrackimages/track" + str(i) + ".png"
		img = cv2.imread(filename)
		cd_color_segmentation(img, "template holder")

		
