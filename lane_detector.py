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
	LANE_SLOPE_MIN = 0.336
	LANE_SLOPE_MAX = 1
	########## YOUR CODE STARTS HERE ##########
	hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)
	#image_print(hsv_img)
	# orange ranges in hues from 30-45
	# settings for good performance on testing: (0, 220, 90), (45, 255, 255)
	mask = cv2.inRange(img, (150, 150, 150), (255, 255, 255))
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
	print(lines.shape)
	slopes = (lines[:,:, 3] - lines[:,:, 2] / lines[:,:, 1] - lines[:,:, 0])
	THRESHOLD = 0.1
	dup_ids = set()
	for i in range(len(slopes)):
		for j in range(i+1, len(slopes)):
			if abs(slopes[i] - slopes[j]) < THRESHOLD:
				dup_ids.add(j)
	print(dup_ids)
	singleton_ids = [i for i in range(len(slopes)) if i not in dup_ids]
	lines = lines[singleton_ids]

	if lines is not None:
		for line in lines:
			for x1,y1,x2,y2 in line:
				line_slope = abs((y2 - y1) / (x2 - x1))
				# and line_slope < LANE_SLOPE_MAX
				if (line_slope > LANE_SLOPE_MIN):
					cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)


	cv2.line(line_image, (0, 0), (10, 100), 5)

	# Draw the lines on the  image
	lines_edges = cv2.addWeighted(img, 0.8, line_image, 1, 0)

	cv2.imshow('test',lines_edges)
	# waits for user to press any key
	# (this is necessary to avoid Python kernel form crashing)
	cv2.waitKey(0) 
	# closing all open windows
	cv2.destroyAllWindows()
	

if __name__ == "__main__":
	for i in range(1, 60):
		filename = "./testtrackimages/track" + str(i) + ".png"
		img = cv2.imread(filename)
		cd_color_segmentation(img, "template holder")
