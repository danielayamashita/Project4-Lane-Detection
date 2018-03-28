#Project 4 - Advanced Lane Detection
#27th March 2018
#Author: Daniela Yassuda Yamashita

#======================================================================
#-----------------------IMPORT LIBRARIES-------------------------------
#======================================================================
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

#======================================================================
#---------------------------USER VARIABLES-----------------------------
#======================================================================
ENABLE_CAMERA_CALIBRATION = True #enable the camera calibration process
TEST_ON_VIDEO = False
#======================================================================
#----------------------FUNCTIONS DEFINITION----------------------------
#======================================================================
def CameraObjectsAndPoints(images,nx,xy):
	objpoints = [] #3D points in real world space
	imgpoints = [] #2D points in image plane
	
	# Prepare object points
	objp = np.zeros((nx*ny,3),np.float32)
	objp[:,:,2] = np.mgrid[0:nx,0:ny].T.reshape[-1,2] #x, y coordinates
	
	for fname in images:
		#Read each image
		img =mpimg.imread(fname)
		
		#Convert image to grayscale
		gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		
		#Find the chessboard corners
		ret, corners = cv2.findChessboardCorners(gray, (nx,ny),None)
		
		if ret == True:
			imgpoints.append(corners)
			objpoints.append(objp)
			#draw and display the corners
			img = cv2.drawChessboardCorners(img,(nx,ny),corners,ret)
			plt.imshow(img)
			
	return objpoints,imgpoints

	
def cal_undistort(img, objpoints, imgpoints):
    # Use cv2.calibrateCamera() and cv2.undistort()
    #undist = np.copy(img)  # Delete this line
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist
	
def CreateBinaryImage(img,thresh_min,thresh_max):
	
	# Convert to HLS color space and separate the S channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
	
	# Grayscale image
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	
	# Sobel x
	sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0) # Take the derivative in x
	abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
	scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
	
	# Threshold x gradient
	sxbinary = np.zeros_like(scaled_sobel)
	sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

	# Stack each channel to view their individual contributions in green and blue respectively
	# This returns a stack of the two binary images, whose components you can see as different colors
	color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255

	# Combine the two binary thresholds
	combined_binary = np.zeros_like(sxbinary)
	combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
	
	return color_binary
#======================================================================
#-----------------------CAMERA CALIBRATION-----------------------------
#======================================================================

if ENABLE_CAMERA_CALIBRATION == True:
	#Read in a calibration image
	original_images = glob.glob('./CarND-Advanced-Lane-Lines-master/camera_cal/calibration*.jpg')

	#Number of row and coluns in the chess board
	nx = 8
	ny = 6

	#Shape of the image
	img_sample = mpimg.imread(original_images[1])
	shapeImg = img_sample.shape

	#Calibrate the camera
	objpoints,imgpoints = CameraObjectsAndPoints(original_images,nx,ny)


	#undistorT a test image
	undistorted = cal_undistort(img_sample,objpoints,imgpoints)
	plt.imshow(undistorted)

	#Show the image
	f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
	f.tight_layout()
	ax1.imshow(img_sample)
	ax1.set_title('Original Image', fontsize=50)
	ax2.imshow(undistorted)
	ax2.set_title('Undistorted Image', fontsize=50)
	plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
else: 
	print('ENABLE_CAMERA_CALIBRATION is disabled')

#======================================================================
#-----------------------CREATING A BINARY IMAGE------------------------
#======================================================================

if TEST_ON_VIDEO  == True:
    #Read the frames of the video
    VideoPath = './CarND-Advanced-Lane-Lines-master/project_video.mp4'
    video = cv2.VideoCapture(VideoPath)
    ret, frame =video.read()
else:
    #Read the image tests
    testPath = './/CarND-Advanced-Lane-Lines-master/test_images/straight_lines1.jpg'
    img = mpimg.imread(testPath)
    
	
	
	#Undistort the image
    undist = cal_undistort(img_sample,objpoints,imgpoints)
	
	#Define threshold max and min for binary image construction
	thresh_min = 20
	thresh_max = 100
	
	#Creating binary image
	color_binary = CreateBinaryImage(img,thresh_min,thresh_max)
	
	# Plotting thresholded images
	f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
	ax1.set_title('Stacked thresholds')
	ax1.imshow(color_binary)

	ax2.set_title('Combined S channel and gradient thresholds')
	ax2.imshow(combined_binary, cmap='gray')
#======================================================================
#-----------------------PERSPECTIVE TRANSFORMING-----------------------
#======================================================================
