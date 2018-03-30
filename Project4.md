
# **Advanced Lane Finding Project**

**Author: Daniela Yassuda Yamashita**

**29th March 2018**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./imagens/original_image.png "Original Image"
[image2]: ./imagens/original_undistort.png "Original and Undistort"
[image3]: ./imagens/corners_detection.png "Corners Detection"
[image4]: ./imagens/Binary_image_chS_chL_grad.png "Binary_image_chS_chL_grad"
[image5]: ./imagens/perspective_transform.png "performance_shadow"
[image6]: ./imagens/warp_unwarp.png "botton_unwarp"
[image7]: ./imagens/botton_histogram.png "first_histogram"
[image8]: ./imagens/up_histogram.png "up_histogram"

[image9]: ./imagens/sliding_window.png "sliding_window"

[image10]: ./imagens/road_detection.png "road_detection"

[image11]: ./imagens/final_road_detection.png "final_road_detection"

[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the fourth code cell of the IPython notebook located in "./Project4.ipynb".  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 
![alt text][image3]
![alt text][image2]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines # through # in the cell 6 of title 'Step 4: Creating binary images').  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image4]

I build the binary image by using both the channel S and channel L of the HSL color maping image. In the video class, it is recommended to use only the channel S, but I used also the channel because of the bad performance of the algorithm in the shadow. 
 
#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warp()`, which appears in the third cell of the file `Project4.ipynb`.  The `warp()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
offset = 400
#Source coordinates
src1 = [200,combined_binary.shape[0]]
src2 = [550,480]
src3 = [735,480]
src4 = [1120,combined_binary.shape[0]]
src = np.float32([src1,src2,src3,src4])

#Destination coordinates
offset = 400
dst1 =[combined_binary.shape[1]/2-offset,combined_binary.shape[0]]
dst2 = [combined_binary.shape[1]/2-offset,100]
dst3 = [combined_binary.shape[1]/2+offset,100]
dst4 = [combined_binary.shape[1]/2+offset,combined_binary.shape[0]]
dst = np.float32([dst1,dst2,dst3,dst4])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 200, 720      | 240, 720       | 
| 550, 480      | 240, 100      |
| 735, 480     | 640, 100      |
| 1120, 720      | 640, 720        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image5]
![alt text][image6]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

This step is located mostly in the function findLaneLines() in the seventh cell in the IPython file (Project4.ipnb).

From the binary image, I detected where is more likely to the lane start from the botton. To do that, I calculate the histogram of the half botton image and took the indice of the two maximuns(left side an right side):

#### Botton image
![alt text][image7]

However, in this specific image, we can notice that the left max is in the wrong position, because it is almost in the center of the image. To fix this problem, I evaluate the position of the picks and if they are close to the center, I use the half up of the image to take a better approximation of the start of lane line.

#### Up image

![alt text][image8]

Once detected where the lane lane start, I slide 9 windows toward the image, by recentering them in the case of abrupt change in the center of the non-zero points.

![alt text][image9]

Doing that, I accuratelly selected the left (red) and right(blue) lane line points. Afterwards, I used a 2nd order polynominal interpolation (function np.polyfit())in order to draw the best fit lane line for this road. Next, by using the function cv2.fillPoly, I fill the polygon that represents the road - see figure below.


![alt text][image10]

Finally, I unwarped the road mask and combined it with the original image. As a result, I got the following image:

![alt text][image11]


#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

This step is located mostly in the function _calculateRadius()_ and _calculatePositions()_in the tenth cell in the IPython file (Project4.ipnb).
 
I calculate the radius and the position by using the transformation between pixels and meters in the real world.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in the third cell of title _Step 2: Functions definition_ in my code in `Project4.ipnb` in the function `unwarp()`.  Here is an example of my result on a test image:

![alt text][image11]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./result4.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.

##### 1 - Shadow of trees

As mentioned above, the shadow of tress was a problem in the construction of the binary image. Indeed, only the channel S was not enough to identify the lane lines clearly. Therefore, I used a combination of gradient and color using S-channel and L-channel in order to overcome this problem. Since the shadows are normally darker than in the normal envoronment, I added a L-channel threshold to ignore the shadows.

##### 2 - Close maximuns of the histogram

As dicussed above, I observed that in somme videos's frames the lane detection was not perfect due to the flaw in the detection of the start point of the lane lines. As a result, the calculus of the position of the first windown (of the sliding windown algorithm) was not very accurated. In order to face this problem, I verify the position of the first windown and evaluated it. If the position was too close to the center, I used the upper half image to estimate the initial position of the lane line windown. 

