# Stereo Matching using Semi-Global Matching - occlusion penalties approach
# Computes a disparity map from a rectified stereo pair using Semi-Global Matching

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

MAX_INT = 2147483647

# Set parameters
dispLevels = 16 #disparity range: 0 to dispLevels-1
p1 = 10 #occlusion penalty 1
p2 = 20 #occlusion penalty 2

# Define matching cost function
computeMatchingCost = lambda left,right: np.absolute(left-right) #absolute differences

# Compute minimum cost paths
def computeDirectionalCosts(input_):
    minInput = np.amin(input_,axis=2)
    possibleOutput = np.zeros((input_.shape[0],input_.shape[1],input_.shape[2],4),dtype=np.int32)
    possibleOutput[:,:,:,0] = input_
    possibleOutput[:,:,:,1] = np.roll(input_,1,2) + p1; possibleOutput[:,:,0,1] = MAX_INT
    possibleOutput[:,:,:,2] = np.roll(input_,-1,2) + p1; possibleOutput[:,:,-1,2] = MAX_INT
    possibleOutput[:,:,:,3] = (minInput + p2)[:,:,np.newaxis]
    output = np.amin(possibleOutput,axis=3)
    output = output - minInput[:,:,np.newaxis] #normalize costs
    return output

# Load left and right images in grayscale
leftImg = cv.imread("left.png",cv.IMREAD_GRAYSCALE)
rightImg = cv.imread("right.png",cv.IMREAD_GRAYSCALE)

# Apply a Gaussian filter
leftImg = cv.GaussianBlur(leftImg,(5,5),0.6)
rightImg = cv.GaussianBlur(rightImg,(5,5),0.6)

# Get the size
(rows,cols) = leftImg.shape

# Convert to int32
leftImg = leftImg.astype(np.int32)
rightImg = rightImg.astype(np.int32)

# Compute pixel-based matching costs
matchingCosts = np.zeros((rows,cols,dispLevels),dtype=np.int32)
for d in range(dispLevels):
    rightImgShifted = np.roll(rightImg,d,1)
    matchingCosts[:,:,d] = computeMatchingCost(leftImg,rightImgShifted)

# Initialize minimum cost paths for the 4 directions
fromLeft = np.zeros((rows,cols,dispLevels),dtype=np.int32)
fromRight = np.zeros((rows,cols,dispLevels),dtype=np.int32)
fromUp = np.zeros((rows,cols,dispLevels),dtype=np.int32)
fromDown = np.zeros((rows,cols,dispLevels),dtype=np.int32)

# Compute minimum cost paths for left to right direction
for x in range(cols-1):
    costs = (matchingCosts[:,x,:] + fromLeft[:,x,:])[:,np.newaxis,:]
    fromLeft[:,x+1,:] = computeDirectionalCosts(costs)[:,0,:]

# Compute minimum cost paths for right to left direction
for x in range(cols-1,0,-1):
    costs = (matchingCosts[:,x,:] + fromRight[:,x,:])[:,np.newaxis,:]
    fromRight[:,x-1,:] = computeDirectionalCosts(costs)[:,0,:]

# Compute minimum cost paths for up to down direction
for y in range(rows-1):
    costs = (matchingCosts[y,:,:] + fromUp[y,:,:])[np.newaxis,:,:]
    fromUp[y+1,:,:] = computeDirectionalCosts(costs)[0,:,:]

# Compute minimum cost paths for down to up direction
for y in range(rows-1,0,-1):
    costs = (matchingCosts[y,:,:] + fromDown[y,:,:])[np.newaxis,:,:]
    fromDown[y-1,:,:] = computeDirectionalCosts(costs)[0,:,:]

# Compute total costs
totalCosts = fromLeft + fromRight + fromUp + fromDown

# Compute the disparity map
dispMap = np.argmin(totalCosts,axis=2)

# Normalize the disparity map for display
scaleFactor = 256/dispLevels
dispImg = (dispMap*scaleFactor).astype(np.uint8)

# Show disparity map
plt.imshow(dispImg,cmap="gray")
plt.show(block=False)
plt.pause(0.01)

# Save disparity map
cv.imwrite("disparity3b_SGM.png",dispImg)

plt.show()
