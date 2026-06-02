# Stereo Matching using Dynamic Programming
# Computes a disparity map from a rectified stereo pair using Dynamic Programming

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# Set parameters
dispLevels = 16 #disparity range: 0 to dispLevels-1
lambda_ = 5 #weight of smoothness cost
trunc = 4 #truncation of smoothness cost

# Define matching cost function
computeMatchingCost = lambda left,right: np.absolute(left-right) #absolute differences

# Define smoothness cost function
computeSmoothnessCost = lambda d1,d2: lambda_*np.minimum(np.absolute(d1-d2),trunc)

# Compute minimum cost paths and transitions
def computeMinSumCosts(costs):
    sumCosts = costs[:,:,:,np.newaxis] + smoothnessCosts4d
    minSumCosts = np.amin(sumCosts,axis=2)
    minSumCosts = minSumCosts - np.amin(minSumCosts,axis=2)[:,:,np.newaxis] #normalize costs
    transitions = np.argmin(sumCosts,axis=2)
    return minSumCosts,transitions

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

# Compute smoothness costs
d = np.arange(dispLevels)
smoothnessCosts = computeSmoothnessCost(d,d[np.newaxis,:].T)
smoothnessCosts4d = smoothnessCosts[np.newaxis,np.newaxis,:,:].astype(np.int32)

# Initialize minimum cost paths and transitions for the left to right direction
fromLeft = np.zeros((rows,cols,dispLevels),dtype=np.int32)
transitions = np.zeros((rows,cols,dispLevels),dtype=np.int32)

# Compute minimum cost paths and transitions for left to right direction
for x in range(cols-1):
    costs = (matchingCosts[:,x,:] + fromLeft[:,x,:])[:,np.newaxis,:]
    C,T = computeMinSumCosts(costs)
    fromLeft[:,x+1,:] = C[:,0,:]
    transitions[:,x+1,:] = T[:,0,:]

# Compute the disparity map - Backtracking
dispMap = np.zeros((rows,cols))
ind = np.argmin(fromLeft[:,cols-1,:],axis=1)
for x in range(cols-1,-1,-1):
    dispMap[:,x] = ind
    ind = transitions[np.arange(rows),x,ind] #get the disparity transitions

# Normalize the disparity map for display
scaleFactor = 256/dispLevels
dispImg = (dispMap*scaleFactor).astype(np.uint8)

# Show disparity map
plt.imshow(dispImg,cmap="gray")
plt.show(block=False)
plt.pause(0.01)

# Save disparity map
cv.imwrite("disparity2_DP.png",dispImg)

plt.show()
