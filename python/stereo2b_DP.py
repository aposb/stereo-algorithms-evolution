# Stereo Matching using Dynamic Programming - occlusion penalties approach
# Computes a disparity map from a rectified stereo pair using Dynamic Programming

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

# Compute minimum cost paths and transitions
def computeMinSumCosts(costs):
    minCosts = np.amin(costs,axis=2)
    minCostsTransitions = np.argmin(costs,axis=2)
    sumCosts = np.zeros((costs.shape[0],costs.shape[1],costs.shape[2],4),dtype=np.int32)
    sumCosts[:,:,:,0] = costs
    sumCosts[:,:,:,1] = np.roll(costs,1,2) + p1; sumCosts[:,:,0,1] = MAX_INT
    sumCosts[:,:,:,2] = np.roll(costs,-1,2) + p1; sumCosts[:,:,-1,2] = MAX_INT
    sumCosts[:,:,:,3] = (minCosts + p2)[:,:,np.newaxis]
    minSumCosts = np.amin(sumCosts,axis=3)
    ind = np.argmin(sumCosts,axis=3)
    minSumCosts = minSumCosts - minCosts[:,:,np.newaxis] #normalize costs
    match = np.arange(costs.shape[2])[np.newaxis,np.newaxis,:] + np.zeros(costs.shape,dtype=np.int32)
    minCostsTransitions3d = minCostsTransitions[:,:,np.newaxis] + np.zeros(costs.shape,dtype=np.int32)
    transitions = np.zeros(costs.shape,dtype=np.int32)
    transitions[ind==0] = match[ind==0];
    transitions[ind==1] = match[ind==1]-1;
    transitions[ind==2] = match[ind==2]+1;
    transitions[ind==3] = minCostsTransitions3d[ind==3];
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
cv.imwrite("disparity2b_DP.png",dispImg)

plt.show()
