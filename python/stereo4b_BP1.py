# Stereo Matching using Belief Propagation (Sequential) - occlusion penalties approach
# Computes a disparity map from a rectified stereo pair using Belief Propagation (Sequential)

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

MAX_INT = 2147483647

# Set parameters
dispLevels = 16 #disparity range: 0 to dispLevels-1
p1 = 10 #occlusion penalty 1
p2 = 20 #occlusion penalty 2
iterations = 20

# Define matching cost function
computeMatchingCost = lambda left,right: np.absolute(left-right) #absolute differences

# Compute messages
def computeMinSumCosts(costs):
    minCosts = np.amin(costs,axis=2)
    sumCosts = np.zeros((costs.shape[0],costs.shape[1],costs.shape[2],4),dtype=np.int32)
    sumCosts[:,:,:,0] = costs
    sumCosts[:,:,:,1] = np.roll(costs,1,2) + p1; sumCosts[:,:,0,1] = MAX_INT
    sumCosts[:,:,:,2] = np.roll(costs,-1,2) + p1; sumCosts[:,:,-1,2] = MAX_INT
    sumCosts[:,:,:,3] = (minCosts + p2)[:,:,np.newaxis]
    minSumCosts = np.amin(sumCosts,axis=3)
    minSumCosts = minSumCosts - minCosts[:,:,np.newaxis] #normalize messages
    return minSumCosts

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

# Compute pixel-based matching costs (data cost)
matchingCosts = np.zeros((rows,cols,dispLevels),dtype=np.int32)
for d in range(dispLevels):
    rightImgShifted = np.roll(rightImg,d,1)
    matchingCosts[:,:,d] = computeMatchingCost(leftImg,rightImgShifted)

# Initialize messages for the 4 directions
fromLeft = np.zeros((rows,cols,dispLevels),dtype=np.int32)
fromRight = np.zeros((rows,cols,dispLevels),dtype=np.int32)
fromUp = np.zeros((rows,cols,dispLevels),dtype=np.int32)
fromDown = np.zeros((rows,cols,dispLevels),dtype=np.int32)

for it in range(iterations):
    # Left to right pass (horizontal forward) - Send messages right
    for x in range(cols-1):
        costs = (matchingCosts[:,x,:] + fromLeft[:,x,:] + fromUp[:,x,:] + fromDown[:,x,:])[:,np.newaxis,:]
        fromLeft[:,x+1,:] = computeMinSumCosts(costs)[:,0,:]

    # Right to left pass (horizontal backward) - Send messages left
    for x in range(cols-1,0,-1):
        costs = (matchingCosts[:,x,:] + fromRight[:,x,:] + fromUp[:,x,:] + fromDown[:,x,:])[:,np.newaxis,:]
        fromRight[:,x-1,:] = computeMinSumCosts(costs)[:,0,:]

    # Up to down pass (vertical forward) - Send messages down
    for y in range(rows-1):
        costs = (matchingCosts[y,:,:] + fromUp[y,:,:] + fromLeft[y,:,:] + fromRight[y,:,:])[np.newaxis,:,:]
        fromUp[y+1,:,:] = computeMinSumCosts(costs)[0,:,:]

    # Down to up pass (vertical backward) - Send messages up
    for y in range(rows-1,0,-1):
        costs = (matchingCosts[y,:,:] + fromDown[y,:,:] + fromLeft[y,:,:] + fromRight[y,:,:])[np.newaxis,:,:]
        fromDown[y-1,:,:] = computeMinSumCosts(costs)[0,:,:]

    # Compute total costs (belief)
    totalCosts = fromLeft + fromRight + fromUp + fromDown

    # Compute the disparity map
    dispMap = np.argmin(totalCosts,axis=2)

    # Normalize the disparity map for display
    scaleFactor = 256/dispLevels
    dispImg = (dispMap*scaleFactor).astype(np.uint8)

    # Show disparity map
    plt.cla()
    plt.imshow(dispImg,cmap="gray")
    plt.show(block=False)
    plt.pause(0.01)

    # Show iterations
    print("iteration: {0}/{1}".format(it+1,iterations))

# Save disparity map
cv.imwrite("disparity4b_BP1.png",dispImg)

plt.show()
