# Stereo Matching using Belief Propagation (Synchronous)
# Computes a disparity map from a rectified stereo pair using Belief Propagation (Synchronous)

import time
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# Set parameters
dispLevels = 16 #disparity range: 0 to dispLevels-1
lambda_ = 5 #weight of smoothness cost
trunc = 4 #truncation of smoothness cost
iterations = 80

# Define matching cost function
computeMatchingCost = lambda left,right: np.absolute(left-right) #absolute differences

# Define smoothness cost function
computeSmoothnessCost = lambda d1,d2: lambda_*np.minimum(np.absolute(d1-d2),trunc)

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

# Compute smoothness costs
d = np.arange(dispLevels)
smoothnessCosts = computeSmoothnessCost(d,d[np.newaxis,:].T)
smoothnessCosts4d = smoothnessCosts[np.newaxis,np.newaxis,:,:].astype(np.int32)

# Initialize messages for the 4 directions
fromLeft = np.zeros((rows,cols,dispLevels),dtype=np.int32)
fromRight = np.zeros((rows,cols,dispLevels),dtype=np.int32)
fromUp = np.zeros((rows,cols,dispLevels),dtype=np.int32)
fromDown = np.zeros((rows,cols,dispLevels),dtype=np.int32)

for it in range(iterations):
    # Create messages to right
    sumCosts = (matchingCosts + fromUp + fromDown + fromLeft)[:,:,:,np.newaxis] + smoothnessCosts4d
    minSumCosts = np.amin(sumCosts,axis=2)
    normalizedCosts = minSumCosts - np.amin(minSumCosts,axis=2)[:,:,np.newaxis]
    toRight = normalizedCosts

    # Create messages to left
    sumCosts = (matchingCosts + fromUp + fromDown + fromRight)[:,:,:,np.newaxis] + smoothnessCosts4d
    minSumCosts = np.amin(sumCosts,axis=2)
    normalizedCosts = minSumCosts - np.amin(minSumCosts,axis=2)[:,:,np.newaxis]
    toLeft = normalizedCosts

    # Create messages to down
    sumCosts = (matchingCosts + fromUp + fromRight + fromLeft)[:,:,:,np.newaxis] + smoothnessCosts4d
    minSumCosts = np.amin(sumCosts,axis=2)
    normalizedCosts = minSumCosts - np.amin(minSumCosts,axis=2)[:,:,np.newaxis]
    toDown = normalizedCosts

    # Create messages to up
    sumCosts = (matchingCosts + fromDown + fromRight + fromLeft)[:,:,:,np.newaxis] + smoothnessCosts4d
    minSumCosts = np.amin(sumCosts,axis=2)
    normalizedCosts = minSumCosts - np.amin(minSumCosts,axis=2)[:,:,np.newaxis]
    toUp = normalizedCosts

    # Send all messages
    fromLeft = np.roll(toRight,1,1) #shift right
    fromRight = np.roll(toLeft,-1,1) #shift left
    fromUp = np.roll(toDown,1,0) #shift down
    fromDown = np.roll(toUp,-1,0) #shift up

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
cv.imwrite("disparity5_BP2.png",dispImg)

plt.show()
