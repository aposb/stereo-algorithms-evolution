# Stereo Matching using Belief Propagation (Sequential)
# Computes a disparity map from a rectified stereo pair using Belief Propagation (Sequential)

import time
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# Set parameters
dispLevels = 16 #disparity range: 0 to dispLevels-1
lambda_ = 5 #weight of smoothness cost
trunc = 4 #truncation of smoothness cost
iterations = 20

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
smoothnessCosts3H = smoothnessCosts[np.newaxis,:,:].astype(np.int32)
smoothnessCosts3V = np.moveaxis(smoothnessCosts3H,0,1)

# Initialize messages for the 4 directions
fromLeft = np.zeros((rows,cols,dispLevels),dtype=np.int32)
fromRight = np.zeros((rows,cols,dispLevels),dtype=np.int32)
fromUp = np.zeros((rows,cols,dispLevels),dtype=np.int32)
fromDown = np.zeros((rows,cols,dispLevels),dtype=np.int32)

for it in range(iterations):
    # Left to right pass (horizontal forward) - Send messages right
    for x in range(cols-1):
        sumCosts = (matchingCosts[:,x,:] + fromUp[:,x,:] + fromDown[:,x,:] + fromLeft[:,x,:])[:,np.newaxis,:] + smoothnessCosts3H
        minSumCosts = np.amin(sumCosts,axis=2)
        normalizedCosts = minSumCosts - np.amin(minSumCosts,axis=1)[:,np.newaxis]
        fromLeft[:,x+1,:] = normalizedCosts

    # Right to left pass (horizontal backward) - Send messages left
    for x in range(cols-1,0,-1):
        sumCosts = (matchingCosts[:,x,:] + fromUp[:,x,:] + fromDown[:,x,:] + fromRight[:,x,:])[:,np.newaxis,:] + smoothnessCosts3H
        minSumCosts = np.amin(sumCosts,axis=2)
        normalizedCosts = minSumCosts - np.amin(minSumCosts,axis=1)[:,np.newaxis]
        fromRight[:,x-1,:] = normalizedCosts

    # Up to down pass (vertical forward) - Send messages down
    for y in range(rows-1):
        sumCosts = (matchingCosts[y,:,:] + fromUp[y,:,:] + fromRight[y,:,:] + fromLeft[y,:,:])[np.newaxis,:,:] + smoothnessCosts3V
        minSumCosts = np.amin(sumCosts,axis=2).T
        normalizedCosts = minSumCosts - np.amin(minSumCosts,axis=1)[:,np.newaxis]
        fromUp[y+1,:,:] = normalizedCosts

    # Down to up pass (vertical backward) - Send messages up
    for y in range(rows-1,0,-1):
        sumCosts = (matchingCosts[y,:,:] + fromDown[y,:,:] + fromRight[y,:,:] + fromLeft[y,:,:])[np.newaxis,:,:] + smoothnessCosts3V
        minSumCosts = np.amin(sumCosts,axis=2).T
        normalizedCosts = minSumCosts - np.amin(minSumCosts,axis=1)[:,np.newaxis]
        fromDown[y-1,:,:] = normalizedCosts

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
cv.imwrite("disparity4_BP1.png",dispImg)

plt.show()
