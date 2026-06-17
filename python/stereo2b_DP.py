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
def computeDirectionalCosts(input_):
    minInput = np.amin(input_,axis=2)
    ind0 = np.argmin(input_,axis=2)
    possibleOutput = np.zeros((input_.shape[0],input_.shape[1],input_.shape[2],4),dtype=np.int32)
    possibleOutput[:,:,:,0] = input_
    possibleOutput[:,:,:,1] = np.roll(input_,1,2) + p1; possibleOutput[:,:,0,1] = MAX_INT
    possibleOutput[:,:,:,2] = np.roll(input_,-1,2) + p1; possibleOutput[:,:,-1,2] = MAX_INT
    possibleOutput[:,:,:,3] = (minInput + p2)[:,:,np.newaxis]
    output = np.amin(possibleOutput,axis=3)
    ind = np.argmin(possibleOutput,axis=3)
    output = output - minInput[:,:,np.newaxis] #normalize costs
    match = np.arange(input_.shape[2])[np.newaxis,np.newaxis,:] + np.zeros(input_.shape,dtype=np.int32)
    near1 = match-1; near2 = match+1
    far = ind0[:,:,np.newaxis] + np.zeros(input_.shape,dtype=np.int32)
    transitions = np.zeros(input_.shape,dtype=np.int32)
    transitions[ind==0] = match[ind==0]
    transitions[ind==1] = near1[ind==1]
    transitions[ind==2] = near2[ind==2]
    transitions[ind==3] = far[ind==3]
    return output,transitions

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
    C,T = computeDirectionalCosts(costs)
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
