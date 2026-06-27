# Stereo Matching using Dynamic Programming - occlusion penalties approach
# Computes a disparity map from a rectified stereo pair using Dynamic Programming

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

MAX_INT = 2147483647

def main():
    # Set parameters
    dispLevels = 16 #disparity range: 0 to dispLevels-1
    p1 = 10 #occlusion penalty 1
    p2 = 20 #occlusion penalty 2

    # Define matching cost function
    computeMatchingCost = lambda left,right: np.absolute(left-right) #absolute differences

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
        rightImgShifted = shiftRight(rightImg,d,0)
        matchingCosts[:,:,d] = computeMatchingCost(leftImg,rightImgShifted)

    # Initialize minimum cost paths and transitions for the left to right direction
    fromLeft = np.zeros((rows,cols,dispLevels),dtype=np.int32)
    transitions = np.zeros((rows,cols,dispLevels),dtype=np.int32)

    # Compute minimum cost paths and transitions for left to right direction
    for x in range(cols-1):
        currentCosts = (matchingCosts[:,x,:] + fromLeft[:,x,:])[:,np.newaxis,:]
        C,T = computeDirectionalCosts(currentCosts,(p1,p2))
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

# Compute minimum cost paths and transitions
# ------------------------------------------
def computeDirectionalCosts(currentCosts,occPenalties):
    minInput = np.amin(currentCosts,axis=2)
    ind0 = np.argmin(currentCosts,axis=2)
    currentCostsP1 = currentCosts + occPenalties[0]
    possibleOutput = np.zeros((currentCosts.shape[0],currentCosts.shape[1],currentCosts.shape[2],4),dtype=np.int32)
    possibleOutput[:,:,:,0] = currentCosts
    possibleOutput[:,:,:,1] = shiftForward(currentCostsP1,1,MAX_INT)
    possibleOutput[:,:,:,2] = shiftBackward(currentCostsP1,1,MAX_INT)
    possibleOutput[:,:,:,3] = (minInput + occPenalties[1])[:,:,np.newaxis]
    output = np.amin(possibleOutput,axis=3)
    ind = np.argmin(possibleOutput,axis=3)
    output = output - minInput[:,:,np.newaxis] #normalize
    match = np.arange(currentCosts.shape[2])[np.newaxis,np.newaxis,:] + np.zeros(currentCosts.shape,dtype=np.int32)
    near1 = match-1; near2 = match+1
    far = ind0[:,:,np.newaxis] + np.zeros(currentCosts.shape,dtype=np.int32)
    transitions = np.zeros(currentCosts.shape,dtype=np.int32)
    transitions[ind==0] = match[ind==0]
    transitions[ind==1] = near1[ind==1]
    transitions[ind==2] = near2[ind==2]
    transitions[ind==3] = far[ind==3]
    return output,transitions

# Shift Functions (Down/Up/Right/Left/Forward/Backward)
# -----------------------------------------------------
def shiftDown(A,n,fillValue):
    B = np.roll(A,n,0)
    B[:n] = fillValue
    return B

def shiftUp(A,n,fillValue):
    B = np.roll(A,-n,0)
    B[-n:] = fillValue
    return B

def shiftRight(A,n,fillValue):
    B = np.roll(A,n,1)
    B[:,:n] = fillValue
    return B

def shiftLeft(A,n,fillValue):
    B = np.roll(A,-n,1)
    B[:,-n:] = fillValue
    return B

def shiftForward(A,n,fillValue):
    B = np.roll(A,n,2)
    B[:,:,:n] = fillValue
    return B

def shiftBackward(A,n,fillValue):
    B = np.roll(A,-n,2)
    B[:,:,-n:] = fillValue
    return B

if __name__ == "__main__":
    main()
