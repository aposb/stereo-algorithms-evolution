# Stereo Matching using Belief Propagation (Directional) - occlusion penalties approach
# Computes a disparity map from a rectified stereo pair using Belief Propagation (Directional)

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

MAX_INT = 2147483647

def main():
    # Set parameters
    dispLevels = 16 #disparity range: 0 to dispLevels-1
    p1 = 10 #occlusion penalty 1
    p2 = 20 #occlusion penalty 2
    iterations = 20

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

    # Compute pixel-based matching costs (data cost)
    matchingCosts = np.zeros((rows,cols,dispLevels),dtype=np.int32)
    for d in range(dispLevels):
        rightImgShifted = shiftRight(rightImg,d,0)
        matchingCosts[:,:,d] = computeMatchingCost(leftImg,rightImgShifted)

    # Initialize messages for the 4 directions
    fromLeft = np.zeros((rows,cols,dispLevels),dtype=np.int32)
    fromRight = np.zeros((rows,cols,dispLevels),dtype=np.int32)
    fromUp = np.zeros((rows,cols,dispLevels),dtype=np.int32)
    fromDown = np.zeros((rows,cols,dispLevels),dtype=np.int32)

    for it in range(iterations):
        # Left to right pass (horizontal forward) - Send messages right
        for x in range(cols-1):
            currentCosts = (matchingCosts[:,x,:] + fromLeft[:,x,:] + fromUp[:,x,:] + fromDown[:,x,:])[:,np.newaxis,:]
            fromLeft[:,x+1,:] = computeDirectionalCosts(currentCosts,(p1,p2))[:,0,:]

        # Right to left pass (horizontal backward) - Send messages left
        for x in range(cols-1,0,-1):
            currentCosts = (matchingCosts[:,x,:] + fromRight[:,x,:] + fromUp[:,x,:] + fromDown[:,x,:])[:,np.newaxis,:]
            fromRight[:,x-1,:] = computeDirectionalCosts(currentCosts,(p1,p2))[:,0,:]

        # Up to down pass (vertical forward) - Send messages down
        for y in range(rows-1):
            currentCosts = (matchingCosts[y,:,:] + fromUp[y,:,:] + fromLeft[y,:,:] + fromRight[y,:,:])[np.newaxis,:,:]
            fromUp[y+1,:,:] = computeDirectionalCosts(currentCosts,(p1,p2))[0,:,:]

        # Down to up pass (vertical backward) - Send messages up
        for y in range(rows-1,0,-1):
            currentCosts = (matchingCosts[y,:,:] + fromDown[y,:,:] + fromLeft[y,:,:] + fromRight[y,:,:])[np.newaxis,:,:]
            fromDown[y-1,:,:] = computeDirectionalCosts(currentCosts,(p1,p2))[0,:,:]

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

# Compute messages
# ----------------
def computeDirectionalCosts(currentCosts,occPenalties):
    minInput = np.amin(currentCosts,axis=2)
    currentCostsP1 = currentCosts + occPenalties[0]
    possibleOutput = np.zeros((currentCosts.shape[0],currentCosts.shape[1],currentCosts.shape[2],4),dtype=np.int32)
    possibleOutput[:,:,:,0] = currentCosts
    possibleOutput[:,:,:,1] = shiftForward(currentCostsP1,1,MAX_INT)
    possibleOutput[:,:,:,2] = shiftBackward(currentCostsP1,1,MAX_INT)
    possibleOutput[:,:,:,3] = (minInput + occPenalties[1])[:,:,np.newaxis]
    output = np.amin(possibleOutput,axis=3)
    output = output - minInput[:,:,np.newaxis] #normalize
    return output

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
