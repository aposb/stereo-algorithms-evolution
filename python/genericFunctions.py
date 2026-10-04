import numpy as np

MAX_INT = 2147483647

def shiftDown(A,n,fillValue):
    """shiftDown Shift an array DOWN with specific padding.
    B = shiftDown(A,n,fillValue) shifts the elements in the array A
    from up to down by n positions and fills the empy places with
    new fillValue elements."""
    B = np.roll(A,n,0)
    B[:n] = fillValue
    return B

def shiftUp(A,n,fillValue):
    """shiftUp Shift an array UP with specific padding.
    B = shiftUp(A,n,fillValue) shifts the elements in the array A
    from down to up by n positions and fills the empy places with
    new fillValue elements."""
    B = np.roll(A,-n,0)
    B[-n:] = fillValue
    return B

def shiftRight(A,n,fillValue):
    """shiftRight Shift an array RIGHT with specific padding.
    B = shiftRight(A,n,fillValue) shifts the elements in the array A
    from left to right by n positions and fills the empy places with
    new fillValue elements."""
    B = np.roll(A,n,1)
    B[:,:n] = fillValue
    return B

def shiftLeft(A,n,fillValue):
    """shiftLeft Shift an array LEFT with specific padding.
    B = shiftLeft(A,n,fillValue) shifts the elements in the array A
    from right to left by n positions and fills the empy places with
    new fillValue elements."""
    B = np.roll(A,-n,1)
    B[:,-n:] = fillValue
    return B

def shiftForward(A,n,fillValue):
    """shiftForward Shift an array FORWARD with specific padding.
    B = shiftForward(A,n,fillValue) shifts the elements in the array A
    from backward to forward by n positions and fills the empy places
    with new fillValue elements."""
    B = np.roll(A,n,2)
    B[:,:,:n] = fillValue
    return B

def shiftBackward(A,n,fillValue):
    """shiftBackward Shift an array BACKWARD with specific padding.
    B = shiftBackward(A,n,fillValue) shifts the elements in the array A
    from forward to backward by n positions and fills the empy places
    with new fillValue elements."""
    B = np.roll(A,-n,2)
    B[:,:,-n:] = fillValue
    return B

def computeCosts_BpLike(currentCosts,smoothnessCosts):
    """computeCosts_BpLike Compute minimum cost paths (messages) - Approach A (BP-Like)."""
    sum_ = currentCosts[:,:,:,np.newaxis] + smoothnessCosts
    costs = np.amin(sum_,axis=2)
    costs = costs - np.amin(costs,axis=2)[:,:,np.newaxis] #normalize
    return costs

def computeCostsAndTransitions_BpLike(currentCosts,smoothnessCosts):
    """computeCostsAndTransitions_BpLike Compute minimum cost paths (messages) and transitions - Approach A (BP-Like)."""
    sum_ = currentCosts[:,:,:,np.newaxis] + smoothnessCosts
    costs = np.amin(sum_,axis=2)
    costs = costs - np.amin(costs,axis=2)[:,:,np.newaxis] #normalize
    transitions = np.argmin(sum_,axis=2).astype(np.int32)
    return costs,transitions

def computeCosts_SgmLike(currentCosts,occPenalties):
    """computeCosts_SgmLike Compute minimum cost paths (messages) - Approach B (SGM-Like)."""
    minInput = np.amin(currentCosts,axis=2)
    currentCostsP1 = currentCosts + occPenalties[0]
    possibleOutput = np.zeros((currentCosts.shape[0],currentCosts.shape[1],currentCosts.shape[2],4),dtype=np.int32)
    possibleOutput[:,:,:,0] = currentCosts
    possibleOutput[:,:,:,1] = shiftForward(currentCostsP1,1,MAX_INT)
    possibleOutput[:,:,:,2] = shiftBackward(currentCostsP1,1,MAX_INT)
    possibleOutput[:,:,:,3] = (minInput + occPenalties[1])[:,:,np.newaxis]
    costs = np.amin(possibleOutput,axis=3)
    costs = costs - minInput[:,:,np.newaxis] #normalize
    return costs

def computeCostsAndTransitions_SgmLike(currentCosts,occPenalties):
    """computeCostsAndTransitions_SgmLike Compute minimum cost paths (messages) and transitions - Approach B (SGM-Like)."""
    minInput = np.amin(currentCosts,axis=2)
    ind0 = np.argmin(currentCosts,axis=2)
    currentCostsP1 = currentCosts + occPenalties[0]
    possibleOutput = np.zeros((currentCosts.shape[0],currentCosts.shape[1],currentCosts.shape[2],4),dtype=np.int32)
    possibleOutput[:,:,:,0] = currentCosts
    possibleOutput[:,:,:,1] = shiftForward(currentCostsP1,1,MAX_INT)
    possibleOutput[:,:,:,2] = shiftBackward(currentCostsP1,1,MAX_INT)
    possibleOutput[:,:,:,3] = (minInput + occPenalties[1])[:,:,np.newaxis]
    costs = np.amin(possibleOutput,axis=3)
    ind = np.argmin(possibleOutput,axis=3)
    costs = costs - minInput[:,:,np.newaxis] #normalize
    match = np.arange(currentCosts.shape[2])[np.newaxis,np.newaxis,:] + np.zeros(currentCosts.shape,dtype=np.int32)
    near1 = match-1; near2 = match+1
    far = ind0[:,:,np.newaxis] + np.zeros(currentCosts.shape,dtype=np.int32)
    transitions = np.zeros(currentCosts.shape,dtype=np.int32)
    transitions[ind==0] = match[ind==0]
    transitions[ind==1] = near1[ind==1]
    transitions[ind==2] = near2[ind==2]
    transitions[ind==3] = far[ind==3]
    return costs,transitions
