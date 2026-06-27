% Stereo Matching using Semi-Global Matching - occlusion penalties approach
% Computes a disparity map from a rectified stereo pair using Semi-Global Matching

function stereo3b_SGM

    % Set parameters
    dispLevels = 16; %disparity range: 0 to dispLevels-1
    p1 = 10; %occlusion penalty 1
    p2 = 20; %occlusion penalty 2

    % Define matching cost function
    computeMatchingCost = @(left,right) abs(left-right); %absolute differences

    % Load left and right images in grayscale
    leftImg = rgb2gray(imread('left.png'));
    rightImg = rgb2gray(imread('right.png'));

    % Apply a Gaussian filter
    leftImg = imgaussfilt(leftImg,0.6,'FilterSize',5);
    rightImg = imgaussfilt(rightImg,0.6,'FilterSize',5);

    % Get the size
    [rows,cols] = size(leftImg);

    % Convert to int32
    leftImg = int32(leftImg);
    rightImg = int32(rightImg);

    % Compute pixel-based matching costs
    matchingCosts = zeros(rows,cols,dispLevels,'int32');
    for d = 0:dispLevels-1
        rightImgShifted = shiftRight(rightImg,d,0);
        matchingCosts(:,:,d+1) = computeMatchingCost(leftImg,rightImgShifted);
    end

    % Initialize minimum cost paths for the 4 directions
    fromLeft = zeros(rows,cols,dispLevels,'int32');
    fromRight = zeros(rows,cols,dispLevels,'int32');
    fromUp = zeros(rows,cols,dispLevels,'int32');
    fromDown = zeros(rows,cols,dispLevels,'int32');

    % Compute minimum cost paths for left to right direction
    for x = 1:cols-1
        currentCosts = matchingCosts(:,x,:) + fromLeft(:,x,:);
        fromLeft(:,x+1,:) = computeDirectionalCosts(currentCosts,[p1 p2]);
    end

    % Compute minimum cost paths for right to left direction
    for x = cols:-1:2
        currentCosts = matchingCosts(:,x,:) + fromRight(:,x,:);
        fromRight(:,x-1,:) = computeDirectionalCosts(currentCosts,[p1 p2]);
    end

    % Compute minimum cost paths for up to down direction
    for y = 1:rows-1
        currentCosts = matchingCosts(y,:,:) + fromUp(y,:,:);
        fromUp(y+1,:,:) = computeDirectionalCosts(currentCosts,[p1 p2]);
    end

    % Compute minimum cost paths for down to up direction
    for y = rows:-1:2
        currentCosts = matchingCosts(y,:,:) + fromDown(y,:,:);
        fromDown(y-1,:,:) = computeDirectionalCosts(currentCosts,[p1 p2]);
    end

    % Compute total costs
    totalCosts = fromLeft + fromRight + fromUp + fromDown;

    % Compute the disparity map
    [~,ind] = min(totalCosts,[],3);
    dispMap = ind-1;

    % Normalize the disparity map for display
    scaleFactor = 256/dispLevels;
    dispImg = uint8(dispMap*scaleFactor);

    % Show disparity map
    figure; imshow(dispImg)

    % Save disparity map
    imwrite(dispImg,'disparity3b_SGM.png')
end

% Compute minimum cost paths
% --------------------------
function output = computeDirectionalCosts(currentCosts,occPenalties)
    minInput = min(currentCosts,[],3);
    currentCostsP1 = currentCosts + occPenalties(1);
    possibleOutput = zeros([size(currentCosts),4],'int32');
    possibleOutput(:,:,:,1) = currentCosts;
    possibleOutput(:,:,:,2) = shiftForward(currentCostsP1,1,intmax);
    possibleOutput(:,:,:,3) = shiftBackward(currentCostsP1,1,intmax);
    possibleOutput(:,:,:,4) = minInput + occPenalties(2) + zeros(size(currentCosts),'int32');
    output = min(possibleOutput,[],4);
    output = output - minInput; %normalize
end

% Shift Functions (Down/Up/Right/Left/Forward/Backward)
% -----------------------------------------------------
function B = shiftDown(A,n,fillValue)
    B = circshift(A,n,1);
    B(1:min(n,end),:,:) = fillValue;
end
function B = shiftUp(A,n,fillValue)
    B = circshift(A,-n,1);
    B(max(end-n+1,1):end,:,:) = fillValue;
end
function B = shiftRight(A,n,fillValue)
    B = circshift(A,n,2);
    B(:,1:min(n,end),:) = fillValue;
end
function B = shiftLeft(A,n,fillValue)
    B = circshift(A,-n,2);
    B(:,max(end-n+1,1):end,:) = fillValue;
end
function B = shiftForward(A,n,fillValue)
    B = circshift(A,n,3);
    B(:,:,1:min(n,end)) = fillValue;
end
function B = shiftBackward(A,n,fillValue)
    B = circshift(A,-n,3);
    B(:,:,max(end-n+1,1):end) = fillValue;
end
