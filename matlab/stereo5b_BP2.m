% Stereo Matching using Belief Propagation (Synchronous) - occlusion penalties approach
% Computes a disparity map from a rectified stereo pair using Belief Propagation (Synchronous)

% Set parameters
dispLevels = 16; %disparity range: 0 to dispLevels-1
p1 = 10; %occlusion penalty 1
p2 = 20; %occlusion penalty 2
iterations = 60;

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

% Compute pixel-based matching costs (data cost)
matchingCosts = zeros(rows,cols,dispLevels,'int32');
for d = 0:dispLevels-1
    rightImgShifted = shiftRight(rightImg,d,0);
    matchingCosts(:,:,d+1) = computeMatchingCost(leftImg,rightImgShifted);
end

% Initialize messages for the 4 directions
fromLeft = zeros(rows,cols,dispLevels,'int32');
fromRight = zeros(rows,cols,dispLevels,'int32');
fromUp = zeros(rows,cols,dispLevels,'int32');
fromDown = zeros(rows,cols,dispLevels,'int32');

figure
for it = 1:iterations
    % Create messages to right
    currentCosts = matchingCosts + fromLeft + fromUp + fromDown;
    toRight = computeCosts_SgmLike(currentCosts,[p1 p2]);

    % Create messages to left
    currentCosts = matchingCosts + fromRight + fromUp + fromDown;
    toLeft = computeCosts_SgmLike(currentCosts,[p1 p2]);

    % Create messages to down
    currentCosts = matchingCosts + fromUp + fromLeft + fromRight;
    toDown = computeCosts_SgmLike(currentCosts,[p1 p2]);

    % Create messages to up
    currentCosts = matchingCosts + fromDown + fromLeft + fromRight;
    toUp = computeCosts_SgmLike(currentCosts,[p1 p2]);

    % Send all messages
    fromLeft = shiftRight(toRight,1,0);
    fromRight = shiftLeft(toLeft,1,0);
    fromUp = shiftDown(toDown,1,0);
    fromDown = shiftUp(toUp,1,0);

    % Compute total costs (belief)
    totalCosts = fromLeft + fromRight + fromUp + fromDown;

    % Compute the disparity map
    [~,ind] = min(totalCosts,[],3);
    dispMap = ind-1;

    % Normalize the disparity map for display
    scaleFactor = 256/dispLevels;
    dispImg = uint8(dispMap*scaleFactor);

    % Show disparity map
    imshow(dispImg)

    % Show iterations
    fprintf('iteration: %d/%d\n',it,iterations)
end

% Save disparity map
imwrite(dispImg,'disparity5b_BP2.png')
