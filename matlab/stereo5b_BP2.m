% Stereo Matching using Belief Propagation (Synchronous) - occlusion penalties approach
% Computes a disparity map from a rectified stereo pair using Belief Propagation (Synchronous)

global p1 p2

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
    rightImgShifted = circshift(rightImg,d,2);
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
    costs = matchingCosts + fromLeft + fromUp + fromDown;
    toRight = computeMinSumCosts(costs);

    % Create messages to left
    costs = matchingCosts + fromRight + fromUp + fromDown;
    toLeft = computeMinSumCosts(costs);

    % Create messages to down
    costs = matchingCosts + fromUp + fromLeft + fromRight;
    toDown = computeMinSumCosts(costs);

    % Create messages to up
    costs = matchingCosts + fromDown + fromLeft + fromRight;
    toUp = computeMinSumCosts(costs);

    % Send all messages
    fromLeft = circshift(toRight,1,2); %shift right
    fromRight = circshift(toLeft,-1,2); %shift left
    fromUp = circshift(toDown,1,1); %shift down
    fromDown = circshift(toUp,-1,1); %shift up

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

% Compute messages
function minSumCosts = computeMinSumCosts(costs)
    global p1 p2
    minCosts = min(costs,[],3);
    sumCosts = zeros([size(costs),4],'int32');
    sumCosts(:,:,:,1) = costs;
    sumCosts(:,:,:,2) = circshift(costs,1,3) + p1; sumCosts(:,:,1,2) = intmax;
    sumCosts(:,:,:,3) = circshift(costs,-1,3) + p1; sumCosts(:,:,end,3) = intmax;
    sumCosts(:,:,:,4) = minCosts + p2 + zeros(size(costs),'int32');
    minSumCosts = min(sumCosts,[],4);
    minSumCosts = minSumCosts - minCosts; %normalize messages
end
