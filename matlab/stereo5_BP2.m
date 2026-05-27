% Stereo Matching using Belief Propagation (Synchronous)
% Computes a disparity map from a rectified stereo pair using Belief Propagation (Synchronous)

% Set parameters
dispLevels = 16; %disparity range: 0 to dispLevels-1
lambda = 5; %weight of smoothness cost
trunc = 4; %truncation of smoothness cost
iterations = 80;

% Define matching cost function
computeMatchingCost = @(left,right) abs(left-right); %absolute differences

% Define smoothness cost function
computeSmoothnessCost = @(d1,d2) lambda*min(abs(d1-d2),trunc);

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

% Compute smoothness costs
d = 0:dispLevels-1;
smoothnessCosts = computeSmoothnessCost(d,d.');
smoothnessCosts4 = zeros(1,1,dispLevels,dispLevels,'int32');
smoothnessCosts4(1,1,:,:) = smoothnessCosts;

% Initialize messages for the 4 directions
fromLeft = zeros(rows,cols,dispLevels,'int32');
fromRight = zeros(rows,cols,dispLevels,'int32');
fromUp = zeros(rows,cols,dispLevels,'int32');
fromDown = zeros(rows,cols,dispLevels,'int32');

figure
for it = 1:iterations
    % Create messages to right
    sumCosts = matchingCosts + fromUp + fromDown + fromLeft + smoothnessCosts4;
    minSumCosts = squeeze(min(sumCosts,[],3));
    normalizedCosts = minSumCosts - min(minSumCosts,[],3);
    toRight = normalizedCosts;

    % Create messages to left
    sumCosts = matchingCosts + fromUp + fromDown + fromRight + smoothnessCosts4;
    minSumCosts = squeeze(min(sumCosts,[],3));
    normalizedCosts = minSumCosts - min(minSumCosts,[],3);
    toLeft = normalizedCosts;

    % Create messages to down
    sumCosts = matchingCosts + fromUp + fromRight + fromLeft + smoothnessCosts4;
    minSumCosts = squeeze(min(sumCosts,[],3));
    normalizedCosts = minSumCosts - min(minSumCosts,[],3);
    toDown = normalizedCosts;

    % Create messages to up
    sumCosts = matchingCosts + fromDown + fromRight + fromLeft + smoothnessCosts4;
    minSumCosts = squeeze(min(sumCosts,[],3));
    normalizedCosts = minSumCosts - min(minSumCosts,[],3);
    toUp = normalizedCosts;

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
imwrite(dispImg,'disparity5_BP2.png')
