% Stereo Matching using Belief Propagation (Sequential)
% Computes a disparity map from a rectified stereo pair using Belief Propagation (Sequential)

global smoothnessCosts4d

% Set parameters
dispLevels = 16; %disparity range: 0 to dispLevels-1
lambda = 5; %weight of smoothness cost
trunc = 4; %truncation of smoothness cost
iterations = 20;

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
smoothnessCosts4d = permute(int32(smoothnessCosts),[3 4 1 2]);

% Initialize messages for the 4 directions
fromLeft = zeros(rows,cols,dispLevels,'int32');
fromRight = zeros(rows,cols,dispLevels,'int32');
fromUp = zeros(rows,cols,dispLevels,'int32');
fromDown = zeros(rows,cols,dispLevels,'int32');

figure
for it = 1:iterations
    % Left to right pass (horizontal forward) - Send messages right
    for x = 1:cols-1
        costs = matchingCosts(:,x,:) + fromLeft(:,x,:) + fromUp(:,x,:) + fromDown(:,x,:);
        fromLeft(:,x+1,:) = computeMinSumCosts(costs);
    end

    % Right to left pass (horizontal backward) - Send messages left
    for x = cols:-1:2
        costs = matchingCosts(:,x,:) + fromRight(:,x,:) + fromUp(:,x,:) + fromDown(:,x,:);
        fromRight(:,x-1,:) = computeMinSumCosts(costs);
    end

    % Up to down pass (vertical forward) - Send messages down
    for y = 1:rows-1
        costs = matchingCosts(y,:,:) + fromUp(y,:,:) + fromLeft(y,:,:) + fromRight(y,:,:);
        fromUp(y+1,:,:) = computeMinSumCosts(costs);
    end

    % Down to up pass (vertical backward) - Send messages up
    for y = rows:-1:2
        costs = matchingCosts(y,:,:) + fromDown(y,:,:) + fromLeft(y,:,:) + fromRight(y,:,:);
        fromDown(y-1,:,:) = computeMinSumCosts(costs);
    end

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
imwrite(dispImg,'disparity4_BP1.png')

% Compute messages
function minSumCosts = computeMinSumCosts(costs)
    global smoothnessCosts4d
    sumCosts = costs + smoothnessCosts4d;
    minSumCosts = permute(min(sumCosts,[],3),[1 2 4 3]);
    minSumCosts = minSumCosts - min(minSumCosts,[],3); %normalize messages
end
