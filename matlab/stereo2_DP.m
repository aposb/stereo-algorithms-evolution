% Stereo Matching using Dynamic Programming
% Computes a disparity map from a rectified stereo pair using Dynamic Programming

% Set parameters
dispLevels = 16; %disparity range: 0 to dispLevels-1
lambda = 5; %weight of smoothness cost
trunc = 4; %truncation of smoothness cost

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

% Compute pixel-based matching costs
matchingCosts = zeros(rows,cols,dispLevels,'int32');
for d = 0:dispLevels-1
    rightImgShifted = circshift(rightImg,d,2);
    matchingCosts(:,:,d+1) = computeMatchingCost(leftImg,rightImgShifted);
end

% Compute smoothness costs
d = 0:dispLevels-1;
smoothnessCosts = computeSmoothnessCost(d,d.');
smoothnessCosts3H = zeros(1,dispLevels,dispLevels,'int32');
smoothnessCosts3H(1,:,:) = smoothnessCosts;

% Initialize minimum cost paths and transitions for the left to right direction
fromLeft = zeros(rows,cols,dispLevels,'int32');
transitions = zeros(rows,cols,dispLevels,'int32');

% Compute minimum cost paths and transitions for left to right direction
for x = 1:cols-1
    sumCosts = matchingCosts(:,x,:) + fromLeft(:,x,:) + smoothnessCosts3H;
    [minSumCosts,ind] = min(sumCosts,[],3);
    fromLeft(:,x+1,:) = minSumCosts;
    transitions(:,x+1,:) = ind;
end

% Compute the disparity map - Backtracking
dispMap = zeros(rows,cols);
[~,ind] = min(fromLeft(:,cols,:),[],3);
for x = cols:-1:1
    dispMap(:,x) = ind-1;
    ind = transitions(sub2ind(size(transitions),(1:rows).',x*ones(rows,1),ind)); %get the disparity transitions
end

% Normalize the disparity map for display
scaleFactor = 256/dispLevels;
dispImg = uint8(dispMap*scaleFactor);

% Show disparity map
figure; imshow(dispImg)

% Save disparity map
imwrite(dispImg,'disparity2_DP.png')
