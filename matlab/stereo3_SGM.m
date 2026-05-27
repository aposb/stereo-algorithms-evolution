% Stereo Matching using Semi-Global Matching
% Computes a disparity map from a rectified stereo pair using Semi-Global Matching

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
smoothnessCosts3V = permute(smoothnessCosts3H,[2 1 3]);

% Initialize minimum cost paths for the 4 directions
fromLeft = zeros(rows,cols,dispLevels,'int32');
fromRight = zeros(rows,cols,dispLevels,'int32');
fromUp = zeros(rows,cols,dispLevels,'int32');
fromDown = zeros(rows,cols,dispLevels,'int32');

% Compute minimum cost paths for left to right direction
for x = 1:cols-1
    sumCosts = matchingCosts(:,x,:) + fromLeft(:,x,:) + smoothnessCosts3H;
    minSumCosts = min(sumCosts,[],3);
    normalizedCosts = minSumCosts - min(minSumCosts,[],2);
    fromLeft(:,x+1,:) = normalizedCosts;
end

% Compute minimum cost paths for right to left direction
for x = cols:-1:2
    sumCosts = matchingCosts(:,x,:) + fromRight(:,x,:) + smoothnessCosts3H;
    minSumCosts = min(sumCosts,[],3);
    normalizedCosts = minSumCosts - min(minSumCosts,[],2);
    fromRight(:,x-1,:) = normalizedCosts;
end

% Compute minimum cost paths for up to down direction
for y = 1:rows-1
    sumCosts = matchingCosts(y,:,:) + fromUp(y,:,:) + smoothnessCosts3V;
    minSumCosts = min(sumCosts,[],3).';
    normalizedCosts = minSumCosts - min(minSumCosts,[],2);
    fromUp(y+1,:,:) = normalizedCosts;
end

% Compute minimum cost paths for down to up direction
for y = rows:-1:2
    sumCosts = matchingCosts(y,:,:) + fromDown(y,:,:) + smoothnessCosts3V;
    minSumCosts = min(sumCosts,[],3).';
    normalizedCosts = minSumCosts - min(minSumCosts,[],2);
    fromDown(y-1,:,:) = normalizedCosts;
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
imwrite(dispImg,'disparity3_SGM.png')
