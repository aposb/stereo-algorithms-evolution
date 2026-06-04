% Stereo Matching using Semi-Global Matching - occlusion penalties approach
% Computes a disparity map from a rectified stereo pair using Semi-Global Matching

global p1 p2

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
    rightImgShifted = circshift(rightImg,d,2);
    matchingCosts(:,:,d+1) = computeMatchingCost(leftImg,rightImgShifted);
end

% Initialize minimum cost paths for the 4 directions
fromLeft = zeros(rows,cols,dispLevels,'int32');
fromRight = zeros(rows,cols,dispLevels,'int32');
fromUp = zeros(rows,cols,dispLevels,'int32');
fromDown = zeros(rows,cols,dispLevels,'int32');

% Compute minimum cost paths for left to right direction
for x = 1:cols-1
    costs = matchingCosts(:,x,:) + fromLeft(:,x,:);
    fromLeft(:,x+1,:) = computeMinSumCosts(costs);
end

% Compute minimum cost paths for right to left direction
for x = cols:-1:2
    costs = matchingCosts(:,x,:) + fromRight(:,x,:);
    fromRight(:,x-1,:) = computeMinSumCosts(costs);
end

% Compute minimum cost paths for up to down direction
for y = 1:rows-1
    costs = matchingCosts(y,:,:) + fromUp(y,:,:);
    fromUp(y+1,:,:) = computeMinSumCosts(costs);
end

% Compute minimum cost paths for down to up direction
for y = rows:-1:2
    costs = matchingCosts(y,:,:) + fromDown(y,:,:);
    fromDown(y-1,:,:) = computeMinSumCosts(costs);
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

% Compute minimum cost paths
function minSumCosts = computeMinSumCosts(costs)
    global p1 p2
    minCosts = min(costs,[],3);
    sumCosts = zeros([size(costs),4],'int32');
    sumCosts(:,:,:,1) = costs;
    sumCosts(:,:,:,2) = circshift(costs,1,3) + p1; sumCosts(:,:,1,2) = intmax;
    sumCosts(:,:,:,3) = circshift(costs,-1,3) + p1; sumCosts(:,:,end,3) = intmax;
    sumCosts(:,:,:,4) = minCosts + p2 + zeros(size(costs),'int32');
    minSumCosts = min(sumCosts,[],4);
    minSumCosts = minSumCosts - minCosts; %normalize costs
end
