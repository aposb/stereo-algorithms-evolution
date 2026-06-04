% Stereo Matching using Dynamic Programming - occlusion penalties approach
% Computes a disparity map from a rectified stereo pair using Dynamic Programming

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

% Initialize minimum cost paths and transitions for the left to right direction
fromLeft = zeros(rows,cols,dispLevels,'int32');
transitions = zeros(rows,cols,dispLevels,'int32');

% Compute minimum cost paths and transitions for left to right direction
for x = 1:cols-1
    costs = matchingCosts(:,x,:) + fromLeft(:,x,:);
    [C,T] = computeMinSumCosts(costs);
    fromLeft(:,x+1,:) = C;
    transitions(:,x+1,:) = T;
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
imwrite(dispImg,'disparity2b_DP.png')

% Compute minimum cost paths and transitions
function [minSumCosts,transitions] = computeMinSumCosts(costs)
    global p1 p2
    [minCosts,minCostsTransitions] = min(costs,[],3);
    sumCosts = zeros([size(costs),4],'int32');
    sumCosts(:,:,:,1) = costs;
    sumCosts(:,:,:,2) = circshift(costs,1,3) + p1; sumCosts(:,:,1,2) = intmax;
    sumCosts(:,:,:,3) = circshift(costs,-1,3) + p1; sumCosts(:,:,end,3) = intmax;
    sumCosts(:,:,:,4) = minCosts + p2 + zeros(size(costs),'int32');
    [minSumCosts,ind] = min(sumCosts,[],4);
    minSumCosts = minSumCosts - minCosts; %normalize costs
    match = permute(int32(1:size(costs,3)),[3 1 2]) + zeros(size(costs),'int32');
    minCostsTransitions3d = int32(minCostsTransitions) + zeros(size(costs),'int32');
    transitions = zeros(size(costs),'int32');
    transitions(ind==1) = match(ind==1);
    transitions(ind==2) = match(ind==2)-1;
    transitions(ind==3) = match(ind==3)+1;
    transitions(ind==4) = minCostsTransitions3d(ind==4);
end
