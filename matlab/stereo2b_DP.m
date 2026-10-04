% Stereo Matching using Dynamic Programming - occlusion penalties approach
% Computes a disparity map from a rectified stereo pair using Dynamic Programming

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

% Initialize minimum cost paths and transitions for the left to right direction
fromLeft = zeros(rows,cols,dispLevels,'int32');
transitions = zeros(rows,cols,dispLevels,'int32');

% Compute minimum cost paths and transitions for left to right direction
for x = 1:cols-1
    currentCosts = matchingCosts(:,x,:) + fromLeft(:,x,:);
    [C,T] = computeCostsAndTransitions_SgmLike(currentCosts,[p1 p2]);
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
