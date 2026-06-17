% Stereo Matching using Belief Propagation (Directional) - occlusion penalties approach
% Computes a disparity map from a rectified stereo pair using Belief Propagation (Directional)

global p1 p2

% Set parameters
dispLevels = 16; %disparity range: 0 to dispLevels-1
p1 = 10; %occlusion penalty 1
p2 = 20; %occlusion penalty 2
iterations = 20;

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
    % Left to right pass (horizontal forward) - Send messages right
    for x = 1:cols-1
        costs = matchingCosts(:,x,:) + fromLeft(:,x,:) + fromUp(:,x,:) + fromDown(:,x,:);
        fromLeft(:,x+1,:) = computeDirectionalCosts(costs);
    end

    % Right to left pass (horizontal backward) - Send messages left
    for x = cols:-1:2
        costs = matchingCosts(:,x,:) + fromRight(:,x,:) + fromUp(:,x,:) + fromDown(:,x,:);
        fromRight(:,x-1,:) = computeDirectionalCosts(costs);
    end

    % Up to down pass (vertical forward) - Send messages down
    for y = 1:rows-1
        costs = matchingCosts(y,:,:) + fromUp(y,:,:) + fromLeft(y,:,:) + fromRight(y,:,:);
        fromUp(y+1,:,:) = computeDirectionalCosts(costs);
    end

    % Down to up pass (vertical backward) - Send messages up
    for y = rows:-1:2
        costs = matchingCosts(y,:,:) + fromDown(y,:,:) + fromLeft(y,:,:) + fromRight(y,:,:);
        fromDown(y-1,:,:) = computeDirectionalCosts(costs);
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
imwrite(dispImg,'disparity4b_BP1.png')

% Compute messages
function output = computeDirectionalCosts(input)
    global p1 p2
    minInput = min(input,[],3);
    possibleOutput = zeros([size(input),4],'int32');
    possibleOutput(:,:,:,1) = input;
    possibleOutput(:,:,:,2) = circshift(input,1,3) + p1; possibleOutput(:,:,1,2) = intmax;
    possibleOutput(:,:,:,3) = circshift(input,-1,3) + p1; possibleOutput(:,:,end,3) = intmax;
    possibleOutput(:,:,:,4) = minInput + p2 + zeros(size(input),'int32');
    output = min(possibleOutput,[],4);
    output = output - minInput; %normalize messages
end
