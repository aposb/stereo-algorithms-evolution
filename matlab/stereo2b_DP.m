% Stereo Matching using Dynamic Programming - occlusion penalties approach
% Computes a disparity map from a rectified stereo pair using Dynamic Programming

function stereo2b_DP()

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
        [C,T] = computeDirectionalCosts(currentCosts,[p1 p2]);
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
end

% Compute minimum cost paths and transitions
% ------------------------------------------
function [output,transitions] = computeDirectionalCosts(currentCosts,occPenalties)
    [minInput,ind0] = min(currentCosts,[],3);
    currentCostsP1 = currentCosts + occPenalties(1);
    possibleOutput = zeros([size(currentCosts),4],'int32');
    possibleOutput(:,:,:,1) = currentCosts;
    possibleOutput(:,:,:,2) = shiftForward(currentCostsP1,1,intmax);
    possibleOutput(:,:,:,3) = shiftBackward(currentCostsP1,1,intmax);
    possibleOutput(:,:,:,4) = minInput + occPenalties(2) + zeros(size(currentCosts),'int32');
    [output,ind] = min(possibleOutput,[],4);
    output = output - minInput; %normalize
    match = permute(int32(1:size(currentCosts,3)),[3 1 2]) + zeros(size(currentCosts),'int32');
    near1 = match-1; near2 = match+1;
    far = int32(ind0) + zeros(size(currentCosts),'int32');
    transitions = zeros(size(currentCosts),'int32');
    transitions(ind==1) = match(ind==1);
    transitions(ind==2) = near1(ind==2);
    transitions(ind==3) = near2(ind==3);
    transitions(ind==4) = far(ind==4);
end

% Shift Functions (Down/Up/Right/Left/Forward/Backward)
% -----------------------------------------------------
function B = shiftDown(A,n,fillValue)
    B = circshift(A,n,1);
    B(1:min(n,end),:,:) = fillValue;
end
function B = shiftUp(A,n,fillValue)
    B = circshift(A,-n,1);
    B(max(end-n+1,1):end,:,:) = fillValue;
end
function B = shiftRight(A,n,fillValue)
    B = circshift(A,n,2);
    B(:,1:min(n,end),:) = fillValue;
end
function B = shiftLeft(A,n,fillValue)
    B = circshift(A,-n,2);
    B(:,max(end-n+1,1):end,:) = fillValue;
end
function B = shiftForward(A,n,fillValue)
    B = circshift(A,n,3);
    B(:,:,1:min(n,end)) = fillValue;
end
function B = shiftBackward(A,n,fillValue)
    B = circshift(A,-n,3);
    B(:,:,max(end-n+1,1):end) = fillValue;
end
