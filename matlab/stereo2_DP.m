% Stereo Matching using Dynamic Programming - smoothness cost function approach
% Computes a disparity map from a rectified stereo pair using Dynamic Programming

function stereo2_DP()

    % Set parameters
    dispLevels = 16; %disparity range: 0 to dispLevels-1
    lambda = 10; %weight of smoothness cost
    trunc = 2; %truncation of smoothness cost

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
        rightImgShifted = shiftRight(rightImg,d,0);
        matchingCosts(:,:,d+1) = computeMatchingCost(leftImg,rightImgShifted);
    end

    % Compute smoothness costs
    d = 0:dispLevels-1;
    smoothnessCosts = computeSmoothnessCost(d,d.');
    smoothnessCosts = permute(int32(smoothnessCosts),[3 4 1 2]);

    % Initialize minimum cost paths and transitions for the left to right direction
    fromLeft = zeros(rows,cols,dispLevels,'int32');
    transitions = zeros(rows,cols,dispLevels,'int32');

    % Compute minimum cost paths and transitions for left to right direction
    for x = 1:cols-1
        currentCosts = matchingCosts(:,x,:) + fromLeft(:,x,:);
        [C,T] = computeDirectionalCosts(currentCosts,smoothnessCosts);
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
    imwrite(dispImg,'disparity2_DP.png')
end

% Compute minimum cost paths and transitions
% ------------------------------------------
function [output,transitions] = computeDirectionalCosts(currentCosts,smoothnessCosts)
    sum = currentCosts + smoothnessCosts;
    output = permute(min(sum,[],3),[1 2 4 3]);
    output = output - min(output,[],3); %normalize
    [~,ind] = min(sum,[],3);
    transitions = int32(permute(ind,[1 2 4 3]));
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
