% Stereo Matching using Block Matching
% Computes a disparity map from a rectified stereo pair using Block Matching

function stereo1_BM()

    % Set parameters
    dispLevels = 16; %disparity range: 0 to dispLevels-1
    windowSize = 3;

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

    % Aggregate the matching costs
    totalCosts = int32(convn(matchingCosts,ones(windowSize,windowSize,1),'same'));

    % Compute the disparity map
    [~,ind] = min(totalCosts,[],3);
    dispMap = ind-1;

    % Normalize the disparity map for display
    scaleFactor = 256/dispLevels;
    dispImg = uint8(dispMap*scaleFactor);

    % Show disparity map
    figure; imshow(dispImg)

    % Save disparity map
    imwrite(dispImg,'disparity1_BM.png')
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
