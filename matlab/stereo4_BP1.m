% Stereo Matching using Belief Propagation (Directional) - smoothness cost function approach
% Computes a disparity map from a rectified stereo pair using Belief Propagation (Directional)

function stereo4_BP1()

    % Set parameters
    dispLevels = 16; %disparity range: 0 to dispLevels-1
    lambda = 10; %weight of smoothness cost
    trunc = 2; %truncation of smoothness cost
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
        rightImgShifted = shiftRight(rightImg,d,0);
        matchingCosts(:,:,d+1) = computeMatchingCost(leftImg,rightImgShifted);
    end

    % Compute smoothness costs
    d = 0:dispLevels-1;
    smoothnessCosts = computeSmoothnessCost(d,d.');
    smoothnessCosts = permute(int32(smoothnessCosts),[3 4 1 2]);

    % Initialize messages for the 4 directions
    fromLeft = zeros(rows,cols,dispLevels,'int32');
    fromRight = zeros(rows,cols,dispLevels,'int32');
    fromUp = zeros(rows,cols,dispLevels,'int32');
    fromDown = zeros(rows,cols,dispLevels,'int32');

    figure
    for it = 1:iterations
        % Left to right pass (horizontal forward) - Send messages right
        for x = 1:cols-1
            currentCosts = matchingCosts(:,x,:) + fromLeft(:,x,:) + fromUp(:,x,:) + fromDown(:,x,:);
            fromLeft(:,x+1,:) = computeDirectionalCosts(currentCosts,smoothnessCosts);
        end

        % Right to left pass (horizontal backward) - Send messages left
        for x = cols:-1:2
            currentCosts = matchingCosts(:,x,:) + fromRight(:,x,:) + fromUp(:,x,:) + fromDown(:,x,:);
            fromRight(:,x-1,:) = computeDirectionalCosts(currentCosts,smoothnessCosts);
        end

        % Up to down pass (vertical forward) - Send messages down
        for y = 1:rows-1
            currentCosts = matchingCosts(y,:,:) + fromUp(y,:,:) + fromLeft(y,:,:) + fromRight(y,:,:);
            fromUp(y+1,:,:) = computeDirectionalCosts(currentCosts,smoothnessCosts);
        end

        % Down to up pass (vertical backward) - Send messages up
        for y = rows:-1:2
            currentCosts = matchingCosts(y,:,:) + fromDown(y,:,:) + fromLeft(y,:,:) + fromRight(y,:,:);
            fromDown(y-1,:,:) = computeDirectionalCosts(currentCosts,smoothnessCosts);
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
end

% Compute messages
% ----------------
function output = computeDirectionalCosts(currentCosts,smoothnessCosts)
    sum = currentCosts + smoothnessCosts;
    output = permute(min(sum,[],3),[1 2 4 3]);
    output = output - min(output,[],3); %normalize
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
