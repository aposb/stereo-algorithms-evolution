function [costs,transitions] = computeCostsAndTransitions_SgmLike(currentCosts,occPenalties)
    %computeCostsAndTransitions_SgmLike Compute minimum cost paths (messages) and transitions - Approach B (SGM-Like).
    [minInput,ind0] = min(currentCosts,[],3);
    currentCostsP1 = currentCosts + occPenalties(1);
    possibleOutput = zeros([size(currentCosts),4],'int32');
    possibleOutput(:,:,:,1) = currentCosts;
    possibleOutput(:,:,:,2) = shiftForward(currentCostsP1,1,intmax);
    possibleOutput(:,:,:,3) = shiftBackward(currentCostsP1,1,intmax);
    possibleOutput(:,:,:,4) = minInput + occPenalties(2) + zeros(size(currentCosts),'int32');
    [costs,ind] = min(possibleOutput,[],4);
    costs = costs - minInput; %normalize
    match = permute(int32(1:size(currentCosts,3)),[3 1 2]) + zeros(size(currentCosts),'int32');
    near1 = match-1; near2 = match+1;
    far = int32(ind0) + zeros(size(currentCosts),'int32');
    transitions = zeros(size(currentCosts),'int32');
    transitions(ind==1) = match(ind==1);
    transitions(ind==2) = near1(ind==2);
    transitions(ind==3) = near2(ind==3);
    transitions(ind==4) = far(ind==4);
end
