function costs = computeCosts_SgmLike(currentCosts,occPenalties)
    %computeCosts_SgmLike Compute minimum cost paths (messages) - Approach B (SGM-Like).
    minInput = min(currentCosts,[],3);
    currentCostsP1 = currentCosts + occPenalties(1);
    possibleOutput = zeros([size(currentCosts),4],'int32');
    possibleOutput(:,:,:,1) = currentCosts;
    possibleOutput(:,:,:,2) = shiftForward(currentCostsP1,1,intmax);
    possibleOutput(:,:,:,3) = shiftBackward(currentCostsP1,1,intmax);
    possibleOutput(:,:,:,4) = minInput + occPenalties(2) + zeros(size(currentCosts),'int32');
    costs = min(possibleOutput,[],4);
    costs = costs - minInput; %normalize
end
