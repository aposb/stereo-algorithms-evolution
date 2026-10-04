function [costs,transitions] = computeCostsAndTransitions_BpLike(currentCosts,smoothnessCosts)
    %computeCostsAndTransitions_BpLike Compute minimum cost paths (messages) and transitions - Approach A (BP-Like).
    sum = currentCosts + smoothnessCosts;
    [minsum,ind] = min(sum,[],3);
    costs = permute(minsum,[1 2 4 3]);
    costs = costs - min(costs,[],3); %normalize
    transitions = int32(permute(ind,[1 2 4 3]));
end
