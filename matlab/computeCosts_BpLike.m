function costs = computeCosts_BpLike(currentCosts,smoothnessCosts)
    %computeCosts_BpLike Compute minimum cost paths (messages) - Approach A (BP-Like).
    sum = currentCosts + smoothnessCosts;
    minsum = min(sum,[],3);
    costs = permute(minsum,[1 2 4 3]);
    costs = costs - min(costs,[],3); %normalize
end
