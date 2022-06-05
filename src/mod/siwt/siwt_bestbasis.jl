function bestbasistree!(siwtObj::ShiftInvariantWaveletTransformObject{N,T₁,T₂}) where
                       {N, T₁<:Integer, T₂<:AbstractFloat}
    rootNodeIndex = (0,0,0)
    bestbasis_treeselection!(siwtObj, rootNodeIndex)
    siwtObj.MinCost = siwtObj.Nodes[rootNodeIndex].Cost
    return siwtObj.BestTree
end

function bestbasis_treeselection!(siwtObj::ShiftInvariantWaveletTransformObject{N,T₁,T₂},
                                  index::NTuple{3,T₁}) where
                                 {N, T₁<:Integer, T₂<:AbstractFloat}
    # Base case: Check if tree contains desired index
    if index ∉ siwtObj.BestTree
        return nothing
    end

    # Get the cost of current node and its children nodes
    nodeDepth, nodeIndexAtDepth, nodeTransformShift = index
    child1Index = (nodeDepth+1, nodeIndexAtDepth<<1, nodeTransformShift)
    child2Index = (nodeDepth+1, nodeIndexAtDepth<<1+1, nodeTransformShift)
    shiftedChild1Index = (nodeDepth+1, nodeIndexAtDepth<<1, nodeTransformShift+(1<<nodeDepth))
    shiftedChild2Index = (nodeDepth+1, nodeIndexAtDepth<<1+1, nodeTransformShift+(1<<nodeDepth))
    
    nodeCost = siwtObj.Nodes[index].Cost
    child1Cost = bestbasis_treeselection!(siwtObj, child1Index)
    child2Cost = bestbasis_treeselection!(siwtObj, child2Index)
    shiftedChild1Cost = bestbasis_treeselection!(siwtObj, shiftedChild1Index)
    shiftedChild2Cost = bestbasis_treeselection!(siwtObj, shiftedChild2Index)

    nonShiftedChildrenCost = (all ∘ broadcast)(isnothing, [child1Cost, child2Cost]) ? nothing : child1Cost+child2Cost
    shiftedChildrenCost = (all ∘ broadcast)(isnothing, [shiftedChild1Cost, shiftedChild2Cost]) ? nothing : shiftedChild1Cost+shiftedChild2Cost

    # Compare costs and delete node(s)
    hasNonShiftedChildren = !isnothing(nonShiftedChildrenCost)
    hasShiftedChildren = !isnothing(shiftedChildrenCost)
    hasChildren = hasNonShiftedChildren && hasShiftedChildren
    hasNoChildren = !hasNonShiftedChildren && !hasShiftedChildren

    isNodeCostLessThanNonShiftedChildrenCost = hasNonShiftedChildren && (nodeCost < nonShiftedChildrenCost)
    isNodeCostLessThanShiftedChildrenCost = hasShiftedChildren && (nodeCost < shiftedChildrenCost)
    isNonShiftedChildrenCostLessThanShiftedChildrenCost = (hasNonShiftedChildren && !hasShiftedChildren) || (hasChildren && nonShiftedChildrenCost<shiftedChildrenCost)

    isNodeCostMinimum = hasNoChildren || (isNodeCostLessThanNonShiftedChildrenCost && isNodeCostLessThanShiftedChildrenCost)
    isNonShiftedChildrenCostMinimum = !isNodeCostMinimum && isNonShiftedChildrenCostLessThanShiftedChildrenCost
    
    if isNodeCostMinimum
        delete_node!(siwtObj, child1Index)
        delete_node!(siwtObj, child2Index)
        delete_node!(siwtObj, shiftedChild1Index)
        delete_node!(siwtObj, shiftedChild2Index)
    elseif isNonShiftedChildrenCostMinimum
        delete_node!(siwtObj, shiftedChild1Index)
        delete_node!(siwtObj, shiftedChild2Index)
        siwtObj.Nodes[index].Cost = nonShiftedChildrenCost
    else    # Shifted children cost is minimum
        delete_node!(siwtObj, child1Index)
        delete_node!(siwtObj, child2Index)
        siwtObj.Nodes[index].Cost = shiftedChildrenCost
    end

    return siwtObj.Nodes[index].Cost
end

function delete_node!(siwtObj::ShiftInvariantWaveletTransformObject{N,T₁,T₂},
                      index::NTuple{3,T₁}) where
                     {N, T₁<:Integer, T₂<:AbstractFloat}
    # Node unavailable
    if index ∉ siwtObj.BestTree
        return nothing
    end

    # Delete node
    delete!(siwtObj.Nodes, index)
    deleteat!(siwtObj.BestTree, findall(x -> x==index, siwtObj.BestTree))

    # Delete children nodes
    nodeDepth, nodeIndexAtDepth, nodeTransformShift = index
    child1Index = (nodeDepth+1, nodeIndexAtDepth<<1, nodeTransformShift)
    child2Index = (nodeDepth+1, nodeIndexAtDepth<<1+1, nodeTransformShift)
    shiftedChild1Index = (nodeDepth+1, nodeIndexAtDepth<<1, nodeTransformShift+(1<<nodeDepth))
    shiftedChild2Index = (nodeDepth+1, nodeIndexAtDepth<<1+1, nodeTransformShift+(1<<nodeDepth))
    delete_node!(siwtObj, child1Index)
    delete_node!(siwtObj, child2Index)
    delete_node!(siwtObj, shiftedChild1Index)
    delete_node!(siwtObj, shiftedChild2Index)
    
    return nothing
end