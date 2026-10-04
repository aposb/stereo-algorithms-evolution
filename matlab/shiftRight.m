function B = shiftRight(A,n,fillValue)
    %shiftRight Shift an array RIGHT with specific padding.
    %   B = shiftRight(A,n,fillValue) shifts the elements in the array A
    %   from left to right by n positions and fills the empy places with
    %   new fillValue elements.
    B = circshift(A,n,2);
    B(:,1:min(n,end),:) = fillValue;
end
