function B = shiftDown(A,n,fillValue)
    %shiftDown Shift an array DOWN with specific padding.
    %   B = shiftDown(A,n,fillValue) shifts the elements in the array A
    %   from up to down by n positions and fills the empy places with
    %   new fillValue elements.
    B = circshift(A,n,1);
    B(1:min(n,end),:,:) = fillValue;
end
