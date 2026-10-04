function B = shiftForward(A,n,fillValue)
    %shiftForward Shift an array FORWARD with specific padding.
    %   B = shiftForward(A,n,fillValue) shifts the elements in the array A
    %   from backward to forward by n positions and fills the empy places
    %   with new fillValue elements.
    B = circshift(A,n,3);
    B(:,:,1:min(n,end)) = fillValue;
end
