function B = shiftUp(A,n,fillValue)
    %shiftUp Shift an array UP with specific padding.
    %   B = shiftUp(A,n,fillValue) shifts the elements in the array A
    %   from down to up by n positions and fills the empy places with
    %   new fillValue elements.
    B = circshift(A,-n,1);
    B(max(end-n+1,1):end,:,:) = fillValue;
end
