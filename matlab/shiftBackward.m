function B = shiftBackward(A,n,fillValue)
    %shiftBackward Shift an array BACKWARD with specific padding.
    %   B = shiftBackward(A,n,fillValue) shifts the elements in the array A
    %   from forward to backward by n positions and fills the empy places
    %   with new fillValue elements.
    B = circshift(A,-n,3);
    B(:,:,max(end-n+1,1):end) = fillValue;
end
