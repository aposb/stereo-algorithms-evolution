function B = shiftLeft(A,n,fillValue)
    %shiftLeft Shift an array LEFT with specific padding.
    %   B = shiftLeft(A,n,fillValue) shifts the elements in the array A
    %   from right to left by n positions and fills the empy places with
    %   new fillValue elements.
    B = circshift(A,-n,2);
    B(:,max(end-n+1,1):end,:) = fillValue;
end
