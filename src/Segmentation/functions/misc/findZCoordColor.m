function cl = findZCoordColor( currValue, startValue, endValue, cmap)

if nargin < 4
    cmap == 'jet';
end

n = 100;
currIdx = 1+ round( ( (currValue-1)/(endValue-startValue) )*n);
currIdx( currIdx < 1) = 1; currIdx( currIdx > n) = n;

eval( [ 'cdat = [uint8(' cmap '(' num2str(n) ')*255) uint8(ones(' num2str(n) ',1))].'';'] );

cl = cdat(:, currIdx);

end
