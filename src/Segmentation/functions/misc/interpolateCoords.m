function [coordsNewX, coordsNewY, coordsNew] = interpolateCoords( coordX, coordY, numIntervals)

if any( size(coordX) ~= size(coordY) ), error( 'input size of coords is different'), end

nC = length( coordX);

coordsNewX = [];
coordsNewY = [];

for jC = 1 : nC-1
    cX = linspace( coordX(jC), coordX(jC+1), numIntervals);
    cY = linspace( coordY(jC), coordY(jC+1), numIntervals);
    coordsNewX = [ coordsNewX, cX(1:end-1)];
    coordsNewY = [ coordsNewY, cY(1:end-1)];
    if jC == nC-1
        coordsNewX = [coordsNewX, cX(end)]; coordsNewY = [coordsNewY, cY(end)];
    end

end

if isempty(coordsNewX)
    coordsNewX = coordX;
    coordsNewY = coordY;
end

coordsNew = [coordsNewX ; coordsNewY];

end
