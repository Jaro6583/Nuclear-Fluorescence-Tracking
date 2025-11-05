function fontSize = determineFontSize()
%determineFontSize: Automatically determines the font size ideal for a
% screen by pulling in the screen resolution

pos = get(0, 'ScreenSize');

screenResolution = pos(3)*pos(4);

if screenResolution < (1.0)*10^6
    fontSize = 12;
elseif screenResolution < (1.2)*10^6
    fontSize = 14;
elseif screenResolution < (2.0)*10^6
    fontSize = 16;
elseif screenResolution < (3.0)*10^6
    fontSize = 18;
else
    fontSize = 20;
end

end

