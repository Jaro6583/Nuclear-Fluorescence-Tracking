mainpth = '~/Documents/Projects/ImageAnalysis/FY Datasets/Paper/Monopolar/16-10-17 (6D dynamics random with Piezo)';

% Find all nd2 files
fileList = dir([mainpth, filesep,'*.nd2']);
mov_names = {fileList.name};

% for each nd2 file, find cells, and then launch getAmplitudeAtCellLocations
for jm = 1 : length(mov_names)
    mov_name = mov_names{jm};
    
    % get cell names
    [ ~, onlyName, ~] = fileparts( mov_name);
    cellList = dir([mainpth, filesep, onlyName, '_*.mat']);
    cellList = {cellList.name};
    cellNames = {};
    for jc = 1 : length(cellList)
        if ~endsWith(cellList{jc}, 'amp.mat')
            cellNames = {cellNames{:}, [mainpth,filesep, cellList{jc}]};
        end
    end
    
    fprintf('Working on file %s\n', onlyName);
    getAmplitudeAtCellLocations([mainpth,filesep,mov_name], cellNames);
end