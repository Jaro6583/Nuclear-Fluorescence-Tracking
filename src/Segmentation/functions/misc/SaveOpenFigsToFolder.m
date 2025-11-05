function SaveOpenFigsToFolder( filePath, figFormat, closeAfterSave )
% SaveFigsToFolder : Gets all the figure handles and saves each figure with
% a name specified by its title in a user-specified format. The figures are
% saved in a specified location

    disp('---------------- Saving All Open Figures ---------------- \n')
    
    % check if folder exists in the provided FilePath. If not, then
    % create it
    if exist( filePath, 'dir') ~= 7
        mkdir( filePath)
    end
    % add this path to matlab
    addpath(filePath);
    
    % get all the figure handles
    FigHandles = findobj('Type', 'figure');
    
    % for each handle, access the figure name, set it to screensize, and
    % save it in a folder
    for jFig = 1 : length( FigHandles)
        h = FigHandles( jFig);
        figName = [ h.Name ];
        figPath = [ filePath, filesep, figName];
        
        saveas( h, figPath, figFormat);        
    end
    
    if closeAfterSave
        close all
    end
    
end

