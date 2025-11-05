function dirName = createSaveDirectory( mainPath, expTag, newTime)
%  MainDir : this is where all results/output will be pushed to, e.g. 'results'
%  expTag : In Main Dir, there will be a directory for each experimental tag that is performed
%  expTag will contain a timestamp folder for the moment this software was run.
%  Inside the time folder, will be a cell folder and a frame folder which will contain any cell-frame specific information.
%  if newTiem = 1, the time directory will be created, otherwise the latest time dir will be used

    if ~exist( mainPath)
        mkdir( mainPath);
    end

    [~, expTag, ~] = fileparts( expTag);
    expDir = [ mainPath, filesep, expTag];                                               
    if ~exist( expDir)
        mkdir( expDir)
    end
    if newTime==1
        currTime = datestr( now, 'yymmdd_HHMM');
        timeDir = [ expDir, filesep, currTime];
        mkdir( timeDir);
    else
        dirs = dir( [ expDir, filesep, '*_*']);
        datenums = [dirs.datenum];
        [~, idx] = min( abs( datenums - now) );
        timeDir = [ expDir, filesep, dirs(idx).name ];
    end
    dirName = timeDir;

end
