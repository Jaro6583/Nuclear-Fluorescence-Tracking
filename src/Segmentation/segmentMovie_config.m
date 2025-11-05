function config = segmentMovie_config()
    
    config.cellsToSegment = 'prompt'; % 'prompt' or 'all'
    config.channelForSegmentation = 2;
    config.segmentationRoutine = 'segmentCellsBySpindleBrightness';
    config.cellSize = 150;
    config.manualSeg = 1;

end
