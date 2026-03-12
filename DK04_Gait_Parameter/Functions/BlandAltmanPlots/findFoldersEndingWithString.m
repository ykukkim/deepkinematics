function folders = findFoldersEndingWithString(baseDir, endingStr)
    % Function to find all folders within baseDir that end with endingStr

    % Get a list of all files and folders in the base directory
    dirInfo = dir(baseDir);

    % Filter out non-folder entries and folders that do not end with endingStr
    isFolder = [dirInfo.isdir];
    folderNames = {dirInfo(isFolder).name};
    matchingFolders = folderNames(endsWith(folderNames, endingStr));

    % Remove '.' and '..' from the list
    matchingFolders = matchingFolders(~ismember(matchingFolders, {'.', '..'}));

    % Prepend baseDir to get the full paths of the matching folders
    folders = fullfile(baseDir, matchingFolders);
end