function [VD, cNaNmarkerTrials, cNaNmarkerTrialsnames] =cut_markers(VD, participant, filename, ...
    cNaNmarkerTrials, cNaNmarkerTrialsnames )

    %Cut all data, that no NAN exists in heel and toe marker 
    %import VD
    
    
    %make a matrix of the important markers, include NAN values
%     markerlist={'LTHL','RTHL','LTFB','RTFB'};
%     matrix=[VD.LTHL.Values.z_coord, VD.RTHL.Values.z_coord,...
%         VD.LTFB.Values.z_coord,VD.RTFB.Values.z_coord ];
    

    
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% fill NaN values in z-coord   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% (written by SN)

matrix=[VD.LTHL.Values.z_coord, VD.RTHL.Values.z_coord,...
         VD.LTFB.Values.z_coord,VD.RTFB.Values.z_coord ];
% subplot(2,1,1)    
% plot(VD.RTFB.Values.z_coord, 'g')
% hold on     
B = matrix;
newB = B;

%%% to find the positions of the values next to the NaNs
% finds the positions of the NaN values
C=find(isnan(B));
% numel = number of array elements
% setdiff = set difference of two arrays
% finds positions in which values exist
D=setdiff(1:numel(B(:)),C);
% finds the previous and the next values to the NaNs
% E = values of D before/after the ith NaN value
% prev selects the last values of the list of values before the NaNs
% next selects the first values of the list of values after the NaNs
for i=1:size(C,1)
   E=D(D<C(i));
   prev(i,1)=E(end);
   E=D(D>C(i));
   disp(i);
   next(i,1)=E(1);
end

%%% creates a list of previous and next values and inserts avg values
% prev creates a list of the positions of all previous values with 0 in between
% next fills the gaps (0) with the position of "next" values
all(1:2:2*length(prev))=prev;
all(2:2:2*length(prev))=next;
fun=@(block_struct) mean(block_struct.data);
% B(all) puts in the values for each position
prevNextValues=B(all);
newB(C) = blockproc(prevNextValues,[1 2],fun);
matrix = newB;

VD.LTHL.Values.z_coord=matrix(:,1);
VD.RTHL.Values.z_coord=matrix(:,2);
VD.LTFB.Values.z_coord=matrix(:,3);
VD.RTFB.Values.z_coord=matrix(:,4);

% subplot(2,1,2)
% plot(VD.RTFB.Values.z_coord, 'b')

%%%%%%%%%%%%%%%%%%%%%
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% fill NaN values in y-coord   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

matrix=[VD.LTHL.Values.y_coord, VD.RTHL.Values.y_coord,...
        VD.LTFB.Values.y_coord,VD.RTFB.Values.y_coord ];
    
B = matrix;
newB = B;

%%% to find the positions of the values next to the NaNs
% finds the positions of the NaN values
C=find(isnan(B));
% numel = number of array elements
% setdiff = set difference of two arrays
% finds positions in which values exist
D=setdiff(1:numel(B(:)),C);
% finds the previous and the next values to the NaNs
% E = values of D before/after the ith NaN value
% prev selects the last values of the list of values before the NaNs
% next selects the first values of the list of values after the NaNs
for i=1:size(C,1)
   E=D(D<C(i));
   prev(i,1)=E(end);
   E=D(D>C(i));
   next(i,1)=E(1);
end

%%% creates a list of previous and next values and inserts avg values
% prev creates a list of the positions of all previous values with 0 in between
% next fills the gaps (0) with the position of "next" values
all(1:2:2*length(prev))=prev;
all(2:2:2*length(prev))=next;
fun=@(block_struct) mean(block_struct.data);
% B(all) puts in the values for each position
prevNextValues=B(all);
newB(C) = blockproc(prevNextValues,[1 2],fun);
matrix = newB;

VD.LTHL.Values.y_coord=matrix(:,1);
VD.RTHL.Values.y_coord=matrix(:,2);
VD.LTFB.Values.y_coord=matrix(:,3);
VD.RTFB.Values.y_coord=matrix(:,4);
%%%%%%%%%%%%%%%%%%%%%
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% fill NaN values in x-coord   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

matrix=[VD.LTHL.Values.x_coord, VD.RTHL.Values.x_coord,...
        VD.LTFB.Values.x_coord,VD.RTFB.Values.x_coord ];
    
B = matrix;
newB = B;

%%% to find the positions of the values next to the NaNs
% finds the positions of the NaN values
C=find(isnan(B));
% numel = number of array elements
% setdiff = set difference of two arrays
% finds positions in which values exist
D=setdiff(1:numel(B(:)),C);
% finds the previous and the next values to the NaNs
% E = values of D before/after the ith NaN value
% prev selects the last values of the list of values before the NaNs
% next selects the first values of the list of values after the NaNs
for i=1:size(C,1)
   E=D(D<C(i));
   prev(i,1)=E(end);
   E=D(D>C(i));
   next(i,1)=E(1);
end

%%% creates a list of previous and next values and inserts avg values
% prev creates a list of the positions of all previous values with 0 in between
% next fills the gaps (0) with the position of "next" values
all(1:2:2*length(prev))=prev;
all(2:2:2*length(prev))=next;
fun=@(block_struct) mean(block_struct.data);
% B(all) puts in the values for each position
prevNextValues=B(all);
newB(C) = blockproc(prevNextValues,[1 2],fun);
matrix = newB;

VD.LTHL.Values.x_coord=matrix(:,1);
VD.RTHL.Values.x_coord=matrix(:,2);
VD.LTFB.Values.x_coord=matrix(:,3);
VD.RTFB.Values.x_coord=matrix(:,4);
%%%%%%%%%%%%%%%%%%%%%
% %%
%     
%     dataRel=not(isnan(matrix));
%     %%% Extract data. Only not NaN values are of interest
%     % Data relevant for following computations is the data of the largest
%     % time intervall of the data without NaN values for the relevant
%     % markers
%     % If markerlist is not empty assertion doesn't fail
%     % If markerlist is empty (one of the markers is missing), assertion fails
%     assert(not(isempty(markerlist)));
% %     eval(['d = size(' markersRelListFull{1} ');']);
% %     m = length(markersRelListFull);
% %     dataRel = nan(d .* [1 m]);
% %     for jj = 1:m
% %         eval(['dataRel(:, (jj-1)*d(2)+1:jj*d(2)) = ' ...
% %             markersRelListFull{jj} ';']);
% %     end
% % %     dataRel = [LHEEL RHEEL LTOE RTOE];
% %     
% 
%     
%     dataRel1=(not(isnan(matrix)));
%     start=[];
%     stop=[];
% %     cNaNmarkerTrials={};
% %     cNaNmarkerTrialsnames   = {};
%     
%     for i=1:length(markerlist);
%         dataRel=dataRel1(:,i);
%     % If in every frame some of the four markers (2x heel + 2x toe) is nan
%     if all(not(dataRel))
%         % Set all the markers to NaN
% %         for jj = 1:length(markersListFull)
% %             eval([markersListFull{jj} ...
% %                 ' = nan(size(' markersListFull{jj} '));']);
% %         end
% %%%cNaNmarkerTrials???
%         
%         cNaNmarkerTrials = cNaNmarkerTrials + 1;
%         cNaNmarkerTrialsnames(cNaNmarkerTrials, 1:2) = ...
%             {participant,  filename}; %#ok<AGROW>
%         continue;
%     else % Find the largest interval where all the relevant markers
%          % are valid in every frame
%         
%         fstindx = find(dataRel, 1, 'first');
%         lstindx = find(dataRel, 1, 'last');
%         
%         % Finds NaN-values in the matrix
%         indxofnans = find(not(dataRel));
%         % checks whether NaN-value is between first and last data point
%         indxofnans = indxofnans(fstindx <= indxofnans ...
%             & indxofnans <= lstindx);
%         % defines a column containing the locations of the first value, the NaN-values and
%         % the last value
%         indxofnans = [fstindx-1; indxofnans; lstindx+1];
%         % calculates the differences between the locations of the above
%         % mentioned values = calculates the intervals between the gaps
%         l = indxofnans(2:end) - indxofnans(1:end-1);
% 
%         % Returns the largest interval without gaps and its starting
%         % position index
%         [tmp, indx] = max(l);
%         % cuts at the start of the largest interval
%         startcut = indxofnans(indx)     + 1;
%         % cuts at the end of the largest interval
%         finshcut = indxofnans(indx+1)   - 1;
%         % checks whether the interval is longer than 100 data points
%         %%%cNaNmarkerTrials??? if the file contains less then 100 data
%         %%%points the file is skipped
%         if finshcut - startcut < 100
%             cNaNmarkerTrials = cNaNmarkerTrials + 1;
%             cNaNmarkerTrialsnames(cNaNmarkerTrials, 1:2) = ...
%                 {participant, filename}; %#ok<AGROW>
%             continue;
%         end
%         % defines start and stop of the interval
%         start=[start; startcut];
%         stop=[stop;finshcut];
%         % Cut all the data from to the intervall [startcut, finshcut]
%         %cutVars = union(markersListFull,{'frames', 'data.trajectories.samples'});
% %         for jj = 1:length(cutVars)
% %             eval([cutVars{jj} ' = ' cutVars{jj} '(startcut:finshcut, :);']);
% %         end
%     end
%     end
%     %%% what's the sense of this?
%     startcut=max(start);
%     finshcut=min(stop);
%     
%     % Cut all the data from to the intervall [startcut, finshcut]
%     cutVars=fieldnames(VD);
%     for i=1:33 %only cut for the markers
%         % converts the marker list into a character array (string/ column)
%         marker=char(cutVars(i));
%         % cuts the data according to the previously defined start- and finishcut
%         VD.(genvarname(marker)).Values.x_coord=VD.(genvarname(marker)).Values.x_coord(startcut:finshcut);
%         VD.(genvarname(marker)).Values.y_coord=VD.(genvarname(marker)).Values.y_coord(startcut:finshcut);
%         VD.(genvarname(marker)).Values.z_coord=VD.(genvarname(marker)).Values.z_coord(startcut:finshcut);
%         VD.(genvarname(marker)).Values.resid=VD.(genvarname(marker)).Values.resid(startcut:finshcut);
%     end
%     
% return
% end

    
