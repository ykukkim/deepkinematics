% create an example array
A = randi(100,10,5);
% plot(A, 'r')
% hold on

% criteria to set NaN values in the example array
A(A < 30) = NaN;
B = A;
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
B
newB(C) = blockproc(prevNextValues,[1 2],fun);
% plot(newB)