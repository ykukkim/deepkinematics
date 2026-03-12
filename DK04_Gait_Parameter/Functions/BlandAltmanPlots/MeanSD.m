function [row]=MeanSD(v)

Mean=mean(abs(v));

row(1,1)=mean(v);
row(1,2)=std(v);
row(1,3)=min(v);
row(1,4)=max(v);

end