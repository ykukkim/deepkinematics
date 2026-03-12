function [output] = rem_outliers_2mad(data)

output = data;

median_data = median(output);
mad2_data = 2*mad(output,1);

output(output<(median_data-mad2_data) | output>(median_data+mad2_data)) = [];

end