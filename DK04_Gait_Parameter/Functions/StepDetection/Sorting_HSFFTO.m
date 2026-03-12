function [New_HS,New_MS,New_TO] = Sorting_HSFFTO(HS,FF,TO)

% Calculate the minimum length among HS, MS, and TO
min_length = min([length(HS), length(FF), length(TO)]);

% Truncate HS, MS, and TOto the minimum length
HS = HS(1:min_length)';
FF = FF(1:min_length)';
TO = TO(1:min_length)';

time_limit = 1.5 * 200;

% Combine and sort the timestamps along with their labels
all_events = [HS, repmat(double('A'), min_length, 1); ...
    FF, repmat(double('B'),min_length,1);...
    TO, repmat(double('C'), min_length, 1)];

all_events = sortrows(all_events);

% Initialize containers to store sorted timestamps
sorted_A = [];
sorted_B = [];
sorted_C = [];

% Loop through sorted events
% Initialize variables to keep track of the start and members of each cycle
cycle_members = '';
initial_A_time = NaN;

if (all_events(1,2) == double('A')) ~= 1
    first_LHS = find(all_events(:,2) == double('A'),1);
    all_events(1:first_LHS,:) = [];
end

% Loop through sorted events
for i = 1:size(all_events, 1)
    event_time = all_events(i, 1);
    event_label = char(all_events(i, 2));

    % Reset the cycle if an unexpected event is encountered
    if ~isempty(cycle_members) && ~contains('ABC', event_label)
        cycle_members = '';
        initial_A_time = NaN;
    end

    % Check conditions for adding to sorted arrays based on the sequence A > B > C > D
    if isempty(cycle_members)
        if event_label == 'A'
            cycle_members = 'A';
            sorted_A = [sorted_A; event_time];
            initial_A_time = event_time;
        end
    else
        if contains(cycle_members, 'A') && event_label == 'B'
            cycle_members = [cycle_members, 'B'];
            sorted_B = [sorted_B; event_time];
        elseif contains(cycle_members, 'B') && event_label == 'C'
            if abs(event_time - initial_A_time) <= time_limit
                cycle_members = [cycle_members, 'C'];
                sorted_C = [sorted_C; event_time];

                % Reset cycle_members if all four events have been added in a cycle
                if length(cycle_members) == 3
                    cycle_members = '';
                    initial_A_time = NaN;
                end
            else
                % Reset the cycle if the time limit is exceeded
                cycle_members = '';
                initial_A_time = NaN;
                if ~isempty(sorted_A)
                    sorted_A(end) = [];
                end

                if ~isempty(sorted_B)
                    sorted_B(end) = [];
                end

                if ~isempty(sorted_C)
                    sorted_C(end) = [];
                end
            end
        end
    end

    New_HS = sorted_A;
    New_MS = sorted_B;
    New_TO = sorted_C;

end
end
