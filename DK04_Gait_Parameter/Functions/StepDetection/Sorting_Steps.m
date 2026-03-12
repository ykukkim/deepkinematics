function [LHS,RTO,RHS,LTO] = SortingStep_YK(HSleftlocs,TOrightlocs,HSrightlocs,TOleftlocs)

%% Final Step Check
% Format the data. Start with LHS, RTO, RHS ,LTO

LHS = HSleftlocs';
RTO = TOrightlocs';
RHS = HSrightlocs';
LTO = TOleftlocs';

% Calculate the minimum length among A, B, C, and D
min_length = min([length(LHS), length(RTO), length(RHS), length(LTO)]);

% Truncate A, B, C, and D to the minimum length
LHS = LHS(1:min_length);
RTO = RTO(1:min_length);
RHS = RHS(1:min_length);
LTO = LTO(1:min_length);

time_limit = 1.7 * 200;

% Combine and sort the timestamps along with their labels
all_events = [LHS', repmat(double('A'), min_length, 1); ...
    RTO', repmat(double('B'), min_length, 1); ...
    RHS', repmat(double('C'), min_length, 1); ...
    LTO', repmat(double('D'), min_length, 1)];

all_events = sortrows(all_events);

% Initialize containers to store sorted timestamps
sorted_A = [];
sorted_B = [];
sorted_C = [];
sorted_D = [];

% Loop through sorted events
% Initialize variables to keep track of the start and members of each cycle
cycle_members = '';
initial_A_time = NaN;
first_LHS = find(all_events(:,2) == double('A'),1);
if first_LHS ~= 1
    all_events(1:first_LHS,:) = [];
end
% Loop through sorted events
for i = 1:size(all_events, 1)
    event_time = all_events(i, 1);
    event_label = char(all_events(i, 2));

    % Reset the cycle if an unexpected event is encountered
    if ~isempty(cycle_members) && ~contains('ABCD', event_label)
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
            cycle_members = [cycle_members, 'C'];
            sorted_C = [sorted_C; event_time];
        elseif contains(cycle_members, 'C') && event_label == 'D'
            % Check if the length between 'A' and 'D' is within the set limit
            if abs(event_time - initial_A_time) <= time_limit
                cycle_members = [cycle_members, 'D'];
                sorted_D = [sorted_D; event_time];

                % Reset cycle_members if all four events have been added in a cycle
                if length(cycle_members) == 4
                    cycle_members = '';
                    initial_A_time = NaN;
                end
            else
                % Reset the cycle if the time limit is exceeded
                cycle_members = '';
                initial_A_time = NaN;
                sorted_A(end) = [];
                sorted_B(end) = [];
                sorted_C(end) = [];
            end
        end
    end
end

LHS = sorted_A';
RTO = sorted_B';
RHS = sorted_C';
LTO = sorted_D';
end
