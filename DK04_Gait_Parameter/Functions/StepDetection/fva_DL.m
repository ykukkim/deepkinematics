function [HSlocs, TOlocs,zfootVel] = fva_DL(zheel, ztoe, srate, varargin)

%% Preallocation
HSlocs   = [];
TOlocs   = [];
optargin = size(varargin, 2);
switch optargin
    case 0
        toprdfac  = 0.8;  % Factor for disgarding HS
        hsprdfac  = 0.08; % Factor for disgarding TO
        zrangefac = 0.35; % Heel marker z coordinate by the HS event should
        % be in the range
        % [min(z), min(z) + zrangefac*(max(z) - min(z))]
        forder    = 4;    % Order
        cutfreq   = 7;    % Cut-off frequency
    case 2
        forder      = varargin{1};
        cutfreq     = varargin{2};
    case 3
        toprdfac    = varargin{1};
        hsprdfac    = varargin{2};
        zrangefac   = varargin{3};
    case 5
        toprdfac    = varargin{1};
        hsprdfac    = varargin{2};
        zrangefac   = varargin{3};
        forder      = varargin{4};
        cutfreq     = varargin{5};
    otherwise
        error('Wrong input format.');
end

if length(zheel) < ceil(toprdfac * srate) + 1
    error(['Input singal is too short. Signal should at least ' ...
        num2str(toprdfac * srate + 1) ' samples long']);
    return;
end

%% Filtering of the signal: Butterworth filter with specified parameters
assert(all(not(isnan([zheel; ztoe]))));
[b,a] = butter(forder, cutfreq / srate);
zheel = filtfilt(b, a, zheel);
ztoe  = filtfilt(b, a, ztoe);
%% Estimated center of the foot and Velocity of Foot

zCoordfootcentre = 1/2 * (zheel + ztoe);
zfootVel = diff(zCoordfootcentre) ./ transpose(diff(1:length(zCoordfootcentre))); % Compute velocity of the foot

%% Determine TO events_Original

mpd = floor(toprdfac * srate);
[TOpks, TOlocs] = findpeaks(zfootVel,  'minpeakdistance', 140 * srate / 200);
TOlocs = TOlocs(TOpks > 0.5);         % x = TOlocs
TOpks = zfootVel(TOlocs);             % y = TOpks
% figure; plot(zfootVel);title('TOlocs, TOpks plotted on zfootVel');hold on;
% plot(TOlocs,TOpks, 'or'); hold on;
% figure; plot(ztoe,'r'); title('TOlocs, TOpks plotted on ztoe'); hold on;
% plot(TOlocs,ztoe(TOlocs),'ob');

%% Determine HS events_Original
mpd = floor(hsprdfac * srate);
[HSpks, HSlocs] = findpeaks(-zheel, 'MinPeakDistance',srate,'MinPeakHeight',-80); % the velocity values are multiplied by -1
% HSpks = zfootVel(HSlocs);           % Re-multiply the values with -1
% HSlocs = HSlocs(HSpks < -0.3 & HSpks > -4);      % HS occurs at negative velocity
% HSpks = zfootVel(HSlocs);           % y = HSpks x = HSlocs
% if min(zheel) <= 50
%     thre = abs(40+ (zrangefac * (abs(max(zheel)) - abs(33))));
%     thre_centre =abs(40+ (zrangefac * (abs(max(zCoordfootcentre)) - abs(40))));
% thre_Vel = -(-1.9 +(zrangefac*max(zfootVel)-(-1.9)));
% else
% thre = abs(min(zheel(10000:15000))+ (zrangefac * (abs(max(zheel(10000:15000))) - abs(min(zheel(10000:15000))))));
% thre_centre =abs(min(zCoordfootcentre(10000:15000))+ (zrangefac * (abs(max(zCoordfootcentre(10000:15000))) - abs(min(zCoordfootcentre(10000:15000))))));
% thre_Vel = -(min(zfootVel) +(zrangefac*max(zfootVel)-min(zfootVel)));% By HS z-component of the heel should be in the lower 20% of the range
% % end
% indx = zheel(HSlocs) <= thre & zCoordfootcentre(HSlocs) <= thre_centre & zfootVel(HSlocs)  <= - thre_Vel;
% HSlocs = HSlocs(indx);
% HSpks = HSpks(indx);
% figure; plot(zfootVel);title('HSlocs, HSpks plotted on zfootVel');hold on;
% plot(HSlocs,HSpks, 'or'); hold on;
% figure; plot(zheel); title('HSlocs, HSpks plotted on zheel'); hold on;
% plot(HSlocs,zheel(HSlocs),'or');
% figure; plot(zCoordfootcentre); title('HSlocs, HSpks plotted on zCoordfootcentre'); hold on;
% plot(HSlocs,zCoordfootcentre(HSlocs),'or');

if isempty(TOpks), return; end

% It is possible that up to this point multiple HS events are found between TO events.
% This problem is settled below. The signal is considered as periodical with period from one TO up the
% next TO. In every period there should be not more then one HS event.
%% Only one HS between two TO
HSpkstmp = [];
HSlocstmp = [];
for j = 1:length(TOlocs)-1              % min two TOlocs required
    indx = TOlocs(j) < HSlocs & HSlocs <  TOlocs(j+1); % Wahrheitspr?fung
    HSpksPrd         = HSpks(indx);
    HSlocsPrd        = HSlocs(indx);
    [HSpksPrd, indx] = min(HSpksPrd);   % smaller peak is chosen
    HSlocsPrd        = HSlocsPrd(indx);
    HSpkstmp         = [HSpkstmp HSpksPrd];                 %#ok<AGROW>
    HSlocstmp        = [HSlocstmp HSlocsPrd];               %#ok<AGROW>
end

%% First HS before all TO
indx                 = HSlocs < TOlocs(1);
HSpksPrd             = HSpks(indx);
HSlocsPrd            = HSlocs(indx);
[HSpksPrd, indx]     = min(HSpksPrd);
HSlocsPrd            = HSlocsPrd(indx);
HSpkstmp             = [HSpkstmp HSpksPrd];
HSlocstmp            = [HSlocstmp HSlocsPrd];

%% Last HS after all TO
indx                 =  TOlocs(end) < HSlocs;
HSpksPrd             = HSpks(indx);
HSlocsPrd            = HSlocs(indx);
[HSpksPrd, indx]     = min(HSpksPrd);
HSlocsPrd            = HSlocsPrd(indx);
HSpkstmp             = [HSpkstmp HSpksPrd];
HSlocstmp            = [HSlocstmp HSlocsPrd];

HSpks       = sort(HSpkstmp);
HSlocs      = sort(HSlocstmp);

d = size(TOlocs);
if d(1)>d(2), TOlocs = transpose(TOlocs); end
d = size(HSlocs);
if d(1)>d(2), HSlocs = transpose(HSlocs); end

% figure
% plot(zfootVel, 'black')
% title ('final')
% hold on
% plot(HSlocs, HSpks, 'or')
% hold on
% plot(TOlocs, TOpks, 'sr')
% 
% figure; plot(zheel); title('HSlocs, HSpks plotted on heel height');
% hold on;
% plot(HSlocs,zheel(HSlocs),'or');
% figure; plot(ztoe);  title('TOlocs, TOpks plotted on toe height');
% hold on;
% plot(TOlocs, ztoe(TOlocs),'sr');
return;
end