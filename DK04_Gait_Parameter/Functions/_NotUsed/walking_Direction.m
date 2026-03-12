clc
close all
clear all
walkingDirecfnal = [];
figure_walkdir = figure;
ax_walkdir = gca(figure_walkdir);
figure_walkdirright = figure;
ax_walkdirright = gca(figure_walkdirright);
figure_walkdirleft = figure;
ax_walkdirleft = gca(figure_walkdirleft);
figure_asisx = figure;
ax_walkasisx = gca(figure_asisx);
figure_asisy = figure;
ax_walkasisy = gca(figure_asisy);
figure_asislx = figure;
ax_walkasislx = gca(figure_asislx);
figure_asisly = figure;
ax_walkasisly = gca(figure_asisly);

%% Determine the direction of walking

for datfile = 1:2
    filename = ['GHCE_95_Pre_gt_0', num2str(datfile), '.mat'];
    data = load(filename);
    dataKinematicVar = fieldnames(data);
    dataKinematic = data.(genvarname(dataKinematicVar{2}));

    markernames = fieldnames(dataKinematic);

    for i = 1:length(markernames)-1
        if (strcmp(markernames{i}, 'RTAS') ~= 0)
            asisRx = dataKinematic.(genvarname(markernames{i})).Values.x_coord;
            asisRy = dataKinematic.(genvarname(markernames{i})).Values.y_coord;
        elseif (strcmp(markernames{i}, 'LTAS') ~= 0)
            asisLx = dataKinematic.(genvarname(markernames{i})).Values.x_coord;
            asisLy = dataKinematic.(genvarname(markernames{i})).Values.y_coord;
        end
    end
    data.(genvarname('walkingDirec')) = unwrap(atan2((asisRy-asisLy),(asisRx-asisLx)));
    data.(genvarname('walkingDirec_right')) = unwrap(atan2((asisRy), (asisRx)));
    data.(genvarname('walkingDirec_left')) = unwrap(atan2((asisLy), (asisLx)));
    hold(ax_walkdir, 'on')
    plot(ax_walkdir, data.walkingDirec);
    hold(ax_walkdirright, 'on')
    plot(ax_walkdirright, data.walkingDirec_right);
    hold(ax_walkdirleft, 'on')
    plot(ax_walkdirleft, data.walkingDirec_left);
    hold(ax_walkasisx, 'on');
    plot(ax_walkasisx, asisRx);
    hold(ax_walkasisy, 'on');
    plot(ax_walkasisy, asisRy);
    hold(ax_walkasislx, 'on');
    plot(ax_walkasislx, asisLx);
    hold(ax_walkasisly, 'on');
    plot(ax_walkasisly, asisLy);
end


