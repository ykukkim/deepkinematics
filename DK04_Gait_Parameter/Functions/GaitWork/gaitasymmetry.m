function [GaitAsymmetry_ShortvsLong, GaitAsymmetry_LeftvsRight] ...
    = gaitasymmetry(swingphl, swingphr)

%% Function for Gait Asymmetry: for more information see
 % Plotnik, M. et al. 2007. A new measure for quantifying ... Exp Br Res
 % 181:561-570.

noofPossiblesets = min(size(swingphl,2), size(swingphr,2));

for i = 1:noofPossiblesets
    if swingphl(i) <= swingphr(i)
        swingphshortleg(i) = swingphl(i);
        swingphlongleg(i) = swingphr(i);
    else
        swingphshortleg(i) = swingphr(i);
        swingphlongleg(i) = swingphl(i);
    end
end

swingphl = swingphl(1:noofPossiblesets);
swingphr = swingphr(1:noofPossiblesets);

GaitAsymmetry_ShortvsLong = 100 * abs(log(swingphshortleg./swingphlongleg));
GaitAsymmetry_LeftvsRight = 100 * abs(log(swingphl./swingphr));

return;
end
