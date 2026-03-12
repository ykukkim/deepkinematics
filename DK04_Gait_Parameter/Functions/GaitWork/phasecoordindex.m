function [phaseCoordID_shortvslong, phaseCoordID_leftvsright] = ...
    phasecoordindex(swingphl, swingphr, stridephl, stridephr, stepphl, stepphr)

%% Function for Phase coordination Index: for more information see
 % Plotnik, M. et al. 2007. A new measure for quantifying ... Exp Br Res
 % 181:561-570.

noofPossiblesets_swing = min(size(swingphl,2), size(swingphr,2));
noofPossiblesets_step = min(size(stepphl,2), size(stepphr,2));
noofPossiblesets = min(noofPossiblesets_swing, noofPossiblesets_step);

for i = 1:noofPossiblesets
    if swingphl(i) <= swingphr(i)
        swingphshortleg(i) = swingphl(i);
        stridephshortleg(i) = stridephl(i);
        stepphshortleg(i) = stepphl(i);
        
        swingphlongleg(i) = swingphr(i);
        stridephlongleg(i) = stridephr(i);
        stepphlongleg(i) = stepphr(i);
    elseif swingphl(i) > swingphr(i)
        swingphshortleg(i) = swingphr(i);
        stridephshortleg(i) = stridephr(i);
        stepphshortleg(i) = stepphr(i);
        
        swingphlongleg(i) = swingphl(i);
        stridephlongleg(i) = stridephl(i);
        stepphlongleg(i) = stepphl(i);
    end
end

for j = 1:noofPossiblesets
    pcisvlo(j) = 360 * (stepphshortleg(j)/stridephlongleg(j));
    pcilvr(j) = 360 * (stepphl(j)/stridephr(j));
end

phaseCoordID_shortvslong = pcisvlo;
phaseCoordID_leftvsright = pcilvr;

return
end