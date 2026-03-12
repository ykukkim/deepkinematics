

answer = EinstellungenGUI_NR;
if isempty(answer)
    disp(' '); disp('Programm abgebrochen!'); disp(' ')
    return
else
    if answer(1)
        Drehung = 'G';
        shift1=0; 
    end
    if answer(3)
        Drehung = 'U';
        shift1=0;
    end
    if answer(1) && answer(3)
        Drehung= 'G';
        shift1=1;
    end
end

[Referenz_stat,Referenz,Joints]=LadeReferenzSysteme_Navi(subject,Drehung);
 
% Extrahiere die Segmentmarker aus dem Gang-file:
Segm = segmentextractor_stride_Navi_ReducedFAT(newVD); % create the Segm struct