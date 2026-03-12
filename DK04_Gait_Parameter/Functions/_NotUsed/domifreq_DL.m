function [domfreq] = domifreq_DL(Data)
%  x = Data.gyro(:,2);
%  Fs = Data.SamplingRate; % sampling rate
%  x = Data;
%  Fs = 128;
% x = Data(:,2);
x = Data;
Fs = 200;

N = length(x);
xdft = fft(x);
xdft = xdft(1:floor(N/2+1)); %floor is used therefor that no warning appears
psdx = (1/(Fs*N)) * abs(xdft).^2;
psdx(2:end-1) = 2*psdx(2:end-1);
freq = 0:Fs/length(x):Fs/2;

spectrum = 10*log10(psdx);
figure;
plot(freq,spectrum)
% figure;
% plot(freq, angle(xdft(1:N/2+1)))
% grid on
title('Periodogram Using FFT')
xlabel('Frequency (Hz)')
ylabel('Power/Frequency (dB/Hz)')


% spectrum(1) = 0;
[M,I] = max(spectrum);
 close;
domfreq = freq(I);
end 
