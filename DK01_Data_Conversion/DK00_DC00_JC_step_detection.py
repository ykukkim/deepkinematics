import math
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

""" Values are taken from the step detection function in Yongs Matlab Main_Scripts """
toprdfac = 0.8
hsprdfac = 0.08
zrangefac = 0.4
filterOrder = 4
cutfreqFilter = 25
srate = 200

def contact_phase_detection(heel, toe, sf):
    
    # Filtering the signal:
    sos = signal.butter(filterOrder, cutfreqFilter / srate, 'lp', output='sos')

    zheel = signal.sosfilt(sos, heel[:,2])
    ztoe = signal.sosfilt(sos, toe[:,2])

    zheel = zheel[sf:]
    ztoe = ztoe[sf:]

    # Estimated center of foot and velocity of foot
    zCoordfootcentre = 0.5 * (zheel + ztoe)
    zfootVel = np.divide(np.diff(zCoordfootcentre), np.diff(np.arange(1,len(zCoordfootcentre)+1,1)).T)
    
    # Determine TO events
    mpd =  math.floor(toprdfac * srate)
    TOlocs, _ = signal.find_peaks(zfootVel, distance=mpd) #, threshold=0.5e-3)

    TOpks = zfootVel[TOlocs]
    TOlocs = TOlocs[TOpks > 0.5e-3]
    TOpks = zfootVel[TOlocs]

    # Determine HS event
    mpd = math.floor(hsprdfac * srate)
    HSlocs, _ = signal.find_peaks(-zheel, distance=150, threshold = -80e-3)
    HSpks = zheel[HSlocs]
    #
    # print(HSlocs[60:70])
    # print(TOlocs[60:70])
    #
    # if len(TOlocs) > 307:
    #     TOlocs = np.delete(TOlocs, 307)
    # plt.figure()
    # plt.plot(zheel)
    # plt.plot(HSlocs[0:20], zheel[HSlocs[0:20]], "x")
    # plt.plot(TOlocs[0:20], zheel[TOlocs[0:20]], "o")
    # plt.show()
    # TOlocs = TOlocs[3:]

    # Only one HS between two TO
    HSpkstmp = []
    HSlocstmp = []

    for i in range(len(TOlocs)-2):
        if TOlocs[i] < 1000:
            # Disregard any TO/HS Events before 1000 frames / disregard jump
            continue
        indx = np.logical_and(TOlocs[i] < HSlocs, HSlocs < TOlocs[i + 1])
        HSpksPrd = HSpks[indx]
        HSlocsPrd = HSlocs[indx]  
        HSlocsPrd = HSlocsPrd[np.argmin(HSpksPrd)]
        HSpksPrd = min(HSpksPrd)
        HSpkstmp.append(HSpksPrd)
        HSlocstmp.append(HSlocsPrd)

    # Determine contact phase of Heel and Toe
    HS_contact = np.zeros(zheel.shape[0])
    TO_contact = np.zeros(ztoe.shape[0])

    # Heel contact phase is determined by the vertical distance of the Heel
    for hs in HSlocstmp:
        i = hs
        while zheel[i] < zheel[hs] + 0.02:
            HS_contact[i] = 1
            i = i + 1
            if i == len(zheel)-1:
                break
        
    # Toe contact phase is deteremined by the velocity of the foot
    for to in TOlocs:
        i = to
        while zfootVel[i] > -1e-4:
            TO_contact[i] = 1
            i = i - 1

    HS_contact_bool = HS_contact.astype(dtype=bool)
    TO_contact_bool = TO_contact.astype(dtype=bool)

    # # Set font properties
    # font = {'family': 'Arial',
    #         'weight': 'normal',
    #         'size': 22}
    #
    # plt.rc('font', **font)
    #
    # plt.figure()
    # plt.subplot(211)
    # plt.title('Toe position [z-axis]')
    # plt.ylabel('[m]')
    # # plt.xlabel('frames')
    # plt.plot(ztoe)
    # # plt.plot(HSlocstmp, ztoe[HSlocstmp], "x")
    # # plt.plot(TOlocs, ztoe[TOlocs], "x")
    # plt.plot(np.arange(ztoe.shape[0])[TO_contact_bool], ztoe[TO_contact_bool], "x")
    #
    # plt.subplot(212)
    # plt.title('Heel position [z-axis]')
    # plt.ylabel('[m]')
    # plt.plot(zheel)
    # plt.plot(np.arange(zheel.shape[0])[HS_contact_bool], zheel[HS_contact_bool], "x")
    # # plt.plot(TOlocs, zheel[TOlocs], "x")
    # plt.show()

    contact_phase = np.stack((TO_contact, HS_contact), axis = 1)
    return contact_phase