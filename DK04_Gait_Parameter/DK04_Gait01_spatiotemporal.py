#!/usr/bin/env python3
"""
compute_spatiotemporal.py — Python port of the MATLAB treadmill gait-parameter
chain (GaitWork_Treadmill / strLength_Treadmill / stepwidth / f_approxVelocity_
treadmill / Sorting_Steps), producing per-stride step/stride length & width -> CSV.

Faithful port notes (must match the MATLAB exactly):
- Marker axes: column 0 = X (medio-lateral), column 1 = Y (anterior-posterior,
  = treadmill travel), column 2 = Z (vertical). Units: mm in -> cm out (x 0.1).
- Stride length/width use TREADMILL-CORRECTED positions (belt displacement added
  to Y); step WIDTH uses the RAW positions — exactly as GaitWork_Treadmill calls them.
- "Heel" markers are the ankle joint centres LAJC/RAJC; toe markers LTO3/RTO3.
- Gait events (HSleft/HSright/TOleft/TOright, 1-based like MATLAB) are an INPUT
  (from the foot-IMU detector). They are model-independent, so the same events are
  used for OMC and every model; pass them in via --events-npz.

Belt velocity (f_approxVelocity_treadmill): estimated from the toe-marker Y swing
in frames 10000:12000, averaged L/R with outliers removed, then belt displacement
(mm/frame * frame index) is added to every marker's Y.
"""
import argparse, os, glob, csv, numpy as np
from scipy.signal import find_peaks


def _rm_outliers(a):
    """MATLAB rmoutliers default: median +/- 3 * scaled-MAD."""
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    if a.size == 0: return a
    med = np.median(a); mad = np.median(np.abs(a - med)) * 1.4826
    if mad == 0: return a
    return a[np.abs(a - med) <= 3 * mad]


def approx_velocity_treadmill(markers, SF, corr_markers=("RTO3","LTO3","RAJC","LAJC")):
    """Add treadmill belt displacement to the Y-axis of corr_markers. markers: dict
    name->(T,3). Returns corrected dict. Mirrors f_approxVelocity_treadmill.m."""
    ymr = markers["RTO3"][10000:12000, 1]
    yml = markers["LTO3"][10000:12000, 1]
    mpd = int(round(0.8 * SF))
    def peaks(y):   # returns max-peak locs, min-peak locs (0-based within window)
        mx,_ = find_peaks(y,  distance=mpd)
        mn,_ = find_peaks(-y, distance=mpd)
        return mx, mn
    mxr, mnr = peaks(ymr); mxl, mnl = peaks(yml)
    mnr = mnr[np.searchsorted(mnr, mxr[0], side="right"):] if len(mxr) else mnr
    mnl = mnl[np.searchsorted(mnl, mxl[0], side="right"):] if len(mxl) else mnl
    def swing_speeds(ymid, locmax, locmin):
        n = min(len(locmax), len(locmin)); sp = []
        for i in range(n - 1):
            idx = locmax[locmax > locmin[i]]
            if len(idx) == 0: continue
            a = int(np.floor(idx[0] - 0.05 * SF)); b = int(np.floor(locmin[i] + 0.05 * SF))
            if a < 0 or b < 0 or a >= len(ymid) or b >= len(ymid) or a == b: continue
            sp.append((ymid[a] - ymid[b]) / (a - b) * SF)
        return np.array(sp)
    sr = _rm_outliers(swing_speeds(ymr, mxr, mnr))
    sl = _rm_outliers(swing_speeds(yml, mxl, mnl))
    tm_v = (np.nanmean(sl) + np.nanmean(sr)) / 2.0
    mmpf = tm_v / SF
    out = {k: v.copy() for k, v in markers.items()}
    for name in corr_markers:
        m = markers[name].copy()
        m[:, 1] = m[:, 1] + mmpf * (np.arange(1, len(m) + 1))
        out[name] = m
    return out, float(tm_v)


def str_length_treadmill(LAJC, RAJC, HSL, HSR, SF):
    """stride/step length (cm), stride width (cm), step time (s), cadence. Ankle
    markers already treadmill-corrected. HSL/HSR are 1-based event indices."""
    hl = np.asarray(HSL, int) - 1; hr = np.asarray(HSR, int) - 1
    def xy(M):
        x = M[:, 0].astype(float); y = M[:, 1].astype(float); y = y - y.mean()
        return np.column_stack([x, y])
    L = xy(LAJC); R = xy(RAJC)
    strideL = np.array([0.1*np.linalg.norm(L[hl[j+1]]-L[hl[j]]) for j in range(len(hl)-1)])
    strideR = np.array([0.1*np.linalg.norm(R[hr[j+1]]-R[hr[j]]) for j in range(len(hr)-1)])
    strideWL = np.array([0.1*abs(L[hl[j+1],0]-L[hl[j],0]) for j in range(len(hl)-1)])
    strideWR = np.array([0.1*abs(R[hr[j+1],0]-R[hr[j],0]) for j in range(len(hr)-1)])
    m = min(len(hl), len(hr)); stepR=[]; stepTR=[]
    for j in range(m):
        if hl[j] < hr[j]:
            stepR.append(0.1*np.linalg.norm(np.abs(R[hr[j]]-L[hl[j]]))); stepTR.append((hr[j]-hl[j])/SF)
    stepL=[]; stepTL=[]
    for j in range(m-1):
        if hr[j] < hl[j+1]:
            stepL.append(0.1*np.linalg.norm(np.abs(L[hl[j+1]]-R[hr[j]]))); stepTL.append((hl[j+1]-hr[j])/SF)
    cad = 60*(len(stepTL)+len(stepTR)) / ((hr[-1]-hl[0])/SF) if len(hl) and len(hr) else np.nan
    return dict(strideLengthL=strideL, strideLengthR=strideR, strideWidthL=strideWL,
                strideWidthR=strideWR, stepLengthL=np.array(stepL), stepLengthR=np.array(stepR),
                stepTimeL=np.array(stepTL), stepTimeR=np.array(stepTR), cadence=cad)


def stepwidth(LAJC, RAJC, HSL, HSR):
    """Step width (cm) = perpendicular distance from contralateral heel strike to
    the ipsilateral stride line. Uses RAW (non-treadmill-corrected) positions."""
    hl = np.asarray(HSL, int) - 1; hr = np.asarray(HSR, int) - 1
    def xy(M):
        x = M[:,0].astype(float); y = M[:,1].astype(float); y = y - y.mean()
        return np.column_stack([x, y])
    L = xy(LAJC); R = xy(RAJC); m = min(len(hl), len(hr))
    def perp(p2, p1, q):
        vec = p2 - p1; n = np.linalg.norm(vec)
        if n == 0: return np.nan
        vec = vec/n; w = p1 - q
        return abs(vec[0]*w[1] - vec[1]*w[0])   # |cross| in 2D
    wbL=[]; wbR=[]
    for j in range(m-1):
        if hl[j] < hr[j]:
            wbL.append(0.1*perp(L[hl[j+1]], L[hl[j]], R[hr[j]]))
    for j in range(m-1):
        if hr[j] < hl[j+1]:
            wbR.append(0.1*perp(R[hr[j+1]], R[hr[j]], L[hl[j]]))
    wbL=np.array([v for v in wbL if np.isfinite(v)]); wbR=np.array([v for v in wbR if np.isfinite(v)])
    return dict(stepWidthL=wbL, stepWidthR=wbR)


def compute(markers_raw, events, SF=50.0):
    """markers_raw: dict name->(T,3) mm (model or OMC joint-centre positions).
    events: dict HSleft/HSright (1-based). Returns dict of per-stride param arrays."""
    corr, _ = approx_velocity_treadmill(markers_raw, SF)
    p = str_length_treadmill(corr["LAJC"], corr["RAJC"], events["HSleft"], events["HSright"], SF)
    w = stepwidth(markers_raw["LAJC"], markers_raw["RAJC"], events["HSleft"], events["HSright"])
    p.update(w); return p


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--markers-npz", required=True,
                    help="npz with keys '<subj>|<gt|hat>|<MARKER>' -> (T,3) and "
                         "'<subj>|HSleft'/'<subj>|HSright' event indices (1-based)")
    ap.add_argument("--out", default="spatiotemporal_perstride.csv")
    ap.add_argument("--sf", type=float, default=50.0)
    args = ap.parse_args()
    d = np.load(args.markers_npz)
    subs = sorted({k.split("|")[0] for k in d.files})
    rows = []
    for s in subs:
        for variant in ("gt", "hat"):
            try:
                mk = {m: d[f"{s}|{variant}|{m}"] for m in ("LAJC","RAJC","LTO3","RTO3")}
            except KeyError:
                continue
            ev = {"HSleft": d[f"{s}|HSleft"], "HSright": d[f"{s}|HSright"]}
            p = compute(mk, ev, args.sf)
            for param, arr in p.items():
                if param == "cadence": continue
                for i, val in enumerate(np.atleast_1d(arr)):
                    rows.append({"subject": s, "variant": variant, "parameter": param,
                                 "stride": i, "value_cm": round(float(val), 4)})
    with open(args.out, "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"wrote {args.out}  ({len(rows)} rows, {len(subs)} subjects)")
