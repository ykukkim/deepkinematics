"""
Python port of Functions/BlandAltmanPlots/*.m

Ported functions:
  - findFoldersEndingWithString.m -> find_folders_ending_with_string
  - LogsSONEMS.m                  -> logs_sonems

Only the functions actually used by the converted DK04_GaitXX drivers are
ported here so far (see DK04_Gait_Parameter dependency map). BlandAltmanPlot,
MeanSD, nanmean, and rem_outliers_2mad are NOT yet ported.
"""
import os
import traceback
from datetime import datetime


def find_folders_ending_with_string(base_dir, ending_str):
    """
    Python port of findFoldersEndingWithString.m

    Returns the full paths of the immediate subdirectories of `base_dir`
    whose names end with `ending_str` (case-sensitive, matching MATLAB's
    endsWith). '.' and '..' are never included (os.listdir already omits
    them, unlike MATLAB's dir()).

    Original MATLAB used whatever order dir() returned (filesystem-
    dependent); this returns results sorted by name for determinism.
    """
    if not os.path.isdir(base_dir):
        return []
    matching = []
    for name in sorted(os.listdir(base_dir)):
        full_path = os.path.join(base_dir, name)
        if os.path.isdir(full_path) and name.endswith(ending_str):
            matching.append(full_path)
    return matching


def logs_sonems(logs, subject, exc=None):
    """
    Python port of LogsSONEMS.m

    Appends one entry to `logs` (a list of dicts, standing in for MATLAB's
    struct array) recording either a successful run or a failure for
    `subject`. Pass the caught exception as `exc` to log an error entry
    (this is the Python equivalent of MATLAB's ME / MException argument);
    leave it as None to log a success entry.

    Mutates and returns `logs`.

    Deviation from the original: MATLAB's loop built `error_msg` by
    concatenating `ME.message` once per stack frame (so the same message
    was literally repeated N times, since MException.message doesn't vary
    per frame) -- that looked like an unintended artifact of the original
    loop rather than a deliberate behaviour, so here the message is stored
    once. `ErrorFunction` and `line` are still built to mirror the
    original's outermost-frame-first ordering.
    """
    entry = {'Participant': subject}

    if exc is not None:
        tb = traceback.extract_tb(exc.__traceback__)  # outermost frame first, innermost last
        entry['Note1'] = 'Failed'
        entry['ErrorFunction'] = ' >> '.join(frame.name for frame in tb) + ' >> ' if tb else ''
        entry['line'] = [frame.lineno for frame in tb]
        entry['Message'] = str(exc)
        entry['LastUpdated'] = datetime.now().isoformat()
    else:
        entry['Note1'] = 'Successful'
        entry['LastUpdated'] = datetime.now().isoformat()
        entry['ErrorFunction'] = ''
        entry['Note2'] = ''

    logs.append(entry)
    return logs
