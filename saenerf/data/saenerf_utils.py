
import numpy as np
from pathlib import Path
import h5py
import torch
import os
import cv2
import glob
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from scipy.interpolate import interp1d
from torch import Tensor
from typing import Dict, Optional, Type, List, Any, Literal, Tuple
from torch.utils.data.dataset import Dataset
import threading
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.cameras.camera_utils import quaternion_from_matrix, quaternion_matrix
import time
from tqdm import tqdm
import roma
import numba
import math

@numba.jit(nopython=True)
def accumulate_events(xs, ys, ts, ps, e_frames):
    """
    accumulate negative and positive event frames separately.
    """
    assert len(xs) == len(ys) == len(ts) == len(ps)
    for i in range(len(xs)):
        x, y, t, p = xs[i], ys[i], ts[i], ps[i]
        if p > 0:
            e_frames[y, x] += 1
        else:
            e_frames[y, x] -= 1

def event_split(event_stream: dict, cam_ts: Tensor, h: int, w: int):
    """
    event_stream: 
    cam_ts: camera pose timestamps
    h: image height
    w: image width
    """
    xs, ys, ts, ps = event_stream['x'], event_stream['y'], \
        event_stream['t'], event_stream['p']
    assert len(cam_ts)
    cam_cnt = len(cam_ts)
    event_frames = np.zeros((cam_cnt - 1, h, w), dtype=np.float32)
    
    for i in tqdm(range(1, cam_cnt), desc="Accululating events"):  # t in (i-1, i] event accumulated
        start = np.searchsorted(ts, cam_ts[i-1], side="left")
        end   = np.searchsorted(ts, cam_ts[i],   side="left")
        accumulate_events(xs[start:end], ys[start:end], ts[start:end], \
                          ps[start:end], event_frames[i-1])
    return Tensor(event_frames)

def event_load_eventnerf(event_path: Path):
    """
    load event from event-nerf dataset
    """
    events = np.load(event_path)
    # events = {
    #     'x': events['x'],
    #     'y': events['y'],
    #     't': events['t'],
    #     'p': events['p'],
    # }
    return events

def get_eventnerf_gt_timestamp(ts):
    """
    eventnerf use uniform motion
    """
    gt_len = 1001
    # camera_timestamps = range(gt_len + 1) * (ts.max() - ts.min()) / gt_len + ts.min()
    camera_timestamps = np.linspace(ts.min(), ts.max(), gt_len)
    return camera_timestamps

# from https://github.com/uzh-rpg/DSEC/blob/main/scripts/utils/eventslicer.py
class EventSlicer:
    def __init__(self, h5f: h5py.File):
        self.h5f = h5f

        self.events = dict()
        if 'events/x' in self.h5f.keys():
            for dset_str in ['p', 'x', 'y', 't']:
                self.events[dset_str] = self.h5f['events/{}'.format(dset_str)]
        else:
            for dset_str in ['p', 'x', 'y', 't']:
                self.events[dset_str] = self.h5f['{}'.format(dset_str)]

        # This is the mapping from milliseconds to event index:
        # It is defined such that
        # (1) t[ms_to_idx[ms]] >= ms*1000
        # (2) t[ms_to_idx[ms] - 1] < ms*1000
        # ,where 'ms' is the time in milliseconds and 't' the event timestamps in microseconds.
        #
        # As an example, given 't' and 'ms':
        # t:    0     500    2100    5000    5000    7100    7200    7200    8100    9000
        # ms:   0       1       2       3       4       5       6       7       8       9
        #
        # we get
        #
        # ms_to_idx:
        #       0       2       2       3       3       3       5       5       8       9
        self.ms_to_idx = np.asarray(self.h5f['ms_to_idx'], dtype='int64')

        if "t_offset" in list(h5f.keys()):
            self.t_offset = int(h5f['t_offset'][()])
        else:
            self.t_offset = 0
        self.t_final = int(self.events['t'][-1]) + self.t_offset

    def get_start_time_us(self):
        return self.t_offset

    def get_final_time_us(self):
        return self.t_final

    def get_events(self, t_start_us: int, t_end_us: int) -> Dict[str, np.ndarray]:
        """Get events (p, x, y, t) within the specified time window
        Parameters
        ----------
        t_start_us: start time in microseconds
        t_end_us: end time in microseconds
        Returns
        -------
        events: dictionary of (p, x, y, t) or None if the time window cannot be retrieved
        """
        assert t_start_us < t_end_us

        # We assume that the times are top-off-day, hence subtract offset:
        t_start_us -= self.t_offset
        t_end_us -= self.t_offset

        t_start_ms, t_end_ms = self.get_conservative_window_ms(t_start_us, t_end_us)
        t_start_ms = np.maximum(t_start_ms, 0)
        t_start_ms_idx = self.ms2idx(t_start_ms)
        t_end_ms_idx = self.ms2idx(t_end_ms)
        #if t_end_ms_idx is None:
        #    t_end_ms_idx = self.ms2idx(t_end_ms-1)

        if t_start_ms_idx is None or t_end_ms_idx is None:
            # Cannot guarantee window size anymore
            return None

        events = dict()
        time_array_conservative = np.asarray(self.events['t'][t_start_ms_idx:t_end_ms_idx])
        idx_start_offset, idx_end_offset = self.get_time_indices_offsets(time_array_conservative, t_start_us, t_end_us)
        t_start_us_idx = t_start_ms_idx + idx_start_offset
        t_end_us_idx = t_start_ms_idx + idx_end_offset
        # Again add t_offset to get gps time
        events['t'] = time_array_conservative[idx_start_offset:idx_end_offset] + self.t_offset
        for dset_str in ['p', 'x', 'y']:
            events[dset_str] = np.asarray(self.events[dset_str][t_start_us_idx:t_end_us_idx])
            assert events[dset_str].size == events['t'].size
        return events

    @staticmethod
    def get_conservative_window_ms(ts_start_us: int, ts_end_us) -> Tuple[int, int]:
        """Compute a conservative time window of time with millisecond resolution.
        We have a time to index mapping for each millisecond. Hence, we need
        to compute the lower and upper millisecond to retrieve events.
        Parameters
        ----------
        ts_start_us:    start time in microseconds
        ts_end_us:      end time in microseconds
        Returns
        -------
        window_start_ms:    conservative start time in milliseconds
        window_end_ms:      conservative end time in milliseconds
        """
        assert ts_end_us > ts_start_us
        window_start_ms = math.floor(ts_start_us/1000)
        window_end_ms = math.ceil(ts_end_us/1000)
        return window_start_ms, window_end_ms

    @staticmethod
    @numba.jit(nopython=True)
    def get_time_indices_offsets(
            time_array: np.ndarray,
            time_start_us: int,
            time_end_us: int) -> Tuple[int, int]:
        """Compute index offset of start and end timestamps in microseconds
        Parameters
        ----------
        time_array:     timestamps (in us) of the events
        time_start_us:  start timestamp (in us)
        time_end_us:    end timestamp (in us)
        Returns
        -------
        idx_start:  Index within this array corresponding to time_start_us
        idx_end:    Index within this array corresponding to time_end_us
        such that (in non-edge cases)
        time_array[idx_start] >= time_start_us
        time_array[idx_end] >= time_end_us
        time_array[idx_start - 1] < time_start_us
        time_array[idx_end - 1] < time_end_us
        this means that
        time_start_us <= time_array[idx_start:idx_end] < time_end_us
        """

        assert time_array.ndim == 1

        idx_start = -1
        if time_array[-1] < time_start_us:
            # This can happen in extreme corner cases. E.g.
            # time_array[0] = 1016
            # time_array[-1] = 1984
            # time_start_us = 1990
            # time_end_us = 2000

            # Return same index twice: array[x:x] is empty.
            return time_array.size, time_array.size
        else:
            for idx_from_start in range(0, time_array.size, 1):
                if time_array[idx_from_start] >= time_start_us:
                    idx_start = idx_from_start
                    break
        assert idx_start >= 0

        idx_end = time_array.size
        for idx_from_end in range(time_array.size - 1, -1, -1):
            if time_array[idx_from_end] >= time_end_us:
                idx_end = idx_from_end
            else:
                break

        assert time_array[idx_start] >= time_start_us
        if idx_end < time_array.size:
            assert time_array[idx_end] >= time_end_us
        if idx_start > 0:
            assert time_array[idx_start - 1] < time_start_us
        if idx_end > 0:
            assert time_array[idx_end - 1] < time_end_us
        return idx_start, idx_end

    def ms2idx(self, time_ms: int) -> int:
        assert time_ms >= 0
        if time_ms >= self.ms_to_idx.size:
            return None
        return self.ms_to_idx[time_ms]


# ref: https://github.com/knelk/enerf/utils/pose_utils.py
##################################
# Interpolating
##################################
def interpol_poses_slerp(tss_poses_ns, poses_rots, poses_trans, tss_query_ns):
    """
    Input
    :tss_poses_ns list of known tss
    :poses_rots list of 3x3 np.arrays
    :poses_trans list of 3x1 np.arrays
    :tss_query_ns list of query tss

    Returns:
    :rots list of rots at tss_query_ns
    :trans list of translations at tss_query_ns
    """
    # Setup Rot interpolator
    rot_interpolator = Slerp(tss_poses_ns, R.from_matrix(poses_rots))
    # Query rot interpolator
    rots = rot_interpolator(tss_query_ns).as_matrix()

    # Setup trans interpolator
    trans_interpolator = interp1d(x=tss_poses_ns, y=poses_trans, axis=0, kind="cubic", bounds_error=True)
    # Query trans interpolator
    trans = trans_interpolator(tss_query_ns)

    return rots, trans

def compose_poses(rots, trans):
    poses = []
    for rot, tran in zip(rots, trans):
        pose = np.identity(4)
        pose[:3, :3] = rot[:3, :3]
        # print(pose[:3, 3].shape)
        pose[:3, 3] = tran.flatten()
        poses.append(pose)
    poses = np.array(poses, dtype=np.float32)
    check_rot_batch(poses[:, :3, :])
    return poses

def get_hom_trafos(rots_3_3, trans_3_1):
    N = rots_3_3.shape[0]
    assert rots_3_3.shape == (N, 3, 3)

    if trans_3_1.shape == (N, 3):
        trans_3_1 = np.expand_dims(trans_3_1, axis=-1)
    else:
        assert trans_3_1.shape == (N, 3, 1)
    
    pose_N_4_4 = np.zeros((N, 4, 4))
    hom = np.array([0,0,0,1]).reshape((1, 4)).repeat(N, axis=0).reshape((N, 1, 4))

    pose_N_4_4[:N, :3, :3] = rots_3_3  # (N, 3, 3)
    pose_N_4_4[:N, :3, 3:4] = trans_3_1 # (N, 3, 1)
    pose_N_4_4[:N, 3:4, :] = hom # (N, 1, 4)

    # pose_N_3_4 = np.asarray([np.concatenate((r, t), axis=1) for r, t in zip(rots_3_3, trans_3_1)])
    # pose_N_4_4 = np.asarray([np.vstack((p, np.asarray([0, 0, 0, 1]))) for p in pose_N_3_4])
    return pose_N_4_4

# ref: https://github.com/knelk/enerf/utils/pose_utils.py
def quatList_to_poses_hom_and_tss(quat_list_us):
    """
    quat_list: [[t, px, py, pz, qx, qy, qz, qw], ...]
    """
    tss_all_poses_us = [t[0] for t in quat_list_us]

    all_rots = [R.from_quat(rot[4:]).as_matrix() for rot in quat_list_us]
    all_trans = [trans[1:4] for trans in quat_list_us]
    all_trafos = get_hom_trafos(np.asarray(all_rots), np.asarray(all_trans))

    return tss_all_poses_us, all_trafos

def check_rot(rot, right_handed=True, eps=1e-6):
    """
    Input: 3x3 rotation matrix
    """
    assert rot.shape[0] == 3
    assert rot.shape[1] == 3

    assert np.allclose(rot.transpose() @ rot, np.eye(3), atol=1e-6)
    assert np.linalg.det(rot) - 1 < eps * 2

    if right_handed:
        assert np.abs(np.dot(np.cross(rot[:, 0], rot[:, 1]), rot[:, 2]) - 1.0) < 1e-3
    else:
        assert np.abs(np.dot(np.cross(rot[:, 0], rot[:, 1]), rot[:, 2]) + 1.0) < 1e-3
        
##################################
# Check rotation matrix
##################################
def check_rot_batch(poses, right_handed=True):
    """
    Input: Either (num_poses, 3, 5)-array or list of poses (3, 5) or (3,4)
    """
    assert len(poses) > 0
    assert np.all([p.shape[0] == 3 for p in poses])
    assert np.all([p.shape[1] >= 4 for p in poses])

    for i in range(len(poses)):
        rot = poses[i][:3, :3]
        check_rot(rot, right_handed=right_handed)

##################################
# Coordinate System Conventions
##################################
def rub_from_rdf(poses):
    """
    Input
        :poses (num_poses, 3, 4) as (right, down, front), i.e. the normal convention
    Output: 
        :poses (num_poses, 3, 4) reordered as (right, up, back)
    """
    assert poses.shape[0] > 0
    assert poses.shape[1] == 3
    assert poses.shape[2] >= 4

    poses_ = np.zeros_like(poses)
    poses_ = np.concatenate([poses[:, :, 0:1], -poses[:, :, 1:2:], -poses[:, :, 2:3], poses[:, :, 3:]], 2)

    check_rot_batch(poses_)
    return poses_

def render_ev_accumulation(x: np.ndarray, y: np.ndarray, pol: np.ndarray, H: int, W: int) -> np.ndarray:
    assert x.size == y.size == pol.size
    assert H > 0
    assert W > 0
    img = np.full((H,W,3), fill_value=255,dtype='uint8')
    mask = np.zeros((H,W),dtype='int32')
    pol = pol.astype('int')
    pol[pol==0]=-1
    mask1 = (x>=0)&(y>=0)&(W>x)&(H>y)
    mask[y[mask1].astype(np.int32), x[mask1].astype(np.int32)] = pol[mask1]
    img[mask==0]=[255,255,255]
    img[mask==-1]=[255,0,0]
    img[mask==1]=[0,0,255] 
    return img

def find_files(dir, exts):
    if os.path.isdir(dir):
        files_grabbed = []
        for ext in exts:
            files_grabbed.extend(glob.glob(os.path.join(dir, ext)))
        if len(files_grabbed) > 0:
            files_grabbed = sorted(files_grabbed)
        return files_grabbed
    else:
        return []
            
def compute_ms_to_idx(tss_ns, ms_start=0):
    """
    evs_ns: (N, 4)
    idx_start: Integer
    ms_start: Integer
    """

    ms_to_ns = 1000000
    # tss_sorted, _ = torch.sort(tss_ns) 
    # assert torch.abs(tss_sorted != tss_ns).sum() < 500

    ms_end = int(math.floor(tss_ns.max()) / ms_to_ns)
    assert ms_end >= ms_start
    ms_window = np.arange(ms_start, ms_end + 1, 1).astype(np.uint64)
    ms_to_idx = np.searchsorted(tss_ns, ms_window * ms_to_ns, side="left", sorter=np.argsort(tss_ns))
    
    assert np.all(np.asarray([(tss_ns[ms_to_idx[ms]] >= ms*ms_to_ns) for ms in ms_window]))
    assert np.all(np.asarray([(tss_ns[ms_to_idx[ms]-1] < ms*ms_to_ns) for ms in ms_window if ms_to_idx[ms] >= 1]))
    
    return ms_to_idx

def parse_txt(filename, shape):
    assert os.path.isfile(filename), "file not exist"
    nums = open(filename).read().split()
    return np.array([float(x) for x in nums], dtype=np.float32).reshape(shape)

class EventDictSlicer:
    def __init__(self, events):
        self.xs, self.ys, self.ts, self.ps = events['x'], events['y'], events['t'], events['p']
    
    def get_events(self, t_start_us: int, t_end_us: int) -> Dict[str, np.ndarray]:
        assert t_start_us < t_end_us
        l = np.searchsorted(self.ts, t_start_us, side="left")
        r = np.searchsorted(self.ts, t_end_us,   side="left")
        return {
            'x' : self.xs[l:r],
            'y' : self.ys[l:r],
            't' : self.ts[l:r],
            'p' : self.ps[l:r],
        }
    
    
def create_event_prefix_sum(event_slicer, tss_event_pose, height, width, path=None):
    tss_gt_len = len(tss_event_pose)
    events_prefix_sum = np.zeros((tss_gt_len, height, width), dtype=np.float32)
    for idx in tqdm(range(1, tss_gt_len), desc="create_event_prefix_sum"):
        events = event_slicer.get_events(
            # tss_event_pose[max(0, idx - 10)], 
            tss_event_pose[idx - 1], 
            tss_event_pose[idx]
        )
        if path is not None and os.path.exists(path):
            img = render_ev_accumulation(events['x'], events['y'], events['p'], height, width)
            cv2.imwrite(os.path.join(path, f"{idx - 1:05d}.png"), img)
        if events == None:
            print(f"warning [{tss_event_pose[idx - 1]}, {tss_event_pose[idx]}] no events" )
            continue
        # print(pre, idx, tss_event_pose[pre], tss_event_pose[idx], events['t'][0], events['t'][-1])
        accumulate_events(
            events['x'], 
            events['y'], 
            events['t'], 
            events['p'],
            events_prefix_sum[idx]
        )
        events_prefix_sum[idx] += events_prefix_sum[idx - 1]
    return events_prefix_sum

