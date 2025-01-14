
import os
import glob
import json
import numpy as np
from torch import Tensor
from torchtyping import TensorType
import torch
import cv2
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from scipy.interpolate import interp1d
from tqdm import tqdm
import h5py

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Type, List, Any, Literal
from jaxtyping import Float

from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras import camera_utils
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.configs.config_utils import to_immutable_dict
from nerfstudio.data.dataparsers.base_dataparser import DataParser, DataParserConfig, DataparserOutputs
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.data.utils.dataparsers_utils import (
    get_train_eval_split_all,
    get_train_eval_split_filename,
    get_train_eval_split_fraction,
    get_train_eval_split_interval,
)
from nerfstudio.utils.io import load_from_json

from saenerf.data.saenerf_utils import (
    event_load_eventnerf,
    get_eventnerf_gt_timestamp,
    event_split,
    accumulate_events,
    EventSlicer,
)

@dataclass
class ENeRFDataparserConfig(DataParserConfig):

    _target: Type = field(default_factory=lambda: ENeRFDataparser)
    data: Path = Path("/data/dataset/enerf/esim/ShakeMoon1")
    scale_factor: float = 1.0
    scene_scale: float = 1.0 # 0.35
    auto_scale_poses: bool = True
    eval_mode: Literal["fraction", "filename", "interval", "all"] = "interval"
    train_split_fraction: float = 1.0
    eval_interval: int = 10
    
    mode: Literal["tumvie", "esim"] = "esim"
    orientation_method: Literal["pca", "up", "vertical", "none"] = "up"
    """The method to use for orientation."""
    center_method: Literal["poses", "focus", "none"] = "poses"
    """The method to use to center the poses."""

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

##################################
# Transform Transforms
##################################
def invert_trafo(rot, trans):
    # invert transform from w2cam (esim, colmap) to cam2w
    assert rot.shape[0] == 3
    assert rot.shape[1] == 3
    assert trans.shape[0] == 3

    rot_ = rot.transpose()
    trans_ = -1.0 * np.matmul(rot_, trans)

    check_rot(rot_)
    return rot_, trans_
            
def read_poses_bounds(path_poses_bounds, start_frame=None, end_frame=None, skip_frames=None, invert=False):
    """ Returns: 
    #    :poses np.array (num_poses, 3, 5) in  c2w (even for esim, these poses are already inverted to c2w)
         :bds (num_poses, 2) where [:, 0] = min_depth, and [:, 1] = max_depth
    """
    assert os.path.exists(path_poses_bounds)

    poses_arr = np.load(path_poses_bounds)
    assert poses_arr.shape[0] > 10
    assert poses_arr.shape[1] == 17
    # (num_poses, 17), where  17 = (rot | trans | hwf).ravel(), zmin, zmax
    poses = poses_arr[:, :-2].reshape([-1, 3, 5])  # (num_poses, 17) to (num_poses, 3, 5)
    bds = poses_arr[:, -2:]  # (num_poses, 2)
    num_poses = poses.shape[0]

    if invert:
        for i in range(num_poses):
            rot, trans = poses[i, :3, :3], poses[i, :, 3]
            rot, trans = invert_trafo(rot, trans)
            poses[i, :3, :3] = rot
            poses[i, :3, 3] = trans
        print("** Inverted Poses from ** ", path_poses_bounds)

    # check rotation matrix
    for i in range(num_poses):
        rot = poses[i, :3, :3]
        check_rot(rot, right_handed=True)

    if (start_frame is not None) and (end_frame is not None) and (skip_frames is not None):
        assert end_frame > start_frame
        assert start_frame >= 0
        assert skip_frames > 0
        assert skip_frames < 50

        if end_frame == -1:
            end_frame = poses.shape[0] - 1

        poses = poses[start_frame:end_frame:skip_frames, ...]  # (num_poses, 3, 5)
        bds = bds[start_frame:end_frame:skip_frames, :]

    print("Got total of %d poses" % (poses.shape[0]))
    return poses, bds
            
@dataclass
class ENeRFDataparser(DataParser):

    config: ENeRFDataparserConfig
    
    # ref: https://github.com/knelk/enerf/blob/main/nerf/provider.py
    def _generate_dataparser_outputs(self, split="train", **kwargs: Optional[Dict]):
        
        assert self.config.data.exists(), f"Data directory {self.config.data} does not exist."
        #dir_path = self.config.data / f"{split}"
        root = self.config.data
        print(root)
        
        # load evs
        eventdir = os.path.join(root, "events")
        event_npys = [os.path.join(eventdir, f) for f in sorted(os.listdir(eventdir)) if f.endswith(".npy")]
        # print(event_npys)
        event_batches = []
        for i in range(len(event_npys)):
            evs = np.load(event_npys[i])
            event_batches.append(evs)
        print(event_batches.shape)
        poses_c2w, bds = read_poses_bounds(os.path.join(root, F"poses_bounds.npy"))
        
        return {}
        # loading evs
        h5file = os.path.join(root, 'events.h5')
        evs = h5py.File(h5file, 'r')
        event_slicer = EventSlicer(evs)
        print(f"Total {(event_slicer.get_start_time_us()-event_slicer.t_offset)/1e6}secs \
           to {(event_slicer.get_final_time_us()-event_slicer.t_offset)/1e6}secs.")

        # loadings undistortion 
        calibstr = 'calib0'
        h5file = glob.glob(os.path.join(root, f'rectify_map_{calibstr}.h5'))[0]
        rmap = h5py.File(os.path.join(root, h5file), 'r')
        rectify_map = np.array(rmap['rectify_map'])  # (H, W, 2)
        rmap.close()
        poses_gt_us = np.loadtxt(os.path.join(root, "stamped_groundtruth_us.txt"), skiprows=1)
        
        img0 = cv2.imread(os.path.join(root, "images", "frame_0000000000.png"))
        Height, Width = img0.shape[0], img0.shape[1]     
        # Height = 480
        # Width = 640
        print(Height, Width)
        
        tss_imgs_us = np.loadtxt(os.path.join(root, 'images_timestamps_us.txt'))
        image_filenames = find_files(os.path.join(root, "images"), ["*.png"])
        tss_gt_us = poses_gt_us[:, 0]
        # print(poses_gt_us.shape, tss_gt_us.size, tss_imgs_us.size, tss_imgs_us[-2:])
        bds = np.zeros((len(tss_imgs_us), 2))
        tss_all_poses_ns, all_trafos_c2w = quatList_to_poses_hom_and_tss(poses_gt_us)
        tss_all_poses_ns = [t * 1000 for t in tss_all_poses_ns]
        # all_trafos_c2w = rub_from_rdf(all_trafos_c2w[:, :3, :])
        # check_rot_batch(all_trafos_c2w[:, :3, :])
        tss_imgs_us = tss_imgs_us[430:971] 
        image_filenames = image_filenames[430:971]
        # print(tss_gt_us.shape, tss_imgs_us.shape)
        print(all_trafos_c2w.shape)
        all_trafos_c2w[..., 0:3, 1:3] *= -1
        all_trafos_c2w[..., :, :] = torch.Tensor([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]]) @ all_trafos_c2w[..., :, :]
        all_trafos_c2w, transform_matrix = camera_utils.auto_orient_and_center_poses(
            poses=Tensor(all_trafos_c2w),
            method=self.config.orientation_method,
            center_method=self.config.center_method,
        )
        
        rot_slerp, trans_slerp = interpol_poses_slerp(
            tss_gt_us, all_trafos_c2w[:, :, :3], all_trafos_c2w[:, :, 3:], tss_imgs_us
        )
        # print(rot_slerp.shape, trans_slerp.shape)
        poses = np.concatenate((rot_slerp, trans_slerp), axis=2)
        poses = rub_from_rdf(poses[:, :3, :])
        poses = Tensor(poses)
        
        tss_imgs_us = tss_imgs_us[:np.searchsorted(tss_imgs_us, evs['t'][-1])]
        assert len(tss_imgs_us)
        cam_cnt = len(tss_imgs_us)
        pos_frames = np.zeros((cam_cnt - 1, Height, Width), dtype=np.float32)
        neg_frames = np.zeros((cam_cnt - 1, Height, Width), dtype=np.float32)
        # print(event_slicer.t_offset, evs['t'][-10], tss_imgs_us[-1])
        
        for i in tqdm(range(1, cam_cnt - 1), desc="Accumulating events"):  # t in (i-1, i] event accumulated
            events = event_slicer.get_events(tss_imgs_us[i - 1], tss_imgs_us[i])
            img = render_ev_accumulation(events['x'], events['y'], events['p'], Height, Width)
            # print(os.path.join(path, "img", "%04d" % i + ".png"))
            cv2.imwrite(os.path.join(root, "img", "%04d" % i + ".png"), img)
            accumulate_events(events['x'], events['y'], events['t'], events['p'], pos_frames[i-1], neg_frames[i-1])
            
        metadata = {"event_file": "",
                    "height": Height,
                    "width": Width,
                    'pos_frames': Tensor(pos_frames),
                    'neg_frames': Tensor(neg_frames),
                    'is_color': False,
                    }

        # if self.config.eval_mode == "fraction":
        #     i_train, i_eval = get_train_eval_split_fraction(img_files, self.config.train_split_fraction)
        # elif self.config.eval_mode == "filename":
        #     i_train, i_eval = get_train_eval_split_filename(img_files)
        # elif self.config.eval_mode == "interval":
        #     i_train, i_eval = get_train_eval_split_interval(img_files, self.config.eval_interval)
        # elif self.config.eval_mode == "all":
        #     CONSOLE.log(
        #         "[yellow] Be careful with '--eval-mode=all'. If using camera optimization, the cameras may diverge in the current implementation, giving unpredictable results."
        #     )
        #     i_train, i_eval = get_train_eval_split_all(img_files)
        # else:
        #     raise ValueError(f"Unknown eval mode {self.config.eval_mode}")
        
        # if split == "train":
        #     indices = i_train
        # else:
        #     indices = i_eval

        # transform_matrix = torch.eye(4)

        # Scale poses
        scale_factor = 1.0
        if self.config.auto_scale_poses:
            scale_factor /= float(torch.max(torch.abs(poses[:, :3, 3])))
        scale_factor *= self.config.scale_factor
        #CONSOLE.print(scale_factor)

        poses[:, :3, 3] *= scale_factor

        # Chose image_filenames and poses based on split, but after auto orient and scaling the poses.
        # if split != "train":
        #     img_files = [img_files[i] for i in indices]
        #     idx_tensor = torch.tensor(indices, dtype=torch.long)
        #     poses = poses[idx_tensor]
        #CONSOLE.print(image_filenames)
        #CONSOLE.print(poses)


        # in x,y,z order
        # assumes that the scene is centered at the origin
        aabb_scale = self.config.scene_scale
        scene_box = SceneBox(
            aabb=torch.tensor([[-aabb_scale, -aabb_scale, -aabb_scale], [aabb_scale, aabb_scale, aabb_scale]], dtype=torch.float32)
        )

        # intrinsic files
        calibdata = json.load(open(os.path.join(root, f"calib_undist_{calibstr}.json"), 'r'))['intrinsics_undistorted'][0]
        print(calibdata)
        fx = calibdata['fx']
        fy = calibdata['fy']
        cx = calibdata['cx']
        cy = calibdata['cy']
        distortion_params = torch.zeros(6)

        cameras = Cameras(
            camera_to_worlds=Tensor(poses[:, :3, :4]), 
            fx=fx, fy=fy, cx=cx, cy=cy, 
            distortion_params=distortion_params, 
            height=Height, width=Width,
        )

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
            dataparser_scale=scale_factor,
            dataparser_transform=transform_matrix,
            metadata=metadata,
        )
        return dataparser_outputs


if __name__ == "__main__":
    config = ENeRFDataparserConfig()
    parser = config.setup()
    outputs = parser.get_dataparser_outputs()
    # print(outputs.metadata)