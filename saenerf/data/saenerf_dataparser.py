
import os
import glob
import json
import numpy as np
from torch import Tensor
from torchtyping import TensorType
import torch

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Type, List, Any, Literal
from jaxtyping import Float
from copy import deepcopy

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
from tqdm import tqdm
from nerfstudio.utils.io import load_from_json

from saenerf.data.saenerf_utils import (
    event_load_eventnerf,
    get_eventnerf_gt_timestamp,
    event_split,
    find_files,
    parse_txt,
    check_rot_batch,
    rub_from_rdf,
    interpol_poses_slerp,
    accumulate_events,
    EventDictSlicer,
    create_event_prefix_sum,
    compose_poses,
)

@dataclass
class SaENeRFDataParserConfig(DataParserConfig):

    _target: Type = field(default_factory=lambda: SaENeRFDataParser)
    data: Path = Path("/data/dataset/eventnerf/nextgen/r/")
    # data: Path = Path("/data/dataset/eventnerf/nerf/lego/")
    orientation_method: Literal["pca", "up", "vertical", "none"] = "up"
    """The method to use for orientation."""
    center_method: Literal["poses", "focus", "none"] = "poses"
    """The method to use to center the poses."""
    scale_factor: float = 1.0
    scene_scale: float = 0.35
    auto_scale_poses: bool = False #True
    eval_mode: Literal["fraction", "filename", "interval", "all"] = "interval"
    train_split_fraction: float = 1.0
    eval_interval: int = 10
    
    interpolated_poses_num : int = 9
    interpolated_poses_method: Literal["time", "number"] = "time"
    neg_ratio : float = 0.05
    win_size_ratio : float = 0.05


@dataclass
class SaENeRFDataParser(DataParser):

    config: SaENeRFDataParserConfig

    def _generate_dataparser_outputs(self, split="train", **kwargs: Optional[Dict]):

        assert self.config.data.exists(), f"Data directory {self.config.data} does not exist."
        #dir_path = self.config.data / f"{split}"
        dir_path = self.config.data / "train"
        print(dir_path)
        
        Height = 260
        Width = 346

        # event file
        event_files = find_files(dir_path / "events", ["*.npz"])
        assert len(event_files) == 1, "event files not 1"

        events = event_load_eventnerf(event_files[0])
        event_slicer = EventDictSlicer(events)
        
        tss_gt_poses = get_eventnerf_gt_timestamp(events['t'])
        # print(len(tss_gt))
        # print((tss_gt[-5:]))
        
        if split == "train":
            if events['t'][-1] == 1000:  # synthetic nerf dataset
                tss_event_poses = tss_gt_poses
            else:
                assert self.config.interpolated_poses_num >= 0
                self.config.win_size_ratio = 0.1
                tss_event_poses = [tss_gt_poses[0], ]
                for i in range(1, len(tss_gt_poses)):
                    if self.config.interpolated_poses_method == "number":
                        events = event_slicer.get_events(tss_gt_poses[i - 1], tss_gt_poses[i])
                        uni_space = np.linspace(0, len(events["t"]) - 1, self.config.interpolated_poses_num + 2).astype(int)
                        tss_event_poses.extend(events['t'][uni_space[1:]])
                    else:
                        tss_uni_space = np.linspace(tss_gt_poses[i - 1], tss_gt_poses[i], self.config.interpolated_poses_num + 2)
                        tss_event_poses.extend(tss_uni_space[1:])
            print (len(tss_event_poses), tss_event_poses[:5])
            events_prefix_sum = create_event_prefix_sum(
                event_slicer, tss_event_poses, Height, Width, dir_path / "evs_img"
            )
        else:
            events_prefix_sum = []
            tss_event_poses = tss_gt_poses
        # rgb files
        img_files = find_files(dir_path / "rgb", ["*.png"])

        # pose filies
        pose_files = find_files(dir_path / "pose", ["*.txt"])
        print("pose size: ", len(pose_files))
        poses = []
        for i in range(0, len(pose_files)):
            pose = parse_txt(pose_files[i], (4, 4)).reshape(1, 4, 4)
            poses.append(pose)
        
        poses = np.array(poses).astype(np.float32).reshape((-1, 4, 4))
        poses[:, :3, :] = rub_from_rdf(poses[:, :3, :])
        poses, transform_matrix = camera_utils.auto_orient_and_center_poses(
            poses=Tensor(poses),
            method=self.config.orientation_method,
            center_method=self.config.center_method,
        )
        evs_rot_slerp, evs_trans_slerp = interpol_poses_slerp(
            tss_gt_poses, poses[:, :3, :3], poses[:, :3, 3:], tss_event_poses
        )
        poses_evs = compose_poses(evs_rot_slerp, evs_trans_slerp)
        poses_evs = torch.from_numpy(poses_evs)
        # poses = torch.from_numpy(poses)

        if self.config.eval_mode == "fraction":
            i_train, i_eval = get_train_eval_split_fraction(img_files, self.config.train_split_fraction)
        elif self.config.eval_mode == "filename":
            i_train, i_eval = get_train_eval_split_filename(img_files)
        elif self.config.eval_mode == "interval":
            i_train, i_eval = get_train_eval_split_interval(img_files, self.config.eval_interval)
        elif self.config.eval_mode == "all":
            CONSOLE.log(
                "[yellow] Be careful with '--eval-mode=all'. If using camera optimization, the cameras may diverge in the current implementation, giving unpredictable results."
            )
            i_train, i_eval = get_train_eval_split_all(img_files)
        else:
            raise ValueError(f"Unknown eval mode {self.config.eval_mode}")
        
        # if split == "train":
        #     indices = i_train
        # else:
        indices = i_eval

        # Scale poses
        scale_factor = 1.0
        if self.config.auto_scale_poses:
            scale_factor /= float(torch.max(torch.abs(poses_evs[:, :3, 3])))
        scale_factor *= self.config.scale_factor
        #CONSOLE.print(scale_factor)

        poses_evs[:, :3, 3] *= scale_factor

        # Chose image_filenames and poses based on split, but after auto orient and scaling the poses.
        img_files = [img_files[i] for i in indices]
        idx_tensor = torch.tensor(indices, dtype=torch.long)
        poses = poses[idx_tensor]
        #CONSOLE.print(image_filenames)
        #CONSOLE.print(poses)


        # in x,y,z order
        # assumes that the scene is centered at the origin
        aabb_scale = self.config.scene_scale
        scene_box = SceneBox(
            aabb=torch.tensor([[-aabb_scale, -aabb_scale, -aabb_scale], [aabb_scale, aabb_scale, aabb_scale]], dtype=torch.float32)
        )

        # intrinsic files
        intrinsic_files = find_files(dir_path/"intrinsics", ["*.txt"])
        intrinsic = Tensor(parse_txt(intrinsic_files[0], (-1, 4)))
        fx = intrinsic[0][0].clone()
        fy = intrinsic[1][1].clone()
        cx = intrinsic[0][2].clone()
        cy = intrinsic[1][2].clone()
        distortion_params = torch.zeros(6)
        if intrinsic.shape[0] == 5:
            distortion_params[0] = intrinsic[4][0].clone()
            distortion_params[1] = intrinsic[4][1].clone()

        cameras_event = Cameras(
            camera_to_worlds=poses_evs[:, :3, :4], 
            fx=fx, fy=fy, cx=cx, cy=cy, 
            distortion_params=distortion_params, 
            height=Height, width=Width,
        )
        cameras_rgb = Cameras(
            camera_to_worlds=poses[:, :3, :4], 
            fx=fx, fy=fy, cx=cx, cy=cy, 
            distortion_params=distortion_params, 
            height=Height, width=Width,
        )

        metadata = {
            "ev_cameras": cameras_event,
            "height": Height,
            "width": Width,
            'events_prefix_sum': events_prefix_sum,
            'win_size_ratio': self.config.win_size_ratio,
            "is_color": True,
        }
                
        dataparser_outputs = DataparserOutputs(
            image_filenames=img_files,
            cameras=cameras_rgb,
            scene_box=scene_box,
            dataparser_scale=scale_factor,
            dataparser_transform=transform_matrix,
            metadata=metadata,
        )
        return dataparser_outputs

if __name__ == "__main__":
    #print(Path("Hello ")/"s")
    config = SaENeRFDataParserConfig()
    parser = config.setup()
    outputs = parser.get_dataparser_outputs()
    # print(outputs.metadata)