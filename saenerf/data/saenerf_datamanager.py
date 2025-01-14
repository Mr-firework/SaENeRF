
from saenerf.data.saenerf_dataset import SaENeRFDataset
from saenerf.data.saenerf_dataparser import SaENeRFDataParserConfig
from saenerf.saenerf_sampler import saenerf_sampler

from dataclasses import dataclass, field
from typing import Dict, Literal, Tuple, Type, Union
import torch
from tqdm import tqdm

from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.utils.dataloaders import CacheDataloader
from nerfstudio.model_components.ray_generators import RayGenerator
from nerfstudio.data.utils.dataloaders import CacheDataloader, FixedIndicesEvalDataloader, RandIndicesEvalDataloader
from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManager,
    VanillaDataManagerConfig
)


@dataclass
class SaENeRFDataManagerConfig(VanillaDataManagerConfig):

    _target: Type = field(default_factory=lambda: SaENeRFDataManager)
    """Target class to instantiate."""
    train_num_images_to_sample_from: int = 1000
    """Number of images to sample during training iteration."""
    train_num_times_to_repeat_images: int = 100
    """When not training on all images, number of iterations before picking new
    images. If -1, never pick new images."""
    
    negative_sample_ratio : float = 0.05


class SaENeRFDataManager(VanillaDataManager):

    config: SaENeRFDataManagerConfig
    train_ev_cameras: Cameras

    def __init__(
        self,
        config: SaENeRFDataManagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,  # pylint: disable=unused-argument
    ):
        super().__init__(config=config, device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank)
        
    
    def setup_train(self):
        """Sets up the data loaders for training"""
        assert self.train_dataset is not None
        CONSOLE.print("Setting up training dataset...")
        # self.train_image_dataloader = CacheDataloader(
        #     self.train_dataset,
        #     num_images_to_sample_from=self.config.train_num_images_to_sample_from,
        #     num_times_to_repeat_images=self.config.train_num_times_to_repeat_images,
        #     device=self.device,
        #     num_workers=self.world_size * 4,
        #     pin_memory=True,
        #     collate_fn=self.config.collate_fn,
        #     exclude_batch_keys_from_device=self.exclude_batch_keys_from_device,
        # )
        # self.iter_train_image_dataloader = iter(self.train_image_dataloader)
        # self.train_pixel_sampler = self._get_pixel_sampler(self.train_dataset, self.config.train_num_rays_per_batch)
        # self.train_ray_generator = RayGenerator(self.train_dataset.cameras.to(self.device))

        # event setup
        metadata = self.train_dataparser_outputs.metadata
        if "neg_ratio" in metadata.keys():
            self.config.negative_sample_ratio = metadata["neg_ratio"]
        h, w = metadata["height"], metadata["width"]
        is_colored_events = metadata['is_color']
        
        
        events_prefix_sum = metadata["events_prefix_sum"]
        win_size_ratio = metadata["win_size_ratio"]
        self.event_dataset = SaENeRFDataset(events_prefix_sum, win_size_ratio)
        
        self.train_event_dataloader = CacheDataloader(self.event_dataset,
            num_images_to_sample_from=self.config.train_num_images_to_sample_from,
            num_times_to_repeat_images=self.config.train_num_times_to_repeat_images,
            device=self.device,
            num_workers=self.world_size * 2,
            pin_memory=True,
            collate_fn=self.config.collate_fn,
            exclude_batch_keys_from_device=[],
        )
        self.train_event_iter = iter(self.train_event_dataloader)

        color_mask = torch.zeros((h, w), device=self.device, dtype=torch.bool)[..., None].tile(1, 1, 3)
        if is_colored_events:
            color_mask[0::2, 0::2, 0] = 1 # R
            color_mask[0::2, 1::2, 1] = 1 # G
            color_mask[1::2, 0::2, 1] = 1 # G
            color_mask[1::2, 1::2, 2] = 1 # B
        else:
            color_mask[:, :, :] = 1

        self.color_mask = color_mask
        assert "ev_cameras" in metadata.keys()
        self.train_ev_cameras = self.train_dataparser_outputs.metadata["ev_cameras"]
        self.train_ray_generator_ev = RayGenerator(self.train_ev_cameras.to(self.device))
        # print(self.train_ev_cameras.image_height, self.train_ev_cameras.image_width)
        
    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the train dataloader."""
        self.train_count += 1
        # image_batch = next(self.iter_train_image_dataloader)
        # assert self.train_pixel_sampler is not None
        # assert isinstance(image_batch, dict)
        # batch = self.train_pixel_sampler.sample(image_batch)
        # ray_indices = batch["indices"]
        # ray_bundle = self.train_ray_generator(ray_indices)

        batch = next(self.train_event_iter)
        event_frames = batch["event_frames"]
        splits = batch["splits"]
        # CONSOLE.print("next_train", event_frames.shape, splits.shape)
        # print(event_frames.shape, splits.shape)
        assert event_frames.shape[0] == splits.shape[0] 
        indices = saenerf_sampler(event_frames, step, self.get_train_rays_per_batch(), neg_ratio=self.config.negative_sample_ratio)
        #CONSOLE.print("next_train incices", indices.shape)
        color_mask = self.color_mask[indices[:, 1], indices[:, 2], :]
        # CONSOLE.print(event_frames[indices[:, 0], indices[:, 1], indices[:, 2]].shape)
        event_gt = (event_frames[indices[:, 0], indices[:, 1], indices[:, 2]])[..., None].tile(1, 1, 3)
        ray_indices = torch.concat((
            torch.concat((splits[indices[:, 0], 0].reshape((-1, 1)), indices[:, 1:]), dim=-1),
            torch.concat((splits[indices[:, 0], 1].reshape((-1, 1)), indices[:, 1:]), dim=-1),
        ), dim=0).int()
        # CONSOLE.print("next_train ray_incices", ray_indices.shape, ray_indices[:5])
        # CONSOLE.print(self.train_ev_cameras.camera_to_worlds.shape)
        # CONSOLE.print("camera", self.train_dataparser_outputs.cameras.device, self.device)
        ray_bundle = self.train_ray_generator_ev(ray_indices)
        # CONSOLE.print(self.train_ev_cameras.camera_to_worlds.shape)

        # if False and len(self.train_dataparser_outputs.image_filenames) > 0:
        #     batch["image"] = self.images[indices[:, 0], indices[:, 1], indices[:, 2]]
        # else:
        #     batch["image"] = torch.ones_like(indices).float
        batch["ray_indices"] = indices
        batch["event_gt"] = event_gt
        batch["color_mask"] = color_mask

        return ray_bundle, batch
    
    def get_param_groups(self):
        return {}