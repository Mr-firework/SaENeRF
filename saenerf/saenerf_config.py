
from __future__ import annotations

from saenerf.data.saenerf_dataparser  import SaENeRFDataParserConfig
from saenerf.data.saenerf_datamanager import SaENeRFDataManagerConfig
from saenerf.saenerf_pipeline    import SaENeRFPipelineConfig

from saenerf.saenerf_model   import SaENeRFModelConfig
from saenerf.enerf_model     import ENeRFModelConfig
from saenerf.eventnerf_model import EventNeRFModelConfig

from dataclasses import dataclass
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
from nerfstudio.engine.schedulers import (
    ExponentialDecaySchedulerConfig,
)
from nerfstudio.plugins.registry_dataparser import DataParserSpecification
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification
from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.config_utils import to_immutable_dict


saenerf   = MethodSpecification(
    config=TrainerConfig(
        method_name="saenerf", 
        steps_per_eval_batch=500000,
        steps_per_save=5000,
        max_num_iterations=5000 * 1,
        mixed_precision=True,
        pipeline=SaENeRFPipelineConfig(
            datamanager=SaENeRFDataManagerConfig(
                dataparser=SaENeRFDataParserConfig(),
                train_num_rays_per_batch=1024 * 2,
                eval_num_rays_per_batch=1024 * 2,
            ),
            model=SaENeRFModelConfig(eval_num_rays_per_chunk=8192,),
        ),
        optimizers={
            "fields": {
                "optimizer": RAdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=200000),
            },
            "evs_threashold" : {
                "optimizer": RAdamOptimizerConfig(lr=1e-5, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-7, max_steps=200000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 12,
                            websocket_port_default=7009),
        vis="viewer",
    ),
    description="SaENeRF",
)

enerf     = MethodSpecification(
    config=TrainerConfig(
        method_name="enerf", 
        steps_per_eval_batch=500000,
        steps_per_save=5000,
        max_num_iterations=5000 * 1,
        mixed_precision=True,
        pipeline=SaENeRFPipelineConfig(
            datamanager=SaENeRFDataManagerConfig(
                dataparser=SaENeRFDataParserConfig(),
                train_num_rays_per_batch=1024 * 2,
                eval_num_rays_per_batch=1024 * 2,
            ),
            model=ENeRFModelConfig(eval_num_rays_per_chunk=8192),
        ),
        optimizers={
            # TODO: consider changing the optimizers depending on your custom Model
            "fields": {
                "optimizer": RAdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=200000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 12,
                            websocket_port_default=7007),
        vis="viewer",
    ),
    description="E-NeRF method.",
)

eventnerf = MethodSpecification(
    config=TrainerConfig(
        method_name="eventnerf", 
        steps_per_eval_batch=500000,
        steps_per_save=5000,
        max_num_iterations=5000 * 1,
        mixed_precision=True,
        pipeline=SaENeRFPipelineConfig(
            datamanager=SaENeRFDataManagerConfig(
                dataparser=SaENeRFDataParserConfig(),
                train_num_rays_per_batch=1024 * 2,
                eval_num_rays_per_batch=1024 * 2,
            ),
            model=EventNeRFModelConfig(eval_num_rays_per_chunk=8192),
        ),
        optimizers={
            # TODO: consider changing the optimizers depending on your custom Model
            "fields": {
                "optimizer": RAdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=200000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 12,
                            websocket_port_default=7008),
        vis="viewer",
    ),
    description="Event Nerf method.",
)
