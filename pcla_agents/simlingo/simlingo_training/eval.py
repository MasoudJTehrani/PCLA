import os
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from transformers import AutoProcessor, AutoTokenizer

from simlingo_training.config import TrainConfig
from simlingo_training.utils.logging_project import setup_logging
# from simlingo_training.callbacks.visualise import VisualiseCallback

@hydra.main(config_path=f"config", config_name="config", version_base="1.1")
def main(cfg: TrainConfig):
    
    torch.set_float32_matmul_precision("high")
    pl.seed_everything(42)
    
    # eval_mode = "QA"
    # eval_mode = "commentary"
    eval_mode = "Dreaming"

    qa_dataset = cfg.data_module.qa_dataset
    insteval_dataset = cfg.data_module.insteval_dataset
    load_path = '/YOUR_PATH/outputs/simlingo/checkpoints/epoch=013.ckpt'
    if load_path is not None:
        load_path_config = Path(load_path).parent.parent / '.hydra/config.yaml'
        cfg = OmegaConf.load(load_path_config)
    
    cfg.data_module.qa_dataset = qa_dataset
    cfg.data_module.insteval_dataset = insteval_dataset
    cfg.gpus = 1
    cfg.data_module.num_workers = 8
    cfg.data_module.batch_size = 64

    print(f'Eval mode: {eval_mode}')
    print(f'Checkpoint: {load_path}')
    print(f"Using {cfg.gpus} GPUs")
    
    if eval_mode == "QA" or eval_mode == "commentary":
        cfg.data_module.dreamer_dataset = None
        cfg.data_module.driving_dataset = None
        cfg.data_module.insteval_dataset = None 
    elif eval_mode == "Dreaming":
        cfg.data_module.dreamer_dataset = None
        cfg.data_module.driving_dataset = None
        cfg.data_module.qa_dataset = None
    
    if eval_mode == "QA":
        cfg.data_module.base_dataset.use_commentary = False
        cfg.data_module.base_dataset.use_qa = True
    elif eval_mode == "commentary":
        cfg.data_module.base_dataset.use_commentary = True
        cfg.data_module.base_dataset.use_qa = False
    elif eval_mode == "Dreaming":
        cfg.data_module.base_dataset.use_safety_flag = True
    
    # disable image augmentation
    cfg.data_module.base_dataset.img_augmentation = False
    
    # disable img_shift_augmentation
    cfg.data_module.base_dataset.img_shift_augmentation = False
    
    if "2B" in cfg.model.language_model.variant:
        processor = AutoTokenizer.from_pretrained(cfg.model.language_model.variant, trust_remote_code=True, use_fast=False)
    else:
        processor = AutoProcessor.from_pretrained(cfg.model.language_model.variant, trust_remote_code=True, use_fast=False)
    model_type_name = cfg.model.vision_model.variant.split('/')[1]
    cache_dir = f"pretrained/{(model_type_name)}"
    
    data_module = hydra.utils.instantiate(
        cfg.data_module, 
        processor=processor,
        encoder_variant=cfg.model.vision_model.variant,
        llm_variant=cfg.model.language_model.variant,
        predict=True,
        _recursive_=False
    )
    
    model = hydra.utils.instantiate(
        cfg.model,
        cfg_data_module=cfg.data_module,
        processor=processor,
        cache_dir=cache_dir,
        _recursive_=False
        )

    if cfg.checkpoint is not None:
        if os.path.isdir(cfg.checkpoint):
            state_dict = get_fp32_state_dict_from_zero_checkpoint(cfg.checkpoint)
        else:
            state_dict = torch.load(cfg.checkpoint, map_location="cpu")
        model.load_state_dict(state_dict)

        
    # print config
    print(OmegaConf.to_yaml(cfg))
    os.environ["WANDB_DISABLE_CODE"] = "True"

    
    # setup logging
    setup_logging(cfg)

    # resume training
    resume_path = "./checkpoints/last.ckpt"


    if os.path.exists(resume_path) and cfg.resume:
        resume_path = resume_path
    else:
        resume_path = None
    
    # setup lightning logger
    loggers = []

    strategy = cfg.strategy
    if strategy == "deepspeed_stage_2":
        strategy = pl.strategies.DeepSpeedStrategy(
            stage=2, loss_scale=cfg.fp16_loss_scale, logging_batch_size_per_gpu=cfg.data_module.batch_size
        )
  
    print(f"Number of GPUS: {cfg.gpus}")
    overfit = 0
    
    if cfg.gpus >= 1:
        trainer = Trainer(
            accelerator="gpu",
            benchmark=True,
            devices=cfg.gpus,
            gradient_clip_val=0.3,
            log_every_n_steps=20,
            logger=loggers,
            precision=cfg.precision,
            strategy=strategy,
            sync_batchnorm=True,
            max_epochs=cfg.max_epochs,
            overfit_batches=overfit,
            check_val_every_n_epoch=cfg.val_every_n_epochs,
        )

    if load_path is not None:
        trainer.predict(model, data_module, ckpt_path=f"{load_path}/")
    else:
        trainer.predict(model, data_module)

if __name__ == "__main__":
    main()