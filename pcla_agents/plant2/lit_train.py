import os
import random
import string
import hydra
from pathlib import Path
from omegaconf import OmegaConf

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger, WandbLogger, TensorBoardLogger
from pytorch_lightning import Trainer
import wandb

from util.logging import setup_logging, sync_wandb
# from dataloader import get_dataloader
from lit_module import LitHFLM

from dataset import generate_batch
from dataset import PlanTDataset
from torch.utils.data import DataLoader

@hydra.main(config_path=f"config", config_name="config")
def main(cfg):

    # print config
    print(OmegaConf.to_yaml(cfg))

    # setup debug mode
    overfit = 0.0
    if cfg.debug:
        os.environ["WANDB_MODE"] = "offline"
        cfg.expname = "debug"
        overfit = 5  # use only 5 fixed batches for debugging

    if cfg.overfit > 0:
        overfit = cfg.overfit

    # use data caching for ML-Cloud #TODO
    shared_dict = None
    if cfg.use_caching:
        from diskcache import Cache
        tmp_folder = os.environ.get('DS_LOCAL')
        # if tmp_folder is None:
        #     tmp_folder = f"/tmp/dataset_cache_{''.join(random.choices(string.ascii_uppercase + string.digits, k=5))}" # TODO TODO TODO
        print("Tmp folder for dataset cache: ", tmp_folder)
        # tmp_folder = tmp_folder
        # We use a local diskcache to cache the dataset on the faster SSD drives on our cluster.
        shared_dict = Cache(directory=tmp_folder ,size_limit=int(512 * 1024 ** 3))
        # shared_dict = Cache(size_limit=int(768 * 1024**3))

    # if we use mutliple GPUs and want wandb online it does need too much 
    # time on the MLCLoud and the training freezes or is too slow
    # log only local and sync afterwards with wandb sync [OPTIONS] [PATH]
    if cfg.gpus > 1:
        os.environ["WANDB_MODE"] = "offline"

    # setup logging
    seed = os.environ["SEED"] #cfg.seed
    print("The current seed is"+str(seed))
    pl.seed_everything(int(seed))
    setup_logging(cfg)

    # setup lightning logger
    csvlogger = CSVLogger(cfg.model.training.log_path, "CSVLogger")
    wandb.init(project=cfg.exp_folder_name, name="PlanT_2_" + os.environ.get("CHECKPOINT_ADDON", "") + "_" + str(seed))
    wandblogger = WandbLogger(
        project=cfg.exp_folder_name,
        name=cfg.wandb_name,
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        entity="seqdrive",
    )
    Path(f"{cfg.model.training.log_path}/TBLogger").mkdir(parents=True, exist_ok=True)
    TBlogger = TensorBoardLogger(cfg.model.training.log_path, name="TBLogger")

    # resume training
    resume_path = cfg.resume_path # os.environ["RESUME_PATH"] #"  TODO
    if os.path.exists(resume_path) and cfg.resume:
        resume_path = resume_path
    else:
        resume_path = None
    # checkpoint_path = None

    out_path = "{epoch:03d}_"
    if addon := os.environ.get("CHECKPOINT_ADDON"):
        out_path += addon + "_"
    out_path += str(seed)

    print("Checkpoint path: "+os.path.join(cfg.user.working_dir, "PlanT/checkpoints",out_path))

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=-1,
        dirpath=os.path.join(cfg.user.working_dir, "PlanT/checkpoints"),
        filename=out_path, # TODO config
        save_last=True,
        every_n_epochs=10,
        save_on_train_epoch_end=True
    )
    checkpoint_callback.CHECKPOINT_NAME_LAST = f"last_{seed}"

    dataset = PlanTDataset(os.environ.get('DS')+"/data", cfg, shared_dict=shared_dict)

    train_loader = DataLoader(
        dataset,
        shuffle=True,
        pin_memory=True,
        batch_size=cfg.model.training.batch_size,
        collate_fn=generate_batch,
        num_workers=cfg.model.training.num_workers,
    )

    val_loader = None

    GPT_model = LitHFLM(cfg=cfg)

    wandblogger.watch(GPT_model)

    if cfg.gpus > 1:
        replace_sampler_ddp = not cfg.custom_sampler
        trainer = Trainer(
            callbacks=checkpoint_callback,
            accelerator="gpu",
            devices=cfg.gpus,
            # strategy="ddp_find_unused_parameters_true",
            strategy="ddp",
            # replace_sampler_ddp=replace_sampler_ddp,
            logger=[csvlogger, wandblogger, TBlogger],
            log_every_n_steps=5,
            # resume_from_checkpoint=resume_path,
            check_val_every_n_epoch=2,
            max_epochs=cfg.model.training.max_epochs,
            overfit_batches=overfit,
        )
    else:
        trainer = Trainer(
            callbacks=checkpoint_callback,
            accelerator="gpu",
            devices=1,
            logger=[csvlogger, wandblogger, TBlogger],
            log_every_n_steps=2,
            # resume_from_checkpoint=resume_path,
            check_val_every_n_epoch=2,
            max_epochs=cfg.model.training.max_epochs,
            overfit_batches=overfit,
        )

    torch.set_float32_matmul_precision('high')

    trainer.fit(GPT_model, train_loader, val_loader, ckpt_path=resume_path)

    if cfg.gpus > 1:
        sync_wandb(cfg)
        # os.system('wandb sync ./wandb/offline*')


if __name__ == "__main__":
    main()
