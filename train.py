import warnings
import os
with warnings.catch_warnings(record=True) as w:   
    import torch
    import lightning as L
    import pytorch_lightning as pl
    from model import Unet
    from dataset import FishDataset
    from pytorch_lightning.loggers import TensorBoardLogger
#tensorboard --host 0.0.0.0 --logdir tb_logs
pl
torch.set_float32_matmul_precision('medium')
if __name__ == "__main__":
    with warnings.catch_warnings(record=True) as w:   
        logger = TensorBoardLogger("/Users/charlie/Documents/ML/U-Net/tb_logs", name="U-net", default_hp_metric=False)
        # model = Unet(channels=3, lr=0.09120108393559097)
        model = Unet(channels=3, lr=1e-3)
        dataLoader = FishDataset(
            x_dir="/Users/charlie/Documents/ML/U-Net/Trout/Trout",
            y_dir="/Users/charlie/Documents/ML/U-Net/Trout/Trout GT",
            batch_size=8,
            num_workers=16,
        )
        trainer = pl.Trainer(
            benchmark=True,
            enable_checkpointing=True,
            logger=logger,
            accelerator='gpu',
            devices=1,
            min_epochs=1,
            max_epochs=250,
            precision=32,
        )
        lr_finder = trainer.lr_find(model)
        print(new_lr = lr_finder.suggestion())
        # trainer.fit(model, dataLoader)
