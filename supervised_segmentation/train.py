import argparse
import os
from pathlib import Path
from typing import Dict
import yaml
import numpy as np
from tabulate import tabulate
import pandas as pd

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelSummary
import albumentations as A
import segmentation_models_pytorch as smp
import torchmetrics
import wandb

from data.mapping_dataset import MappingDatasetAlbu, DATASET_MEAN, DATASET_STD
import utils


class CardiacSegmentation(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        self.norm_max_pixel_value = self.config.get("norm_max_pixel_value", 1.0)

        self.num_classes = 3
        self.class_labels = ["background", "epicardial", "endocardial"]

        # Instantiating encoder and loading pretrained weights
        encoder_weights = self.config["model"].get("encoder_weights", None)
        
        # Only supervised ImageNet weights can be loaded when instantiating an smp.Unet:
        if encoder_weights in ['imagenet', 'supervised-imagenet']:
            smp_loaded_encoder_weights = 'imagenet'
        else:
            smp_loaded_encoder_weights = None

        self.model =smp.Unet(
            encoder_name=self.config["model"]["encoder_name"],           # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=smp_loaded_encoder_weights,     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=1,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=self.num_classes,            # model output channels (number of classes in your dataset)
        )

        self.loss = smp.losses.__dict__[self.config["loss"]](smp.losses.MULTICLASS_MODE)
        
        self.train_iou = torchmetrics.JaccardIndex(num_classes=self.num_classes, ignore_index=0) 
        self.val_iou = torchmetrics.JaccardIndex(num_classes=self.num_classes, ignore_index=0)     

        self.example_input_array = torch.zeros((1, 1,224,224))
    

    def forward(self, batch: torch.Tensor) -> torch.Tensor:  # type: ignore
        return self.model(batch)

    def setup(self, stage=0):
        self.train_dataset = MappingDatasetAlbu(self.config["dataset_root"])
        self.val_dataset = MappingDatasetAlbu(self.config["dataset_root"], 
                                              split='val', 
                                              mapping_only=self.config['val_mapping_only'])
        self.test_dataset = MappingDatasetAlbu(self.config["dataset_root"],
                                               split='test',
                                               mapping_only=self.config['test_mapping_only'])
        print("Number of  train samples = ", len(self.train_dataset))
        print("Number of val samples = ", len(self.val_dataset))
        print("Number of test samples = ", len(self.test_dataset))

    def train_dataloader(self):
        train_aug = A.Compose([
            A.RandomResizedCrop(224, 224),
            # A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(mean=(DATASET_MEAN,), std=(DATASET_STD,), max_pixel_value=self.norm_max_pixel_value),
        ])
        self.train_dataset.transforms = train_aug

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config["batch_size"],
            num_workers=self.config["num_workers"],
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )

        print("Train dataloader = ", len(train_loader))
        return train_loader

    def val_dataloader(self):
        val_aug = A.Compose([
            A.RandomResizedCrop(224, 224),
            A.Normalize(mean=(DATASET_MEAN,), std=(DATASET_STD,), max_pixel_value=self.norm_max_pixel_value),
        ])
        self.val_dataset.transforms = val_aug

        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config["batch_size"],
            num_workers=self.config["num_workers"],
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )

        print("Val dataloader = ", len(val_loader))
        return val_loader

    def configure_optimizers(self):
        optimizer = utils.object_from_dict(
            self.config["optimizer"],
            params=[x for x in self.model.parameters() if x.requires_grad],
        )

        scheduler = utils.object_from_dict(self.config["scheduler"], optimizer=optimizer)
        self.optimizers = [optimizer]

        return self.optimizers, [scheduler]

    def training_step(self, batch, batch_idx):
        features, targets = batch
        targets = targets.to(torch.int64)

        logits = self.forward(features)
        total_loss = self.loss(logits, targets)

        self.log("train_loss", total_loss)
        self.log("train_iou", self.train_iou(preds=logits, target=targets), on_epoch=True)
        self.log("lr",  self._get_current_lr())

        return total_loss

    def _get_current_lr(self) -> torch.Tensor:
        lr = [x["lr"] for x in self.optimizers[0].param_groups][0]  # type: ignore
        return torch.Tensor([lr])[0]

    def validation_step(self, batch, batch_id):
        features, targets = batch
        targets = targets.to(torch.int64)

        logits = self.forward(features)
        loss = self.loss(logits, targets)
        # "val_loss" and "val_iou" logged to allow easy comparison with older runs
        self.log("val_loss", loss) 
        self.log("val_iou", self.val_iou(preds=logits, target=targets))
        self.log("val_metrics/loss", loss)
        self.log("val_metrics/mean_iou", self.val_iou(preds=logits, target=targets))
        per_class_ious = torchmetrics.functional.jaccard_index(logits, targets, absent_score=np.NaN, 
                                                     num_classes=self.num_classes, average="none")
        for i in range(self.num_classes):
            self.log(f"val_metrics/{self.class_labels[i]}_iou", per_class_ious[i])
        

        if batch_id==0:
            # Wandb.Image expects a dictionary like {0: "background", 1:"epicardial", 2:"endocardial"}
            # Here we convert the list self.class_labels to such dictionary
            class_labels_dict = {id:label for id, label in enumerate(self.class_labels)}

            def wb_mask(bg_img, pred_mask, true_mask):
                return wandb.Image(bg_img, masks={
                    "prediction" : {"mask_data" : pred_mask, "class_labels" : class_labels_dict},
                    "ground truth" : {"mask_data" : true_mask, "class_labels" : class_labels_dict}})

            bg_img = features[0, 0].detach().cpu().numpy()
            bg_img -= bg_img.min()
            bg_img /= bg_img.max()
            bg_img = np.expand_dims(bg_img, -1)
            # print("bg_img[0].shape", bg_img.shape)
            # print("bg_img[0].min", bg_img.min())
            # print("bg_img[0].max", bg_img.max())
            # print("bg_img[0].dtype", bg_img.dtype)
            pred_mask = torch.argmax(logits[0].detach(), dim=0).cpu().numpy().astype(np.uint8)
            # print("logits[0].shape", logits[0].shape)
            # print("pred_mask.shape", pred_mask.shape)
            # print("pred_mask.min", pred_mask.min())
            # print("pred_mask.max", pred_mask.max())
            # print("pred_mask.dtype", pred_mask.dtype)
            true_mask = targets[0].detach().cpu().numpy().astype(np.uint8)
            # print("true_mask.shape", true_mask.shape)
            # print("true_mask.min", true_mask.min())
            # print("true_mask.max", true_mask.max())
            # print("true_mask.dtype", true_mask.dtype)
            wandb.log({"example_outputs": wb_mask(bg_img, pred_mask, true_mask)})

        return loss

    # def validation_epoch_end(self, outputs):
    #     # outputs. [tensor(-0.1065, device='cuda:0'), tensor(-0.1557, device='cuda:0')]
    #     if len(outputs[0]["val_iou"].shape) == 0:
    #         avg_val_iou = torch.stack([x["val_iou"] for x in outputs]).mean()
    #     else:
    #         avg_val_iou = torch.cat([x["val_iou"] for x in outputs]).mean()

    #     self.log("val_iou", avg_val_iou)

    def on_train_epoch_start(self) -> None:
        if self.current_epoch == (self.config["trainer"]["max_epochs"] - self.config["trainer"]["mapping_only_epochs"]):
            print("=============================")
            print("Running predictions on the test dataset with the best model before training on mapping images")
            from supervised_segmentation.inference import generate_prediction_pdfs
            results_df = generate_prediction_pdfs(checkpoint_path=self.trainer.checkpoint_callback.best_model_path, 
                                                  output_folder=self.trainer.loggers[0].log_dir,
                                                  remove_individual_pdfs=True, 
                                                  filename_suffix=f'_ep={self.current_epoch:03d}')
            resutlts_dict = {"test/" + k: v for k,v in results_df.mean().to_dict().items()}
            resutlts_dict["epoch"] = self.trainer.current_epoch
            wandb.log(resutlts_dict)
            self.train()

            print("\n=============================")
            print("Continue training on mapping images")
            print("=============================")
            self.train_dataset.to_mapping_only()
            self.trainer.reset_train_dataloader(self) # Very important line! If omitted, the trainer won't call the validation loop (without any warning or error!!!)
        # return super().on_epoch_start()

    def on_after_backward(self) -> None:
        # Cancel gradients for the encoder in the first few epochs
        if self.current_epoch < self.config['model'].get("freeze_encoder_weights_epochs", 0):
            for name, p in self.model.named_parameters():
                if "encoder." in name:
                    p.grad = None
        return super().on_after_backward()

def main():
    print("Running on:")
    os.system('/usr/bin/hostname')
    default_config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'config.yaml') 
    config = utils.get_config(default_config_path)

    pipeline = CardiacSegmentation(config)

    tb_logger = pl.loggers.TensorBoardLogger("artifacts", name=config["experiment_name"], log_graph=True)
    wandb_logger = pl.loggers.WandbLogger(name=config["experiment_name"], project='Cardiac-Mapping', save_dir="artifacts")
    #pl.LightningModule.hparams is set by pytorch lightning when calling save_hyperparameters
    tb_logger.log_hyperparams(pipeline.hparams)
    wandb_logger.log_hyperparams(pipeline.hparams)
    checkpoint = pl.callbacks.ModelCheckpoint(
            dirpath=os.path.join(tb_logger.log_dir, "checkpoints"),
            save_last=True,
            save_top_k=1,
            monitor='val_metrics/mean_iou',
            mode='max',
            every_n_epochs=10,
        )
    
    model_summary = ModelSummary(max_depth = 3)


    trainer = pl.Trainer(gpus=0 if config["trainer"]["gpus"] == '0' or not torch.cuda.is_available() else config["trainer"]["gpus"],
                        max_epochs=config["trainer"]["max_epochs"],
                        precision=config["trainer"]["precision"] if torch.cuda.is_available() else 32,
                        logger=[tb_logger, wandb_logger],
                        callbacks=[checkpoint, model_summary],
                        log_every_n_steps=1,
                        gradient_clip_val = config["trainer"]["gradient_clip_val"]
                        )

    trainer.fit(pipeline)

    print("=============================")
    from supervised_segmentation.inference import generate_prediction_pdfs

    if checkpoint.best_model_path == '':
        model_to_eval = checkpoint.last_model_path
        print(f"Running predictions on the test dataset with the last model {model_to_eval}")
    else:
        model_to_eval = checkpoint.best_model_path
        print(f"Running predictions on the test dataset with the best model {model_to_eval}")

    
    results_df = generate_prediction_pdfs(checkpoint_path=model_to_eval, 
                                          dataset_root=config["dataset_root"],
                                          output_folder=tb_logger.log_dir,
                                          remove_individual_pdfs=True,
                                          filename_suffix=f'_ep={trainer.current_epoch:03d}')
    resutlts_dict = {"test/" + k: v for k,v in results_df.mean().to_dict().items()}
    resutlts_dict["epoch"] = trainer.current_epoch
    wandb.log(resutlts_dict)

    summarized_resuts = pd.concat([results_df.mean().transpose(), 
                                   results_df.median().transpose()], 
                                   axis=1)
    summarized_resuts.columns = ["Mean", "Median"]
    print(tabulate(summarized_resuts, headers='keys', tablefmt="pipe"))


if __name__ == "__main__":
    main()
