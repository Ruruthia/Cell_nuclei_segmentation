from typing import Dict, Any, Optional

import pytorch_lightning as pl
from backbones_unet.model.unet import Unet
from pytorch_toolbelt.losses import BinaryFocalLoss
from torch.optim import Adam, lr_scheduler


# TODO: Monitor more meaningful metrics

class UNetLit(pl.LightningModule):
    """ A pytorch lightning wrapper for UNet11.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initializes the model with hyperparameters.

        Args:
            config:
                A dict of model hyperparameters. It should contain following fields:
                lr - learning rate of Adam optimizer
                eps - term added to denominator to improve numerical stability in Adam optimizer
                step_size - period of learning rate decay in scheduler
                gamma - multiplicative factor of learning rate decay in scheduler
        """
        super().__init__()
        if config:
            self.lr = config["lr"]
            self.eps = config["eps"]
            self.step_size = config["step_size"]
            self.gamma = config["gamma"]

        self.model = Unet(
            backbone='efficientnet_b0',  # backbone network name, see https://huggingface.co/docs/timm/results
            in_channels=1,  # input channels (1 for gray-scale images, 3 for RGB, etc.)
            num_classes=1,  # output channels (number of classes in your dataset)
        )
        self.loss_fn = BinaryFocalLoss()

    def forward(self, x):
        outputs = self.model(x)
        return outputs

    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(), lr=self.lr, eps=self.eps)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.float()
        outputs = self(x)
        loss = self.loss_fn(outputs, y)
        preds = (outputs > 0.5).float()
        self.log("train_loss", loss)
        self.log("train_acc", (preds == y).float().mean(), on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.float()
        outputs = self(x)
        loss = self.loss_fn(outputs, y)
        preds = (outputs > 0.5).float()
        self.log("val_loss", loss)
        self.log("val_acc", (preds == y).float().mean(), on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.float()
        outputs = self(x)
        loss = self.loss_fn(outputs, y)
        preds = (outputs > 0.5).float()
        self.log("test_loss", loss)
        self.log("test_acc", (preds == y).float().mean(), on_step=False, on_epoch=True)
        return loss
