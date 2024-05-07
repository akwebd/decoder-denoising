import lightning as pl
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torchmetrics import Metric, JaccardIndex
import numpy as np


class DecoderDenoisingModel(pl.LightningModule):
    def __init__(
        self,
        lr: float = 1e-4,
        optimizer: str = "adam",
        betas: tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 0.0,
        momentum: float = 0.9,
        arch: str = "unet",
        encoder: str = "resnet18",
        in_channels: int = 3,
        num_class: int = 4,
        mode: str = "decoder",
        noise_type: str = "scaled",
        noise_std: float = 0.22,
        loss_type: str = "l2",
        channel_last: bool = False,
    ):
        """Decoder Denoising Pretraining Model

        Args:
            lr: Learning rate
            optimizer: Name of optimizer (adam | adamw | sgd)
            betas: Adam beta parameters
            weight_decay: Optimizer weight decay
            momentum: SGD momentum parameter
            arch: Segmentation model architecture
            encoder: Segmentation model encoder architecture
            in_channels: Number of channels of input image
            mode: Denoising pretraining mode (encoder | encoder+decoder)
            noise_type: Type of noising process (scaled | simple)
            noise_std: Standard deviation/magnitude of gaussian noise
            loss_type: Loss function type (l1 | l2 | huber)
            channel_last: Change to channel last memory format for possible training speed up
        """
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.optimizer = optimizer
        self.betas = betas
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.noise_type = noise_type
        self.noise_std = noise_std
        self.channel_last = channel_last
        self.num_class = num_class

        # Initialize loss function
        self.loss_fn = self.get_loss_fn(loss_type)

        # Initalize network
        self.net = smp.create_model(
            arch,
            encoder_name=encoder,
            in_channels=in_channels,
            classes=num_class,
            encoder_weights="imagenet" if mode == "decoder" else None,
        )

        # Freeze encoder when doing decoder only pretraining
        if mode == "decoder":
            for child in self.net.encoder.children():  # type:ignore
                for param in child.parameters():
                    param.requires_grad = False
        elif mode != "encoder+decoder":
            raise ValueError(
                f"{mode} is not an available training mode. Should be one of ['decoder', 'encoder+decoder']"
            )

        # Change to channel last memory format
        # https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html
        if self.channel_last:
            self = self.to(memory_format=torch.channels_last)

    @staticmethod
    def get_loss_fn(loss_type: str):
        if loss_type == "l1":
            return F.l1_loss
        elif loss_type == "l2":
            return F.mse_loss
        elif loss_type == "huber":
            return F.smooth_l1_loss
        elif loss_type == "ce":
            return F.cross_entropy
            # return torch.nn.CrossEntropyLoss
        else:
            raise ValueError(
                f"{loss_type} is not an available loss function. Should be one of ['l1', 'l2', 'huber', 'ce']"
            )

    @torch.no_grad()
    def add_noise(self, x):
        # Sample noise
        noise = torch.randn_like(x)

        # Add noise to x
        if self.noise_type == "simple":
            x_noise = x + noise * self.noise_std
        elif self.noise_type == "scaled":
            x_noise = ((1 + self.noise_std**2) ** -0.5) * (x + noise * self.noise_std)
        else:
            raise ValueError(
                f"{self.noise_type} is not an available noise type. Should be one of ['simple', 'scaled']"
            )

        return x_noise, noise

    def denoise_step(self, x, mode="train"):
        if self.channel_last:
            x = x.to(memory_format=torch.channels_last)

        # Add noise to x
        x_noise, noise = self.add_noise(x)

        # Predict noise
        pred_noise = self.net(x_noise)

        # Calculate loss
        loss = self.loss_fn(pred_noise, noise)

        # Log
        self.log(f"{mode}_loss", loss, prog_bar=True)

        return loss

    def training_step(self, x, _):
        self.log(
            "lr",
            self.trainer.optimizers[0].param_groups[0]["lr"],  # type:ignore
            prog_bar=True,
        )
        return self.denoise_step(x, mode="train")

    def validation_step(self, x, _):
        return self.denoise_step(x, mode="val")

    def configure_optimizers(self):
        if self.optimizer == "adam":
            optimizer = Adam(
                self.net.parameters(),
                lr=self.lr,
                betas=self.betas,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer == "adamw":
            optimizer = AdamW(
                self.net.parameters(),
                lr=self.lr,
                betas=self.betas,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer == "sgd":
            optimizer = SGD(
                self.net.parameters(),
                lr=self.lr,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
            )
        else:
            raise ValueError(
                f"{self.optimizer} is not an available optimizer. Should be one of ['adam', 'adamw', 'sgd']"
            )

        # scheduler = CosineAnnealingLR(
        #     optimizer, T_max=self.trainer.estimated_stepping_batches  # type:ignore
        # )
        scheduler = ReduceLROnPlateau(optimizer, "min", 0.1, 3)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": "train_loss"
            },
        }

class FineTuningModel(DecoderDenoisingModel):
    def __init__(self, *args, **kwargs):
        """Fine Tuning Model

        Args:
            lr: Learning rate
            optimizer: Name of optimizer (adam | adamw | sgd)
            betas: Adam beta parameters
            weight_decay: Optimizer weight decay
            momentum: SGD momentum parameter
            arch: Segmentation model architecture
            encoder: Segmentation model encoder architecture
            in_channels: Number of channels of input image
            mode: Denoising pretraining mode (encoder | encoder+decoder)
            noise_type: Type of noising process (scaled | simple)
            noise_std: Standard deviation/magnitude of gaussian noise
            loss_type: Loss function type (l1 | l2 | huber)
            channel_last: Change to channel last memory format for possible training speed up
        """
        super().__init__(*args, **kwargs)
        self.acc_fn = JaccardIndex(ignore_index=0, task="multiclass", num_classes=4, average="micro")

    def forward(self, x):
        return self.net(x)

    def step(self, batch, mode="train"):
        x, y = batch
        y = y.squeeze(1).long()

        # Predict mask
        outputs = self.net(x)
        # softmax normalization
        probs = torch.softmax(outputs, dim=1)
        # take the highest probabilities
        pred_y = torch.argmax(probs, dim=1)

        # Calculate loss        
        loss = self.loss_fn(outputs, y)
        
        # update accuracy
        self.acc_fn.update(pred_y, y)

        # Log
        self.log(f"{mode}_loss", loss, prog_bar=True)

        return loss

    def training_step(self, batch):        
        self.log(
            "lr",
            self.trainer.optimizers[0].param_groups[0]["lr"],  # type:ignore
            prog_bar=True,
        )
        return self.step(batch, mode="train")

    def validation_step(self, batch):
        return self.step(batch, mode="val")
    
    def on_trainin_epoch_end(self):
        self.log('train_jaccard', self.acc_fn.compute(), prog_bar=True)
        
    def on_validation_epoch_end(self):
        self.log('val_jaccard', self.acc_fn.compute(), prog_bar=True)
        
    def predict_step(self, batch):
        image, path = batch
        
        # create contribution array
        arrone = torch.ones((1,4,*image.shape[-2:]))
        contribution = self.stitch_patches(self.sliding_window(arrone, 512, 256), arrone.size())
                
        # slice image into overlapping patches
        patches = self.sliding_window(image, 512, 256)
        
        # iterate over patches to get predictions
        for im, img in enumerate(patches):            
            # make prediction
            output = self(img[2].to(self.device))
            # update 
            patches[im][2] = output.to('cpu')
            
            # # check the prediction
            # probs = torch.softmax(output, dim=1)
            # # take the highest probabilities
            # pred_y = torch.argmax(probs, dim=1)
            # print(torch.unique(pred_y))
        
        # stitch back patches and scale contribution
        outputs = self.stitch_patches(patches, arrone.size())/contribution        
        
        # softmax normalization
        probs = torch.softmax(outputs, dim=1)
        # take the highest probabilities
        pred_y = torch.argmax(probs, dim=1)
        # print(torch.unique(pred_y))
        
        # save prediction
        np.save(path[0], pred_y.numpy())
        
        return pred_y
    
    def sliding_window(self, image, window_size, stride):
        patches = []
        _, _, H, W = image.size()

        # Iterate over rows
        for i in range(0, H - stride, stride):
            # Ensure that the last window covers the remaining part of the image
            if i + window_size > H:
                i = H - window_size
            # Iterate over columns
            for j in range(0, W - stride, stride):
                # Ensure that the last window covers the remaining part of the image
                if j + window_size > W:
                    j = W - window_size
                # Extract patch
                patch = image[:, :, i:i+window_size, j:j+window_size]
                patches.append([i, j, patch])  # Store patch position for stitching later

        return patches

    def stitch_patches(self, patches, image_size):
        stitched_image = torch.zeros(*image_size)

        for i, j, patch in patches:
            _, _, h, w = patch.size()
            stitched_image[:, :, i:i+h, j:j+w] += patch

        return stitched_image