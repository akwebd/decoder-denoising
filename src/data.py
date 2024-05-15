import os
from glob import glob
from typing import Callable
import numpy as np

import lightning as pl
import torch
import torch.utils.data as data
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import (Compose, Lambda, RandomCrop,
                                    RandomHorizontalFlip, Resize, 
                                    ToTensor, Normalize, ColorJitter, 
                                    RandomApply, RandomGrayscale)


class SSLDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root: str,
        size: int = 512,
        crop: int = 224,
        num_val: int = 1000,
        batch_size: int = 32,
        workers: int = 4,
    ):
        """Basic data module

        Args:
            root: Path to image directory
            size: Size of resized image
            crop: Size of image crop
            num_val: Number of validation samples
            batch_size: Number of batch samples
            workers: Number of data workers
        """
        super().__init__()
        self.root = root
        self.num_val = num_val
        self.batch_size = batch_size
        self.workers = workers
        

        self.transforms = Compose(
            [
                # Resize(size),
                RandomCrop(crop), #I have enought data not to use crop
                RandomHorizontalFlip(),
                ToTensor(),
                Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                # Lambda(lambda t: (t * 2) - 1),  # Scale to [-1, 1]
            ]
        )

    def setup(self, stage="fit"):
        if stage == "fit":
            dataset = SSLDataset(self.root, self.transforms)

            self.train_dataset, self.val_dataset = data.random_split(
                dataset,
                [len(dataset) - self.num_val, self.num_val],
                generator=torch.Generator().manual_seed(42),
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
            pin_memory=True,
            drop_last=False,
        )


class SSLDataset(data.Dataset):
    def __init__(self, root: str, transforms: Callable):
        """Image dataset from directory

        Args:
            root: Path to directory
            transforms: Image augmentations
        """
        super().__init__()
        self.root = root
        self.paths = [
            f for f in glob(f"{root}/**/*", recursive=True) if os.path.isfile(f) and any(f.lower().endswith(end) for end in ['.jpg', '.png'])
        ]
        self.transforms = transforms

        print(f"Loaded {len(self.paths)} images from {root}")

    def __getitem__(self, index):
        img = Image.open(self.paths[index]).convert("RGB")
        img = self.transforms(img)
        return img

    def __len__(self):
        return len(self.paths)



class SupervisedDataModule(SSLDataModule):
    def __init__(
            self,
            root: str,
            size: int = 512,
            crop: int = 224,
            num_val: int = 1000,
            batch_size: int = 32,
            workers: int = 4,
        ):
        super().__init__(root, size, crop, num_val, batch_size, workers)
        self.transforms = Compose(
            [
                # Resize(size),
                RandomCrop(crop),
                RandomHorizontalFlip(),
                ToTensor(),
                Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                # Lambda(lambda t: (t * 2) - 1),  # Scale to [-1, 1]
            ]
        )

    def setup(self, stage="fit"):
        if stage == "fit":
            dataset = SupervisedDataset(self.root, self.transforms)

            self.train_dataset, self.val_dataset = data.random_split(
                dataset,
                [len(dataset) - self.num_val, self.num_val],
                generator=torch.Generator().manual_seed(42),
            )
        
        
        if stage == "predict":
            trf = Compose([*self.transforms.transforms[-2:]])
            self.predict_dataset = PredictionDataset(self.root, trf)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
            pin_memory=True,
            drop_last=False,
        )
        
    
    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
            pin_memory=True,
            drop_last=False,
        )


class SupervisedDataset(data.Dataset):
    def __init__(self, root: str, transforms: Callable):
        """Image dataset from directory

        Args:
            root: Path to directory
            transforms: Image augmentations
        """
        super().__init__()
        self.root = root
        
        self.paths = [
            f for f in glob(f"{root}/**/*", recursive=True) if os.path.isfile(f)
        ]

        self.images = sorted([x for x in self.paths if ('image' in x) and any(x.lower().endswith(end) for end in ['.jpg', '.png'])])
        
        self.gts = sorted([x for x in self.paths if ('label' in x) and any(x.lower().endswith(end) for end in ['.jpg', '.png', '.npy'])])
        # trf= transforms.transforms
        # trf.pop(1)
        self.transforms = transforms
        self.transforms2 = Compose(self.transforms.transforms[:-1])

        print(f"Loaded {len(self.images)} images from {root}")

    def __getitem__(self, index):
        img_path = self.images[index]
        img = Image.open(img_path).convert("RGB")
        rand_state = torch.get_rng_state()
        img = self.transforms(img)
        
        # Assuming `mask_array` is your NumPy array representing the mask
        mask_path = img_path.replace("image", "label").replace(".jpg", "_gt.npy").replace(".JPG", "_gt.npy")
        if os.path.exists(mask_path):
            mask_array = np.load(mask_path)
        else:
            # not all images have masks, because some samples do not contain segmented minerals
            mask_array = np.zeros(img.shape[-2:])

        # Convert NumPy array to PIL Image
        mask = Image.fromarray(mask_array.astype(np.uint8), mode="L")
        torch.set_rng_state(rand_state)
        mask = self.transforms2(mask)*255
        return img, mask

    def __len__(self):
        return len(self.images)

class PredictionDataset(SupervisedDataset):
    def __init__(self, root: str, transforms: Callable):
        """Image dataset from directory

        Args:
            root: Path to directory
            transforms: Image augmentations
        """
        super().__init__(root, transforms)
        # make directory for predictions
        os.makedirs(os.path.join(self.root, 'label'), exist_ok=True)

    def __getitem__(self, index):
        img_path = self.images[index]
        img = Image.open(img_path).convert("RGB")
        img = self.transforms(img)
        

        save_path = img_path.replace(os.path.join(self.root, 'image'), 
                                     os.path.join(self.root, 'label')).replace(".jpg", 
                                                                               "_pred.npy").replace(".JPG", 
                                                                                                    "_pred.npy")
        
        return img, save_path
    
class SplitMaskDataModule(SSLDataModule):
    def __init__(
        self,
        root: str,
        size: int = 512,
        crop: int = 224,
        num_val: int = 1000,
        batch_size: int = 32,
        workers: int = 4,
        augms = [RandomApply([ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
                 RandomGrayscale(p=0.2)]
                 
            # RandomGrayscale(p=0.2),
    ):
        """Basic data module

        Args:
            root: Path to image directory
            size: Size of resized image
            crop: Size of image crop
            num_val: Number of validation samples
            batch_size: Number of batch samples
            workers: Number of data workers
        """
        super().__init__(root, size, crop, num_val, batch_size, workers)
        s=1
        color_jitter = ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        transforms = [
            # do not use any geometric transforms, because then alignment after encoder is not meaningful
            # transforms.RandomResizedCrop(size=size),
            # transforms.RandomHorizontalFlip(),
            RandomCrop(crop),
            *augms,
            ToTensor(),
            Normalize(mean=[0.5]*3, std=[0.5]*3)
            ]
        # for views we use all transforms except crop
        self.transforms = Compose(transforms)
        

    def setup(self, stage="fit"):
        if stage == "fit":
            dataset = SplitMaskDataset(self.root, self.transforms)

            self.train_dataset, self.val_dataset = data.random_split(
                dataset,
                [len(dataset) - self.num_val, self.num_val],
                generator=torch.Generator().manual_seed(42),
            )

class SplitMaskDataset(SSLDataset):
    '''
    Dataset should return two photometric views of the same image.
    Three variants (scenarios) for decoder loss will be calculated:
    1. L1 loss between predicted masks (the final task)
    2. L1 loss between reconstructed images (image1 VS image2)
    3. L1 loss between masked reconstructed images (imag1, image2 VS image)

    '''
    def __init__(self, root, transforms, tile_size=16):
        super().__init__(root, transforms)
        # for views we use all transforms except cropping
        self.transformsX = Compose(transforms.transforms[1:])
        # for original image we do not use photometric transforms
        self.transforms = Compose(transforms.transforms[-2:])
        # for image mask we do not use any transform, just convertion to tensor
        self.transformsM = transforms.transforms[-2]
        # crop operation
        self.cropT = transforms.transforms[0]
        self.tile_size = tile_size    
    
    def __getitem__(self, index):
        img_path = self.paths[index]
        # load image
        img = Image.open(img_path).convert("RGB")
        # create a crop
        img = self.cropT(img)
        # img_ = np.array(img)
        # create mask
        mask = self.transformsM(self.get_mask(img))
        # apply mask twice
        gt1, gt2 = self.transformsX(img.copy()), self.transformsX(img.copy())
        img1 = gt1*mask
        img2 = gt2*(~mask)
        # transform results and return
        return img1, gt1, img2, gt2, mask 

    def get_mask(self, image):
        tile = (self.tile_size, self.tile_size)
        tilen = image.size[-2]//self.tile_size
        # Create a mask
        mask = np.zeros((tilen, tilen), dtype=bool)

        # Flatten the mask to make it a 1D array
        flat_mask = mask.flatten()

        # Randomly set 50% of the pixels to True
        num_true_pixels = flat_mask.size // 2
        random_indices = np.random.choice(flat_mask.size, 
                                          size=num_true_pixels, 
                                          replace=False)
        flat_mask[random_indices] = True

        # Reshape the flattened mask back to its original shape
        mask = flat_mask.reshape((tilen, tilen))
        mask = np.kron(mask, np.ones(tile)).astype(bool)
        return mask
    