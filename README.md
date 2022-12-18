# pytorchLightning_blog

## CallBacks ```from pytorch_lightning.callbacks import Callback```
If we want to implement Callback, there's 2 ways: 
### (i) Create a Callback CLASS outside of PyTorch Lightning Trainer Module. 
a) Simple
<details>
    <summary> Code </summary>
    <p>
    
    
```Python

from pytorch_lightning.callbacks import Callback
class MyPrintingCallback(Callback):
    def on_train_start(self,trainer,pl_module):
        print("->>>>>>>  Training is starting   <<<<<<<-")
            
    def on_train_end(self,trainer,pl_module):
        print("->>>>>>>  Training is ending  <<<<<<<-")
```       
</p>
</details>
            
            
b) Save Checkpoint Every N Epochs
            <details><summary>Code</summary>
            <p> 
            
                        
```Python
            
from pytorch_lightning.callbacks import Callback
## https://github.com/Lightning-AI/lightning/issues/2534#issuecomment-674582085
class CheckpointEveryNEpochs(Callback):
    """
    Save a checkpoint every N Epochs
    """
    def __init__(self, save_epoch_frequency, prefix="N_Epoch_Checkpoint",
                 use_modelCheckpoint_filename=False):
        super().__init__()
        self.save_epoch_frequency = save_epoch_frequency
        self.prefix = prefix
        self.use_modelCheckpoint_filename = use_modelCheckpoint_filename
    
    #### https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.callbacks.Callback.html#pytorch_lightning.callbacks.Callback.on_train_epoch_end
    def on_train_epoch_end(self, trainer, _):
        epoch = trainer.current_epoch
        if epoch % self.save_epoch_frequency==0:
            if self.use_modelCheckpoint_filename:
                filename = trainer.checkpoint_callback.filename
            else:
                filename= f"{self.prefix}_{epoch}.ckpt"
            
            dir_path = os.path.dirname(trainer.checkpoint_callback.dirpath)
            save_dir = join(dir_path, "saveEvery_%dEpoch"%self.save_epoch_frequency)
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            ckpt_path = join(save_dir, filename)
            trainer.save_checkpoint(ckpt_path)
                         
```
            
</p>
</details>
            
            
### (ii) Create a Callback FUNCTION inside of PyTorch Lightning Trainer Module.
            
<details> 
            <summary> Code </summary>
            <p>
            
```Python
def training_epoch_end(self, outputs):
        """
        outputs is a python list containing the batch_dictionary from each batch
        for the given epoch stacked up against each other. 
        """
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        ##### using tensorboard logger
        self.logger.experiment.add_scalar("Loss", avg_loss,self.current_epoch)
        epoch_dict = {"loss": avg_loss}
        
        
        # print(f"outputs: {outputs}")
        # out_dict = outputs[1]
        # save_dir = "/home/user/output/Painter/allPoses"
        # Path(save_dir).mkdir(parents=True, exist_ok= True)
        # pred_image = out_dict['pred_image']
        # epoch = out_dict["epoch"]
        # img_fpath = join(save_dir,"ep%03d.png" % epoch)
        # # print(f"pred_image: {pred_image.shape}, epoch:{epoch}")
        # # print(f"mask_fpath: {img_fpath}")
        # pred_image = pred_image[0,:,:,:]
        # save_image(pred_image, img_fpath)
        # print(f"layer 0 weight: {torch.sum(self.painter_net.painter_net[0].weight)}")
        # print(f"layer 0 grad: {torch.sum(self.painter_net.painter_net[0].weight.grad)}")

        # print(f"layer 2 grad: {torch.sum(self.density_net.my_net[0].weight.grad)}")
        # print(f"layer 4 grad: {torch.sum(self.density_net.my_net[0].weight.grad)}")
        # print(f"layer 6 grad: {torch.sum(self.density_net.my_net[0].weight.grad)}")
        # print(f"layer 8 grad: {torch.sum(self.density_net.my_net[0].weight.grad)}")

        # print(f"layer 0 grad sum: {torch.sum(self.density_net.my_net[0].weight.grad)}")
```
</p>            
</details>
            
## Optimizers
### Multiple Optimizers for multiple networks [Colab Example](https://colab.research.google.com/drive/1jVPI6as9gBCRxdu7r1Q6RvYu2Jh08OKJ?usp=sharing#scrollTo=jNqCMifazeDX)
<details> <summary> Code ([Github Issue Comment](https://github.com/Lightning-AI/lightning/issues/3346#issuecomment-760642139)   )</summary>
<p>
            
 ```Python
 import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, random_split

import pytorch_lightning as pl

from torchvision import transforms
from torchvision.datasets.mnist import MNIST


# This is just a wrapper so we can observe which optimizer
# gets used in the update
class CustomAdam(torch.optim.Adam):

    def __init__(self, name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = name

    def step(self, *args, **kwargs):
        print("updating", self.name)
        return super().step(*args, **kwargs)


class LitAutoEncoder(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 28 * 28)
        )

    def training_step(self, batch, batch_idx, optimizer_idx):
        print("skipping for batch_idx", batch_idx)
        if optimizer_idx == 1:
            x, y = batch
            x = x.view(x.size(0), -1)
            z = self.encoder(x)
            x_hat = self.decoder(z)
            loss = F.mse_loss(x_hat, x)
            return loss

    # one optimizer for encoder, one for decoder
    def configure_optimizers(self):
        optimizer0 = CustomAdam("encoder opt", self.encoder.parameters(), lr=1e-2)
        optimizer1 = CustomAdam("decoder opt", self.decoder.parameters(), lr=1e-4)
        return optimizer0, optimizer1

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, *args, **kwargs):
        if optimizer_idx == 1:
            for opt in self.optimizers():
                super().optimizer_step(epoch, batch_idx,  opt, optimizer_idx, *args, **kwargs)           
 ```
</p></details>
