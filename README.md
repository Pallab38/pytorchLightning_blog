# pytorchLightning_blog

## CallBacks
If we want to implement Callback, there's 2 ways: 
### (i) Create a Callback CLASS outside of PyTorch Lightning Trainer Module. 
a) Simple
```
class MyPrintingCallback(Callback):
    def on_train_start(self,trainer,pl_module):
        print("->>>>>>>  Training is starting   <<<<<<<-")
    def on_train_end(self,trainer,pl_module):
        print("->>>>>>>  Training is ending  <<<<<<<-")

```
b) Save Checkpoint Every N Epochs
```
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
### (ii) Create a Callback FUNCTION inside of PyTorch Lightning Trainer Module.
