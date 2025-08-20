import shutil
import gc
import os

import numpy as np
import torch

from tqdm import tqdm

from pathlib import Path
from util.stream_helper_SPS import SPSHelper

from loguru import logger


class Trainer():
    def __init__(self, args, model, train_cfg, current_epoch, save_root, device, epoch_ratio=1):
        super(Trainer, self).__init__()
        assert current_epoch > 0
        
        self.args = args
        self.model = model
        self.train_cfg = train_cfg

        if train_cfg is not None:
            self.phase = {}
            for k, v in sorted({v: k for k, v in train_cfg['phase'].items()}.items()):
                self.phase[k] = v
            self.end_epoch = max(self.phase.keys())
        
        if hasattr(self.args, "end_epoch") and self.args.end_epoch is not None:
            self.end_epoch = self.args.end_epoch

        self.current_epoch = current_epoch
        self.current_phase = None
        self.save_root = save_root
        self.num_device = 1 if device == 'cpu' else args.gpus
        self.epoch_ratio = epoch_ratio
        self.device = device

    def get_phase(self, epoch):
        for k in self.phase.keys():
            if epoch <= k:
                return self.phase[k]
                
    def get_prev_phase(self, epoch):
        previous = None
        current = None
        for k in self.phase.keys():
            previous = current
            current = self.phase[k]

            if epoch <= k:
                return previous

    def save_checkpoint(self, state, is_best):
        torch.save(state, self.save_root + f'/epoch={self.current_epoch}.pth.tar')
        if is_best:
            shutil.copyfile(self.save_root + f'/epoch={self.current_epoch}.pth.tar',
                            self.save_root + f'/checkpoint_best_loss.pth.tar')

    def fit(self):
        if not self.args.no_sanity:
            self.before_train()

        start = self.current_epoch
        best_loss = float("inf")

        print("Remember to change the frozen modules function !!!")
        for epoch in range(start, self.end_epoch + 1):
            phase = self.get_phase(epoch)

            if phase != self.current_phase:
                kwargs = {
                    "stage": "fit",
                    "max_num_Pframe": self.train_cfg[phase]['max_num_Pframe'] if 'max_num_Pframe' in self.train_cfg[phase].keys() else 6,
                    "epoch_ratio": self.train_cfg[phase]['train_len'] if 'train_len' in self.train_cfg[phase].keys() else 1
                }

                if self.epoch_ratio != 1:
                    kwargs.update({"epoch_ratio": self.epoch_ratio})
                    
                self.model.setup(**kwargs)

                # setup train dataloader
                self.train_loader = self.model.train_dataloader(self.train_cfg[phase]['batch_size'] * self.num_device)

                # setup val dataloader
                self.val_loader = self.model.val_dataloader(self.num_device)

                # re-calculate the milestones of lr_scheduler
                milestones = np.array(self.train_cfg[phase]['lr_scheduler']['milestones'])
                previous_phase = self.get_prev_phase(epoch)
                if previous_phase is None:
                    diff = self.current_epoch
                else:
                    diff = self.current_epoch - self.train_cfg['phase'][previous_phase]

                milestones = milestones - diff
                done = len(milestones[milestones <= 0])
                milestones = list(milestones[milestones>0])

                lr = self.train_cfg[phase]['lr'] * self.train_cfg[phase]['lr_scheduler']['gamma']**done

                # setup optimizer
                self.model.configure_optimizers(lr, 
                                                include_module_name=self.train_cfg[phase]['include_module_name'] if 'include_module_name' in self.train_cfg[phase] else None,
                                                exclude_module_name=self.train_cfg[phase]['exclude_module_name'] if 'exclude_module_name' in self.train_cfg[phase] else None)
                
                # setup lr_scheduler
                if len(self.train_cfg[phase]['lr_scheduler']['milestones']) != 0:
                    lr_scheduler = self.model.configure_lr_scheduler(milestones, self.train_cfg[phase]['lr_scheduler']['gamma']) 
                else:
                    lr_scheduler = None

                self.current_phase = phase

                print(f'Start {self.current_phase} phase. Batch size={self.train_cfg[phase]["batch_size"]}\n lr: {lr}, milestones: {milestones}, frozen_modules: {self.train_cfg[phase]["frozen_modules"]}')

            self.current_epoch = epoch
            self.model.train()

            # setup train progressbar
            data_len = len(self.train_loader)
            progressbar = tqdm(self.train_loader, total=data_len)
            progressbar.set_description(f'epoch {epoch}')

            for i, batch in enumerate(progressbar, start=1):
                self.model.optimizer.zero_grad()

                loss, logs = self.model.training_step(batch, phase)

                # skip nan loss
                if loss > 10000 or torch.any(torch.isnan(loss)):
                    logger.warning(f"Skip this step: loss={loss.item()}")
                    gc.collect()
                    torch.cuda.empty_cache()
                    del batch, loss, logs
                    continue

                if loss != 0:
                    loss.backward()
                    self.model.optimizer_step()
                    
                self.model.training_step_end(logs, epoch, (epoch - 1) * data_len + i)

                update_txt=f'Loss: {loss.item():.3f}'
                progressbar.set_postfix_str(update_txt, refresh=True)

                del batch, loss, logs
            
            if lr_scheduler is not None:
                lr_scheduler.step()

            gc.collect()
            torch.cuda.empty_cache()

            self.model.eval()
            outputs = []
            val_loss = []
            
            # setup validation progressbar
            progressbar = tqdm(self.val_loader, total=len(self.val_loader), leave=True)
            progressbar.set_description(f'epoch {epoch}')        

            gc.collect()
            torch.cuda.empty_cache()  

            for batch in progressbar:
                logs = self.model.validation_step(batch, epoch)
                outputs.append(logs)
                val_loss.append(np.mean(logs['val/loss']))

                update_txt=f'[Validation Loss: {np.mean(logs["val/loss"]):.3f}]'
                progressbar.set_postfix_str(update_txt, refresh=True)

                del batch
                gc.collect()
                torch.cuda.empty_cache()

            self.model.validation_epoch_end(outputs, epoch)
            del outputs
            gc.collect()
            torch.cuda.empty_cache()

            val_loss = np.mean(val_loss)
            is_best = val_loss < best_loss
            best_loss = min(val_loss, best_loss)

            self.save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": self.model.state_dict(),
                    "loss": val_loss,
                    "optimizer": self.model.optimizer.state_dict(),
                },
                is_best
            )
            
    def test(self):
        torch.backends.cudnn.enabled = True
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(1)
        
        self.model.setup('test')
        test_loader = self.model.test_dataloader()

        self.model.eval()
        outputs = []
        for batch in tqdm(test_loader):
            logs = self.model.test_step(batch)
            outputs.append(logs)

            del batch
            gc.collect()
            torch.cuda.empty_cache()

        self.model.test_epoch_end(outputs)

    def compress(self):
        torch.backends.cudnn.enabled = True
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(1)

        self.model.setup('test')
        test_loader = self.model.test_dataloader()

        os.makedirs(os.path.join(self.args.save_dir, 'bin'), exist_ok=True)
        self.model.bin_path = os.path.join(self.args.save_dir, 'bin', f'{self.args.test_seqs[0]}_{self.args.color_transform}.bin')
        bitstream_path = Path(self.model.bin_path)
        self.model.output_file = bitstream_path.open("wb")

        self.model.sps_helper = SPSHelper()
        self.model.bits = []

        if self.args.YUV_FILE:
            os.makedirs(os.path.join(self.args.save_dir, 'yuv'), exist_ok=True)
            self.model.yuv_path = os.path.join(self.args.save_dir, 'yuv', f"{self.args.test_seqs[0]}_{self.args.color_transform}.yuv")
            self.model.yuv_file = open(self.model.yuv_path, "wb")

        self.model.eval()
        outputs = []
        first = True

        for batch in tqdm(test_loader):
            logs = self.model.compress_step(batch, first)
            outputs.append(logs)
            first = False
        
        self.model.output_file.close()
        if self.args.YUV_FILE:
            self.model.yuv_file.close()
        rate_bin = os.path.getsize(self.model.bin_path) / self.model.shape[0] / self.model.shape[1] * 8 / self.model.num_frames

        for i in range(len(outputs)):
            outputs[i]['metrics']['rate_bin'] = rate_bin
        
        self.model.test_epoch_end(outputs)

    def decompress(self):
        torch.backends.cudnn.enabled = True
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(1)
        
        self.model.setup('test')
        test_loader = self.model.test_dataloader()

        self.model.bin_path = os.path.join(self.args.save_dir, 'bin', f'{self.args.test_seqs[0]}_{self.args.color_transform}.bin')
        bitstream_path = Path(self.model.bin_path)
        self.model.input_file = bitstream_path.open("rb")

        self.model.sps_helper = SPSHelper()

        if self.args.YUV_FILE:
            os.makedirs(os.path.join(self.args.save_dir, 'yuv_recon'), exist_ok=True)
            self.model.yuv_path = os.path.join(self.args.save_dir, 'yuv_recon', f"{self.args.test_seqs[0]}_{self.args.color_transform}.yuv")
            self.model.yuv_file = open(self.model.yuv_path, "wb")

        self.model.eval()
        outputs = []
        first = True
        self.model.gop_count = 0
        for batch in tqdm(test_loader):
            logs = self.model.decompress_step(batch, first)
            outputs.append(logs)
            first = False
            self.model.gop_count += 1
        
        self.model.input_file.close()
        if self.args.YUV_FILE:
            self.model.yuv_file.close()
        rate_bin = os.path.getsize(self.model.bin_path) / self.model.shape[0] / self.model.shape[1] * 8 / self.model.num_frames

        for i in range(len(outputs)):
            outputs[i]['metrics']['rate_bin'] = rate_bin

        self.model.test_epoch_end(outputs)

    def before_train(self):
        self.model.setup('fit')

        self.val_loader = self.model.val_dataloader(self.num_device)

        self.model.eval()
        outputs = []
        val_loss = []
        
        progressbar = tqdm(self.val_loader, total=len(self.val_loader), leave=True)
        progressbar.set_description(f'epoch {self.current_epoch - 1}')

        for batch in progressbar:
            logs = self.model.validation_step(batch, (self.current_epoch - 1))
            outputs.append(logs)
            val_loss.append(np.mean(logs["val/loss"]))

            update_txt=f'[Validation Loss: {np.mean(logs["val/loss"]):.3f}]'
            progressbar.set_postfix_str(update_txt, refresh=True)

            del batch
            gc.collect()
            torch.cuda.empty_cache()

        self.model.validation_epoch_end(outputs, (self.current_epoch - 1))