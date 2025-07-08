import shutil
import gc
from itertools import count
import os

import numpy as np
import torch

from tqdm import tqdm

from pathlib import Path
from util.stream_helper_SPS import SPSHelper


class Trainer():
    def __init__(self, args, model, current_epoch, device, aux_train=True):
        super(Trainer, self).__init__()
        assert current_epoch > 0
        
        self.args = args
        self.model = model
        
        if hasattr(self.args, "end_epoch") and self.args.end_epoch is not None:
            self.end_epoch = self.args.end_epoch

        self.current_epoch = current_epoch
        self.current_phase = None
        self.num_device = 1 if device == 'cpu' else args.gpus
        self.aux_train = aux_train
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