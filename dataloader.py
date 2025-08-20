import math
import os
import random
from glob import glob
from torch import stack
from torch.utils.data import Dataset as torchData
from torchvision import transforms

from util.seed import seed_everything
from util.vision import imgloader, rgb_transform

YCBCR_WEIGHTS = {
    # Spec: (K_r, K_g, K_b) with K_g = 1 - K_r - K_b
    "ITU-R_BT.709": (0.2126, 0.7152, 0.0722)
}

    
class VideoData(torchData):
    """Video Dataset

    Args:
        root
        mode
        frames
        transform
    """

    def __init__(self, root, frames, transform=rgb_transform, epoch_ratio=1):
        super().__init__()
        self.folder = glob(root + 'sequences/*/*/')
        self.frames = frames
        self.transform = transform

        assert 0 < epoch_ratio and epoch_ratio <= 1
        self.epoch_ratio = epoch_ratio

    def __len__(self):
        return int(len(self.folder) * self.epoch_ratio)

    @property
    def info(self):
        gop = self[0]
        return "\nGop size: {}".format(gop.shape)

    def __getitem__(self, index):
        path = self.folder[index]
        seed = random.randint(0, 1e9)
        imgs = []
        for f in range(1, self.frames+1):
            seed_everything(seed)
            file = path + 'im' + str(f) + '.png'
            imgs.append(self.transform(imgloader(file)))

        return stack(imgs)


class BVI_Dataset(torchData):
    def __init__(self, root, frames, transform=rgb_transform, epoch_ratio=1., directions=[1, -1], intervals=[1, 2]):
        super().__init__()
        self.folder = root
        self.video_list = os.listdir(root)
        self.frames = frames
        self.transform = transform
        assert 0 < epoch_ratio and epoch_ratio <= 1
        self.epoch_ratio = epoch_ratio
        self.directions = directions
        self.intervals = intervals

    def __len__(self):
        return int(len(self.video_list) * self.epoch_ratio)

    def __getitem__(self, index):
        seed = random.randint(0, 1e9)
        
        seq_len = len(os.listdir(f"{self.folder}/{self.video_list[index]}"))
        direction = random.choice(self.directions)
        interval = random.choice(self.intervals)

        if direction == 1:
            start_index = random.randint(0, (seq_len - 1) - (self.frames-1) * interval)
        elif direction == -1:
            start_index = random.randint((self.frames-1) * interval, (seq_len - 1))
        
        imgs = []
        for i in range(0, self.frames):
            seed_everything(seed)
            frame_index = start_index + i * direction * interval
            frame_path = f"{self.folder}/{self.video_list[index]}/Frame_{str(frame_index)}.png"
            imgs.append(self.transform(imgloader(frame_path)))
        return stack(imgs)
    

class VideoTestData(torchData):
    def __init__(self, root, first_gop=False, sequence=('U', 'B'), GOP=32, test_seq_len=96, use_seqs=[], full_seq=False, color_transform='BT601'):
        super(VideoTestData, self).__init__()
        
        self.root = root

        self.seq_name = []
        seq_len = []
        gop_size = []
        dataset_name_list = []
        self.color_transform = color_transform
        self.full_seq = full_seq

        if 'UVG' in sequence:
            self.seq_name.extend(['Beauty', 'Bosphorus', 'HoneyBee', 'Jockey', 'ReadySteadyGo', 'ShakeNDry', 'YachtRide'])
            if GOP in [12, 16]:
                seq_len.extend([600, 600, 600, 600, 600, 300, 600])
            else:
                seq_len.extend([test_seq_len]*7)
        
            gop_size.extend([GOP]*7)
            dataset_name_list.extend(['UVG']*7)

        if 'HEVC-B' in sequence:
            self.seq_name.extend(['Kimono1', 'BQTerrace', 'Cactus', 'BasketballDrive', 'ParkScene'])
            if GOP in [12, 16]:
                seq_len.extend([100]*5)
            else:
                seq_len.extend([test_seq_len]*5)

            gop_size.extend([GOP]*5)
            dataset_name_list.extend(['HEVC-B']*5)

        if 'HEVC-C' in sequence:
            self.seq_name.extend(['BasketballDrill', 'BQMall', 'PartyScene', 'RaceHorses'])
            if GOP in [12, 16]:
                seq_len.extend([100]*4)
            else:
                seq_len.extend([test_seq_len]*4)

            gop_size.extend([GOP]*4)
            dataset_name_list.extend(['HEVC-C']*4)

        if 'HEVC-D' in sequence:
            self.seq_name.extend(['BasketballPass', 'BQSquare', 'BlowingBubbles', 'RaceHorses1'])
            if GOP in [12, 16]:
                seq_len.extend([100]*4)
            else:
                seq_len.extend([test_seq_len]*4)

            gop_size.extend([GOP]*4)
            dataset_name_list.extend(['HEVC-D']*4)

        if 'HEVC-E' in sequence:
            self.seq_name.extend(['FourPeople', 'Johnny', 'KristenAndSara'])
            if GOP in [12, 16]:
                seq_len.extend([100]*3)
            else:
                seq_len.extend([test_seq_len]*3)

            gop_size.extend([GOP]*3)
            dataset_name_list.extend(['HEVC-E']*3)

        if 'HEVC-RGB' in sequence:
            self.seq_name.extend(['DucksAndLegs', 'EBULupoCandlelight', 'EBURainFruits', 'Kimono1_10b_rgb', 'OldTownCross', 'ParkScene_10b_rgb'])
            if GOP in [12, 16]:
                seq_len.extend([300, 600, 600, 240, 500, 240])
            elif GOP in [10]:
                seq_len.extend([100]*6)
            else:
                seq_len.extend([test_seq_len]*6)
        
            gop_size.extend([GOP]*6)
            dataset_name_list.extend(['HEVC-RGB']*6)

        if 'MCL-JCV' in sequence:
            MCL_list = []
            for i in range(1, 31):
               MCL_list.append('videoSRC'+str(i).zfill(2))
            
            self.seq_name.extend(MCL_list)
            if GOP in [12, 16]:
               seq_len.extend([150, 150, 150, 150, 125, 125, 125, 125, 125, 150,
                               150, 150, 150, 150, 150, 150, 120, 125, 150, 125,
                               120, 120, 120, 120, 120, 150, 150, 150, 120, 150])
            else:
               seq_len.extend([test_seq_len]*30)
               
            gop_size.extend([GOP]*30)
            dataset_name_list.extend(['MCL-JCV']*30)

        if use_seqs:
            for seq_name in SEQUENCES:
                if seq_name not in self.seq_name:
                    continue
                if seq_name in use_seqs:
                    
                    continue
                idx=self.seq_name.index(seq_name)
                self.seq_name.remove(seq_name)
                gop_size.pop(idx)
                dataset_name_list.pop(idx)
                seq_len.pop(idx)

        seq_len = dict(zip(self.seq_name, seq_len))
        gop_size = dict(zip(self.seq_name, gop_size))
        dataset_name_list = dict(zip(self.seq_name, dataset_name_list))

        self.gop_list = []

        for seq_name in self.seq_name:
            if first_gop:
                gop_num = 1
            elif self.full_seq:
                full_seq_len = DATASETS[dataset_name_list[seq_name]][seq_name]['frameNum']
                gop_num = math.ceil(full_seq_len/gop_size[seq_name])
                last_idx = full_seq_len
            else:
                gop_num = seq_len[seq_name] // gop_size[seq_name]
                
            for gop_idx in range(gop_num):
                if self.full_seq and gop_idx == (gop_num-1):
                    self.gop_list.append([dataset_name_list[seq_name],
                                        seq_name,
                                        1 + gop_size[seq_name] * gop_idx,
                                        1 + last_idx])
                else:
                    self.gop_list.append([dataset_name_list[seq_name],
                                      seq_name,
                                      1 + gop_size[seq_name] * gop_idx,
                                      1 + gop_size[seq_name] * (gop_idx + 1)])
        
    def __len__(self):
        return len(self.gop_list)

    def __getitem__(self, idx):
        dataset_name, seq_name, frame_start, frame_end = self.gop_list[idx]
        imgs = []

        if self.color_transform == 'BT709' and dataset_name != 'HEVC-RGB':
            seq_name += "_BT709"

        for frame_idx in range(frame_start, frame_end):
            raw_path = os.path.join(self.root, dataset_name, seq_name, 'frame_{:d}.png'.format(frame_idx))
            
            imgs.append(transforms.ToTensor()(imgloader(raw_path)))

        return dataset_name, seq_name, stack(imgs), frame_start
    

DATASETS = {
    "UVG": {
        "Beauty":             {"frameWH": (1920, 1080), "frameNum": 600, "frameRate": 120, "vi_name": 'Beauty_1920x1080_120'},
        "Bosphorus":          {"frameWH": (1920, 1080), "frameNum": 600, "frameRate": 120, "vi_name": 'Bosphorus_1920x1080_120'},
        "HoneyBee":           {"frameWH": (1920, 1080), "frameNum": 600, "frameRate": 120, "vi_name": "HoneyBee_1920x1080_120"},
        "Jockey":             {"frameWH": (1920, 1080), "frameNum": 600, "frameRate": 120, "vi_name": "Jockey_1920x1080_120"},
        "ReadySteadyGo":      {"frameWH": (1920, 1080), "frameNum": 600, "frameRate": 120, "vi_name": "ReadySteadyGo_1920x1080_120"},
        "ShakeNDry":          {"frameWH": (1920, 1080), "frameNum": 300, "frameRate": 120, "vi_name": "ShakeNDry_1920x1080_120"},
        "YachtRide":          {"frameWH": (1920, 1080), "frameNum": 600, "frameRate": 120, "vi_name": "YachtRide_1920x1080_120"},
    },
    "HEVC-B": {
        "BasketballDrive":     {"frameWH": (1920, 1080), "frameNum": 500, "frameRate": 50, "vi_name": "BasketballDrive_1920x1080_50", 'Full_Intra_Num': 97},
        "BQTerrace":           {"frameWH": (1920, 1080), "frameNum": 600, "frameRate": 60, "vi_name": "BQTerrace_1920x1080_60", 'Full_Intra_Num': 97},
        "Cactus":              {"frameWH": (1920, 1080), "frameNum": 500, "frameRate": 50, "vi_name": "Cactus_1920x1080_50", 'Full_Intra_Num': 97},
        "Kimono1":             {"frameWH": (1920, 1080), "frameNum": 240, "frameRate": 24, 'Full_Intra_Num': 97, "vi_name": "Kimono1_1920x1080_24", 'Full_Intra_Num': 97},
        "ParkScene":           {"frameWH": (1920, 1080), "frameNum": 240, "frameRate": 24, 'Full_Intra_Num': 97, "vi_name": "ParkScene_1920x1080_24", 'Full_Intra_Num': 97},
    },
    "MCL-JCV": {
        "videoSRC01":      {"frameWH": (1920, 1080), "frameNum": 150, "frameRate": 30, 'Full_Intra_Num': 129, 'vi_name':"videoSRC01_1920x1080_30"},
        "videoSRC02":      {"frameWH": (1920, 1080), "frameNum": 150, "frameRate": 30, 'Full_Intra_Num': 129, 'vi_name':"videoSRC02_1920x1080_30"},
        "videoSRC03":      {"frameWH": (1920, 1080), "frameNum": 150, "frameRate": 30, 'Full_Intra_Num': 129, 'vi_name':"videoSRC03_1920x1080_30"},
        "videoSRC04":      {"frameWH": (1920, 1080), "frameNum": 150, "frameRate": 30, 'Full_Intra_Num': 129, 'vi_name':"videoSRC04_1920x1080_30"},
        "videoSRC05":      {"frameWH": (1920, 1080), "frameNum": 125, "frameRate": 25, 'Full_Intra_Num': 97, 'vi_name':"videoSRC05_1920x1080_25"},
        "videoSRC06":      {"frameWH": (1920, 1080), "frameNum": 125, "frameRate": 25, 'Full_Intra_Num': 97, 'vi_name':"videoSRC06_1920x1080_25"},
        "videoSRC07":      {"frameWH": (1920, 1080), "frameNum": 125, "frameRate": 25, 'Full_Intra_Num': 97, 'vi_name':"videoSRC07_1920x1080_25"},
        "videoSRC08":      {"frameWH": (1920, 1080), "frameNum": 125, "frameRate": 25, 'Full_Intra_Num': 97, 'vi_name':"videoSRC08_1920x1080_25"},
        "videoSRC09":      {"frameWH": (1920, 1080), "frameNum": 125, "frameRate": 25, 'Full_Intra_Num': 97, 'vi_name':"videoSRC09_1920x1080_25"},
        "videoSRC10":      {"frameWH": (1920, 1080), "frameNum": 150, "frameRate": 30, 'Full_Intra_Num': 129, 'vi_name':"videoSRC10_1920x1080_30"},
        "videoSRC11":      {"frameWH": (1920, 1080), "frameNum": 150, "frameRate": 30, 'Full_Intra_Num': 129, 'vi_name':"videoSRC11_1920x1080_30"},
        "videoSRC12":      {"frameWH": (1920, 1080), "frameNum": 150, "frameRate": 30, 'Full_Intra_Num': 129, 'vi_name':"videoSRC12_1920x1080_30"},
        "videoSRC13":      {"frameWH": (1920, 1080), "frameNum": 150, "frameRate": 30, 'Full_Intra_Num': 129, 'vi_name':"videoSRC13_1920x1080_30"},
        "videoSRC14":      {"frameWH": (1920, 1080), "frameNum": 150, "frameRate": 30, 'Full_Intra_Num': 129, 'vi_name':"videoSRC14_1920x1080_30"},
        "videoSRC15":      {"frameWH": (1920, 1080), "frameNum": 150, "frameRate": 30, 'Full_Intra_Num': 129, 'vi_name':"videoSRC15_1920x1080_30"},
        "videoSRC16":      {"frameWH": (1920, 1080), "frameNum": 150, "frameRate": 30, 'Full_Intra_Num': 129, 'vi_name':"videoSRC16_1920x1080_30"},
        "videoSRC17":      {"frameWH": (1920, 1080), "frameNum": 120, "frameRate": 24, 'Full_Intra_Num': 97, 'vi_name':"videoSRC17_1920x1080_24"},
        "videoSRC18":      {"frameWH": (1920, 1080), "frameNum": 125, "frameRate": 25, 'Full_Intra_Num': 97, 'vi_name':"videoSRC18_1920x1080_25"},
        "videoSRC19":      {"frameWH": (1920, 1080), "frameNum": 150, "frameRate": 30, 'Full_Intra_Num': 129, 'vi_name':"videoSRC19_1920x1080_30"},
        "videoSRC20":      {"frameWH": (1920, 1080), "frameNum": 125, "frameRate": 25, 'Full_Intra_Num': 97, 'vi_name':"videoSRC20_1920x1080_25"},
        "videoSRC21":      {"frameWH": (1920, 1080), "frameNum": 120, "frameRate": 24, 'Full_Intra_Num': 97, 'vi_name':"videoSRC21_1920x1080_24"},
        "videoSRC22":      {"frameWH": (1920, 1080), "frameNum": 120, "frameRate": 24, 'Full_Intra_Num': 97, 'vi_name':"videoSRC22_1920x1080_24"},
        "videoSRC23":      {"frameWH": (1920, 1080), "frameNum": 120, "frameRate": 24, 'Full_Intra_Num': 97, 'vi_name':"videoSRC23_1920x1080_24"},
        "videoSRC24":      {"frameWH": (1920, 1080), "frameNum": 120, "frameRate": 24, 'Full_Intra_Num': 97, 'vi_name':"videoSRC24_1920x1080_24"},
        "videoSRC25":      {"frameWH": (1920, 1080), "frameNum": 120, "frameRate": 24, 'Full_Intra_Num': 97, 'vi_name':"videoSRC25_1920x1080_24"},
        "videoSRC26":      {"frameWH": (1920, 1080), "frameNum": 150, "frameRate": 30, 'Full_Intra_Num': 129, 'vi_name':"videoSRC26_1920x1080_30"},
        "videoSRC27":      {"frameWH": (1920, 1080), "frameNum": 150, "frameRate": 30, 'Full_Intra_Num': 129, 'vi_name':"videoSRC27_1920x1080_30"},
        "videoSRC28":      {"frameWH": (1920, 1080), "frameNum": 150, "frameRate": 30, 'Full_Intra_Num': 129, 'vi_name':"videoSRC28_1920x1080_30"},
        "videoSRC29":      {"frameWH": (1920, 1080), "frameNum": 120, "frameRate": 24, 'Full_Intra_Num': 97, 'vi_name':"videoSRC29_1920x1080_24"},
        "videoSRC30":      {"frameWH": (1920, 1080), "frameNum": 150, "frameRate": 30, 'Full_Intra_Num': 129, 'vi_name':"videoSRC30_1920x1080_30"},
    },
    "HEVC-C": {
        "BQMall":               {"frameWH": ( 832,  480), "frameNum": 600, "frameRate":  60, "vi_name": 'BQMall_832x480_60', "Full_Intra_Num": None},
        "BasketballDrill":      {"frameWH": ( 832,  480), "frameNum": 500, "frameRate":  50, "vi_name": 'BasketballDrill_832x480_50', "Full_Intra_Num": None},
        "PartyScene":           {"frameWH": ( 832,  480), "frameNum": 500, "frameRate":  50, "vi_name": 'PartyScene_832x480_50', "Full_Intra_Num": None},
        "RaceHorses":           {"frameWH": ( 832,  480), "frameNum": 300, "frameRate":  30, "vi_name": 'RaceHorses_832x480_30', "Full_Intra_Num": None},
    },
    "HEVC-D": {
        "BQSquare":             {"frameWH": ( 416,  240), "frameNum": 600, "frameRate":  60, "vi_name": 'BQSquare_416x240_60', "Full_Intra_Num": None},
        "BasketballPass":       {"frameWH": ( 416,  240), "frameNum": 500, "frameRate":  50, "vi_name": 'BasketballPass_416x240_50', "Full_Intra_Num": None},
        "BlowingBubbles":       {"frameWH": ( 416,  240), "frameNum": 500, "frameRate":  50, "vi_name": 'BlowingBubbles_416x240_50', "Full_Intra_Num": None},
        "RaceHorses1":          {"frameWH": ( 416,  240), "frameNum": 300, "frameRate":  30, "vi_name": 'RaceHorses_416x240_30', "Full_Intra_Num": None},
    },
    "HEVC-E": {
        "FourPeople":           {"frameWH": (1280,  720), "frameNum": 600, "frameRate":  60, "vi_name": 'FourPeople_1280x720_60', "Full_Intra_Num": None},
        "Johnny":               {"frameWH": (1280,  720), "frameNum": 600, "frameRate":  60, "vi_name": 'Johnny_1280x720_60', "Full_Intra_Num": None},
        "KristenAndSara":       {"frameWH": (1280,  720), "frameNum": 600, "frameRate":  60, "vi_name": 'KristenAndSara_1280x720_60', "Full_Intra_Num": None},
    },
    "HEVC-RGB": {
        "DucksAndLegs":         {"frameWH": (1920, 1080), "frameNum": 300, "frameRate":  30, "vi_name": 'DucksAndLegs_1920x1080_30_10bit_444'},
        "EBULupoCandlelight":   {"frameWH": (1920, 1080), "frameNum": 600, "frameRate":  50, "vi_name": 'EBULupoCandlelight_1920x1080_50_10bit_444'},
        "EBURainFruits":        {"frameWH": (1920, 1080), "frameNum": 600, "frameRate":  50, "vi_name": 'EBURainFruits_1920x1080_50_10bit_444'},
        "Kimono1_10b_rgb":      {"frameWH": (1920, 1080), "frameNum": 240, "frameRate":  24, "vi_name": 'Kimono1_1920x1080_24_10bit_444'},
        "OldTownCross":         {"frameWH": (1920, 1080), "frameNum": 500, "frameRate":  50, "vi_name": 'OldTownCross_1920x1080_50_10bit_444'},
        "ParkScene_10b_rgb":    {"frameWH": (1920, 1080), "frameNum": 240, "frameRate":  24, "vi_name": 'ParkScene_1920x1080_24_10bit_444'},
    },

}

SEQUENCES = {seqName: prop for seqs in DATASETS.values() for seqName, prop in seqs.items()}