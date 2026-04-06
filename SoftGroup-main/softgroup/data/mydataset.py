import os.path as osp
import numpy as np
import torch
from glob import glob
from .custom import CustomDataset

class mydataset(CustomDataset):
    # Ваши классы в порядке от 0 до num_classes-1
    CLASSES = (
        'never_classified',  # 0 - ignore
        'ground',            # 1
        'medium_vegetation', # 2
        'high_vegetation',   # 3
        'building',          # 4
        'wire_conductor',    # 5
        'pillar',            # 6 (thing)
        'traffic_sign',      # 7 (thing)
        'unknown_73',        # 8
        'fence',             # 9
    )
    
    # Фоновые классы (stuff)
    STUFF = ('never_classified', 'ground', 'medium_vegetation', 'high_vegetation', 
             'building', 'wire_conductor', 'unknown_73', 'fence')
    
    # Объектные классы (thing)
    THING = ('pillar', 'traffic_sign')
    
    def __init__(self,
                 data_root,
                 prefix,
                 suffix='.pth',
                 voxel_cfg=None,
                 training=True,
                 with_label=True,
                 repeat=1,
                 logger=None):
        
        self.data_root = data_root
        self.prefix = prefix
        self.suffix = suffix
        self.training = training
        self.with_label = with_label
        self.voxel_cfg = voxel_cfg
        self.repeat = repeat
        self.logger = logger
        
        self.filenames = self.get_filenames()
        
    def get_filenames(self):
        """Получение списка файлов для обучения/тестирования"""
        pattern = osp.join(self.data_root, self.prefix, '*' + self.suffix)
        filenames = glob(pattern)
        if len(filenames) == 0:
            raise ValueError(f"Не найдено файлов по паттерну: {pattern}")
        return sorted(filenames * self.repeat)
    
    def load(self, filename):
        """Загрузка одного .pth файла"""
        data = torch.load(filename, weights_only=True)
        
        # data содержит: (coords, colors, sem_labels, instance_labels)
        coords = data[0].numpy()
        colors = data[1].numpy()
        
        if self.with_label and len(data) >= 4:
            sem_labels = data[2].numpy()
            inst_labels = data[3].numpy()
        else:
            sem_labels = np.zeros(coords.shape[0])
            inst_labels = np.zeros(coords.shape[0])
        
        return coords, colors, sem_labels, inst_labels
    
    def getInstanceInfo(self, xyz, instance_label, semantic_label):
        """Получение информации об экземплярах"""
        return super().getInstanceInfo(xyz, instance_label, semantic_label)