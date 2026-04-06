from .rs10_dataset import RS10_Dataset

__all__ = ['build_dataset', 'S3DISDataset', 'ScanNetDataset', 'KITTIDataset', 'MyDataset']

def build_dataset(cfg, logger=None):
    if cfg.type == 'rs10_class':
        return MyDataset(**cfg)
    # ... остальные условия