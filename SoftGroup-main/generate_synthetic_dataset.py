"""
Generate synthetic 3D point cloud dataset for SoftGroup.
Creates realistic indoor scenes with furniture-like objects for testing.

Usage:
    conda activate softgroup
    python generate_synthetic_dataset.py

Output:
    dataset/synthetic/
    ├── train/   (8 scenes)
    ├── val/     (2 scenes)
    └── val_gt/  (ground truth for evaluation)
"""
import os
import numpy as np
import torch
import random
from pathlib import Path


# ===== Scene generation primitives =====

def make_box_points(center, size, density=500):
    """Generate points on the surface of a box (6 faces)."""
    cx, cy, cz = center
    sx, sy, sz = size
    points = []
    # Each face
    for axis in range(3):
        for sign in [-1, 1]:
            n_face = int(density * size[(axis + 1) % 3] * size[(axis + 2) % 3])
            n_face = max(n_face, 50)
            u = np.random.uniform(-0.5, 0.5, n_face) * size[(axis + 1) % 3]
            v = np.random.uniform(-0.5, 0.5, n_face) * size[(axis + 2) % 3]
            w = np.full(n_face, sign * size[axis] / 2)
            face = np.zeros((n_face, 3))
            face[:, axis] = w
            face[:, (axis + 1) % 3] = u
            face[:, (axis + 2) % 3] = v
            points.append(face + center)
    return np.vstack(points).astype(np.float32)


def make_cylinder_points(center, radius, height, n_points=800):
    """Generate points on the surface of a cylinder."""
    points = []
    # Side surface
    n_side = int(n_points * 0.7)
    theta = np.random.uniform(0, 2 * np.pi, n_side)
    z = np.random.uniform(-height / 2, height / 2, n_side)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    points.append(np.stack([x, y, z], axis=1) + center)
    # Top and bottom discs
    for sign in [-1, 1]:
        n_cap = int(n_points * 0.15)
        r = np.sqrt(np.random.uniform(0, 1, n_cap)) * radius
        theta = np.random.uniform(0, 2 * np.pi, n_cap)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = np.full(n_cap, sign * height / 2)
        points.append(np.stack([x, y, z], axis=1) + center)
    return np.vstack(points).astype(np.float32)


def make_plane_points(origin, normal, size, n_points=5000):
    """Generate points on a plane."""
    # Create basis vectors orthogonal to normal
    normal = np.array(normal, dtype=np.float32)
    normal = normal / np.linalg.norm(normal)
    if abs(normal[0]) < 0.9:
        tangent1 = np.cross(normal, [1, 0, 0])
    else:
        tangent1 = np.cross(normal, [0, 1, 0])
    tangent1 = tangent1 / np.linalg.norm(tangent1)
    tangent2 = np.cross(normal, tangent1)
    u = np.random.uniform(-size[0] / 2, size[0] / 2, n_points)
    v = np.random.uniform(-size[1] / 2, size[1] / 2, n_points)
    points = (np.array(origin) +
              u[:, None] * tangent1 +
              v[:, None] * tangent2)
    # Add small noise along normal for thickness
    points += np.random.normal(0, 0.005, (n_points, 1)) * normal
    return points.astype(np.float32)


# ===== Color generation =====

COLORS = {
    'wall':         ([200, 200, 200], 20),   # gray walls
    'floor':        ([160, 140, 120], 15),   # brown-ish floor
    'ceiling':      ([240, 240, 240], 10),   # white ceiling
    'table':        ([139, 90, 43], 20),     # wood brown
    'chair':        ([60, 60, 60], 15),      # dark gray
    'sofa':         ([140, 50, 50], 25),     # dark red
    'bed':          ([100, 140, 180], 20),   # blue
    'cabinet':      ([120, 80, 40], 15),     # dark wood
    'bookshelf':    ([100, 70, 30], 20),     # wood
    'desk':         ([150, 100, 50], 15),    # light wood
    'door':         ([180, 160, 140], 10),   # light brown
    'window':       ([180, 220, 250], 15),   # light blue
    'curtain':      ([200, 170, 130], 25),   # beige
    'toilet':       ([230, 230, 230], 10),   # white
    'sink':         ([210, 210, 220], 10),   # off-white
    'bathtub':      ([220, 220, 230], 15),   # white-ish
    'otherfurniture': ([100, 100, 80], 20),  # olive
}


def generate_color(obj_type, n_points):
    """Generate realistic colors for an object."""
    base, noise = COLORS.get(obj_type, ([128, 128, 128], 20))
    rgb = np.tile(base, (n_points, 1)).astype(np.float32)
    rgb += np.random.normal(0, noise, (n_points, 3))
    rgb = np.clip(rgb, 0, 255)
    # Normalize to [-1, 1]
    return (rgb / 127.5 - 1).astype(np.float32)


# ===== ScanNet-compatible semantic classes =====
# ScanNet has 20 semantic classes (0=wall, 1=floor, 2-19=instances)
# Instance classes are indices 2-19 (18 classes)
SCANNET_CLASSES = [
    'wall', 'floor',  # 0, 1 (non-instance)
    'cabinet', 'bed', 'chair', 'sofa', 'table', 'door',  # 2-7
    'window', 'bookshelf', 'picture', 'counter', 'desk',  # 8-12
    'curtain', 'refrigerator', 'shower curtain', 'toilet',  # 13-16
    'sink', 'bathtub', 'otherfurniture'  # 17-19
]


def generate_room(room_size=(6.0, 8.0, 3.0), n_objects=None, seed=None):
    """Generate a single synthetic indoor room scene.

    Returns:
        coords: (N, 3) float32 - centered coordinates
        colors: (N, 3) float32 - normalized RGB in [-1, 1]
        sem_labels: (N,) int64 - semantic labels [0, 19]
        inst_labels: (N,) int64 - instance labels, -100 for walls/floor/ceiling
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    rx, ry, rz = room_size
    all_points = []
    all_colors = []
    all_sem = []
    all_inst = []
    instance_id = 0

    # --- Floor (sem=1, inst=-100) ---
    floor = make_plane_points([0, 0, 0], [0, 0, 1], [rx, ry],
                              n_points=int(3000 * rx * ry / 48))
    all_points.append(floor)
    all_colors.append(generate_color('floor', len(floor)))
    all_sem.append(np.full(len(floor), 1, dtype=np.int64))
    all_inst.append(np.full(len(floor), -100, dtype=np.int64))

    # --- Ceiling (sem=0 treated as wall for ScanNet, inst=-100) ---
    ceiling = make_plane_points([0, 0, rz], [0, 0, -1], [rx, ry],
                                n_points=int(2000 * rx * ry / 48))
    all_points.append(ceiling)
    all_colors.append(generate_color('ceiling', len(ceiling)))
    all_sem.append(np.full(len(ceiling), 0, dtype=np.int64))
    all_inst.append(np.full(len(ceiling), -100, dtype=np.int64))

    # --- 4 Walls (sem=0, inst=-100) ---
    walls = [
        ([0, -ry / 2, rz / 2], [0, 1, 0], [rx, rz]),
        ([0, ry / 2, rz / 2], [0, -1, 0], [rx, rz]),
        ([-rx / 2, 0, rz / 2], [1, 0, 0], [ry, rz]),
        ([rx / 2, 0, rz / 2], [-1, 0, 0], [ry, rz]),
    ]
    for origin, normal, size in walls:
        wall_pts = make_plane_points(origin, normal, size,
                                     n_points=int(2000 * size[0] * size[1] / 18))
        all_points.append(wall_pts)
        all_colors.append(generate_color('wall', len(wall_pts)))
        all_sem.append(np.full(len(wall_pts), 0, dtype=np.int64))
        all_inst.append(np.full(len(wall_pts), -100, dtype=np.int64))

    # --- Furniture objects ---
    if n_objects is None:
        n_objects = np.random.randint(6, 15)

    # Define object types with (semantic_class_id, color_key, generator)
    furniture_templates = [
        # Tables (sem=6)
        {
            'sem': 6, 'color': 'table',
            'gen': lambda pos: make_box_points(
                [pos[0], pos[1], 0.4],
                [np.random.uniform(0.8, 1.5), np.random.uniform(0.6, 1.2), 0.8],
                density=600)
        },
        # Chairs (sem=4)
        {
            'sem': 4, 'color': 'chair',
            'gen': lambda pos: np.vstack([
                make_box_points([pos[0], pos[1], 0.25],
                                [0.45, 0.45, 0.5], density=500),
                make_box_points([pos[0], pos[1] - 0.15, 0.65],
                                [0.45, 0.05, 0.6], density=400),
            ])
        },
        # Sofa (sem=5)
        {
            'sem': 5, 'color': 'sofa',
            'gen': lambda pos: make_box_points(
                [pos[0], pos[1], 0.3],
                [np.random.uniform(1.5, 2.5), np.random.uniform(0.7, 1.0), 0.6],
                density=500)
        },
        # Bed (sem=3)
        {
            'sem': 3, 'color': 'bed',
            'gen': lambda pos: make_box_points(
                [pos[0], pos[1], 0.35],
                [np.random.uniform(1.8, 2.2), np.random.uniform(1.2, 1.6), 0.7],
                density=400)
        },
        # Cabinet (sem=2)
        {
            'sem': 2, 'color': 'cabinet',
            'gen': lambda pos: make_box_points(
                [pos[0], pos[1], 0.9],
                [np.random.uniform(0.5, 1.2), 0.5, 1.8],
                density=500)
        },
        # Desk (sem=12)
        {
            'sem': 12, 'color': 'desk',
            'gen': lambda pos: make_box_points(
                [pos[0], pos[1], 0.38],
                [np.random.uniform(1.0, 1.8), np.random.uniform(0.6, 0.9), 0.76],
                density=500)
        },
        # Bookshelf (sem=9)
        {
            'sem': 9, 'color': 'bookshelf',
            'gen': lambda pos: make_box_points(
                [pos[0], pos[1], 0.95],
                [np.random.uniform(0.8, 1.5), 0.35, 1.9],
                density=500)
        },
        # Door (sem=7)
        {
            'sem': 7, 'color': 'door',
            'gen': lambda pos: make_box_points(
                [pos[0], pos[1], 1.05],
                [0.9, 0.08, 2.1],
                density=600)
        },
        # Window (sem=8)
        {
            'sem': 8, 'color': 'window',
            'gen': lambda pos: make_box_points(
                [pos[0], pos[1], 1.5],
                [np.random.uniform(0.8, 1.5), 0.05, np.random.uniform(0.8, 1.2)],
                density=500)
        },
        # OtherFurniture (sem=19)
        {
            'sem': 19, 'color': 'otherfurniture',
            'gen': lambda pos: make_cylinder_points(
                [pos[0], pos[1], 0.5],
                np.random.uniform(0.15, 0.4),
                np.random.uniform(0.5, 1.2),
                n_points=600)
        },
        # Toilet (sem=16)
        {
            'sem': 16, 'color': 'toilet',
            'gen': lambda pos: np.vstack([
                make_cylinder_points([pos[0], pos[1], 0.2], 0.2, 0.4, n_points=400),
                make_box_points([pos[0], pos[1] - 0.1, 0.5], [0.35, 0.1, 0.4],
                                density=400),
            ])
        },
        # Sink (sem=17)
        {
            'sem': 17, 'color': 'sink',
            'gen': lambda pos: make_cylinder_points(
                [pos[0], pos[1], 0.85], 0.25, 0.15, n_points=500)
        },
    ]

    occupied = []

    for _ in range(n_objects):
        template = random.choice(furniture_templates)

        # Find non-overlapping position
        margin = 0.5
        for attempt in range(20):
            px = np.random.uniform(-rx / 2 + margin, rx / 2 - margin)
            py = np.random.uniform(-ry / 2 + margin, ry / 2 - margin)
            pos = (px, py)

            # Simple overlap check
            ok = True
            for ox, oy in occupied:
                if abs(px - ox) < 1.0 and abs(py - oy) < 1.0:
                    ok = False
                    break
            if ok:
                occupied.append(pos)
                break
        else:
            continue

        pts = template['gen'](pos)
        n = len(pts)
        all_points.append(pts)
        all_colors.append(generate_color(template['color'], n))
        all_sem.append(np.full(n, template['sem'], dtype=np.int64))
        all_inst.append(np.full(n, instance_id, dtype=np.int64))
        instance_id += 1

    # Concatenate
    coords = np.vstack(all_points).astype(np.float32)
    colors = np.vstack(all_colors).astype(np.float32)
    sem_labels = np.concatenate(all_sem).astype(np.int64)
    inst_labels = np.concatenate(all_inst).astype(np.int64)

    # Center coordinates
    coords -= coords.mean(0)

    # Add slight noise for realism
    coords += np.random.normal(0, 0.005, coords.shape).astype(np.float32)

    return coords, colors, sem_labels, inst_labels


def generate_gt_files(val_dir, gt_dir, semantic_classes=20):
    """Generate ground truth .txt files for ScanNet evaluation."""
    os.makedirs(gt_dir, exist_ok=True)
    for pth_file in sorted(Path(val_dir).glob('*.pth')):
        data = torch.load(str(pth_file), weights_only=False)
        coords, colors, sem_labels, inst_labels = data
        if isinstance(sem_labels, torch.Tensor):
            sem_labels = sem_labels.numpy()
        if isinstance(inst_labels, torch.Tensor):
            inst_labels = inst_labels.numpy()

        scene_name = pth_file.stem.replace('_inst_nostuff', '')
        gt_file = os.path.join(gt_dir, f'{scene_name}.txt')

        with open(gt_file, 'w') as f:
            for i in range(len(sem_labels)):
                sem = sem_labels[i]
                inst = inst_labels[i]
                if inst == -100:
                    # Background point
                    f.write(f'{sem} 0 0\n')
                else:
                    # Instance point: sem_label instance_id
                    f.write(f'{sem} {inst + 1} 1\n')


def write_config(output_dir, n_train, n_val):
    """Write a SoftGroup config for the synthetic dataset."""
    config = f"""model:
  channels: 32
  num_blocks: 7
  semantic_classes: 20
  instance_classes: 18
  sem2ins_classes: []
  semantic_only: False
  ignore_label: -100
  grouping_cfg:
    score_thr: 0.2
    radius: 0.04
    mean_active: 300
    class_numpoint_mean: [-1., -1., 2500., 8000., 2500.,
                          4000., 2500., 1500., 1000., 3000.,
                          500., 1500., 2500., 2000., 1500.,
                          1500., 1500., 1000., 2000., 1500.]
    npoint_thr: 0.05
    ignore_classes: [0, 1]
  instance_voxel_cfg:
    scale: 50
    spatial_shape: 20
  train_cfg:
    max_proposal_num: 200
    pos_iou_thr: 0.5
  test_cfg:
    x4_split: False
    cls_score_thr: 0.001
    mask_score_thr: -0.5
    min_npoint: 100
    eval_tasks: ['semantic', 'instance']
  fixed_modules: []

data:
  train:
    type: 'scannetv2'
    data_root: '{output_dir}'
    prefix: 'train'
    suffix: '_inst_nostuff.pth'
    training: True
    repeat: 4
    voxel_cfg:
      scale: 50
      spatial_shape: [128, 512]
      max_npoint: 250000
      min_npoint: 5000
  test:
    type: 'scannetv2'
    data_root: '{output_dir}'
    prefix: 'val'
    suffix: '_inst_nostuff.pth'
    training: False
    with_label: True
    voxel_cfg:
      scale: 50
      spatial_shape: [128, 512]
      max_npoint: 250000
      min_npoint: 5000

dataloader:
  train:
    batch_size: 2
    num_workers: 2
  test:
    batch_size: 1
    num_workers: 1

optimizer:
  type: 'Adam'
  lr: 0.001

fp16: False
epochs: 20
step_epoch: 14
save_freq: 2
pretrain: ''
work_dir: ''
"""
    config_path = os.path.join('configs', 'softgroup', 'softgroup_synthetic.yaml')
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w') as f:
        f.write(config)
    print(f'  Config: {config_path}')
    return config_path


def main():
    print('=' * 60)
    print(' SoftGroup Synthetic Dataset Generator')
    print('=' * 60)

    output_dir = 'dataset/synthetic'
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    gt_dir = os.path.join(output_dir, 'val_gt')

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    n_train = 8
    n_val = 2

    # Room size variations
    room_configs = [
        {'room_size': (6, 8, 3), 'n_objects': None},     # standard room
        {'room_size': (4, 5, 2.8), 'n_objects': None},   # small room
        {'room_size': (8, 10, 3.2), 'n_objects': None},  # large room
        {'room_size': (5, 6, 3), 'n_objects': None},     # medium room
        {'room_size': (7, 7, 3), 'n_objects': None},     # square room
        {'room_size': (10, 12, 3.5), 'n_objects': None}, # very large
        {'room_size': (4, 4, 2.7), 'n_objects': None},   # tiny room
        {'room_size': (6, 9, 3.1), 'n_objects': None},   # long room
        {'room_size': (5, 7, 3), 'n_objects': None},     # val room 1
        {'room_size': (8, 8, 3.2), 'n_objects': None},   # val room 2
    ]

    # Generate training scenes
    print(f'\n  Generating {n_train} training scenes...')
    for i in range(n_train):
        cfg = room_configs[i % len(room_configs)]
        coords, colors, sem_labels, inst_labels = generate_room(
            seed=42 + i, **cfg)

        n_instances = len(np.unique(inst_labels[inst_labels >= 0]))
        scene_name = f'scene_synth_{i:04d}'
        path = os.path.join(train_dir, f'{scene_name}_inst_nostuff.pth')
        torch.save((coords, colors, sem_labels, inst_labels), path)
        print(f'    {scene_name}: {len(coords):,} points, '
              f'{n_instances} instances, '
              f'room {cfg["room_size"][0]}x{cfg["room_size"][1]}m')

    # Generate validation scenes
    print(f'\n  Generating {n_val} validation scenes...')
    for i in range(n_val):
        cfg = room_configs[n_train + i]
        coords, colors, sem_labels, inst_labels = generate_room(
            seed=1000 + i, **cfg)

        n_instances = len(np.unique(inst_labels[inst_labels >= 0]))
        scene_name = f'scene_synth_{n_train + i:04d}'
        path = os.path.join(val_dir, f'{scene_name}_inst_nostuff.pth')
        torch.save((coords, colors, sem_labels, inst_labels), path)
        print(f'    {scene_name}: {len(coords):,} points, '
              f'{n_instances} instances, '
              f'room {cfg["room_size"][0]}x{cfg["room_size"][1]}m')

    # Generate ground truth
    print('\n  Generating ground truth files...')
    generate_gt_files(val_dir, gt_dir)
    gt_count = len(list(Path(gt_dir).glob('*.txt')))
    print(f'    {gt_count} ground truth files created')

    # Write config
    print('\n  Writing config file...')
    config_path = write_config(output_dir, n_train, n_val)

    # Summary
    print('\n' + '=' * 60)
    print(' Dataset generation complete!')
    print('=' * 60)

    total_train_pts = 0
    for f in Path(train_dir).glob('*.pth'):
        data = torch.load(str(f), weights_only=False)
        total_train_pts += len(data[0])

    total_val_pts = 0
    for f in Path(val_dir).glob('*.pth'):
        data = torch.load(str(f), weights_only=False)
        total_val_pts += len(data[0])

    print(f"""
  Output: {output_dir}/
  Train:  {n_train} scenes, {total_train_pts:,} total points
  Val:    {n_val} scenes, {total_val_pts:,} total points
  GT:     {gt_dir}/
  Config: {config_path}

  Train command:
    python tools/train.py {config_path}

  Test command:
    python tools/test.py {config_path} --pretrain <checkpoint.pth>
""")


if __name__ == '__main__':
    main()
