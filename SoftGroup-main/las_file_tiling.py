import laspy
import numpy as np
import os
from pathlib import Path

def split_pointcloud_to_tiles(input_las, output_dir, grid_size=5):
    """
    Разбивает облако точек на тайлы по сетке grid_size x grid_size
    
    Args:
        input_las: путь к входному LAS файлу
        output_dir: директория для сохранения тайлов
        grid_size: размер сетки (по умолчанию 5, получится 5x5=25 тайлов)
    """
    
    # Создаем выходную директорию
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Загрузка файла: {input_las}")
    las = laspy.read(input_las)
    
    # Получаем координаты
    x = las.x
    y = las.y
    z = las.z
    
    print(f"Всего точек: {len(x)}")
    
    # Определяем границы облака
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    z_min, z_max = z.min(), z.max()
    
    print(f"Границы облака:")
    print(f"  X: [{x_min:.2f}, {x_max:.2f}]")
    print(f"  Y: [{y_min:.2f}, {y_max:.2f}]")
    print(f"  Z: [{z_min:.2f}, {z_max:.2f}]")
    
    # Вычисляем размеры тайлов
    x_range = x_max - x_min
    y_range = y_max - y_min
    
    tile_width = x_range / grid_size
    tile_height = y_range / grid_size
    
    print(f"\nРазмер одного тайла:")
    print(f"  Ширина (X): {tile_width:.2f}")
    print(f"  Высота (Y): {tile_height:.2f}")
    
    # Создаем массив для хранения индексов тайлов для каждой точки
    tile_indices = np.zeros(len(x), dtype=np.int32)
    
    # Вычисляем индексы тайлов для каждой точки
    x_idx = np.floor((x - x_min) / tile_width).astype(np.int32)
    y_idx = np.floor((y - y_min) / tile_height).astype(np.int32)
    
    # Ограничиваем индексы (на случай погрешности)
    x_idx = np.clip(x_idx, 0, grid_size - 1)
    y_idx = np.clip(y_idx, 0, grid_size - 1)
    
    # Вычисляем уникальный индекс тайла (0-24)
    tile_indices = y_idx * grid_size + x_idx
    
    unique_tiles = np.unique(tile_indices)
    print(f"\nУникальных тайлов с точками: {len(unique_tiles)} из {grid_size*grid_size}")
    
    # Сохраняем каждый тайл
    print("\nНачинаем сохранение тайлов...")
    
    for tile_id in range(grid_size * grid_size):
        # Маска точек в этом тайле
        mask = tile_indices == tile_id
        point_count = np.sum(mask)
        
        if point_count == 0:
            print(f"Тайл {tile_id:02d} (строка {tile_id // grid_size}, колонка {tile_id % grid_size}): 0 точек - пропущен")
            continue
        
        # Создаем новый LAS файл для тайла
        # Создаем заголовок с теми же форматами, что и исходный
        header = laspy.LasHeader(point_format=las.header.point_format, version=las.header.version)
        
        # Устанавливаем смещения и масштабы для этого тайла
        points_in_tile = las.points[mask]
        
        # Получаем минимальные координаты для этого тайла
        x_tile = las.x[mask]
        y_tile = las.y[mask]
        z_tile = las.z[mask]
        
        header.offsets = [np.min(x_tile), np.min(y_tile), np.min(z_tile)]
        header.scales = las.header.scales
        
        # Создаем новый LAS объект
        las_tile = laspy.LasData(header)
        
        # Копируем точки
        las_tile.points = points_in_tile
        
        # Сохраняем файл
        output_file = output_path / f"tile_{tile_id:02d}_row{tile_id // grid_size}_col{tile_id % grid_size}.las"
        las_tile.write(output_file)
        
        print(f"Тайл {tile_id:02d} (строка {tile_id // grid_size}, колонка {tile_id % grid_size}): {point_count} точек -> {output_file.name}")
    
    print(f"\nГотово! Тайлы сохранены в: {output_dir}")
    
    # Выводим статистику
    print("\nСтатистика по тайлам:")
    print("-" * 50)
    for row in range(grid_size):
        row_tiles = []
        for col in range(grid_size):
            tile_id = row * grid_size + col
            mask = tile_indices == tile_id
            count = np.sum(mask)
            row_tiles.append(f"{count:>10,}")
        print(f"Строка {row}: " + " | ".join(row_tiles))
    
    return output_dir

def split_pointcloud_3d(input_las, output_dir, grid_xy=5, grid_z=1):
    """
    Разбивает облако точек на 3D тайлы (по X, Y и Z)
    
    Args:
        input_las: путь к входному LAS файлу
        output_dir: директория для сохранения тайлов
        grid_xy: количество тайлов по X и Y (по умолчанию 5)
        grid_z: количество тайлов по Z (по умолчанию 1, только XY разбиение)
    """
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Загрузка файла: {input_las}")
    las = laspy.read(input_las)
    
    x = las.x
    y = las.y
    z = las.z
    
    print(f"Всего точек: {len(x)}")
    
    # Определяем границы
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    z_min, z_max = z.min(), z.max()
    
    print(f"Границы облака:")
    print(f"  X: [{x_min:.2f}, {x_max:.2f}]")
    print(f"  Y: [{y_min:.2f}, {y_max:.2f}]")
    print(f"  Z: [{z_min:.2f}, {z_max:.2f}]")
    
    # Размеры тайлов
    tile_width = (x_max - x_min) / grid_xy
    tile_height = (y_max - y_min) / grid_xy
    tile_depth = (z_max - z_min) / grid_z if grid_z > 1 else (z_max - z_min)
    
    print(f"\nРазмер одного тайла:")
    print(f"  Ширина (X): {tile_width:.2f}")
    print(f"  Высота (Y): {tile_height:.2f}")
    if grid_z > 1:
        print(f"  Глубина (Z): {tile_depth:.2f}")
    
    # Вычисляем индексы
    x_idx = np.floor((x - x_min) / tile_width).astype(np.int32)
    y_idx = np.floor((y - y_min) / tile_height).astype(np.int32)
    
    x_idx = np.clip(x_idx, 0, grid_xy - 1)
    y_idx = np.clip(y_idx, 0, grid_xy - 1)
    
    if grid_z > 1:
        z_idx = np.floor((z - z_min) / tile_depth).astype(np.int32)
        z_idx = np.clip(z_idx, 0, grid_z - 1)
        tile_indices = z_idx * (grid_xy * grid_xy) + y_idx * grid_xy + x_idx
        total_tiles = grid_xy * grid_xy * grid_z
    else:
        tile_indices = y_idx * grid_xy + x_idx
        total_tiles = grid_xy * grid_xy
    
    unique_tiles = np.unique(tile_indices)
    print(f"\nУникальных тайлов с точками: {len(unique_tiles)} из {total_tiles}")
    
    # Сохраняем тайлы
    print("\nНачинаем сохранение тайлов...")
    
    for tile_id in unique_tiles:
        mask = tile_indices == tile_id
        point_count = np.sum(mask)
        
        # Создаем новый LAS файл
        header = laspy.LasHeader(point_format=las.header.point_format, version=las.header.version)
        
        x_tile = las.x[mask]
        y_tile = las.y[mask]
        z_tile = las.z[mask]
        
        header.offsets = [np.min(x_tile), np.min(y_tile), np.min(z_tile)]
        header.scales = las.header.scales
        
        las_tile = laspy.LasData(header)
        las_tile.points = las.points[mask]
        
        if grid_z > 1:
            zi = tile_id // (grid_xy * grid_xy)
            rest = tile_id % (grid_xy * grid_xy)
            yi = rest // grid_xy
            xi = rest % grid_xy
            output_file = output_path / f"tile_z{zi:02d}_y{yi:02d}_x{xi:02d}.las"
        else:
            yi = tile_id // grid_xy
            xi = tile_id % grid_xy
            output_file = output_path / f"tile_y{yi:02d}_x{xi:02d}.las"
        
        las_tile.write(output_file)
        print(f"Тайл {tile_id:03d}: {point_count} точек -> {output_file.name}")
    
    print(f"\nГотово! Тайлы сохранены в: {output_dir}")

if __name__ == "__main__":
    # Путь к входному файлу
    input_file = "./data_las/RS10.las"
    
    # Проверяем существование файла
    if not os.path.exists(input_file):
        print(f"Ошибка: файл {input_file} не найден!")
        exit(1)
    
    # Вариант 1: Простое разбиение на сетку 5x5 (25 тайлов)
    output_directory = "./data_las/RS10_tiles_2x2"
    split_pointcloud_to_tiles(input_file, output_directory, grid_size=2)
    
    # Вариант 2: Разбиение на сетку 5x5 по XY (25 тайлов) с указанием имени
    # split_pointcloud_to_tiles(input_file, "RS10_tiles_5x5_custom", grid_size=5)
    
    # Вариант 3: 3D разбиение (по X, Y и Z) - раскомментируйте если нужно
    # split_pointcloud_3d(input_file, "RS10_tiles_3d", grid_xy=5, grid_z=1)