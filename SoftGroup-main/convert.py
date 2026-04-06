import laspy
import numpy as np

if __name__ == "__main__":

    # 1. Загружаем облако точек с RGB
    print("Начинаем загрузку файла RS10.las.")
    las = laspy.read('./data_las/RS10_tiles_2x2/tile_01_row0_col1.las')
    scale = 6
    print("Извлекаем координаты.")
    # 2. Извлекаем координаты
    xyz = np.vstack((las.x, las.y, las.z)).T
    
    # НОРМАЛИЗАЦИЯ КООРДИНАТ: сдвигаем к нулю
    print("Нормализуем координаты (сдвигаем к нулю).")
    xyz_min = xyz.min(axis=0)
    xyz_normalized = xyz - xyz_min
    print(f"Исходный минимум координат: X={xyz_min[0]:.2f}, Y={xyz_min[1]:.2f}, Z={xyz_min[2]:.2f}")
    print(f"Нормализованный диапазон X: [{xyz_normalized[:,0].min():.2f}, {xyz_normalized[:,0].max():.2f}]")
    print(f"Нормализованный диапазон Y: [{xyz_normalized[:,1].min():.2f}, {xyz_normalized[:,1].max():.2f}]")
    print(f"Нормализованный диапазон Z: [{xyz_normalized[:,2].min():.2f}, {xyz_normalized[:,2].max():.2f}]")
    
    # Проверка координат на NaN и Inf
    print("Проверка координат на NaN/Inf:")
    print(f"  NaN в координатах: {np.isnan(xyz_normalized).any()}")
    print(f"  Inf в координатах: {np.isinf(xyz_normalized).any()}")
    
    if np.isnan(xyz_normalized).any() or np.isinf(xyz_normalized).any():
        print("  Обнаружены некорректные значения! Заменяем на 0...")
        xyz_normalized = np.nan_to_num(xyz_normalized, nan=0.0, posinf=0.0, neginf=0.0)
    
    print("Извлекаем интенсивность.")
    # 3. Извлекаем интенсивность (если есть поле intensity)
    if hasattr(las, 'intensity'):
        intensity = las.intensity.astype(np.float32)
        # Нормализуем интенсивность, если нужно (обычно в LAS интенсивность 0-65535)
        if intensity.max() > 1.0:
            intensity = intensity / 65535.0
        print(f"Интенсивность извлечена из поля 'intensity'")
    else:
        # Если поля intensity нет, используем яркость из RGB
        print("Поле 'intensity' не найдено, вычисляем яркость из RGB.")
        r = las.red / 65535.0
        g = las.green / 65535.0
        b = las.blue / 65535.0
        intensity = (r + g + b) / 3.0
    
    # Проверяем интенсивность на проблемные значения
    print("Проверка интенсивности на NaN/Inf:")
    print(f"  NaN в интенсивности: {np.isnan(intensity).any()}")
    print(f"  Inf в интенсивности: {np.isinf(intensity).any()}")
    print(f"  Диапазон интенсивности: [{intensity.min():.4f}, {intensity.max():.4f}]")
    
    intensity = np.nan_to_num(intensity, nan=0.0, posinf=1.0, neginf=0.0)
    
    print("Собираем массив.")
    # 4. Собираем массив [N, 4] с нормализованными координатами
    points = np.column_stack((xyz_normalized, intensity)).astype(np.float32)
    
    # 5. Уменьшаем количество точек (берем каждую 8-ю точку)
    original_count = len(points)
    points_downsampled = points[::scale]  # шаг 8 - берем каждую 8-ю точку
#     points_downsampled = points
    downsampled_count = len(points_downsampled)
    print(f"Оригинальное количество точек: {original_count}")
    print(f"Количество точек после уменьшения в 8 раз: {downsampled_count}")
    
    # Финальная проверка всех данных
    print("Финальная проверка данных:")
    print(f"  Общий размер данных: {points_downsampled.nbytes / 1024 / 1024:.2f} MB")
    print(f"  NaN в финальных данных: {np.isnan(points_downsampled).any()}")
    print(f"  Inf в финальных данных: {np.isinf(points_downsampled).any()}")
    print(f"  Диапазон X: [{points_downsampled[:,0].min():.2f}, {points_downsampled[:,0].max():.2f}]")
    print(f"  Диапазон Y: [{points_downsampled[:,1].min():.2f}, {points_downsampled[:,1].max():.2f}]")
    print(f"  Диапазон Z: [{points_downsampled[:,2].min():.2f}, {points_downsampled[:,2].max():.2f}]")
    print(f"  Диапазон интенсивности: [{points_downsampled[:,3].min():.4f}, {points_downsampled[:,3].max():.4f}]")
    
    print("Сохраняем в бинарный файл.")
    # 6. Сохраняем в бинарный файл
    points_downsampled.tofile('dataset/kitti/sequences/00/velodyne/000000.bin')

    print(f"Готово! Сохранено {downsampled_count} точек в dataset/kitti/sequences/00/velodyne/000000.bin")
    
    # 7. Создаем файл-заглушку для меток с новым количеством точек
    dummy_labels = np.zeros(downsampled_count, dtype=np.uint32)
    dummy_labels.tofile('dataset/kitti/sequences/00/labels/000000.label')

    print(f"Создан файл-заглушка для {downsampled_count} точек")
    
    # 8. Сохраняем уменьшенное облако в LAS формате для визуализации
    print("\nСохраняем уменьшенное облако в LAS формате...")
    
    # Создаем новый LAS файл
    header = laspy.LasHeader(point_format=3, version="1.4")
    header.offsets = np.min(xyz_normalized[::8], axis=0)
    header.scales = [0.001, 0.001, 0.001]
    
    las_out = laspy.LasData(header)
    
    # Извлекаем координаты из даунсемплированных данных
    xyz_down = points_downsampled[:, :3]
    intensity_down = points_downsampled[:, 3]
    
    las_out.x = xyz_down[:, 0]
    las_out.y = xyz_down[:, 1]
    las_out.z = xyz_down[:, 2]
    
    # Интенсивность обратно в 0-65535
    intensity_normalized = np.clip(intensity_down * 65535, 0, 65535).astype(np.uint16)
    las_out.intensity = intensity_normalized
    
    # Добавляем цвета (серый по интенсивности)
    rgb = np.stack([intensity_normalized, intensity_normalized, intensity_normalized], axis=1)
    las_out.red = rgb[:, 0]
    las_out.green = rgb[:, 1]
    las_out.blue = rgb[:, 2]
    
    # Сохраняем LAS файл
    output_las = 'RS10_downsampled_8x.las'
    las_out.write(output_las)
    print(f"Готово! Уменьшенное облако сохранено в {output_las}")
    print(f"Всего точек в LAS файле: {downsampled_count}")