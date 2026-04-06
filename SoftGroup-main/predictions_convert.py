import laspy
import numpy as np
import os

def load_kitti_bin(bin_path):
    """Загрузка точек из .bin файла KITTI формата"""
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    xyz = points[:, :3]
    intensity = points[:, 3]
    return xyz, intensity

def load_kitti_label(label_path):
    """Загрузка меток из .label файла KITTI формата"""
    labels = np.fromfile(label_path, dtype=np.uint32)
    semantic_labels = labels & 0xFFFF  # нижние 16 бит
    instance_ids = labels >> 16         # верхние 16 бит
    return semantic_labels, instance_ids

def get_color_for_class(sem_id):
    """Получить цвет для класса (в формате RGB 0-65535)"""
    colors = {
        0: [0, 0, 0],
        1: [0, 0, 255],
        10: [245, 150, 100],
        11: [245, 230, 100],
        13: [250, 80, 100],
        15: [150, 60, 30],
        16: [255, 0, 0],
        18: [180, 30, 80],
        20: [255, 0, 0],
        30: [30, 30, 255],
        31: [200, 40, 255],
        32: [90, 30, 150],
        40: [255, 0, 255],
        44: [255, 150, 255],
        48: [75, 0, 75],
        49: [75, 0, 175],
        50: [0, 200, 255],
        51: [50, 120, 255],
        52: [0, 150, 255],
        60: [170, 255, 150],
        70: [0, 175, 0],
        71: [0, 60, 135],
        72: [80, 240, 150],
        80: [150, 240, 255],
        81: [0, 0, 255],
        99: [255, 255, 50],
        252: [245, 150, 100],
        253: [200, 40, 255],
        254: [30, 30, 255],
        255: [90, 30, 150],
        256: [255, 0, 0],
        257: [250, 80, 100],
        258: [180, 30, 80],
        259: [255, 0, 0]
    }
    rgb = colors.get(sem_id, [100, 100, 100])
    # Конвертируем в формат LAS (0-65535)
    return [int(rgb[0] * 257), int(rgb[1] * 257), int(rgb[2] * 257)]

if __name__ == "__main__":
    # Пути к файлам
    input_bin = 'dataset/kitti/sequences/00/velodyne/000000.bin'
    pred_label = 'results_my/panoptic/sequences/00/predictions/000000.label'
    output_las = 'output_predictions.las'
    
    # Загружаем точки из .bin
    print("Загрузка точек из .bin...")
    xyz, intensity = load_kitti_bin(input_bin)
    print(f"Загружено {len(xyz)} точек из .bin")
    
    # Загружаем предсказания
    print("Загрузка предсказаний...")
    if os.path.exists(pred_label):
        semantic_labels, instance_ids = load_kitti_label(pred_label)
        print(f"Загружено {len(semantic_labels)} предсказаний")
        
        # Проверка соответствия
        if len(semantic_labels) != len(xyz):
            print(f"ОШИБКА: количество предсказаний ({len(semantic_labels)}) не совпадает с количеством точек ({len(xyz)})")
            print("Ищем правильный .bin файл...")
            
            # Поиск .bin с правильным количеством точек
            import glob
            for f in glob.glob("**/*.bin", recursive=True):
                try:
                    test_points = np.fromfile(f, dtype=np.float32).reshape(-1, 4)
                    if len(test_points) == len(semantic_labels):
                        print(f"Найден подходящий .bin: {f}")
                        input_bin = f
                        xyz, intensity = load_kitti_bin(input_bin)
                        print(f"Загружено {len(xyz)} точек из нового .bin")
                        break
                except:
                    continue
            else:
                print("Не найден .bin с правильным количеством точек")
                exit(1)
    else:
        print(f"Файл предсказаний не найден: {pred_label}")
        exit(1)
    
    # Статистика по классам
    unique, counts = np.unique(semantic_labels, return_counts=True)
    print(f"\nУникальные классы: {len(unique)}")
    print("Распределение классов:")
    for u, c in zip(unique, counts):
        print(f"  {u}: {c} точек ({c/len(semantic_labels)*100:.1f}%)")
    
    # Создаем LAS файл
    print("\nСоздание LAS файла...")
    
    # Создаем новый LAS файл
    header = laspy.LasHeader(point_format=3, version="1.4")
    header.offsets = np.min(xyz, axis=0)
    header.scales = [0.001, 0.001, 0.001]
    
    las_out = laspy.LasData(header)
    
    # Добавляем координаты
    las_out.x = xyz[:, 0]
    las_out.y = xyz[:, 1]
    las_out.z = xyz[:, 2]
    
    # Интенсивность
    intensity_normalized = np.clip(intensity * 65535, 0, 65535).astype(np.uint16)
    las_out.intensity = intensity_normalized
    
    # Применяем цвета на основе семантических меток
    print("Применение цветов к точкам...")
    colors = np.array([get_color_for_class(label) for label in semantic_labels])
    las_out.red = colors[:, 0]
    las_out.green = colors[:, 1]
    las_out.blue = colors[:, 2]
    
    # Добавляем метки как дополнительные поля
    las_out.add_extra_dim(laspy.ExtraBytesParams(name="semantic", type=np.uint32))
    las_out.add_extra_dim(laspy.ExtraBytesParams(name="instance", type=np.uint32))
    las_out.semantic = semantic_labels
    las_out.instance = instance_ids
    
    # Сохраняем
    las_out.write(output_las)
    print(f"\nГотово! Результат сохранен в {output_las}")
    print(f"Всего точек: {len(xyz)}")
    print(f"Размер файла: {os.path.getsize(output_las) / 1024 / 1024:.2f} MB")