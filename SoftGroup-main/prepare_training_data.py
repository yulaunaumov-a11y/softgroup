import laspy
import numpy as np
import os
import torch
from pathlib import Path

def load_las_with_labels(las_points_path, las_labels_path, downsample_factor=8):
    """Загрузка облака точек и меток из LAS файлов с даунсемплингом"""
    print(f"Загрузка точек: {las_points_path}")
    las_points = laspy.read(las_points_path)
    
    # Координаты
    xyz = np.vstack((las_points.x, las_points.y, las_points.z)).T
    
    # RGB цвета (нормализуем в [-1, 1] для лучшего обучения)
    if hasattr(las_points, 'red') and hasattr(las_points, 'green') and hasattr(las_points, 'blue'):
        r = las_points.red / 65535.0 * 2 - 1
        g = las_points.green / 65535.0 * 2 - 1
        b = las_points.blue / 65535.0 * 2 - 1
        colors = np.column_stack((r, g, b))
    else:
        # Если нет RGB, используем интенсивность
        if hasattr(las_points, 'intensity'):
            intensity = las_points.intensity / 65535.0
            colors = np.column_stack([intensity, intensity, intensity])
        else:
            colors = np.ones((len(xyz), 3)) * 0.5
    
    print(f"Загрузка меток: {las_labels_path}")
    las_labels = laspy.read(las_labels_path)
    
    # Извлекаем семантические классы (поле classification)
    if hasattr(las_labels, 'classification'):
        semantic_labels = las_labels.classification.astype(np.int32)
    else:
        raise ValueError("Файл меток не содержит поле 'classification'")
    
    # Создаем instance ID
    instance_labels = np.zeros_like(semantic_labels, dtype=np.int32)
    
    # Для объектов (thing classes) назначаем уникальные instance ID
    thing_classes = [65, 66]  # pillar и traffic_sign
    current_instance_id = 1
    
    for class_id in thing_classes:
        mask = semantic_labels == class_id
        if mask.any():
            instance_labels[mask] = current_instance_id
            current_instance_id += 1
    
    original_count = len(xyz)
    print(f"Оригинальное количество точек: {original_count}")
    
    # ПРИМЕНЯЕМ ДАУНСЕМПЛИНГ КО ВСЕМ МАССИВАМ
    if downsample_factor > 1:
        indices = np.arange(0, original_count, downsample_factor)  # каждую N-ю точку
        xyz = xyz[indices]
        colors = colors[indices]
        semantic_labels = semantic_labels[indices]
        instance_labels = instance_labels[indices]
        print(f"После даунсемплинга (фактор {downsample_factor}): {len(xyz)} точек")
    
    print(f"Уникальные классы: {np.unique(semantic_labels)}")
    print(f"Количество instance объектов: {current_instance_id - 1}")
    
    return xyz, colors, semantic_labels, instance_labels

def create_class_mapping():
    """Создание маппинга ваших классов в ID SoftGroup"""
    # Ваши классы -> целевые ID (1-9, 0 - ignore)
    class_mapping = {
        0: 0,   # never_classified -> ignore
        2: 1,   # ground
        4: 2,   # medium_vegetation
        5: 3,   # high_vegetation
        6: 4,   # building
        7: 0,   # low_point -> ignore
        14: 5,  # wire_conductor
        65: 6,  # pillar (thing)
        66: 7,  # traffic_sign (thing)
        73: 8,  # unknown_73
        79: 9,  # fence
    }
    return class_mapping

def save_as_pth(xyz, colors, semantic_labels, instance_labels, output_path):
    """Сохранение в формате .pth для SoftGroup"""
    # Нормализуем координаты (центрируем)
    xyz_center = xyz.mean(axis=0)
    xyz_normalized = xyz - xyz_center
    
    # Применяем маппинг классов
    class_mapping = create_class_mapping()
    semantic_mapped = np.vectorize(class_mapping.get)(semantic_labels)
    
    # Проверяем, что нет None
    if semantic_mapped is None:
        raise ValueError("Ошибка маппинга классов")
    
    # Создаем кортеж для сохранения
    data_tuple = (
        torch.tensor(xyz_normalized, dtype=torch.float32),
        torch.tensor(colors, dtype=torch.float32),
        torch.tensor(semantic_mapped, dtype=torch.long),
        torch.tensor(instance_labels, dtype=torch.long)
    )
    
    torch.save(data_tuple, output_path)
    print(f"Сохранено {len(xyz)} точек в {output_path}")

if __name__ == "__main__":
    # Пути к вашим файлам
    points_file = "RS10.las"
    labels_file = "RS10_class.las"
    
    # Создаем структуру папок для обучения
    train_dir = "dataset/mydataset/preprocess"
    os.makedirs(train_dir, exist_ok=True)
    
    # Конвертируем данные
    print("=" * 50)
    print("Начинаем конвертацию данных для обучения")
    print("=" * 50)
    
    xyz, colors, semantic_labels, instance_labels = load_las_with_labels(points_file, labels_file)
    
    # Сохраняем в .pth
    output_pth = os.path.join(train_dir, "RS10.pth")
    save_as_pth(xyz, colors, semantic_labels, instance_labels, output_pth)
    
    print("\n" + "=" * 50)
    print("Готово! Данные подготовлены для обучения")
    print(f"Файл сохранен: {output_pth}")
    print("=" * 50)