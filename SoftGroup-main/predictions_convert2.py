# import laspy
# import numpy as np
# import os
# from scipy.spatial import KDTree

# if __name__ == "__main__":
#     # Пути
#     original_las = "RS10.las"
#     pred_label = "results_my/panoptic/sequences/00/predictions/000000.label"
#     out_las = "RS10_pred_full.las"
    
#     # Сначала найдем правильный .bin файл
#     import glob
#     pred = np.fromfile(pred_label, dtype=np.uint32)
#     print(f"Предсказаний: {len(pred)}")
    
#     # Ищем .bin файл с таким же количеством точек
#     bin_file = None
#     for f in glob.glob("**/*.bin", recursive=True):
#         try:
#             test_points = np.fromfile(f, dtype=np.float32).reshape(-1, 4)
#             if len(test_points) == len(pred):
#                 bin_file = f
#                 points_bin = test_points
#                 print(f"Найден соответствующий .bin: {bin_file}")
#                 break
#         except:
#             continue
    
#     if bin_file is None:
#         print("Ошибка: не найден .bin файл с правильным количеством точек")
#         exit(1)
    
#     xyz_bin = points_bin[:, :3]
    
#     # Загружаем полное облако
#     print("Загрузка полного облака...")
#     las_full = laspy.read(original_las)
#     xyz_full = np.vstack((las_full.x, las_full.y, las_full.z)).T
#     print(f"Полное облако: {len(xyz_full)} точек")
    
#     # Создаем KDTree для поиска ближайших соседей
#     print("Построение KDTree...")
#     tree = KDTree(xyz_bin)
    
#     # Для каждой точки в полном облаке находим ближайшую в даунсемплированном
#     print("Сопоставление предсказаний...")
#     distances, indices = tree.query(xyz_full, k=1)
    
#     # Применяем предсказания
#     sem_full = (pred[indices] & 0xFFFF).astype(np.uint16)
#     ins_full = (pred[indices] >> 16).astype(np.uint32)
    
#     print(f"Уникальные классы в полном облаке: {np.unique(sem_full)}")
    
#     # Цвета для классов
#     def get_color_for_class(sem_id):
#         colors = {
#             0: [0, 0, 0], 1: [255,0,0], 2: [0,255,0], 3: [0,0,255],
#             4: [255,255,0], 5: [255,0,255], 6: [0,255,255], 7: [128,128,128],
#             8: [128,0,0], 9: [0,128,0], 10: [0,0,128], 11: [128,128,0],
#             12: [128,0,128], 13: [0,128,128], 14: [192,192,192], 15: [192,192,0],
#             16: [192,0,192], 17: [0,192,192], 18: [192,0,0], 19: [0,192,0]
#         }
#         rgb = colors.get(sem_id, [100,100,100])
#         return [int(rgb[0]*257), int(rgb[1]*257), int(rgb[2]*257)]
    
#     # Применяем цвета к полному облаку
#     print("Применение цветов...")
#     colors_rgb = np.array([get_color_for_class(s) for s in sem_full])
#     las_full.red = colors_rgb[:, 0]
#     las_full.green = colors_rgb[:, 1]
#     las_full.blue = colors_rgb[:, 2]
    
#     # Добавляем метки
#     las_full.add_extra_dim(laspy.ExtraBytesParams(name="semantic_pred", type=np.uint16))
#     las_full.add_extra_dim(laspy.ExtraBytesParams(name="instance_pred", type=np.uint32))
#     las_full.semantic_pred = sem_full
#     las_full.instance_pred = ins_full
    
#     # Сохраняем
#     las_full.write(out_las)
#     print(f"Готово! Сохранено в {out_las}")
#     print(f"Всего точек: {len(xyz_full)}")

import laspy
import numpy as np

if __name__ == "__main__":
    in_las = "RS10_downsampled_8x.las"
    pred_label = "results_my/panoptic/sequences/00/velodyne/000000.label"
    out_las = "RS10_pred.las"

    las = laspy.read(in_las)
    pred = np.fromfile(pred_label, dtype=np.uint32)

    assert len(pred) == len(las.x), f"points mismatch: {len(pred)} vs {len(las.x)}"

    sem = (pred & 0xFFFF).astype(np.uint16)
    ins = (pred >> 16).astype(np.uint32)

    # добавить extra dims
    las.add_extra_dim(laspy.ExtraBytesParams(name="semantic_pred", type=np.uint16))
    las.add_extra_dim(laspy.ExtraBytesParams(name="instance_pred", type=np.uint32))

    las.semantic_pred = sem
    las.instance_pred = ins

    las.write(out_las)
    print("saved:", out_las)