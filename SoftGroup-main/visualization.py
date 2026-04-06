import laspy
import numpy as np
import pyvista as pv
import os

def load_las_with_classes(las_path):
    """Загрузка LAS файла с классами"""
    print(f"Загрузка файла: {las_path}")
    las = laspy.read(las_path)
    
    # Координаты
    xyz = np.vstack((las.x, las.y, las.z)).T
    
    # Семантические классы (если есть)
    if hasattr(las, 'semantic_pred'):
        semantic = las.semantic_pred
        print(f"Найдено поле 'semantic_pred' с классами")
    elif hasattr(las, 'semantic'):
        semantic = las.semantic
        print(f"Найдено поле 'semantic' с классами")
    else:
        print("Поле с семантическими классами не найдено")
        semantic = None
    
    # Instance классы
    if hasattr(las, 'instance_pred'):
        instance = las.instance_pred
        print(f"Найдено поле 'instance_pred' с instance ID")
    elif hasattr(las, 'instance'):
        instance = las.instance
        print(f"Найдено поле 'instance' с instance ID")
    else:
        instance = None
    
    # RGB цвета
    if hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue'):
        colors = np.column_stack((las.red, las.green, las.blue)) / 65535.0
        print("Найдены RGB цвета")
    else:
        colors = None
    
    print(f"Загружено {len(xyz)} точек")
    
    return xyz, semantic, instance, colors

def create_class_mapping():
    """Создание маппинга ID классов в названия и цвета"""
    class_info = {
        # Основные классы
        0: {'name': 'never_classified', 'color': [128, 128, 128], 'description': 'Не классифицировано'},
        2: {'name': 'ground', 'color': [139, 69, 19], 'description': 'Земля'},
        4: {'name': 'medium_vegetation', 'color': [0, 128, 0], 'description': 'Средняя растительность'},
        5: {'name': 'high_vegetation', 'color': [0, 255, 0], 'description': 'Высокая растительность'},
        6: {'name': 'building', 'color': [255, 0, 0], 'description': 'Здание'},
        7: {'name': 'low_point', 'color': [0, 0, 0], 'description': 'Низкая точка / шум'},
        14: {'name': 'wire_conductor', 'color': [255, 255, 0], 'description': 'Проводник/фаза'},
        65: {'name': 'pillar', 'color': [128, 0, 128], 'description': 'Столбы'},
        66: {'name': 'traffic_sign', 'color': [0, 0, 255], 'description': 'Дорожные знаки'},
        73: {'name': 'unknown_73', 'color': [255, 165, 0], 'description': 'Неизвестный класс 73'},
        79: {'name': 'fence', 'color': [139, 69, 19], 'description': 'Забор'},
    }
    return class_info

def visualize_pointcloud_html(xyz, semantic, instance, colors, output_html, class_info):
    """Визуализация облака точек в HTML с помощью pyvista"""
    
    print("\nСоздание HTML визуализации...")
    
    # Подготовка данных для визуализации
    if semantic is not None:
        # Создаем цвета на основе классов
        point_colors = np.zeros((len(xyz), 3))
        unique_classes = np.unique(semantic)
        
        for class_id in unique_classes:
            mask = semantic == class_id
            if class_id in class_info:
                color = np.array(class_info[class_id]['color']) / 255.0
                point_colors[mask] = color
            else:
                # Для неизвестных классов используем серый цвет
                point_colors[mask] = [0.5, 0.5, 0.5]
        
        print(f"Визуализация с семантическими классами: {len(unique_classes)} уникальных классов")
        print(f"Найденные классы: {sorted(unique_classes)}")
    elif colors is not None:
        point_colors = colors
        print("Визуализация с исходными RGB цветами")
    else:
        intensity = np.linalg.norm(xyz, axis=1)
        intensity = (intensity - intensity.min()) / (intensity.max() - intensity.min())
        point_colors = np.column_stack([intensity, intensity, intensity])
        print("Визуализация с градиентом высоты")
    
    # Уменьшаем количество точек для производительности
    max_points = 500000
    if len(xyz) > max_points:
        print(f"Слишком много точек ({len(xyz)}), уменьшаем до {max_points}...")
        indices = np.random.choice(len(xyz), max_points, replace=False)
        xyz = xyz[indices]
        point_colors = point_colors[indices]
        if semantic is not None:
            semantic = semantic[indices]
    
    # Создаем полигон для визуализации
    poly_data = pv.PolyData(xyz)
    poly_data['colors'] = point_colors
    
    # Создаем сцену
    plotter = pv.Plotter(window_size=[1280, 800], off_screen=True)
    
    # Добавляем точки
    plotter.add_points(poly_data, scalars='colors', rgb=True, point_size=2, render_points_as_spheres=False)
    
    # Добавляем сетку
    plotter.show_grid(color='lightgray', font_size=10)
    
    # Создаем красивую HTML легенду
    if semantic is not None:
        # Группируем классы по категориям
        categories = {
            'Фон/Неразмеченные': [0, 7],
            'Рельеф и земля': [2],
            'Растительность': [4, 5],
            'Сооружения': [6, 79, 65],
            'Инфраструктура': [14, 66],
            'Прочее': [73]
        }
        
        legend_html = """
        <div style="
            position: absolute;
            top: 10px;
            right: 10px;
            background: rgba(30, 30, 30, 0.95);
            color: white;
            padding: 15px;
            border-radius: 8px;
            font-family: Arial, sans-serif;
            font-size: 12px;
            max-height: 80%;
            overflow-y: auto;
            min-width: 280px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.3);
            border: 1px solid #555;
            z-index: 1000;
        ">
            <h3 style="margin: 0 0 10px 0; text-align: center; color: #fff;">📊 Легенда классов</h3>
        """
        
        for category, class_ids in categories.items():
            # Добавляем только классы, которые присутствуют в данных
            present_classes = [cid for cid in class_ids if cid in unique_classes]
            if present_classes:
                legend_html += f"<h4 style='margin: 10px 0 5px 0; color: #ddd; border-bottom: 1px solid #555;'>{category}</h4>"
                for class_id in present_classes:
                    if class_id in class_info:
                        info = class_info[class_id]
                        color = info['color']
                        legend_html += f"""
                        <div style="display: flex; align-items: center; margin-bottom: 8px;">
                            <div style="
                                width: 24px;
                                height: 24px;
                                background-color: rgb({color[0]}, {color[1]}, {color[2]});
                                margin-right: 12px;
                                border: 1px solid white;
                                border-radius: 4px;
                                flex-shrink: 0;
                            "></div>
                            <div style="flex: 1;">
                                <b>{class_id}</b> - {info['name']}
                                <span style="color: #aaa; font-size: 10px; display: block;">{info['description']}</span>
                            </div>
                        </div>
                        """
        
        legend_html += """
            <hr style="margin: 10px 0; border-color: #555;">
            <div style="font-size: 10px; color: #aaa; text-align: center;">
                📍 Всего точек: {total_points:,}<br>
                🏷️ Уникальных классов: {unique_classes}
            </div>
        </div>
        """.format(total_points=len(xyz), unique_classes=len(unique_classes))
        
        # Добавляем HTML аннотацию через PyVista
        plotter.add_text(legend_html, position='upper_right', font_size=12, color='white')
    
    # Добавляем информацию о количестве точек
    info_text = f"Точек: {len(xyz):,}"
    if semantic is not None:
        info_text += f" | Классов: {len(unique_classes)}"
    plotter.add_text(info_text, position='lower_left', font_size=10, color='white')
    
    # Добавляем инструкцию
    plotter.add_text("🖱️ ЛКМ - вращать | ПКМ - панорамировать | Колесо - масштаб", 
                     position='lower_right', font_size=10, color='white')
    
    # Сохраняем в HTML
    print(f"Сохраняем HTML в {output_html}...")
    plotter.export_html(output_html)
    
    # Закрываем plotter
    plotter.close()
    
    print(f"✅ Готово! Откройте {output_html} в браузере для просмотра")

def visualize_pointcloud_plotly(xyz, semantic, instance, colors, output_html, class_info):
    """Альтернативная визуализация с использованием Plotly (интерактивная)"""
    try:
        import plotly.graph_objects as go
        import plotly.express as px
        import pandas as pd
    except ImportError:
        print("Plotly не установлен. Установите: pip install plotly")
        return
    
    print("\nСоздание HTML визуализации с Plotly...")
    
    # Уменьшаем количество точек
    max_points = 200000
    if len(xyz) > max_points:
        print(f"Уменьшаем количество точек с {len(xyz)} до {max_points}...")
        indices = np.random.choice(len(xyz), max_points, replace=False)
        xyz = xyz[indices]
        if semantic is not None:
            semantic = semantic[indices]
    
    # Создаем DataFrame для Plotly
    df = pd.DataFrame({
        'x': xyz[:, 0],
        'y': xyz[:, 1],
        'z': xyz[:, 2]
    })
    
    if semantic is not None:
        # Добавляем классы
        class_names = [class_info.get(c, {'name': f'Class {c}'})['name'] for c in semantic]
        class_colors = [class_info.get(c, {'color': [128, 128, 128]})['color'] for c in semantic]
        
        df['class'] = semantic
        df['class_name'] = class_names
        df['color'] = [f'rgb({c[0]},{c[1]},{c[2]})' for c in class_colors]
        
        # Создаем 3D scatter plot
        fig = px.scatter_3d(
            df, x='x', y='y', z='z',
            color='class_name',
            color_discrete_map={info['name']: f'rgb({info["color"][0]},{info["color"][1]},{info["color"][2]})' 
                                for cid, info in class_info.items() if cid in np.unique(semantic)},
            title='Облако точек с семантической сегментацией',
            opacity=0.7,
            size_max=2
        )
    else:
        fig = px.scatter_3d(
            df, x='x', y='y', z='z',
            title='Облако точек',
            opacity=0.7,
            size_max=2
        )
    
    # Настройка внешнего вида
    fig.update_traces(marker=dict(size=1))
    fig.update_layout(
        scene=dict(
            xaxis_title='X (м)',
            yaxis_title='Y (м)',
            zaxis_title='Z (м)',
            aspectmode='data',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        title=dict(text='Визуализация облака точек', x=0.5, font=dict(size=16)),
        showlegend=True,
        legend=dict(
            title='Классы',
            yanchor='top',
            y=0.99,
            xanchor='left',
            x=0.01,
            bgcolor='rgba(0,0,0,0.8)',
            font=dict(color='white', size=10)
        ),
        paper_bgcolor='rgba(0,0,0,0.1)'
    )
    
    # Сохраняем
    fig.write_html(output_html)
    print(f"✅ Готово! Сохранено в {output_html}")

if __name__ == "__main__":
    # Путь к файлу с предсказаниями
    las_file = "RS10_pred.las"
    output_html = "./tile_predict_html/pointcloud_visualization_tile_01_row0_col1.html"
    
    # Проверяем существование файла
    if not os.path.exists(las_file):
        print(f"❌ Ошибка: файл {las_file} не найден!")
        # Пробуем альтернативные имена
        alternatives = ["RS10_pred (1).las", "RS10_pred_full.las", "output_predictions.las"]
        for alt in alternatives:
            if os.path.exists(alt):
                las_file = alt
                print(f"Найден альтернативный файл: {las_file}")
                break
        else:
            exit(1)
    
    # Загружаем данные
    xyz, semantic, instance, colors = load_las_with_classes(las_file)
    
    # Загружаем информацию о классах
    class_info = create_class_mapping()
    
    # Выводим статистику по классам
    if semantic is not None:
        unique, counts = np.unique(semantic, return_counts=True)
        print("\nСтатистика по классам:")
        for u, c in zip(unique, counts):
            if u in class_info:
                name = class_info[u]['name']
                print(f"  Класс {u}: {name} - {c} точек ({c/len(semantic)*100:.1f}%)")
            else:
                print(f"  Класс {u}: неизвестный - {c} точек ({c/len(semantic)*100:.1f}%)")
    
    # Выбираем метод визуализации
    try:
        visualize_pointcloud_html(xyz, semantic, instance, colors, output_html, class_info)
    except Exception as e:
        print(f"⚠️ PyVista не удалось создать HTML: {e}")
        print("Пробуем Plotly...")
        
        # Метод 2: Plotly (альтернатива)
        visualize_pointcloud_plotly(xyz, semantic, instance, colors, output_html, class_info)
    
    print(f"\n✨ Откройте {output_html} в браузере для просмотра интерактивного облака точек")