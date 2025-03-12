import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from shapely.geometry import Polygon
from shapely.ops import unary_union
import os
import matplotlib.font_manager as fm

# 設定中文字體
try:
    # Mac系統
    font_path = '/System/Library/Fonts/PingFang.ttc'
    if os.path.exists(font_path):
        font_prop = fm.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = font_prop.get_name()
    else:
        # Windows系統
        plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
except Exception as e:
    print(f"警告: 字體設定問題 - {e}")

# 定義台灣區域（依縣市）
TAIWAN_REGIONS = {
    'north': ['臺北市', '台北市', '新北市', '基隆市', '桃園市', '新竹市', '新竹縣', '宜蘭縣'],
    'central': ['苗栗縣', '臺中市', '台中市', '彰化縣', '南投縣', '雲林縣'],
    'south': ['嘉義市', '嘉義縣', '臺南市', '台南市', '高雄市', '屏東縣', '澎湖縣'],
    'east': ['花蓮縣', '臺東縣', '台東縣']
}

# 處理感測器資料函數
def process_sensor_data(sensor_df):
    """
    處理感測器資料計算:
    1. 每個鄉鎮市的獨立感測站數量
    2. 每個鄉鎮市的平均淹水深度
    
    參數:
    sensor_df: 具有感測器資料的DataFrame
    
    返回:
    tuple: (station_count_dict, avg_depth_dict)
    """
    # 計算每個鄉鎮市的獨立感測站數量
    station_count = sensor_df.groupby('TOWNNAME')['station_id'].nunique()
    station_count_dict = station_count.to_dict()
    
    # 計算每個鄉鎮市的平均深度
    avg_depth = sensor_df.groupby('TOWNNAME')['value'].mean()
    avg_depth_dict = avg_depth.to_dict()
    
    return station_count_dict, avg_depth_dict

# 處理雨量資料函數
def process_rainfall_data(rainfall_df):
    """
    處理雨量資料計算每個鄉鎮市的平均雨量
    
    參數:
    rainfall_df: 具有雨量資料的DataFrame，需包含鄉鎮市名稱和雨量資料
    
    返回:
    dict: 每個鄉鎮市的平均雨量 {鄉鎮市名稱: 平均雨量}
    """
    # rainfall_df有'TOWNNAME'和'Past24hr'欄位
    avg_rainfall = rainfall_df.groupby('TownName')['Past24hr'].mean()
    avg_rainfall_dict = avg_rainfall.to_dict()
    
    return avg_rainfall_dict

def process_flood_loss_data(damage_df):
    
    # 計算每個鄉鎮市的平均淹水損失
    avg_damage = damage_df.groupby('TOWNNAME')['total_flood_loss'].mean()
    total_damage_dict = avg_damage.to_dict()
    
    return total_damage_dict

# 繪製鄉鎮市熱力圖
def plot_township_heatmap(sf, data_dict, metric_name,
                         x_lim=(120, 122), y_lim=(21.8, 25.3),
                         figsize=(15, 20), cmap='YlOrRd', title=None,
                         output_path=None):
    """
    繪製鄉鎮市熱力圖並標示縣市
    
    參數:
    sf: shapefile Reader物件
    data_dict: 包含鄉鎮市資料的字典 {鄉鎮市名稱: 數值}
    metric_name: 測量指標名稱（用於色標）
    region: 可選區域過濾器('north', 'central', 'south', 'east')
    x_lim, y_lim: 地圖邊界
    figsize: 圖形大小
    cmap: 顏色地圖
    title: 可選自定義標題
    output_path: 可選儲存圖形的路徑
    
    返回:
    Figure和axis物件
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # 設定地圖邊界
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    
    # 獲取欄位索引
    fields = sf.fields[1:]
    field_names = [field[0] for field in fields]
    county_idx = field_names.index('COUNTYNAME')
    
    # 尋找鄉鎮市欄位
    town_idx = None
    for possible_name in ['TOWNNAME', 'TOWNSHIPNAME', 'TOWNSHIP']:
        if possible_name in field_names:
            town_idx = field_names.index(possible_name)
            break
    
    # 準備資料結構
    township_polygons = {}  # {鄉鎮市名稱: shapely_polygon}
    township_values = {}    # {鄉鎮市名稱: 數值}
    county_polygons = {}    # {縣市名稱: [shapely_polygons]}
    township_to_county = {} # {鄉鎮市名稱: 縣市名稱}
    
    # 處理形狀
    for shape in sf.shapeRecords():
        county_name = shape.record[county_idx]
            
        # 如果可用，獲取鄉鎮市名稱
        town_name = None
        if town_idx is not None:
            town_name = shape.record[town_idx]
        
        # 獲取座標
        x = [p[0] for p in shape.shape.points]
        y = [p[1] for p in shape.shape.points]
        
        # 如果在我們區域之外，則跳過
        x0, y0 = np.mean(x), np.mean(y)
        if not (x_lim[0] <= x0 <= x_lim[1] and y_lim[0] <= y0 <= y_lim[1]):
            continue
            
        # 創建多邊形
        try:
            points = list(zip(x, y))
            poly = Polygon(points)
            if not poly.is_valid:
                poly = poly.buffer(0)
                
            # 儲存縣市多邊形
            county_polygons.setdefault(county_name, []).append(poly)
            
            # 如果可用，儲存鄉鎮市資料
            if town_name and not poly.is_empty:
                township_polygons[town_name] = poly
                township_to_county[town_name] = county_name
                
                # 如果此鄉鎮市有資料，則儲存值
                if town_name in data_dict:
                    township_values[town_name] = data_dict[town_name]
        except Exception as e:
            print(f"處理形狀時出錯: {e}")
    
    # 準備顏色映射
    if township_values:
        vmin = min(township_values.values())
        vmax = max(township_values.values())
    else:
        vmin, vmax = 0, 1
    
    norm = plt.Normalize(vmin, vmax)
    color_mapper = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    
    # 繪製鄉鎮市邊界（淺灰色）
    for town_name, poly in township_polygons.items():
        if poly.geom_type == 'Polygon':
            x, y = poly.exterior.xy
            ax.plot(x, y, color='lightgray', linewidth=0.5, zorder=1)
        elif poly.geom_type == 'MultiPolygon':
            for subpoly in poly.geoms:
                x, y = subpoly.exterior.xy
                ax.plot(x, y, color='lightgray', linewidth=0.5, zorder=1)
    
    # 填充有資料的鄉鎮市
    for town_name, poly in township_polygons.items():
        if town_name in township_values:
            value = township_values[town_name]
            color = color_mapper.to_rgba(value)
            
            if poly.geom_type == 'Polygon':
                x, y = poly.exterior.xy
                ax.fill(x, y, color=color, alpha=0.7, zorder=2)
            elif poly.geom_type == 'MultiPolygon':
                for subpoly in poly.geoms:
                    x, y = subpoly.exterior.xy
                    ax.fill(x, y, color=color, alpha=0.7, zorder=2)
    
    # 繪製縣市邊界（黑色）
    county_centroids = {}  # 用於縣市標籤
    for county, polys in county_polygons.items():
        union_poly = unary_union(polys)
        if union_poly.is_empty:
            continue
            
        # 儲存重心
        county_centroids[county] = union_poly.centroid
        
        # 繪製邊界
        if union_poly.geom_type == 'Polygon':
            x, y = union_poly.exterior.xy
            ax.plot(x, y, color='black', linewidth=1.5, zorder=4)
        elif union_poly.geom_type == 'MultiPolygon':
            for subpoly in union_poly.geoms:
                x, y = subpoly.exterior.xy
                ax.plot(x, y, color='black', linewidth=1.5, zorder=4)
    
    # 添加帶值的鄉鎮市標籤
    for town_name, poly in township_polygons.items():
        if town_name in township_values:
            value = township_values[town_name]
            centroid = poly.centroid
            
            # 格式化值
            if isinstance(value, float):
                if value > 100:  # 大值
                    formatted_value = f"{value:.0f}"
                elif value > 10:  # 中值
                    formatted_value = f"{value:.1f}"
                else:  # 小值
                    formatted_value = f"{value:.2f}"
            else:
                formatted_value = f"{value}"
            
            label = f"{town_name}\n{formatted_value}"
            
            # 根據背景亮度確定文字顏色
            color_value = color_mapper.to_rgba(value)
            brightness = 0.299*color_value[0] + 0.587*color_value[1] + 0.114*color_value[2]
            text_color = 'black' if brightness > 0.5 else 'white'
            
            ax.text(centroid.x, centroid.y, label,
                   fontsize=8, ha='center', va='center',
                   color=text_color, zorder=5, fontweight='bold')
    
    # 添加帶箭頭的縣市標籤
    # 按位置分組縣市
    map_width = x_lim[1] - x_lim[0]
    map_height = y_lim[1] - y_lim[0]
    
    # 按位置分組縣市
    west_counties = []
    east_counties = []
    north_counties = []
    south_counties = []
    
    for county, centroid in county_centroids.items():
        # 確定最近邊緣
        dist_to_left = centroid.x - x_lim[0]
        dist_to_right = x_lim[1] - centroid.x
        dist_to_bottom = centroid.y - y_lim[0]
        dist_to_top = y_lim[1] - centroid.y
        
        min_dist = min(dist_to_left, dist_to_right, dist_to_bottom, dist_to_top)
        
        if min_dist == dist_to_left:
            west_counties.append((county, centroid))
        elif min_dist == dist_to_right:
            east_counties.append((county, centroid))
        elif min_dist == dist_to_bottom:
            south_counties.append((county, centroid))
        elif min_dist == dist_to_top:
            north_counties.append((county, centroid))
    
    # 按位置排序
    west_counties.sort(key=lambda x: x[1].y, reverse=True)
    east_counties.sort(key=lambda x: x[1].y, reverse=True)
    north_counties.sort(key=lambda x: x[1].x)
    south_counties.sort(key=lambda x: x[1].x)
    
    # 箭頭樣式
    arrow_style = dict(arrowstyle="->", color='black', linewidth=1.5, connectionstyle="arc3,rad=0.2")
    text_style = dict(fontsize=12, fontweight='bold', color='black', 
                     bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.3'))
    
    # 添加縣市標籤
    # 西側
    for i, (county, centroid) in enumerate(west_counties):
        label_x = x_lim[0] + map_width * 0.05
        label_y = y_lim[0] + map_height * (0.9 - i * 0.05)
        
        ax.annotate(county, xy=(centroid.x, centroid.y), xytext=(label_x, label_y),
                   arrowprops=arrow_style, zorder=6, **text_style)
    
    # 東側
    for i, (county, centroid) in enumerate(east_counties):
        label_x = x_lim[1] - map_width * 0.05
        label_y = y_lim[0] + map_height * (0.9 - i * 0.05)
        
        ax.annotate(county, xy=(centroid.x, centroid.y), xytext=(label_x, label_y),
                   arrowprops=arrow_style, zorder=6, **text_style)
    
    # 北側
    for i, (county, centroid) in enumerate(north_counties):
        label_x = x_lim[0] + map_width * (0.1 + i * 0.1)
        label_y = y_lim[1] - map_height * 0.05
        
        ax.annotate(county, xy=(centroid.x, centroid.y), xytext=(label_x, label_y),
                   arrowprops=arrow_style, zorder=6, **text_style)
    
    # 南側
    for i, (county, centroid) in enumerate(south_counties):
        label_x = x_lim[0] + map_width * (0.1 + i * 0.1)
        label_y = y_lim[0] + map_height * 0.05
        
        ax.annotate(county, xy=(centroid.x, centroid.y), xytext=(label_x, label_y),
                   arrowprops=arrow_style, zorder=6, **text_style)
    
    # 添加網格
    grid_step = 0.1
    x_ticks = np.arange(np.floor(x_lim[0] * 10) / 10, np.ceil(x_lim[1] * 10) / 10 + grid_step, grid_step)
    y_ticks = np.arange(np.floor(y_lim[0] * 10) / 10, np.ceil(y_lim[1] * 10) / 10 + grid_step, grid_step)
    
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.grid(True, linestyle='--', alpha=0.3, color='gray')
    
    # 格式化軸標籤
    def format_degree(value, pos):
        return f'{value:.1f}°'
    
    ax.xaxis.set_major_formatter(plt.FuncFormatter(format_degree))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(format_degree))
    
    # 添加色標
    cbar = plt.colorbar(color_mapper, ax=ax, pad=0.01)
    cbar.set_label(metric_name)
    
    plt.title(title, fontsize=16)
    
    plt.xlabel("經度", fontsize=12)
    plt.ylabel("緯度", fontsize=12)
    
    plt.tight_layout()
    
    # 如果提供輸出路徑，則儲存
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"已儲存地圖到 {output_path}")
    
    return fig, ax

# 繪製淹水熱力圖
def plot_flood_heatmap(sf, flood_dict, year, region=None, output_path=None):
    """
    繪製淹水深度熱力圖
    
    參數:
    sf: shapefile Reader物件
    flood_dict: 包含淹水深度資料的字典 {鄉鎮市名稱: 淹水深度}
    year: 年份
    region: 可選區域過濾器('north', 'central', 'south', 'east')
    output_path: 可選儲存圖形的路徑
    
    返回:
    Figure物件
    """
    region_name = ""

    title = f"{year}年鄉鎮市平均淹水深度熱點圖"
        
    fig, ax = plot_township_heatmap(
        sf, 
        flood_dict, 
        "平均淹水深度 (cm)",
        cmap="Blues",
        title=title,
        figsize=(15, 20)
    )
    
    # 如果提供輸出路徑，則儲存
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"已儲存淹水熱力圖到 {output_path}")
    
    return fig

# 繪製雨量熱力圖
def plot_rainfall_heatmap(sf, rainfall_dict, year, region=None, output_path=None):
    """
    繪製雨量熱力圖
    
    參數:
    sf: shapefile Reader物件
    rainfall_dict: 包含雨量資料的字典 {鄉鎮市名稱: 雨量}
    year: 年份
    output_path: 可選儲存圖形的路徑
    
    返回:
    Figure物件
    """
    
    title = f"{year}年鄉鎮市平均雨量熱點圖"
    
    fig, ax = plot_township_heatmap(
        sf, 
        rainfall_dict, 
        "平均日雨量 (mm)",
        cmap="Greens",
        title=title,
        figsize=(15, 20)
    )
    
    # 如果提供輸出路徑，則儲存
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"已儲存雨量熱力圖到 {output_path}")
    
    return fig

# 時間序列分析函數
def analyze_time_series(sensor_df, rainfall_df, township, output_path=None):
    """
    分析特定鄉鎮市的淹水和雨量時間序列
    
    參數:
    sensor_df: 淹水感測器資料DataFrame
    rainfall_df: 雨量資料DataFrame
    township: 要分析的鄉鎮市名稱
    output_path: 可選儲存圖形的路徑
    
    返回:
    Figure物件
    """
    # 篩選指定鄉鎮市的資料
    town_sensor_data = sensor_df[sensor_df['TOWNNAME'] == township]
    town_rainfall_data = rainfall_df[rainfall_df['TownName'] == township]
    
    if town_sensor_data.empty or town_rainfall_data.empty:
        print(f"找不到 {township} 的足夠資料")
        return None
    
    # 確保日期格式
    town_sensor_data['file_date'] = pd.to_datetime(town_sensor_data['file_date'])
    town_rainfall_data['date'] = pd.to_datetime(town_rainfall_data['date'])
    
    # 按日期重新取樣
    daily_depth = town_sensor_data.groupby(town_sensor_data['file_date'].dt.date)['value'].mean()
    daily_rainfall = town_rainfall_data.groupby(town_rainfall_data['date'].dt.date)["Past24hr"].mean()
    
    # 重新索引以確保數據對齊
    date_range = pd.date_range(min(daily_depth.index.min(), daily_rainfall.index.min()),
                              max(daily_depth.index.max(), daily_rainfall.index.max()))
    
    daily_depth = daily_depth.reindex(date_range)
    daily_rainfall = daily_rainfall.reindex(date_range)
    
    # 創建圖形（帶雙Y軸）
    fig, ax1 = plt.subplots(figsize=(15, 8))
    
    # 淹水深度軸（左側）
    color = 'tab:blue'
    ax1.set_xlabel('日期')
    ax1.set_ylabel('淹水深度 (cm)', color=color)
    ax1.plot(date_range, daily_depth, color=color, marker='.', markersize=3, label='淹水深度')
    ax1.tick_params(axis='y', labelcolor=color)
    
    # 雨量軸（右側）
    ax2 = ax1.twinx()
    color = 'tab:green'
    ax2.set_ylabel('日雨量 (mm)', color=color)
    ax2.plot(date_range, daily_rainfall, color=color, marker='.', markersize=3, label='日雨量')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # 添加標題和圖例
    plt.title(f"{township} 淹水深度與雨量時間序列分析", fontsize=16)
    
    # 組合圖例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.tight_layout()
    
    # 如果提供輸出路徑，則儲存
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"已儲存時間序列圖到 {output_path}")
    
    return fig

# 確保 load_shapefile 函數正確返回 shapefile Reader 物件
def load_shapefile(shp_path):
    """載入shapefile並返回shapefile.Reader物件"""
    import shapefile as shp
    try:
        sf = shp.Reader(shp_path)
        # 簡單測試確保它是 shapefile Reader 物件
        test = sf.fields[1:]  # 如果這行出錯，表示不是有效的 Reader 物件
        return sf
    except Exception as e:
        print(f"讀取 shapefile 時出錯: {e}")
        print(f"檢查路徑是否正確: {shp_path}")
        raise

# 定義一個函數來處理特定地區的異常值
def remove_max_outlier(df, town_name):
    # 獲取指定地區的資料索引
    town_index = df[df['TOWNNAME'] == town_name].index
    
    # 如果該地區沒有資料，直接返回
    if len(town_index) == 0:
        print(f"{town_name} 沒有資料")
        return df
    
    # 獲取該地區的淹水深度值
    town_values = df.loc[town_index, 'value']
    
    # 計算四分位數
    Q1 = np.percentile(town_values, 25)
    Q3 = np.percentile(town_values, 75)
    IQR = Q3 - Q1
    
    # 定義異常值閾值
    upper_bound = Q3 + 1.5 * IQR
    
    # 找出超過上界的值的索引
    outlier_index = town_index[town_values > upper_bound]
    
    # 如果找到異常值，則處理
    if len(outlier_index) > 0:
        print(f"移除 {town_name} 的 {len(outlier_index)} 個異常值")
        # 將異常值設為 NaN
        df.loc[outlier_index, 'value'] = np.nan
    else:
        print(f"{town_name} 沒有發現異常值")
    
    return df

# 繪製淹水損失熱力圖
def plot_flood_loss_heatmap(sf, damage_dict, year, region=None, output_path=None):
    """
    繪製淹水損失熱力圖
    
    參數:
    sf: shapefile Reader物件
    damage_dict: 包含淹水損失資料的字典 {鄉鎮市名稱: 淹水損失總額}
    year: 年份
    output_path: 可選儲存圖形的路徑
    
    返回:
    Figure物件
    """
    
    title = f"{year}年鄉鎮市淹水損失總額熱點圖"
        
    fig, ax = plot_township_heatmap(
        sf, 
        damage_dict, 
        "淹水損失總額 (元)",
        cmap="Reds",
        title=title,
        figsize=(15, 20)
    )
    
    # 如果提供輸出路徑，則儲存
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"已儲存淹水損失熱力圖到 {output_path}")
    
    return fig

# 修改 process_yearly_data 函數，移除北中南東部地圖繪製
def process_yearly_data(base_path, year, sensor_df, rainfall_df, damage_df, output_dir):
    """
    處理指定年份的淹水、雨量和損失數據，生成熱力圖
    
    參數:
    base_path: 基本路徑
    year: 年份
    sensor_df: 淹水感測資料DataFrame
    rainfall_df: 雨量資料DataFrame
    damage_df: 淹水損失資料DataFrame
    output_dir: 輸出目錄
    """
    # 先載入 shapefile
    print("讀取地圖檔案...")
    try:
        # 檢查 shapefile 路徑是否存在
        shp_path = os.path.join(base_path, '鄉鎮市區界線(TWD97經緯度)/TOWN_MOI_1120317.shp')
        if not os.path.exists(shp_path):
            # 嘗試找到正確的 shapefile 路徑
            potential_paths = [
                os.path.join(base_path, 'map/TOWN_MOI_1120317.shp'),
                os.path.join(base_path, 'maps/TOWN_MOI_1120317.shp'),
                os.path.join(base_path, 'shp/TOWN_MOI_1120317.shp'),
                '/Users/andy/Box Sync/Andy/Project/Intern/Formosa/dataset/鄉鎮市區界線(TWD97經緯度)/TOWN_MOI_1120317.shp'
            ]
            
            for path in potential_paths:
                if os.path.exists(path):
                    shp_path = path
                    print(f"找到替代的 shapefile 路徑: {shp_path}")
                    break
            
            if not os.path.exists(shp_path):
                print(f"錯誤：無法找到 shapefile 檔案。已嘗試路徑: {shp_path}")
                return
        
        sf = load_shapefile(shp_path)  # 確保這裡返回的是 shapefile Reader 物件
    except Exception as e:
        print(f"載入 shapefile 時發生錯誤: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 處理淹水資料
    station_count_dict, avg_depth_dict = process_sensor_data(sensor_df)
    
    # 處理雨量資料
    avg_rainfall_dict = process_rainfall_data(rainfall_df)
    
    # 處理淹水損失資料
    total_damage_dict = process_flood_loss_data(damage_df)
    
    # 繪製全台灣淹水深度熱力圖
    plot_flood_heatmap(
        sf, 
        avg_depth_dict, 
        year,
        output_path=os.path.join(output_dir, f"{year}_taiwan_avg_depth.png")
    )
    
    # 繪製全台灣雨量熱力圖
    plot_rainfall_heatmap(
        sf, 
        avg_rainfall_dict, 
        year,
        output_path=os.path.join(output_dir, f"{year}_taiwan_avg_rainfall.png")
    )
    
    # 繪製全台灣淹水損失熱力圖
    plot_flood_loss_heatmap(
        sf, 
        total_damage_dict, 
        year,
        output_path=os.path.join(output_dir, f"{year}_taiwan_total_flood_loss.png")
    )

        
# 修改主程式部分
if __name__ == "__main__":
    base_path = os.path.normpath(os.path.join('..', 'dataset'))
    senser_path = os.path.normpath(os.path.join(base_path, 'sensor'))
    rain_station_path = os.path.join(base_path, 'rain_station')
    
    output_dir = os.path.join("..", "output", "maps")
    
    # 創建輸出目錄
    os.makedirs(output_dir, exist_ok=True)
    
    # 指定要處理的年份
    years_to_process = [2020, 2021, 2022, 2023, 2024, 2025]

    for year in years_to_process:
        print(f"\n處理 {year} 年資料...")
        
        # 讀取淹水損失資料
        flood_loss_path = os.path.join("..", "output", f"town_flood_loss_statistics_{year}.csv")
        damage_df = pd.read_csv(flood_loss_path)
        sensor_df = pd.read_csv(os.path.join(senser_path, f"flood_data_{year}_merged_filled.csv"))
        rainfall_df = pd.read_csv(os.path.join(rain_station_path, f"rain_station_data_{year}_merged.csv"))
        
        # 移除異常值
        if year == 2024:
            sensor_df = remove_max_outlier(sensor_df, '大村鄉')
            sensor_df = remove_max_outlier(sensor_df, '員林市')
        
        # 運行主函數
        process_yearly_data(base_path, year, sensor_df, rainfall_df, damage_df, output_dir)