import os
import glob
import zipfile
import re
import pandas as pd
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import shapefile
from shapely.geometry import shape, Point  # 讀取 shp 與建立點物件

# 全域變數：基本欄位（依據原始 CSV 欄位，若原始資料中 PQ_name 需要轉成 PQ_fullname）
BASIC_COLUMNS = ["station_id", "PQ_fullname", "value"]

def extract_date_from_filename(filename):
    """
    從檔案名稱中提取日期，假設檔名格式可能為：
      wra_iow_水利署_淹水感測器_YYYYMMDD.zip
    或者
      wra_iow_水利署_淹水感測器_YYYYMMDD_QC.zip
    如果有 "QC"，則優先使用 "QC" 前面的部分作為日期。
    特別處理 7 位數字的情況，例如 "2024011" 代表 "2024 01 01"。
    """
    base = os.path.splitext(os.path.basename(filename))[0]
    parts = base.split('_')
    
    candidate = None
    if "QC" in parts:
        idx = parts.index("QC")
        if idx > 0:
            candidate = parts[idx-1]
    if candidate is None or not candidate.isdigit() or (len(candidate) not in [7,8]):
        for part in parts:
            if part.isdigit() and len(part) in [7,8]:
                candidate = part
                break

    if candidate:
        if len(candidate) == 7:
            candidate = candidate[:4] + candidate[4:6] + candidate[6:].zfill(2)
        try:
            return datetime.strptime(candidate, "%Y%m%d").date()
        except Exception as e:
            print(f"解析日期 {candidate} 失敗: {e}")
    print(f"無法從檔案名稱提取日期: {filename}")
    return None

def process_zip(zip_file):
    """
    處理單一 zip 檔案：
      - 從 zip 中找出所有 CSV 檔案，若有 _QC 檔案則優先使用。
      - 讀取 CSV，過濾 PQ_unit 為 "cm" 的資料，並只保留指定欄位。
      - 從 zip 檔名中提取日期，將其存入 file_date 欄位。
      - 若資料所屬年份 >= 2024 且 CSV 中有 "Longitude" 與 "Latitude" 欄位，則也保留這兩個欄位。
    回傳該 zip 檔案合併後的 DataFrame，否則回傳 None。
    """
    dfs_local = []
    file_date = extract_date_from_filename(zip_file)
    extra_cols = []
    if file_date is not None and file_date.year >= 2024:
        extra_cols = ["Longitude", "Latitude"]
    
    try:
        with zipfile.ZipFile(zip_file, 'r') as zf:
            csv_files = [f for f in zf.namelist() if f.endswith('.csv')]
            qc_files = [f for f in csv_files if "_QC" in os.path.basename(f)]
            if qc_files:
                csv_files = qc_files
            for csv_file in csv_files:
                try:
                    with zf.open(csv_file) as file_obj:
                        df = pd.read_csv(file_obj, low_memory=False)
                        df = df[df["PQ_unit"] == "cm"]
                        if "PQ_name" in df.columns and "PQ_fullname" not in df.columns:
                            df.rename(columns={"PQ_name": "PQ_fullname"}, inplace=True)
                        df['file_date'] = file_date
                        keep_cols = BASIC_COLUMNS + extra_cols + ['file_date']
                        keep_cols = [col for col in keep_cols if col in df.columns]
                        df = df[keep_cols]
                        dfs_local.append(df)
                except Exception as e:
                    print(f"讀取 {csv_file} 時發生錯誤 ({zip_file}): {e}")
    except Exception as e:
        print(f"處理 {zip_file} 時發生錯誤: {e}")
    
    if dfs_local:
        return pd.concat(dfs_local, ignore_index=True)
    else:
        return None

def merge_flood_data(year, base_path):
    """
    讀取指定年度的淹水資料，資料結構位於：
      base_path/year/「年份+月份」/每日的 zip 檔
    使用平行處理合併所有 zip 檔後，對相同 station_id 與 file_date 的 value 取平均，
    並依據 file_date 排序，回傳處理好的 DataFrame。
    """
    dfs = []
    months = range(1, 4) if year == 2025 else range(1, 13)
    zip_files_all = []
    for month in months:
        month_str = f"{year}{month:02d}"
        month_folder = os.path.join(base_path, str(year), month_str)
        pattern = os.path.join(month_folder, "*.zip")
        zip_files = glob.glob(pattern)
        if not zip_files:
            print(f"找不到符合 {pattern} 的檔案")
        else:
            zip_files_all.extend(zip_files)
    print(f"【{year}】總共有 {len(zip_files_all)} 個 zip 檔案需要處理")
    
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_zip, zip_file): zip_file for zip_file in zip_files_all}
        for future in as_completed(futures):
            zip_file = futures[future]
            try:
                df_zip = future.result()
                if df_zip is not None:
                    dfs.append(df_zip)
            except Exception as exc:
                print(f"{zip_file} 產生例外: {exc}")
    dfs = [df for df in dfs if not df.empty]
    if dfs:
        result_df = pd.concat(dfs, ignore_index=True)
        agg_dict = {"value": "mean", "PQ_fullname": "first"}
        if "Longitude" in result_df.columns:
            agg_dict["Longitude"] = "first"
        if "Latitude" in result_df.columns:
            agg_dict["Latitude"] = "first"
        result_df = result_df.groupby(["station_id", "file_date"], as_index=False).agg(agg_dict)
        result_df = result_df.sort_values(by="file_date").reset_index(drop=True)
    else:
        print("沒有找到符合條件的檔案！")
        result_df = pd.DataFrame()
    return result_df

def load_station_data(base_path):
    """
    讀取 station 資料，僅保留 station_id, Longitude, Latitude
    假設 station CSV 位於 base_path 下。
    """
    station_file = os.path.join(base_path, "station_水利署_淹水感測器.csv")
    station_df = pd.read_csv(station_file, encoding="utf-8", low_memory=False)
    station_df = station_df[["station_id", "Longitude", "Latitude"]]
    return station_df

def load_township_polygons():
    """
    讀取鄉鎮市 shp 檔案，建立一個列表，每個元素為 (township_name, county_name, polygon)。
    假設鄉鎮市名稱欄位為 "TOWNNAME"，縣市名稱欄位為 "COUNTYNAME"，請根據實際情況調整。
    此處使用相對路徑：假設 shp 位於 ../dataset/鄉鎮市區界線(TWD97經緯度)/TOWN_MOI_1120317.shp
    """
    shp_path = os.path.join("..", "dataset", "鄉鎮市區界線(TWD97經緯度)", "TOWN_MOI_1120317.shp")
    sf = shapefile.Reader(shp_path)
    fields = [field[0] for field in sf.fields[1:]]  # 跳過刪除旗標
    records = sf.records()
    shapes = sf.shapes()
    polygons = []
    for rec, shp_obj in zip(records, shapes):
        props = dict(zip(fields, rec))
        town_name = props.get("TOWNNAME")
        county_name = props.get("COUNTYNAME")  # 這裡加入縣市名稱
        poly = shape(shp_obj.__geo_interface__)
        polygons.append((town_name, county_name, poly))
    return polygons

def add_township_info(df, polygons):
    """
    對 df 中的每一筆資料（必須包含 Longitude 與 Latitude 欄位），
    判斷該點所屬的鄉鎮市與縣市，並將結果存入新欄位 "TOWNNAME" 與 "COUNTYNAME"。
    """
    def get_location(row):
        pt = Point(row["Longitude"], row["Latitude"])
        for town, county, poly in polygons:
            if poly.contains(pt):
                return town, county
        return None, None
    
    # 創建新的欄位，預設為 None
    df["TOWNNAME"] = None
    df["COUNTYNAME"] = None
    
    # 只對有效的經緯度資料進行處理
    valid_indices = df.dropna(subset=["Longitude", "Latitude"]).index
    
    # 如果沒有有效的經緯度資料，直接返回原 DataFrame
    if len(valid_indices) == 0:
        return df
    
    # 對每一筆有效資料，計算鄉鎮市與縣市，並直接更新原 DataFrame
    for idx in valid_indices:
        town, county = get_location(df.loc[idx])
        df.loc[idx, "TOWNNAME"] = town
        df.loc[idx, "COUNTYNAME"] = county
    
    return df

def process_years(base_path, year):
    """
    對指定年度範圍內的淹水資料進行合併：
      - 2019～2023 年：使用 station.csv 合併經緯度資訊。
      - 2024 與 2025 年：假設資料本身已有經緯度，直接使用。
    合併後，利用 shp 檔將每筆資料的 Longitude 與 Latitude 對應到鄉鎮市與縣市，
    並新增欄位 "TOWNNAME" 與 "COUNTYNAME"。
    最後將結果儲存為 CSV，檔名格式為 flood_data_{year}_merged.csv。
    """
    # 讀取 station 資料（僅供 2019~2023 使用）
    station_df = load_station_data(base_path)
    # 讀取鄉鎮市多邊形
    polygons = load_township_polygons()
    
    
    print(f"處理 {year} 年的 sensor 資料 ...")
    flood_df = merge_flood_data(year, base_path)
    if not flood_df.empty:
        if year <= 2023:
            merged_df = pd.merge(flood_df, station_df, on="station_id", how="left")
        else:
            merged_df = flood_df.copy()
        # 若資料中已有 Longitude, Latitude，則加入鄉鎮市與縣市資訊
        if "Longitude" in merged_df.columns and "Latitude" in merged_df.columns:
            merged_df = add_township_info(merged_df, polygons)
        out_file = os.path.join(base_path, f"flood_data_{year}_merged.csv")
        merged_df.to_csv(out_file, index=False, encoding="utf-8")
        print(f"{year} 合併完成，儲存於: {out_file}")
    else:
        print(f"{year} 年找不到符合條件的資料！")
        
    return merged_df

def txt_to_dict(file_path):
    """
    從 txt 檔案讀取經緯度資訊，轉換成字典格式。
    txt 格式為 key: [value1, value2]。
    """
    data_dict = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:  # 過濾空白行
                continue
            # 使用冒號分割 key 與 value
            key, value_str = line.split(":", 1)
            # 去除左右空白並移除中括號
            
            value_str = value_str.strip().strip("[]")
            values = []
            for num in value_str.split(","):
                num = num.strip()
                # 檢查是否為 "None"
                if num == "None":
                    values.append(None)
                else:
                    try:
                        values.append(float(num))
                    except ValueError:
                        print(f"警告：無法將 {num} 轉換成 float")
            data_dict[key] = values
    return data_dict

def fill_missing_lon_lat(year, missing_locations_all_years, base_path):
    """
    讀取合併後的資料，對缺少經緯度的資料，使用 missing_locations_all_years 中的資訊進行填補。
    然後使用經緯度資訊，對資料加入鄉鎮市與縣市名稱。
    最後將負值的 value 設為 0。
    """
    file_name = f"flood_data_{year}_merged.csv"
    df = pd.read_csv(os.path.join(base_path, file_name))
    
    # 建立經度與緯度的 mapping 字典
    longitude_map = {k: v[0] for k, v in missing_locations_all_years.items()}
    latitude_map = {k: v[1] for k, v in missing_locations_all_years.items()}
    
    # 利用 map 方法，若 PQ_fullname 存在於 mapping 中，則用 mapping 的值，否則保留原值
    df['Longitude'] = df['PQ_fullname'].map(longitude_map).fillna(df['Longitude'])
    df['Latitude'] = df['PQ_fullname'].map(latitude_map).fillna(df['Latitude'])
    
    # 使用更新後的經緯度加入鄉鎮市與縣市資訊
    polygons = load_township_polygons()
    df = add_township_info(df, polygons)
    
    # 過濾掉沒有鄉鎮市資訊的資料
    df = df.dropna(subset=["TOWNNAME"])
    
    # 將負值的 value 設為 0
    df.loc[df['value'] < 0, 'value'] = 0
    
    return df

if __name__ == '__main__':
    base_path = os.path.join("..", "dataset", "sensor")
    
    # 處理原始資料並生成合併資料
    for year in range(2019, 2026):
        merged_df = process_years(base_path, year)
    
    # 讀取缺失位置資訊
    missing_locations_all_years_path = os.path.join(base_path, "missing_locations_all_years.txt")
    missing_locations_all_years = txt_to_dict(missing_locations_all_years_path)
    
    # 填補缺失位置並儲存結果
    for year in range(2019, 2026):
        merged_df = fill_missing_lon_lat(year, missing_locations_all_years, base_path)
        output_file = os.path.join(base_path, f"flood_data_{year}_merged_filled.csv")
        merged_df.to_csv(output_file, index=False, encoding="utf-8")
        print(f"{year} 年補充缺失經緯度完成，儲存於: {output_file}")