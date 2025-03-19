import ee
import pandas as pd
from datetime import datetime, timedelta
import os
import math

ee.Initialize()
# 도시 추가
locations = {
    "Seoul": [126.9780, 37.5665],
    "Tokyo": [139.6917, 35.6895],
    "Beijing": [116.4074, 39.9042],
    "Delhi": [77.1025, 28.7041],
    "Bangkok": [100.5018, 13.7563],
    "Busan": [129.0756, 35.1796],
    "Daegu": [128.5911, 35.8704],
    "Osaka": [135.5023, 34.6937],
    "Nagoya": [136.9066, 35.1802],
    "Sapporo": [141.3545, 43.0618],
    "Fukuoka": [130.4017, 33.5904],
    "Kyoto": [135.7681, 35.0116],
    "Shanghai": [121.4737, 31.2304],
    "Guangzhou": [113.2644, 23.1291],
    "Chongqing": [106.5516, 29.5630],
    "Tianjin": [117.1902, 39.1256],
    "Wuhan": [114.3055, 30.5928],
    "Nanjing": [118.7969, 32.0603],
    "Hangzhou": [120.1551, 30.2741],
    "Chengdu": [104.0668, 30.5728],
    "Xi'an": [108.9355, 34.3416],
    "Shenyang": [123.4291, 41.7968],
    "Almaty": [76.8512, 43.2220],
    "Bishkek": [74.5698, 42.8746],
    "Dushanbe": [68.7864, 38.5481],
    "Kathmandu": [85.3240, 27.7172],
    "Yangon": [96.1951, 16.8409],
    "Guwahati": [91.7362, 26.1445],
    "Ulaanbaatar": [106.9057, 47.8864],
    "Irkutsk": [104.2964, 52.2869],
}

start_date = datetime(2023, 4, 1) 
end_date = datetime(2025, 3, 1)

pm_dataset = ee.ImageCollection("ECMWF/CAMS/NRT")
wind_dataset = ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY")

data = []
current_date = start_date
current_month = current_date.strftime("%Y-%m")
save_dir = "pm25_pm10_wind_data"
os.makedirs(save_dir, exist_ok=True)

while current_date <= end_date:
    date_str = current_date.strftime("%Y-%m-%d")
    print(f"Processing: {date_str}")
    next_date_str = (current_date + timedelta(days=1)).strftime("%Y-%m-%d")

    try:
        daily_pm = pm_dataset.filterDate(date_str, next_date_str).mean()
        daily_wind = wind_dataset.filterDate(date_str, next_date_str).mean()

        for city, coord in locations.items():
            point = ee.Geometry.Point(coord)
            try:
                # PM2.5, PM10 데이터
                pm_values = daily_pm.reduceRegion(reducer=ee.Reducer.mean(), geometry=point, scale=10000).getInfo()
                pm25_value = pm_values.get("particulate_matter_d_less_than_25_um_surface", 0) * 1e9 if pm_values else 0
                pm10_value = pm_values.get("particulate_matter_d_less_than_10_um_surface", 0) * 1e9 if pm_values else 0

                # 풍속, 풍향 데이터
                wind_values = daily_wind.reduceRegion(reducer=ee.Reducer.mean(), geometry=point, scale=10000).getInfo()
                u = wind_values.get("u_component_of_wind_10m", 0) if wind_values and "u_component_of_wind_10m" in wind_values else 0
                v = wind_values.get("v_component_of_wind_10m", 0) if wind_values and "v_component_of_wind_10m" in wind_values else 0
                
                if u is None or v is None:
                    print(f"⚠️ {city} 풍속 데이터 누락 (u={u}, v={v})")
                    u = 0
                    v = 0
                
                wind_speed = math.sqrt(u**2 + v**2)
                wind_direction = (180 + (180 / math.pi) * math.atan2(v, u)) % 360

                data.append({
                    "Date": date_str,
                    "City": city,
                    "Longitude": coord[0],
                    "Latitude": coord[1],
                    "PM2.5 (µg/m³)": pm25_value,
                    "PM10 (µg/m³)": pm10_value,
                    "Wind Speed (m/s)": wind_speed,
                    "Wind Direction (degrees)": wind_direction
                })

            except Exception as city_error:
                print(f"{city} 데이터 수집 실패 ({date_str}): {city_error}")
                continue

    except Exception as daily_error:
        print(f"{date_str} 데이터 수집 실패: {daily_error}")
        current_date += timedelta(days=1)
        continue

    current_date += timedelta(days=1)

    if current_date.strftime("%Y-%m") != current_month or current_date > end_date:
        if data:
            csv_filename = os.path.join(save_dir, f"pm25_pm10_wind_{current_month}.csv")
            df = pd.DataFrame(data)
            df.to_csv(csv_filename, index=False)
            print(f"{csv_filename} 저장 완료!")
        data = []
        current_month = current_date.strftime("%Y-%m")
