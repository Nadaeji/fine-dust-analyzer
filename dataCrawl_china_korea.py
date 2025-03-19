import ee
import pandas as pd
from datetime import datetime, timedelta
import os
import math

ee.Initialize()

# 중국-한국 범위에서 넓은 간격의 격자 좌표 생성
locations = {}
latitudes = [20.0, 32.5, 45.0]  # 위도 12.5° 간격
longitudes = [100.0, 111.7, 123.4, 135.0]  # 경도 약 11.7° 간격

for lat in latitudes:
    for lon in longitudes:
        location_name = f"Grid_{lat}_{lon}"
        locations[location_name] = [lon, lat]

# 결과적으로 총 12개 좌표 생성
print(f"총 좌표 수: {len(locations)}")  # 확인용: 12

# 날짜 설정
start_date = datetime(2018, 1, 1)
end_date = datetime(2025, 2, 1)

pm_dataset = ee.ImageCollection("ECMWF/CAMS/NRT")
wind_dataset = ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY")

data = []
current_date = start_date
current_month = current_date.strftime("%Y-%m")
save_dir = "pm25_pm10_wind_data_cn_kr_wide"
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
                    "Location": city,
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
