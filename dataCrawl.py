import ee
import pandas as pd
from datetime import datetime, timedelta
import os

ee.Initialize()

locations = {
    "Seoul": [126.9780, 37.5665],
    "Tokyo": [139.6917, 35.6895],
    "Beijing": [116.4074, 39.9042],
    "Delhi": [77.1025, 28.7041],
    "Bangkok": [100.5018, 13.7563]
}


start_date = datetime(2015, 1, 1)
end_date = datetime(2025, 3, 10)

pm_dataset = ee.ImageCollection("ECMWF/CAMS/NRT")

data = []
current_date = start_date
current_month = current_date.strftime("%Y-%m") 


save_dir = "pm25_pm10_data"
os.makedirs(save_dir, exist_ok=True)

while current_date <= end_date:
    date_str = current_date.strftime("%Y-%m-%d")
    print(f"Processing: {date_str}")
    next_date_str = (current_date + timedelta(days=1)).strftime("%Y-%m-%d")


    month = current_date.strftime("%Y-%m")

    try:
    
        daily_pm = pm_dataset.filterDate(date_str, next_date_str).mean()

        for city, coord in locations.items():
            point = ee.Geometry.Point(coord)
            try:
                pm_values = daily_pm.reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=point,
                    scale=10000
                ).getInfo()

            
                pm25_value = pm_values.get("particulate_matter_d_less_than_25_um_surface", 0) * 1e9 if pm_values else 0
                pm10_value = pm_values.get("particulate_matter_d_less_than_10_um_surface", 0) * 1e9 if pm_values else 0

                
                data.append({
                    "Date": date_str,
                    "City": city,
                    "Longitude": coord[0],
                    "Latitude": coord[1],
                    "PM2.5 (µg/m³)": pm25_value,
                    "PM10 (µg/m³)": pm10_value
                })

            except Exception as city_error:
                print(f"{city} 데이터 수집 실패 ({date_str}): {city_error}")
                continue  # 오류 발생 시 해당 도시 데이터 건너뛰기

    except Exception as daily_error:
        print(f"{date_str} 데이터 수집 실패: {daily_error}")
        current_date += timedelta(days=1)
        continue  # 오류 발생 시 해당 날짜 건너뛰기

    # 하루 증가
    current_date += timedelta(days=1)

    # 다음 달로 변경되면 데이터 저장 후 리스트 초기화
    if current_date.strftime("%Y-%m") != current_month or current_date > end_date:
        # CSV 파일 저장
        if data:
            csv_filename = os.path.join(save_dir, f"pm25_pm10_{current_month}.csv")
            df = pd.DataFrame(data)
            df.to_csv(csv_filename, index=False)
            print(f"{csv_filename} 저장 완료!")

        # 리스트 초기화
        data = []
        current_month = current_date.strftime("%Y-%m")