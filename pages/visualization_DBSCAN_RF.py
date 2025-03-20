import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from utils.model_cache import ModelCache
from sklearn.metrics import pairwise_distances_argmin_min

# 캐싱된 모델 가져오기
try:
    scaler_pm25 = ModelCache.get_model('rf_db/scaler_pm25')
    scaler_pm10 = ModelCache.get_model('rf_db/scaler_pm10')
    dbscan_pm25 = ModelCache.get_model('rf_db/dbscan_pm25')  # DBSCAN (참고용)
    dbscan_pm10 = ModelCache.get_model('rf_db/dbscan_pm10')  # DBSCAN (참고용)
    season_wind = ModelCache.get_model('rf_db/season_wind')
    evaluation_scores_pm25 = ModelCache.get_model('rf_db/evaluation_scores_pm25')
    evaluation_scores_pm10 = ModelCache.get_model('rf_db/evaluation_scores_pm10')
    cluster_labels_pm25 = ModelCache.get_model('rf_db/cluster_labels_pm25')
    cluster_labels_pm10 = ModelCache.get_model('rf_db/cluster_labels_pm10')
except KeyError as e:
    st.error(f"필요한 모델이 캐시에 없습니다: {e}")
    st.write("현재 캐시된 모델 목록:", ModelCache.list_cached_models())
    st.stop()

# Random Forest 모델 로드
seasons = ['봄', '여름', '가을', '겨울']
nearby_cities = ['Seoul', 'Tokyo', 'Delhi', 'Bangkok', 'Busan', 'Daegu', 'Osaka', 
                 'Sapporo', 'Fukuoka', 'Kyoto', 'Almaty', 'Bishkek', 'Dushanbe', 
                 'Kathmandu', 'Yangon', 'Guwahati', 'Ulaanbaatar', 'Irkutsk']
rf_models_pm25 = {season: {} for season in seasons}
rf_models_pm10 = {season: {} for season in seasons}

for season in seasons:
    for city in nearby_cities:
        try:
            rf_models_pm25[season][city] = ModelCache.get_model(f'rf_db/rf_pm25_{season}_{city}')
            rf_models_pm10[season][city] = ModelCache.get_model(f'rf_db/rf_pm10_{season}_{city}')
        except KeyError:
            continue

# 도시 이름 매핑 및 좌표
city_names_kr = {
    "Seoul": "서울", "Tokyo": "도쿄", "Beijing": "베이징", "Delhi": "델리", 
    "Bangkok": "방콕", "Busan": "부산", "Daegu": "대구", "Osaka": "오사카", 
    "Sapporo": "삿포로", "Fukuoka": "후쿠오카", "Kyoto": "교토", "Shanghai": "상하이", 
    "Guangzhou": "광저우", "Chongqing": "충칭", "Wuhan": "우한", "Nanjing": "난징", 
    "Hangzhou": "항저우", "Chengdu": "청두", "Almaty": "알마티", "Bishkek": "비슈케크", 
    "Dushanbe": "두샨베", "Kathmandu": "카트만두", "Yangon": "양곤", "Guwahati": "구와하티", 
    "Ulaanbaatar": "울란바토르", "Irkutsk": "이르쿠츠크"
}

city_coords = {
    "Seoul": [37.5665, 126.9780], "Tokyo": [35.6895, 139.6917], "Beijing": [39.9042, 116.4074],
    "Delhi": [28.7041, 77.1025], "Bangkok": [13.7563, 100.5018], "Busan": [35.1796, 129.0756],
    "Daegu": [35.8704, 128.5911], "Osaka": [34.6937, 135.5023], "Sapporo": [43.0618, 141.3545],
    "Fukuoka": [33.5904, 130.4017], "Kyoto": [35.0116, 135.7681], "Shanghai": [31.2304, 121.4737],
    "Guangzhou": [23.1291, 113.2644], "Chongqing": [29.5630, 106.5516], "Wuhan": [30.5928, 114.3055],
    "Nanjing": [32.0603, 118.7969], "Hangzhou": [30.2741, 120.1551], "Chengdu": [30.5728, 104.0668],
    "Almaty": [43.2220, 76.8512], "Bishkek": [42.8746, 74.5698], "Dushanbe": [38.5481, 68.7864],
    "Kathmandu": [27.7172, 85.3240], "Yangon": [16.8409, 96.1951], "Guwahati": [26.1445, 91.7362],
    "Ulaanbaatar": [47.8864, 106.9057], "Irkutsk": [52.2869, 104.2964],
}

# 예측 함수
def predict_all_cities(season, china_value, pollutant='PM2.5'):
    if pollutant == 'PM2.5':
        rf_models = rf_models_pm25
        scaler = scaler_pm25
        dbscan = dbscan_pm25
    else:
        rf_models = rf_models_pm10
        scaler = scaler_pm10
        dbscan = dbscan_pm10
    
    if season not in rf_models:
        return None, None
    
    wind_x = season_wind['Wind_X'][season]
    wind_y = season_wind['Wind_Y'][season]
    
    # 입력 데이터 준비 (DBSCAN 군집 레이블 추정 필요)
    input_data = pd.DataFrame([[china_value, wind_x, wind_y]], 
                              columns=[f'{pollutant} (µg/m³)', 'Wind_X', 'Wind_Y'])
    input_scaled = scaler.transform(input_data)
    
    # DBSCAN 군집 레이블 추정 (임시로 -1로 설정, 실제로는 학습 데이터 필요)
    # 주의: DBSCAN은 predict 없음, 여기서는 RF만 사용
    cluster = -1  # 노이즈로 가정 (DBSCAN 학습 데이터 없음)
    
    # RF 입력에 군집 피처 추가 (학습 시와 동일한 피처 구조)
    input_with_cluster = np.hstack((input_scaled, [[cluster]]))
    
    predictions = {}
    for city in nearby_cities:
        if city in rf_models[season]:
            predictions[city] = rf_models[season][city].predict(input_with_cluster)[0]
    
    return predictions, cluster

# 등급 및 색상 계산 함수
def get_grade(value, pollutant='PM2.5'):
    if pollutant == 'PM2.5':
        if value <= 10:
            return "좋음", "green"
        elif value <= 25:
            return "보통", "blue"
        elif value <= 50:
            return "나쁨", "orange"
        else:
            return "매우 나쁨", "red"
    elif pollutant == 'PM10':
        if value <= 30:
            return "좋음", "green"
        elif value <= 80:
            return "보통", "blue"
        elif value <= 150:
            return "나쁨", "orange"
        else:
            return "매우 나쁨", "red"

# Streamlit 인터페이스
st.title("중국 미세먼지가 주변국에 미치는 영향 (DBSCAN + RF)")

tab1, tab2 = st.tabs(["PM2.5 예측", "PM10 예측"])

with tab1:
    st.header("PM2.5 예측")
    season = st.selectbox("계절 선택 (PM2.5)", seasons, key="pm25_season")
    china_pm25 = st.slider("중국 PM2.5 (µg/m³)", 0.0, 200.0, 50.0, key="pm25_input")

    predictions, cluster = predict_all_cities(season, china_pm25, pollutant='PM2.5')

    if predictions:
        with st.expander(f"{season}의 주변국 PM2.5 예측"):
            pred_table_data = {
                "도시": [city_names_kr[city] for city in predictions.keys()],
                "PM2.5 (µg/m³)": [f"{pm25:.2f}" for pm25 in predictions.values()],
                "상태": [get_grade(pm25, 'PM2.5')[0] for pm25 in predictions.values()]
            }
            pred_df = pd.DataFrame(pred_table_data)
            st.dataframe(pred_df, use_container_width=True)
            st.write(f"입력값의 클러스터: {cluster} ({cluster_labels_pm25.get(cluster, 'N/A')})")

        with st.expander(f"{season}의 모델 평가 점수 (PM2.5)"):
            eval_table_data = {
                "도시": [city_names_kr[city] for city in predictions.keys()],
                "MSE": [evaluation_scores_pm25[season][city]['MSE'] if city in evaluation_scores_pm25[season] else None for city in predictions.keys()],
                "RMSE": [evaluation_scores_pm25[season][city]['RMSE'] if city in evaluation_scores_pm25[season] else None for city in predictions.keys()],
                "MAE": [evaluation_scores_pm25[season][city]['MAE'] if city in evaluation_scores_pm25[season] else None for city in predictions.keys()],
                "R² 스코어": [evaluation_scores_pm25[season][city]['R²'] if city in evaluation_scores_pm25[season] else None for city in predictions.keys()]
            }
            eval_df = pd.DataFrame(eval_table_data)
            st.dataframe(eval_df, use_container_width=True)

        st.subheader("지도 (PM2.5)")
        m = folium.Map(location=[35, 120], zoom_start=4)
        for city, pm25 in predictions.items():
            lat, lon = city_coords[city]
            grade, color = get_grade(pm25, 'PM2.5')
            folium.CircleMarker(
                location=[lat, lon],
                radius=pm25 / 10,
                popup=f"{city_names_kr[city]}: {pm25:.2f} µg/m³ ({grade})",
                color=color,
                fill=True,
                fill_opacity=0.7
            ).add_to(m)
        st_folium(m, width=700, height=500, key=f"map_pm25_{season}")
    else:
        st.error(f"{season}에 대한 모델이 없습니다 (PM2.5).")

with tab2:
    st.header("PM10 예측")
    season = st.selectbox("계절 선택 (PM10)", seasons, key="pm10_season")
    china_pm10 = st.slider("중국 PM10 (µg/m³)", 0.0, 300.0, 75.0, key="pm10_input")

    predictions, cluster = predict_all_cities(season, china_pm10, pollutant='PM10')

    if predictions:
        with st.expander(f"{season}의 주변국 PM10 예측"):
            pred_table_data = {
                "도시": [city_names_kr[city] for city in predictions.keys()],
                "PM10 (µg/m³)": [f"{pm10:.2f}" for pm10 in predictions.values()],
                "상태": [get_grade(pm10, 'PM10')[0] for pm10 in predictions.values()]
            }
            pred_df = pd.DataFrame(pred_table_data)
            st.dataframe(pred_df, use_container_width=True)
            st.write(f"입력값의 클러스터: {cluster} ({cluster_labels_pm10.get(cluster, 'N/A')})")

        with st.expander(f"{season}의 모델 평가 점수 (PM10)"):
            eval_table_data = {
                "도시": [city_names_kr[city] for city in predictions.keys()],
                "MSE": [evaluation_scores_pm10[season][city]['MSE'] if city in evaluation_scores_pm10[season] else None for city in predictions.keys()],
                "RMSE": [evaluation_scores_pm10[season][city]['RMSE'] if city in evaluation_scores_pm10[season] else None for city in predictions.keys()],
                "MAE": [evaluation_scores_pm10[season][city]['MAE'] if city in evaluation_scores_pm10[season] else None for city in predictions.keys()],
                "R² 스코어": [evaluation_scores_pm10[season][city]['R²'] if city in evaluation_scores_pm10[season] else None for city in predictions.keys()]
            }
            eval_df = pd.DataFrame(eval_table_data)
            st.dataframe(eval_df, use_container_width=True)

        st.subheader("지도 (PM10)")
        m = folium.Map(location=[35, 120], zoom_start=4)
        for city, pm10 in predictions.items():
            lat, lon = city_coords[city]
            grade, color = get_grade(pm10, 'PM10')
            folium.CircleMarker(
                location=[lat, lon],
                radius=pm10 / 10,
                popup=f"{city_names_kr[city]}: {pm10:.2f} µg/m³ ({grade})",
                color=color,
                fill=True,
                fill_opacity=0.7
            ).add_to(m)
        st_folium(m, width=700, height=500, key=f"map_pm10_{season}")
    else:
        st.error(f"{season}에 대한 모델이 없습니다 (PM10).")