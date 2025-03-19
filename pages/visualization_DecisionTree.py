# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import folium
from streamlit_folium import st_folium

# 1. 저장된 모델 및 데이터 로드
scaler_pm25 = joblib.load('models/dt/scaler_pm25.pkl')  # PM2.5용 StandardScaler
scaler_pm10 = joblib.load('models/dt/scaler_pm10.pkl')  # PM10용 StandardScaler
season_wind = joblib.load('models/dt/season_wind.pkl')
evaluation_scores_pm25 = joblib.load('models/dt/evaluation_scores_pm25.pkl')
evaluation_scores_pm10 = joblib.load('models/dt/evaluation_scores_pm10.pkl')

# DecisionTree 모델 로드 (PM2.5와 PM10)
seasons = ['봄', '여름', '가을', '겨울']
nearby_cities = ['Seoul', 'Tokyo', 'Delhi', 'Bangkok', 'Busan', 'Daegu', 'Osaka', 
                 'Sapporo', 'Fukuoka', 'Kyoto', 'Almaty', 'Bishkek', 'Dushanbe', 
                 'Kathmandu', 'Yangon', 'Guwahati', 'Ulaanbaatar', 'Irkutsk']
dt_models_pm25 = {}
dt_models_pm10 = {}

for season in seasons:
    dt_models_pm25[season] = {}
    dt_models_pm10[season] = {}
    for city in nearby_cities:
        try:
            dt_models_pm25[season][city] = joblib.load(f'models/dt/dt_pm25_{season}_{city}.pkl')
            dt_models_pm10[season][city] = joblib.load(f'models/dt/dt_pm10_{season}_{city}.pkl')
        except FileNotFoundError:
            continue

# 도시 이름 매핑 (영어 → 한국어)
city_names_kr = {
    "Seoul": "서울", "Tokyo": "도쿄", "BeijingSeveral": "베이징", "Delhi": "델리", 
    "Bangkok": "방콕", "Busan": "부산", "Daegu": "대구", "Osaka": "오사카", 
    "Sapporo": "삿포로", "Fukuoka": "후쿠오카", "Kyoto": "교토", "Shanghai": "상하이", 
    "Guangzhou": "광저우", "Chongqing": "충칭", "Wuhan": "우한", "Nanjing": "난징", 
    "Hangzhou": "항저우", "Chengdu": "청두", "Almaty": "알마티", "Bishkek": "비슈케크", 
    "Dushanbe": "두샨베", "Kathmandu": "카트만두", "Yangon": "양곤", "Guwahati": "구와하티", 
    "Ulaanbaatar": "울란바토르", "Irkutsk": "이르쿠츠크"
}

# 도시 좌표 (위도, 경도)
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
        dt_models = dt_models_pm25
        scaler = scaler_pm25
    else:
        dt_models = dt_models_pm10
        scaler = scaler_pm10
    
    if season not in dt_models:
        return None
    
    wind_x = season_wind['Wind_X'][season]
    wind_y = season_wind['Wind_Y'][season]
    
    # 예측 시 DataFrame으로 입력 데이터 생성 (피처 이름 유지)
    input_data = pd.DataFrame([[china_value, wind_x, wind_y]], 
                              columns=[f'{pollutant} (µg/m³)', 'Wind_X', 'Wind_Y'])
    input_scaled = scaler.transform(input_data)
    
    predictions = {}
    for city in nearby_cities:
        if city in dt_models[season]:
            prediction = dt_models[season][city].predict(input_scaled)[0]
            predictions[city] = prediction
    return predictions

# 등급 및 색상 계산 함수 (train.py의 cluster_labels에 맞춤)
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
st.title("중국 미세먼지가 주변국에 미치는 영향")

# 탭 생성
tab1, tab2 = st.tabs(["PM2.5 예측", "PM10 예측"])

# PM2.5 탭
with tab1:
    st.header("PM2.5 예측")

    # 계절 선택
    season = st.selectbox("계절 선택 (PM2.5)", seasons, key="pm25_season")

    # 입력값 수집
    china_pm25 = st.slider("중국 PM2.5 (µg/m³)", 0.0, 200.0, 50.0, key="pm25_input")

    # 예측 및 지도 표시
    predictions = predict_all_cities(season, china_pm25, pollutant='PM2.5')

    if predictions:
        # 예측 결과 섹션 (expander로 묶음)
        with st.expander(f"{season}의 주변국 PM2.5 예측"):
            pred_table_data = {
                "도시": [city_names_kr[city] for city in predictions.keys()],
                "PM2.5 (µg/m³)": [f"{pm25:.2f}" for pm25 in predictions.values()],
                "상태": [get_grade(pm25, 'PM2.5')[0] for pm25 in predictions.values()]
            }
            pred_df = pd.DataFrame(pred_table_data)
            st.dataframe(pred_df, use_container_width=True)

        # 모델 평가 점수 섹션 (expander로 묶음)
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

        # 지도 생성
        st.subheader("지도 (PM2.5)")
        m = folium.Map(location=[35, 120], zoom_start=4)
        
        # 예측값을 지도에 표시
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
        
        # 지도 표시
        st_folium(m, width=700, height=500, key=f"map_pm25_{season}")
    else:
        st.error(f"{season}에 대한 모델이 없습니다 (PM2.5).")

# PM10 탭
with tab2:
    st.header("PM10 예측")

    # 계절 선택
    season = st.selectbox("계절 선택 (PM10)", seasons, key="pm10_season")

    # 입력값 수집
    china_pm10 = st.slider("중국 PM10 (µg/m³)", 0.0, 300.0, 75.0, key="pm10_input")

    # 예측 및 지도 표시
    predictions = predict_all_cities(season, china_pm10, pollutant='PM10')

    if predictions:
        # 예측 결과 섹션 (expander로 묶음)
        with st.expander(f"{season}의 주변국 PM10 예측"):
            pred_table_data = {
                "도시": [city_names_kr[city] for city in predictions.keys()],
                "PM10 (µg/m³)": [f"{pm10:.2f}" for pm10 in predictions.values()],
                "상태": [get_grade(pm10, 'PM10')[0] for pm10 in predictions.values()]
            }
            pred_df = pd.DataFrame(pred_table_data)
            st.dataframe(pred_df, use_container_width=True)

        # 모델 평가 점수 섹션 (expander로 묶음)
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

        # 지도 생성
        st.subheader("지도 (PM10)")
        m = folium.Map(location=[35, 120], zoom_start=4)
        
        # 예측값을 지도에 표시
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
        
        # 지도 표시
        st_folium(m, width=700, height=500, key=f"map_pm10_{season}")
    else:
        st.error(f"{season}에 대한 모델이 없습니다 (PM10).")