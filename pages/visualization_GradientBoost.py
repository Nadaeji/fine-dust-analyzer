import pandas as pd
import streamlit as st
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import folium
from streamlit_folium import st_folium
import numpy as np

# 데이터 로드 및 전처리
@st.cache_data
def load_data():
    data = pd.read_csv("./data/pm25_pm10_merged_wind.csv")  # 파일 경로 수정 필요
    data['Date'] = pd.to_datetime(data['Date'])
    return data

# PM2.5 등급 변환 함수
def pm25_to_grade(pm25):
    if pm25 <= 15:
        return "좋음"
    elif pm25 <= 35:
        return "보통"
    else:
        return "나쁨"

# 모델 학습 (도시별 분류 모델)
def train_model(data):
    pivot_data = data.pivot(index='Date', columns='City', values='PM2.5 (µg/m³)').reset_index().fillna(0)
    X = pivot_data[['Beijing']]  # 입력 변수
    target_cities = [col for col in pivot_data.columns if col not in ['Date', 'Beijing']]
    
    # 타겟 데이터 등급으로 변환
    y_grades = {}
    le = LabelEncoder()
    for city in target_cities:
        grades = [pm25_to_grade(pm25) for pm25 in pivot_data[city]]
        y_grades[city] = le.fit_transform(grades)
    
    # 클래스 분포 확인 및 모델 학습
    models = {}
    evaluation_scores = {}
    for city in target_cities:
        unique_classes = np.unique(y_grades[city])
        if len(unique_classes) < 2:
            st.warning(f"{city}의 데이터에 클래스가 1개뿐입니다. 모델을 학습시키지 않습니다.")
            continue
        
        # 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_grades[city], test_size=0.2, random_state=42)
        
        # 모델 학습
        model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
        model.fit(X_train, y_train)
        models[city] = model
        # 평가 점수 계산
        y_pred = model.predict(X_test)
        r2 = model.score(X_test, y_test)
        mae = (abs(y_pred - y_test)).mean()
        mse = ((y_pred - y_test) ** 2).mean()
        rmse = mse ** 0.5
        evaluation_scores[city] = {'R²': r2, 'MAE': mae, 'MSE': mse, 'RMSE': rmse}
    
    return models, evaluation_scores, target_cities, le

# 예측 함수
def predict_pm25_grade(models, beijing_pm25, label_encoder):
    input_value = [[beijing_pm25]]
    predictions = {}
    for city, model in models.items():
        pred_encoded = model.predict(input_value)[0]
        predictions[city] = label_encoder.inverse_transform([pred_encoded])[0]
    predictions['Beijing'] = pm25_to_grade(beijing_pm25)
    return predictions

# 등급에 따른 색상 계산 함수
def get_color(grade):
    if grade == "좋음":
        return "green"
    elif grade == "보통":
        return "blue"
    else:
        return "orange"

# 도시 좌표 딕셔너리 (기존 코드에서 재사용)
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

# 도시 이름 매핑 (영어 → 한국어, 기존 코드에서 재사용)
city_names_kr = {
    "Seoul": "서울", "Tokyo": "도쿄", "Beijing": "베이징", "Delhi": "델리", 
    "Bangkok": "방콕", "Busan": "부산", "Daegu": "대구", "Osaka": "오사카", 
    "Sapporo": "삿포로", "Fukuoka": "후쿠오카", "Kyoto": "교토", "Shanghai": "상하이", 
    "Guangzhou": "광저우", "Chongqing": "충칭", "Wuhan": "우한", "Nanjing": "난징", 
    "Hangzhou": "항저우", "Chengdu": "청두", "Almaty": "알마티", "Bishkek": "비슈케크", 
    "Dushanbe": "두샨베", "Kathmandu": "카트만두", "Yangon": "양곤", "Guwahati": "구와하티", 
    "Ulaanbaatar": "울란바토르", "Irkutsk": "이르쿠츠크"
}

# Streamlit 앱
st.title("베이징 미세먼지가 주변국에 미치는 영향 (등급 예측)")

# 데이터 로드 및 모델 학습
data = load_data()
models, evaluation_scores, target_cities, label_encoder = train_model(data)

# 탭 생성
tab1 = st.tabs(["PM2.5 등급 예측"])[0]

# PM2.5 등급 예측 탭
with tab1:
    st.header("PM2.5 등급 예측")

    # 입력값 수집
    beijing_pm25 = st.slider("베이징 PM2.5 (µg/m³)", 0.0, 200.0, 50.0, key="pm25_input")

    # 예측 및 지도 표시
    predictions = predict_pm25_grade(models, beijing_pm25, label_encoder)

    if predictions:
        # 예측 결과 섹션
        with st.expander("주변국 PM2.5 등급 예측"):
            pred_table_data = {
                "도시": [city_names_kr.get(city, city) for city in predictions.keys()],
                "PM2.5 등급": [grade for grade in predictions.values()]
            }
            pred_df = pd.DataFrame(pred_table_data)
            st.dataframe(pred_df, use_container_width=True)

        # 모델 평가 점수 섹션
        with st.expander("모델 평가 점수 (PM2.5 등급)"):
            eval_table_data = {
                "도시": [city_names_kr.get(city, city) for city in models.keys()],
                "MSE": [evaluation_scores[city]['MSE'] for city in models.keys()],
                "RMSE": [evaluation_scores[city]['RMSE'] for city in models.keys()],
                "MAE": [evaluation_scores[city]['MAE'] for city in models.keys()],
                "R² 스코어": [evaluation_scores[city]['R²'] for city in models.keys()]
            }
            eval_df = pd.DataFrame(eval_table_data)
            st.dataframe(eval_df, use_container_width=True)

        # 지도 생성
        st.subheader("지도 (PM2.5 등급)")
        m = folium.Map(location=[35, 120], zoom_start=4)
        
        # 예측값을 지도에 표시
        for city, grade in predictions.items():
            if city in city_coords:
                lat, lon = city_coords[city]
                color = get_color(grade)
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=10,  # 고정된 크기 사용
                    popup=f"{city_names_kr.get(city, city)}: {grade}",
                    color=color,
                    fill=True,
                    fill_opacity=0.7
                ).add_to(m)
        
        # 지도 표시
        st_folium(m, width=700, height=500, key="map_pm25")
    else:
        st.error("모델이 없습니다 (PM2.5 등급).")

# 데이터 정보
with st.expander("원본 데이터 미리보기"):
    st.dataframe(data.head())
    st.write(f"총 데이터 행 수: {len(data)}")