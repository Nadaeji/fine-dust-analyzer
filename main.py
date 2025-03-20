import streamlit as st
from utils.model_cache import ModelCache
import os

@st.cache_resource
def initialize_models():
    base_models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    
    # 로딩 상태 표시
    with st.spinner("모델을 로드 중입니다. 잠시만 기다려 주세요..."):
        status = st.empty()
        progress_bar = st.progress(0)
        status.text("모델 가져오는 중...")
        
        try:
            ModelCache.load_all_models(base_models_dir, progress_bar, status)
            status.text("모델 로드 완료!")
            st.success("모델 로드 완료!")
        except FileNotFoundError as e:
            st.error(f"모델 디렉토리 오류: {e}")
            return False
        except Exception as e:
            st.error(f"모델 로드 중 오류 발생: {e}")
            return False
    
    # 캐시 디버깅
    cached_models = ModelCache.list_cached_models()
    st.write("로드된 모델 수:", len(cached_models))
    st.write("로드된 모델 목록:", cached_models)
    return True

if __name__ == "__main__":
    st.title("미세먼지 예측 대시보드")
    
    # 모델 초기화 상태 확인
    if "models_initialized" not in st.session_state:
        st.session_state.models_initialized = False
    
    if not st.session_state.models_initialized:
        st.session_state.models_initialized = initialize_models()
    
    # 모델 로드가 완료된 경우에만 메뉴 표시
    if st.session_state.models_initialized:
        st.write("모델이 로드되었습니다. 좌측 메뉴에서 페이지를 선택하세요.")
    else:
        st.info("모델을 로드 중입니다. 잠시만 기다려 주세요...")
        st.stop()  # 로딩 중에는 추가 렌더링 중지