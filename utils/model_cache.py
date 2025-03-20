import joblib
import os
from typing import Dict, Any
import streamlit as st

class ModelCache:
    _instance = None
    _models: Dict[str, Any] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelCache, cls).__new__(cls)
        return cls._instance
    
    @staticmethod
    def load_model(model_name: str, model_path: str):
        """단일 모델을 로드하고 캐싱합니다."""
        if model_name not in ModelCache._models:
            model_file = os.path.join(model_path, f"{model_name}.pkl")
            if os.path.exists(model_file):
                ModelCache._models[model_name] = joblib.load(model_file)
            else:
                raise FileNotFoundError(f"Model file not found: {model_file}")
        return ModelCache._models[model_name]
    
    @staticmethod
    def load_all_models(base_path: str, progress_bar=None, status=None):
        """지정된 기본 경로 내 모든 하위 디렉토리의 .pkl 파일을 로드하고 캐싱합니다."""
        if not os.path.exists(base_path):
            raise FileNotFoundError(f"Base model directory not found: {base_path}")
        
        # 모델 파일 목록 수집
        model_files = []
        for model_dir_name in os.listdir(base_path):
            model_dir = os.path.join(base_path, model_dir_name)
            if os.path.isdir(model_dir):
                for filename in os.listdir(model_dir):
                    if filename.endswith(".pkl"):
                        model_files.append((model_dir_name, filename, os.path.join(model_dir, filename)))
        
        total_files = len(model_files)
        if total_files == 0:
            if status:
                status.text("로드할 모델 파일이 없습니다.")
            return ModelCache._models
        
        # 진행률 표시와 함께 모델 로드
        for i, (model_dir_name, filename, model_file) in enumerate(model_files):
            model_name = f"{model_dir_name}/{filename[:-4]}"  # 디렉토리명/파일명으로 키 생성
            if model_name not in ModelCache._models:
                try:
                    ModelCache._models[model_name] = joblib.load(model_file)
                except Exception as e:
                    print(f"Error loading {model_name}: {e}")
            
            # 진행률 업데이트
            if progress_bar and status:
                progress_bar.progress((i + 1) / total_files)
                status.text(f"모델 로드 중... ({i + 1}/{total_files})")
        
        return ModelCache._models
    
    @staticmethod
    def get_model(model_name: str):
        """캐싱된 모델을 반환합니다."""
        if model_name in ModelCache._models:
            return ModelCache._models[model_name]
        raise KeyError(f"Model '{model_name}' not found in cache")
    
    @staticmethod
    def clear_cache():
        """캐시를 초기화합니다."""
        ModelCache._models.clear()
    
    @staticmethod
    def list_cached_models():
        """캐싱된 모델 목록을 반환합니다."""
        return list(ModelCache._models.keys())