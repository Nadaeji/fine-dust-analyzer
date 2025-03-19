import os
import subprocess

def run_train_scripts(directory="train"):
    # train 폴더 내 모든 .py 파일 가져오기
    script_files = sorted([f for f in os.listdir(directory) if f.endswith(".py")])
    
    if not script_files:
        print("No training scripts found in the directory.")
        return
    
    for script in script_files:
        script_path = os.path.join(directory, script)
        print(f"Running: {script_path}")
        result = subprocess.run(["python", script_path], capture_output=True, text=True)
        
        # 실행 결과 출력
        print("Output:")
        print(result.stdout)
        print("Error:")
        print(result.stderr)
        print("-" * 50)

if __name__ == "__main__":
    run_train_scripts()