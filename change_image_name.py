import os
import shutil

# 대상 폴더의 경로
folder_path = r'C:\Users\sungwoo\Downloads\data_hashAttack\input\category2_train'
save_path = r'C:\Users\sungwoo\Downloads\hashAttack\InputImages'
# 대상 폴더 안에 있는 모든 파일 가져오기
files = os.listdir(folder_path)

# 파일 이름 변경
for i, file_name in enumerate(files):
    # 새로운 파일 이름 생성
    new_file_name = f"id{i:04d}.png"
    
    # 파일의 현재 경로와 새로운 경로 생성
    current_path = os.path.join(folder_path, file_name)
    new_path = os.path.join(save_path, new_file_name)
    
    # 파일 이름 변경
    shutil.copyfile(current_path, new_path)


