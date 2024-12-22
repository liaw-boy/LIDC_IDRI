import os
import shutil

def find_largest_dcm_folder(base_dir):
    largest_folder = None
    max_dcm_count = 0

    for root, dirs, files in os.walk(base_dir):
        dcm_count = sum(1 for file in files if file.endswith('.dcm'))
        if dcm_count > max_dcm_count:
            max_dcm_count = dcm_count
            largest_folder = root

    return largest_folder, max_dcm_count

def move_files_to_target(largest_folder, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for item in os.listdir(largest_folder):
        source_path = os.path.join(largest_folder, item)
        target_path = os.path.join(target_dir, item)

        if os.path.isdir(source_path):
            shutil.move(source_path, target_path)
        else:
            shutil.move(source_path, target_path)

def delete_subfolders(folder_path):
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isdir(item_path):
            shutil.rmtree(item_path)

def process_all_directories(base_dir):
    for subdir in os.listdir(base_dir):
        subdir_path = os.path.join(base_dir, subdir)

        if os.path.isdir(subdir_path) and subdir.startswith("LIDC-IDRI"):
            print(f"處理資料夾: {subdir}")
            largest_folder, dcm_count = find_largest_dcm_folder(subdir_path)

            if largest_folder:
                target_dir = os.path.join(base_dir, subdir)
                print(f"找到 DICOM 檔案最多的資料夾: {largest_folder}，共有 {dcm_count} 個 .dcm 檔案。")
                move_files_to_target(largest_folder, target_dir)
                print(f"已將檔案移動到目標目錄: {target_dir}")

            # 刪除 LIDC-IDRI-xxxx 資料夾中的所有子資料夾
            delete_subfolders(subdir_path)
            print(f"已刪除 {subdir_path} 中的所有子資料夾。")
        else:
            print(f"未找到任何含有 .dcm 檔案的資料夾於 {subdir}。")

def main():
    base_dir = "C:/Users/Liawboy/Desktop/LIDC_IDRI"  # 替換為您的根目錄路徑
    process_all_directories(base_dir)

if __name__ == "__main__":
    main()