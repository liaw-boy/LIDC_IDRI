import os
import cv2
import numpy as np
import pydicom as dicom
import lxml.etree as etree
from datetime import datetime
from tqdm import tqdm
import csv
import re
from collections import defaultdict
import math
import shutil
import concurrent.futures
import threading

# 設定參數
ROI_SIZE = 32  # ROI大小為30x30的ROI
MIN_EDGE_MAPS = 2  # 最少需要的 edgemap 數量
MAX_GAP = 1  # 允許的最大間隔，若超過此值則視為不連續
MAX_SHIFT_DISTANCE = 20  # 允許的最大偏移距離（像素），超過此值視為大幅度偏移
NUM_WORKERS = 8  # 多線程處理的工作線程數量

# 用於線程安全的結果收集
result_lock = threading.Lock()
csv_lock = threading.Lock()


def ensure_directory(directory):
    """確保資料夾存在，若不存在則建立"""
    if not os.path.exists(directory):
        os.makedirs(directory)


def find_dicom_file(sop_uid, directory):
    """在指定資料夾中找到符合 sop_uid 的 DICOM 檔案路徑"""
    for file in os.listdir(directory):
        if file.endswith('.dcm'):
            try:
                ds = dicom.dcmread(os.path.join(directory, file))
                if ds.SOPInstanceUID == sop_uid:
                    return os.path.join(directory, file)
            except:
                continue
    return None


def extract_slice_number(dicom_path):
    """從 DICOM 檔案名稱中提取切片編號"""
    filename = os.path.basename(dicom_path)
    
    # 嘗試從檔案名稱中提取數字部分（例如 1-047 中的 047）
    match = re.search(r'-(\d+)', filename)
    if match:
        return int(match.group(1))
    
    # 如果沒有 '-' 分隔符，嘗試找出檔案名稱中的最後一組數字
    match = re.search(r'(\d+)(?!.*\d)', filename)
    if match:
        return int(match.group(1))
    
    # 如果檔案名稱中沒有數字，嘗試從DICOM的InstanceNumber獲取
    try:
        ds = dicom.dcmread(dicom_path)
        if hasattr(ds, 'InstanceNumber'):
            return int(ds.InstanceNumber)
    except:
        pass
    
    return None


def extract_roi(image, bbox, size=ROI_SIZE):
    """
    從原始影像中擷取一個固定大小 (size x size) 的 ROI。
    bbox = [x_min, y_min, w, h]
    取 bbox 中心點為中心, 向外擴展 size//2。
    """
    x_min, y_min, w, h = bbox
    center_x = x_min + w // 2
    center_y = y_min + h // 2

    half = size // 2
    x1 = max(0, center_x - half)
    y1 = max(0, center_y - half)
    x2 = min(image.shape[1], center_x + half)
    y2 = min(image.shape[0], center_y + half)

    roi = image[y1:y2, x1:x2]
    # 若擷取區域不是 (size, size), 則 resize 到 (size, size)
    roi = cv2.resize(roi, (size, size))
    return roi


def extract_nodule_characteristics(nodule, ns):
    """從 XML 中提取結節的九個特徵值"""
    characteristics = nodule.find('ns:characteristics', ns)
    if characteristics is None:
        return None
    
    char_dict = {}
    for char in ['subtlety', 'internalStructure', 'calcification', 'sphericity', 
                 'margin', 'lobulation', 'spiculation', 'texture', 'malignancy']:
        elem = characteristics.find(f'ns:{char}', ns)
        if elem is not None:
            char_dict[char] = elem.text
        else:
            char_dict[char] = ''
    
    return char_dict


def parse_xml_to_slices(xml_path, dicom_dir):
    """解析 XML 並提取所有切片資訊，忽略只有一個 edgemap 的標記"""
    ns = {'ns': 'http://www.nih.gov'}
    tree = etree.parse(xml_path)
    root = tree.getroot()

    slices = []  # 用於存放所有切片資訊
    reading_sessions = root.findall(".//ns:readingSession", ns)

    for session_idx, session in enumerate(reading_sessions):
        reader_id = session_idx + 1  # 讀者ID（從1開始）
        unblinded_nodules = session.findall(".//ns:unblindedReadNodule", ns)
        for nodule in unblinded_nodules:
            # 提取結節特徵值
            characteristics = extract_nodule_characteristics(nodule, ns)
            
            rois = nodule.findall(".//ns:roi", ns)
            for roi in rois:
                image_sop_uid_elem = roi.find('ns:imageSOP_UID', ns)
                if image_sop_uid_elem is None:
                    continue
                image_sop_uid = image_sop_uid_elem.text
                edge_maps = roi.findall('ns:edgeMap', ns)
                
                # 忽略只有一個 edgemap 的標記
                if len(edge_maps) < MIN_EDGE_MAPS:
                    continue
                    
                seg_coords = []
                for edge_map in edge_maps:
                    x = int(edge_map.find('ns:xCoord', ns).text)
                    y = int(edge_map.find('ns:yCoord', ns).text)
                    seg_coords.append([x, y])

                # 提取 z 軸座標
                dcm_file = find_dicom_file(image_sop_uid, dicom_dir)
                if dcm_file:
                    slice_number = extract_slice_number(dcm_file)
                    dicom_filename = os.path.splitext(os.path.basename(dcm_file))[0]
                    slices.append({
                        "image_sop_uid": image_sop_uid,
                        "segmentation": seg_coords,
                        "slice_number": slice_number,
                        "dicom_file": dcm_file,
                        "dicom_filename": dicom_filename,
                        "characteristics": characteristics,
                        "reader_id": reader_id
                    })

    return slices


def classify_by_malignancy(slice_data):
    """
    根據惡性程度對切片進行分類
    返回分類結果：'benign'（良性）, 'malignant'（惡性）, 'uncertain'（不確定）或 'unknown'（未知）
    """
    if 'characteristics' not in slice_data or slice_data['characteristics'] is None:
        return 'unknown'
    
    malignancy = slice_data['characteristics'].get('malignancy', '')
    
    try:
        malignancy = int(malignancy)
        # 根據LIDC-IDRI標準：1-2為良性，4-5為惡性，3為不確定
        if malignancy <= 2:
            return 'benign'
        elif malignancy >= 4:
            return 'malignant'
        else:
            return 'uncertain'
    except (ValueError, TypeError):
        return 'unknown'


def determine_majority_malignancy(slice_annotations):
    """
    根據多位醫生的標註，決定一個切片的主要惡性程度分類
    使用投票機制，若平手則優先選擇：惡性 > 不確定 > 良性 > 未知
    """
    # 計算各分類的票數
    votes = {'malignant': 0, 'uncertain': 0, 'benign': 0, 'unknown': 0}
    
    for annotation in slice_annotations:
        malignancy_class = classify_by_malignancy(annotation)
        votes[malignancy_class] += 1
    
    # 找出得票最多的分類
    max_votes = max(votes.values())
    winners = [cls for cls, vote in votes.items() if vote == max_votes]
    
    # 如果只有一個得票最多的分類，直接返回
    if len(winners) == 1:
        return winners[0]
    
    # 如果有平手，按優先順序選擇
    priority = ['malignant', 'uncertain', 'benign', 'unknown']
    for cls in priority:
        if cls in winners:
            return cls
    
    # 預設返回未知
    return 'unknown'


def calculate_centroid(segmentation):
    """計算分割區域的中心點"""
    points = np.array(segmentation)
    centroid_x = np.mean(points[:, 0])
    centroid_y = np.mean(points[:, 1])
    return (centroid_x, centroid_y)


def calculate_overlap(seg1, seg2):
    """
    使用OpenCV計算兩個分割區域的重疊度（IoU, Intersection over Union）
    """
    try:
        # 確保分割區域至少有3個點形成多邊形
        if len(seg1) < 3 or len(seg2) < 3:
            return 0.0
        
        # 創建空白圖像
        width, height = 1000, 1000  # 假設最大圖像尺寸為1000x1000
        mask1 = np.zeros((height, width), dtype=np.uint8)
        mask2 = np.zeros((height, width), dtype=np.uint8)
        
        # 將分割區域轉換為OpenCV格式的點
        pts1 = np.array(seg1, dtype=np.int32).reshape((-1, 1, 2))
        pts2 = np.array(seg2, dtype=np.int32).reshape((-1, 1, 2))
        
        # 填充多邊形
        cv2.fillPoly(mask1, [pts1], 1)
        cv2.fillPoly(mask2, [pts2], 1)
        
        # 計算交集和並集
        intersection = cv2.bitwise_and(mask1, mask2)
        union = cv2.bitwise_or(mask1, mask2)
        
        # 計算IoU
        intersection_area = np.sum(intersection)
        union_area = np.sum(union)
        
        if union_area > 0:
            return intersection_area / union_area
        else:
            return 0.0
    except Exception as e:
        print(f"計算重疊度時發生錯誤: {e}")
        return 0.0


def choose_most_representative_annotation(annotations):
    """
    選擇最具代表性的標記（基於一致性）
    計算每個標記的分割區域與其他標記的平均重疊度，選擇重疊度最高的
    """
    # 如果只有一個標記，直接返回
    if len(annotations) == 1:
        return annotations[0]
    
    # 如果沒有標記，返回None
    if len(annotations) == 0:
        return None
    
    max_avg_overlap = -1
    best_annotation = None
    
    # 計算每個標記與其他標記的平均重疊度
    for i, ann_i in enumerate(annotations):
        seg_i = ann_i["segmentation"]
        total_overlap = 0
        valid_comparisons = 0
        
        for j, ann_j in enumerate(annotations):
            if i != j:
                seg_j = ann_j["segmentation"]
                # 計算重疊度
                overlap = calculate_overlap(seg_i, seg_j)
                total_overlap += overlap
                valid_comparisons += 1
        
        # 計算平均重疊度
        avg_overlap = total_overlap / valid_comparisons if valid_comparisons > 0 else 0
        
        # 更新最具代表性的標記
        if avg_overlap > max_avg_overlap:
            max_avg_overlap = avg_overlap
            best_annotation = ann_i
    
    return best_annotation


def export_roi_info_and_images_by_malignancy(slices, xml_filename, output_dir, csv_writer, patient_id):
    """
    將處理結果輸出到指定資料夾中，根據惡性程度進行分類
    每個切片只保留一個最具代表性的標記，並且每張切片只輸出一次到CSV
    同時輸出30x30的ROI圖片和完整切片，使用指定的命名規則
    """
    if not slices:
        print(f"在 {xml_filename} 找不到有效的結節")
        return 0, defaultdict(int)

    # 按SOP UID對切片進行分組，處理同一切片的多個標註
    sop_uid_to_slices = defaultdict(list)
    for slice_data in slices:
        sop_uid_to_slices[slice_data["image_sop_uid"]].append(slice_data)
    
    # 用於統計切片數量
    malignancy_slice_counts = defaultdict(int)
    processed_slices = 0
    
    # 處理每一個唯一的SOP UID（即每一個切片）
    for sop_uid, slice_annotations in sop_uid_to_slices.items():
        # 確定這個切片的主要惡性程度分類（通過多數決）
        malignancy_class = determine_majority_malignancy(slice_annotations)
        
        # 從符合這個分類的標記中選擇最具代表性的標記
        matching_annotations = [ann for ann in slice_annotations 
                               if classify_by_malignancy(ann) == malignancy_class]
        
        # 如果沒有符合的標記（極少數情況），就使用所有標記
        annotations_to_consider = matching_annotations if matching_annotations else slice_annotations
        
        # 選擇最具代表性的標記
        chosen_annotation = choose_most_representative_annotation(annotations_to_consider)
        
        if chosen_annotation is None:
            continue
        
        # 統計切片數量（線程安全）
        with result_lock:
            malignancy_slice_counts[malignancy_class] += 1
            processed_slices += 1
        
        # 獲取選定標註的基本信息
        dcm_file = chosen_annotation["dicom_file"]
        
        if dcm_file:
            try:
                # 讀取DICOM檔案
                ds = dicom.dcmread(dcm_file)
                pixel_array = ds.pixel_array.astype(np.float32)
                # Convert to HU then apply lung window (WC=-600, WW=1500)
                # Must match predictor.py inference normalization exactly
                slope = float(getattr(ds, "RescaleSlope", 1))
                intercept = float(getattr(ds, "RescaleIntercept", 0))
                hu = pixel_array * slope + intercept
                wc, ww = -600, 1500
                lo, hi = wc - ww // 2, wc + ww // 2
                image = np.clip(hu, lo, hi)
                image = ((image - lo) / ww * 255).astype(np.uint8)
                
                # 使用完整的UID作為檔案名稱基礎
                safe_uid = sop_uid.replace('.', '_')
                
                # 處理分割
                points = np.array(chosen_annotation["segmentation"]).reshape(-1, 2)
                if len(points) > 0:
                    # 提取並儲存30x30 ROI圖像
                    x_min = int(min(points[:, 0]))
                    y_min = int(min(points[:, 1]))
                    w = int(max(points[:, 0]) - x_min)
                    h = int(max(points[:, 1]) - y_min)
                    
                    # 確保寬度和高度至少為1
                    w = max(1, w)
                    h = max(1, h)
                    
                    bbox = [x_min, y_min, w, h]
                    roi_image = extract_roi(image, bbox, size=ROI_SIZE)
                    
                    # 建立分類資料夾路徑
                    malignancy_dir = os.path.join(output_dir, malignancy_class)
                    ensure_directory(malignancy_dir)
                    
                    # 保存ROI圖像，以UID+roi命名
                    roi_filename = f"{safe_uid}_roi.png"
                    roi_path = os.path.join(malignancy_dir, roi_filename)
                    cv2.imwrite(roi_path, roi_image)
                    
                    # 保存完整切片，以UID+ct命名
                    ct_filename = f"{safe_uid}_ct.png"
                    ct_path = os.path.join(malignancy_dir, ct_filename)
                    cv2.imwrite(ct_path, image)
                    
                    # 寫入 CSV 檔案（線程安全）- 每張切片只輸出一次
                    with csv_lock:
                        # 記錄病人編號、SOP UID、惡性程度分類
                        row = [patient_id, sop_uid, malignancy_class]
                        
                        # 收集所有讀者對此切片的惡性程度評分，並標記讀者ID
                        reader_malignancy_details = []
                        for ann in slice_annotations:
                            if ann['characteristics'] and 'malignancy' in ann['characteristics']:
                                reader_id = ann['reader_id']
                                score = ann['characteristics']['malignancy']
                                if score:  # 確保評分不為空
                                    reader_malignancy_details.append(f"R{reader_id}:{score}")
                        
                        # 將所有評分合併為一個字串，用分號分隔
                        reader_scores_str = ';'.join(reader_malignancy_details)
                        row.append(reader_scores_str)
                        
                        # 加入九個特徵值（使用最具代表性標註的特徵值）
                        if chosen_annotation['characteristics']:
                            for char in ['subtlety', 'internalStructure', 'calcification', 'sphericity', 
                                        'margin', 'lobulation', 'spiculation', 'texture', 'malignancy']:
                                row.append(chosen_annotation['characteristics'].get(char, ''))
                        else:
                            # 如果沒有特徵值，則填入空值
                            row.extend([''] * 9)
                        
                        csv_writer.writerow(row)
                
            except Exception as e:
                print(f"處理 DICOM {dcm_file} 時發生錯誤: {e}")
    
    return processed_slices, malignancy_slice_counts

def process_xml_file(xml_path, output_dir, csv_writer):
    """
    處理單個XML檔案的工作函數，用於多線程處理
    """
    # 病人資料夾名稱
    patient_folder_name = os.path.basename(os.path.dirname(xml_path))
    
    # 使用資料夾名稱作為病人編號
    patient_id = patient_folder_name
    
    try:
        start_time = datetime.now()
        slices = parse_xml_to_slices(xml_path, os.path.dirname(xml_path))
        
        # 根據惡性程度分類並輸出，獲取處理的切片數量和分類統計
        processed_slices, malignancy_counts = export_roi_info_and_images_by_malignancy(
            slices, xml_path, output_dir, csv_writer, patient_id)
        
        # 計算標註數量
        annotation_count = len(slices)
        
        processing_time = datetime.now() - start_time
        
        # 使用線程安全的方式輸出處理結果
        with result_lock:
            print(f"\n完成處理 {xml_path}，耗時 {processing_time}")
            print(f"共找到 {annotation_count} 個標註，分佈在 {processed_slices} 個不同的切片上")
            print(f"惡性程度分類統計（切片數量）: 良性: {malignancy_counts['benign']}, 惡性: {malignancy_counts['malignant']}, "
                 f"不確定: {malignancy_counts['uncertain']}, 未知: {malignancy_counts['unknown']}")
        
        return processed_slices, malignancy_counts, annotation_count
    
    except Exception as e:
        with result_lock:
            print(f"處理 XML {xml_path} 時發生錯誤: {e}")
        return 0, defaultdict(int), 0


def main():
    root_dir = os.environ.get('LUNA16_DIR', './LUNA16')
    output_dir = './outputct+roi_0406'  # 統一輸出資料夾
    ensure_directory(output_dir)

    # 建立分類資料夾
    malignancy_dirs = ['benign', 'malignant', 'uncertain', 'unknown']
    for malignancy_dir in malignancy_dirs:
        ensure_directory(os.path.join(output_dir, malignancy_dir))

    # 建立一個全域的 CSV 檔案
    csv_filename = os.path.join(output_dir, "all_nodules_data.csv")
    
    # 掃描 XML 文件
    xml_files = []
    print("掃描 XML 檔案...")
    for dirpath, _, filenames in os.walk(root_dir):
        for file in filenames:
            if file.endswith('.xml'):
                xml_files.append(os.path.join(dirpath, file))

    total_files = len(xml_files)
    print(f"共找到 {total_files} 個 XML 檔案待處理")

    # 用於統計總數
    total_slices = 0
    total_annotations = 0
    total_malignancy_counts = defaultdict(int)

    # 開啟全域 CSV 檔案
    with open(csv_filename, mode='w', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file)
        # 寫入 CSV 標頭 - 修改標頭以反映新的資料結構
        csv_writer.writerow(["Patient ID", "SOP UID", "Majority Malignancy Class", "All Reader Malignancy Scores (ReaderID:Score)", 
                             "subtlety", "internalStructure", "calcification", "sphericity", 
                             "margin", "lobulation", "spiculation", "texture", "malignancy"])
        # 使用線程池處理XML檔案
        with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            # 提交所有任務
            futures = [executor.submit(process_xml_file, xml_path, output_dir, csv_writer) 
                      for xml_path in xml_files]
            
            # 使用tqdm顯示進度
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="處理XML檔案"):
                processed_slices, malignancy_counts, annotation_count = future.result()
                
                # 更新總統計
                total_slices += processed_slices
                total_annotations += annotation_count
                for cls, count in malignancy_counts.items():
                    total_malignancy_counts[cls] += count

    # 輸出總統計
    print("\n=== 處理完成 ===")
    print(f"總共處理了 {total_slices} 個切片，共 {total_annotations} 個標註")
    print(f"惡性程度分類總統計（切片數量）:")
    print(f"  - 良性: {total_malignancy_counts['benign']}")
    print(f"  - 惡性: {total_malignancy_counts['malignant']}")
    print(f"  - 不確定: {total_malignancy_counts['uncertain']}")
    print(f"  - 未知: {total_malignancy_counts['unknown']}")


if __name__ == "__main__":
    main()
