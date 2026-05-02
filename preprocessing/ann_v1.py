import os
import json
import cv2
import numpy as np
import pydicom as dicom
import lxml.etree as etree
import shutil
from datetime import datetime
from tqdm import tqdm
from collections import defaultdict
import csv
import statistics

###############################################################
# 修改：
# 1. 有結節切片使用 bounding box 標記
# 2. 只記錄每個特性的中位數整數值
# 3. 不將無結節切片記錄到 CSV 檔案中
# 4. 只保留至少有2位醫生標記的結節切片
# 5. 移除非結節切片的JSON檔案生成，只保留非結節切片的圖像
###############################################################

# 設定 ROI 圖像大小
ROI_SIZE = 50
# 設定最少需要幾位醫生標記才納入考慮
MIN_DOCTORS = 2


def ensure_directory(directory):
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

def select_most_significant_nodule(nodules):
    """選取 segmentation 面積最大的結節。"""
    max_area = 0
    selected_nodule = None
    for nodule in nodules:
        seg = np.array(nodule["segmentation"])
        area = cv2.contourArea(seg)
        if area > max_area:
            max_area = area
            selected_nodule = nodule
    return selected_nodule

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

def parse_xml_to_coco_format(xml_path):
    """解析 LIDC-IDRI XML, 回傳每個切片 (SOPUID) 所對應的結節資訊及非結節資訊。"""
    from collections import defaultdict
    ns = {'ns': 'http://www.nih.gov'}
    tree = etree.parse(xml_path)
    root = tree.getroot()

    # 初始化資料結構，用於存儲每個切片的結節特性評分
    slice_nodules = defaultdict(lambda: {"segmentations": [], "characteristics": defaultdict(list), "doctor_count": 0})
    non_nodule_slices = set()  # 存放 non-nodule 的切片 SOP UID
    
    reading_sessions = root.findall(".//ns:readingSession", ns)

    # 追蹤每個切片被多少醫生標記
    doctor_marked_slices = defaultdict(set)
    
    for i, session in enumerate(tqdm(reading_sessions, desc="Processing reading sessions", leave=False)):
        doctor_id = i + 1  # 使用 session 索引作為醫生 ID
        
        # 處理 non-nodule 標記
        non_nodules = session.findall(".//ns:nonNodule", ns)
        for non_nodule in non_nodules:
            image_sop_uid_elem = non_nodule.find('ns:imageSOP_UID', ns)
            if image_sop_uid_elem is not None:
                non_nodule_slices.add(image_sop_uid_elem.text)

        # 處理 nodule 標記
        unblinded_nodules = session.findall(".//ns:unblindedReadNodule", ns)
        for nodule in unblinded_nodules:
            characteristics = nodule.find(".//ns:characteristics", ns)
            if characteristics is None:
                continue
                
            # 獲取所有結節特性資訊
            nodule_characteristics = {}
            for feature in ['subtlety', 'internalStructure', 'calcification', 'sphericity', 
                           'margin', 'lobulation', 'spiculation', 'texture', 'malignancy']:
                elem = characteristics.find(f'ns:{feature}', ns)
                if elem is not None:
                    try:
                        nodule_characteristics[feature] = int(elem.text)
                    except:
                        nodule_characteristics[feature] = None
                else:
                    nodule_characteristics[feature] = None

            rois = nodule.findall(".//ns:roi", ns)
            for roi in rois:
                image_sop_uid_elem = roi.find('ns:imageSOP_UID', ns)
                if image_sop_uid_elem is None:
                    continue
                image_sop_uid = image_sop_uid_elem.text
                edge_maps = roi.findall('ns:edgeMap', ns)
                
                # 修改: 如果 edgemap 少於兩個，則不算是有結節
                if len(edge_maps) < 2:
                    continue
                    
                seg_coords = []
                for edge_map in edge_maps:
                    x = int(edge_map.find('ns:xCoord', ns).text)
                    y = int(edge_map.find('ns:yCoord', ns).text)
                    seg_coords.append([x, y])

                # 記錄此醫生標記了這個切片
                doctor_marked_slices[image_sop_uid].add(doctor_id)
                
                # 將這個醫生的分割和特性評分添加到對應的切片中
                slice_nodules[image_sop_uid]["segmentations"].append(seg_coords)
                
                # 將這個醫生對每個特性的評分添加到對應特性的列表中
                for feature, value in nodule_characteristics.items():
                    if value is not None:
                        slice_nodules[image_sop_uid]["characteristics"][feature].append(value)

    # 對於每個 SOP UID, 處理所有醫生的評分和分割
    unique_slices = []
    for sop_uid, nodule_data in slice_nodules.items():
        segmentations = nodule_data["segmentations"]
        characteristics = nodule_data["characteristics"]
        
        # 檢查有多少醫生標記了這個切片
        doctor_count = len(doctor_marked_slices[sop_uid])
        
        # 如果醫生數量少於最低要求，則跳過此切片
        if doctor_count < MIN_DOCTORS or not segmentations:
            continue
        
        # 選擇面積最大的分割
        max_area = 0
        selected_segmentation = None
        
        for seg in segmentations:
            seg_array = np.array(seg)
            area = cv2.contourArea(seg_array)
            if area > max_area:
                max_area = area
                selected_segmentation = seg_array
        
        if selected_segmentation is not None:
            # 計算 bounding box
            x_min = int(np.min(selected_segmentation[:, 0]))
            y_min = int(np.min(selected_segmentation[:, 1]))
            x_max = int(np.max(selected_segmentation[:, 0]))
            y_max = int(np.max(selected_segmentation[:, 1]))
            bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
            area = bbox[2] * bbox[3]
            segmentation_flat = selected_segmentation.flatten().tolist()
            
            # 計算每個特性的中位數整數值
            median_characteristics = {}
            
            for feature, ratings in characteristics.items():
                if ratings:
                    # 使用 statistics.median 計算中位數，並轉為整數
                    median_characteristics[feature] = int(statistics.median(ratings))
                else:
                    median_characteristics[feature] = None
            
            unique_slices.append({
                "image_sop_uid": sop_uid,
                "doctor_count": doctor_count,  # 記錄有多少醫生標記了這個切片
                "nodules": [{
                    "segmentation": segmentation_flat,
                    "bbox": bbox,
                    "area": area,
                    "characteristics": median_characteristics  # 只保存中位數整數值
                }]
            })

    return unique_slices, non_nodule_slices  # 回傳結節切片和非結節切片


def export_roi_info_and_images(slices, non_nodule_slices, xml_filename, external_dir, csv_writer, patient_id):
    """
    處理結節和非結節切片，並儲存相關資訊和影像。
    """
    if not slices and not non_nodule_slices:
        print(f"在 {xml_filename} 找不到有效的 slice")
        return

    # 建立各子資料夾
    mask_images_dir = os.path.join(external_dir, "mask_images")
    overlay_images_dir = os.path.join(external_dir, "overlay_images")
    sop_images_dir = os.path.join(external_dir, "sop_images")
    coco_json_dir = os.path.join(external_dir, "coco_json")
    roi_images_dir = os.path.join(external_dir, "roi_images")
    non_nodule_dir = os.path.join(external_dir, "non_nodule_images")  # 存放 non-nodule 切片的資料夾

    ensure_directory(mask_images_dir)
    ensure_directory(overlay_images_dir)
    ensure_directory(sop_images_dir)
    ensure_directory(coco_json_dir)
    ensure_directory(roi_images_dir)
    ensure_directory(non_nodule_dir)

    # 處理有結節的切片
    total_slices = len(slices)
    for i, slice_data in enumerate(tqdm(slices, desc="Processing nodule slices", leave=False)):
        sop_uid = slice_data["image_sop_uid"]
        doctor_count = slice_data["doctor_count"]
        dcm_file = find_dicom_file(sop_uid, os.path.dirname(xml_filename))
        if dcm_file:
            try:
                ds = dicom.dcmread(dcm_file)
                pixel_array = ds.pixel_array
                # 將影像標準化至 0~255
                ptp = pixel_array.max() - pixel_array.min()
                image = ((pixel_array - pixel_array.min()) / (ptp if ptp > 0 else 1) * 255).astype(np.uint8)

                nodule = slice_data["nodules"][0]
                points = np.array(nodule["segmentation"]).reshape(-1, 2)
                
                # 檢查是否有結節 (segmentation 不為空)
                has_nodule = len(points) > 0
                
                if has_nodule:
                    mask_image = np.zeros_like(image)
                    overlay_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                    cv2.fillPoly(mask_image, [points], 255)
                    cv2.fillPoly(overlay_image, [points], (0, 0, 255))

                    tqdm.write(f"儲存第 {i+1}/{total_slices} 個 slice (SOP UID: {sop_uid}, 醫生數: {doctor_count}) 的影像檔案")

                    mask_filename = f"{sop_uid}_mask.png"
                    overlay_filename = f"{sop_uid}_overlay.png"
                    sop_image_filename = f"{sop_uid}.png"
                    coco_json_filename = f"{sop_uid}.json"
                    roi_filename = f"{sop_uid}_roi.png"

                    # 輸出 mask, overlay, sop 原圖
                    cv2.imwrite(os.path.join(mask_images_dir, mask_filename), mask_image)
                    cv2.imwrite(os.path.join(overlay_images_dir, overlay_filename), overlay_image)
                    cv2.imwrite(os.path.join(sop_images_dir, sop_image_filename), image)

                    # 擷取 ROI 圖 (固定大小)
                    bbox = nodule["bbox"]  # [x_min, y_min, w, h]
                    roi_img = extract_roi(image, bbox, size=ROI_SIZE)
                    cv2.imwrite(os.path.join(roi_images_dir, roi_filename), roi_img)

                    # 產生 COCO 格式 JSON 檔案 (使用 bounding box 標記)
                    annotations = []
                    ann_nodule = {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 1,  # 1 = nodule
                        "bbox": nodule["bbox"],
                        "area": nodule["area"],
                        "iscrowd": 0
                    }
                    annotations.append(ann_nodule)

                    coco_data = {
                        "images": [{
                            "id": 1,
                            "file_name": sop_image_filename,
                            "height": int(image.shape[0]),
                            "width": int(image.shape[1])
                        }],
                        "annotations": annotations,
                        "categories": [
                            {"id": 1, "name": "nodule"}
                        ]
                    }
                    with open(os.path.join(coco_json_dir, coco_json_filename), 'w') as f:
                        json.dump(coco_data, f, indent=4)

                    # 寫入 CSV 檔案 (只加入中位數特性資訊)
                    nodule_no = i + 1
                    characteristics = nodule.get("characteristics", {})
                    
                    # 準備 CSV 行數據
                    row_data = [
                        patient_id, 
                        nodule_no, 
                        sop_uid,
                        doctor_count  # 添加醫生數量資訊
                    ]
                    
                    # 添加每個特性的中位數整數值
                    features = ['subtlety', 'internalStructure', 'calcification', 'sphericity', 
                               'margin', 'lobulation', 'spiculation', 'texture', 'malignancy']
                    
                    for feature in features:
                        # 添加中位數整數值
                        value = characteristics.get(feature)
                        row_data.append(str(value) if value is not None else "")
                    
                    # 寫入 CSV
                    csv_writer.writerow(row_data)
                else:
                    # 將無結節的切片存入 non_nodule 資料夾
                    non_nodule_filename = f"{sop_uid}.png"
                    cv2.imwrite(os.path.join(non_nodule_dir, non_nodule_filename), image)
                    tqdm.write(f"儲存無結節的切片 (SOP UID: {sop_uid}) 到 non_nodule 資料夾")

            except Exception as e:
                tqdm.write(f"處理 DICOM {dcm_file} 時發生錯誤: {e}")

    # 處理 non-nodule 切片 (只保存圖像，不生成 JSON)
    total_non_nodules = len(non_nodule_slices)
    for i, sop_uid in enumerate(tqdm(non_nodule_slices, desc="Processing non-nodule slices", leave=False)):
        dcm_file = find_dicom_file(sop_uid, os.path.dirname(xml_filename))
        if dcm_file:
            try:
                ds = dicom.dcmread(dcm_file)
                pixel_array = ds.pixel_array
                # 將影像標準化至 0~255
                ptp = pixel_array.max() - pixel_array.min()
                image = ((pixel_array - pixel_array.min()) / (ptp if ptp > 0 else 1) * 255).astype(np.uint8)

                # 儲存 non-nodule 切片
                non_nodule_filename = f"{sop_uid}.png"
                cv2.imwrite(os.path.join(non_nodule_dir, non_nodule_filename), image)
                tqdm.write(f"儲存 non-nodule 切片 {i+1}/{total_non_nodules} (SOP UID: {sop_uid})")
                
                # 不再為非結節切片生成 JSON 檔案
                
            except Exception as e:
                tqdm.write(f"處理 DICOM {dcm_file} 時發生錯誤: {e}")


def process_patient_data(patient_dir, external_dir, csv_writer):
    """處理單一病患的資料"""
    patient_id = os.path.basename(patient_dir)
    
    # 找出 XML 檔案
    xml_files = []
    for root, dirs, files in os.walk(patient_dir):
        for file in files:
            if file.endswith('.xml'):
                xml_files.append(os.path.join(root, file))
    
    for xml_file in xml_files:
        try:
            # 解析 XML 檔案
            slices, non_nodule_slices = parse_xml_to_coco_format(xml_file)
            # 輸出結節資訊和影像
            export_roi_info_and_images(slices, non_nodule_slices, xml_file, external_dir, csv_writer, patient_id)
        except Exception as e:
            print(f"處理 {xml_file} 時發生錯誤: {e}")


def main():
    root_dir = '.'
    # 定義外部統一輸出資料夾 (所有影像、JSON 檔案與 CSV 均存放在此資料夾內)
    external_dir = './root_ndu_images'
    ensure_directory(external_dir)

    # 將 CSV 檔案也存放在外部資料夾中，並加入結節特性欄位
    csv_filename = os.path.join(external_dir, "patient_sop_uids.csv")
    with open(csv_filename, mode='w', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file)
        
        # 修改 CSV 表頭，只包含中位數，並加入醫生數量
        header = ["Patient ID", "Nodule No", "SOP UID", "Doctor Count"]
        
        # 為每個特性添加中位數的列
        features = ['subtlety', 'internalStructure', 'calcification', 'sphericity', 
                   'margin', 'lobulation', 'spiculation', 'texture', 'malignancy']
        
        for feature in features:
            header.append(feature)  # 只保留特性名稱，不再加上 _median 後綴
        
        csv_writer.writerow(header)

        xml_files = []
        print("掃描 XML 檔案...")
        for dirpath, _, filenames in os.walk(root_dir):
            for file in filenames:
                if file.endswith('.xml'):
                    xml_files.append(os.path.join(dirpath, file))

        total_files = len(xml_files)
        print(f"共找到 {total_files} 個 XML 檔案待處理")

        for i, xml_path in enumerate(tqdm(xml_files, desc="Processing XML files", unit="file")):
            # 修改：從 XML 檔案所在的資料夾名稱提取病人 ID
            parent_folder = os.path.basename(os.path.dirname(xml_path))
            
            # 提取數字部分並格式化為 LIDC-IDRI-xxxx
            try:
                # 如果資料夾名稱已經是 LIDC-IDRI-xxxx 格式
                if parent_folder.startswith('LIDC-IDRI-'):
                    # 提取數字部分
                    patient_number = ''.join(filter(str.isdigit, parent_folder))
                    # 確保是四位數字格式
                    patient_number = patient_number.zfill(4)
                    patient_id = f"LIDC-IDRI-{patient_number}"
                else:
                    # 如果資料夾名稱是其他格式，嘗試提取數字部分
                    patient_number = ''.join(filter(str.isdigit, parent_folder))
                    if patient_number:
                        # 確保是四位數字格式
                        patient_number = patient_number.zfill(4)
                        patient_id = f"LIDC-IDRI-{patient_number}"
                    else:
                        # 如果無法提取數字，則使用原始資料夾名稱
                        patient_id = f"LIDC-IDRI-{parent_folder}"
            except:
                # 發生錯誤時的備用方案
                patient_id = f"LIDC-IDRI-{parent_folder}"
                
            tqdm.write(f"\n處理第 {i+1}/{total_files} 個 XML 檔案: {xml_path}")
            tqdm.write(f"病人 ID: {patient_id}")
            
            try:
                start_time = datetime.now()
                # 解析 XML 檔案，獲取結節和非結節切片
                slices, non_nodule_slices = parse_xml_to_coco_format(xml_path)
                # 傳遞非結節切片資訊
                export_roi_info_and_images(slices, non_nodule_slices, xml_path, external_dir, csv_writer, patient_id)
                processing_time = datetime.now() - start_time
                tqdm.write(f"完成處理 {xml_path}，耗時 {processing_time}")
                tqdm.write(f"共找到 {len(slices)} 個具有結節的 slice 和 {len(non_nodule_slices)} 個非結節切片")
            except Exception as e:
                tqdm.write(f"處理 XML {xml_path} 時發生錯誤: {e}")


if __name__ == "__main__":
    main()
