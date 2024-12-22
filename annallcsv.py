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

def ensure_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def find_dicom_file(sop_uid, directory):
    for file in os.listdir(directory):
        if file.endswith('.dcm'):
            try:
                ds = dicom.dcmread(os.path.join(directory, file))
                if ds.SOPInstanceUID == sop_uid:
                    return os.path.join(directory, file)
            except:
                continue
    return None

def select_most_significant_contour(contours):
    """Select the contour most closely resembling the lesion boundary."""
    max_area = 0
    selected_contour = None
    for contour in contours:
        contour = np.array(contour)
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            selected_contour = contour
    return selected_contour

def parse_xml_to_coco_format(xml_path):
    """
    Parse LIDC-IDRI XML format and extract nodule information
    """
    ns = {'ns': 'http://www.nih.gov'}
    tree = etree.parse(xml_path)
    root = tree.getroot()

    slice_nodules = defaultdict(list)
    reading_sessions = root.findall(".//ns:readingSession", ns)

    for session in tqdm(reading_sessions, desc="Processing reading sessions", leave=False):
        unblinded_nodules = session.findall(".//ns:unblindedReadNodule", ns)

        for nodule in unblinded_nodules:
            characteristics = nodule.find(".//ns:characteristics", ns)
            if characteristics is None:
                continue

            rois = nodule.findall(".//ns:roi", ns)
            for roi in rois:
                image_sop_uid = roi.find('ns:imageSOP_UID', ns)
                if image_sop_uid is None:
                    continue

                image_sop_uid = image_sop_uid.text
                edge_maps = roi.findall('ns:edgeMap', ns)

                if not edge_maps:
                    continue

                segmentation_coords = []
                for edge_map in edge_maps:
                    x = int(edge_map.find('ns:xCoord', ns).text)
                    y = int(edge_map.find('ns:yCoord', ns).text)
                    segmentation_coords.append([x, y])

                slice_nodules[image_sop_uid].append(segmentation_coords)

    unique_slices = []
    for sop_uid, nodules in slice_nodules.items():
        selected_segmentation = []
        if nodules:
            selected_contour = select_most_significant_contour(nodules)
            if selected_contour is not None:
                selected_segmentation = selected_contour

        bbox = [
            int(np.min(selected_segmentation[:, 0])),
            int(np.min(selected_segmentation[:, 1])),
            int(np.max(selected_segmentation[:, 0]) - np.min(selected_segmentation[:, 0])),
            int(np.max(selected_segmentation[:, 1]) - np.min(selected_segmentation[:, 1]))
        ]

        area = bbox[2] * bbox[3]

        unique_slices.append({
            "image_sop_uid": sop_uid,
            "nodules": [{
                "segmentation": selected_segmentation.flatten().tolist() if len(selected_segmentation) > 0 else [],
                "bbox": bbox,
                "area": area
            }]
        })

    return unique_slices

def copy_files_to_root(sop_dir, root_dir):
    """
    複製 SOP_image 資料夾內所有檔案到最外層的目錄。
    """
    for file_name in os.listdir(sop_dir):
        file_path = os.path.join(sop_dir, file_name)
        if os.path.isfile(file_path):  # 只處理檔案
            shutil.copy(file_path, os.path.join(root_dir, file_name))
            print(f"已複製: {file_name} 到 {root_dir}")

def export_roi_info_and_images(slices, xml_filename, output_dir, root_sop_dir, csv_writer, patient_id):
    if not slices:
        print(f"No valid slices found in {xml_filename}")
        return

    mask_images_dir = os.path.join(output_dir, "mask_images")
    overlay_images_dir = os.path.join(output_dir, "overlay_images")
    sop_images_dir = os.path.join(output_dir, "sop_images")
    ensure_directory(mask_images_dir)
    ensure_directory(overlay_images_dir)
    ensure_directory(sop_images_dir)

    total_slices = len(slices)

    for i, slice_data in enumerate(tqdm(slices, desc="Processing slices", leave=False)):
        sop_uid = slice_data["image_sop_uid"]
        dcm_file = find_dicom_file(sop_uid, os.path.dirname(xml_filename))

        if dcm_file:
            try:
                ds = dicom.dcmread(dcm_file)
                pixel_array = ds.pixel_array
                image = ((pixel_array - pixel_array.min()) / \
                         (pixel_array.max() - pixel_array.min()) * 255).astype(np.uint8)

                mask_image = np.zeros_like(image)
                overlay_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

                for nodule in slice_data["nodules"]:
                    points = np.array(nodule["segmentation"]).reshape(-1, 2)
                    cv2.fillPoly(mask_image, [points], 255)
                    cv2.fillPoly(overlay_image, [points], (0, 0, 255))

                tqdm.write(f"Saving images for slice {i+1}/{total_slices} (SOP UID: {sop_uid})")

                mask_filename = f"{sop_uid}_mask.png"
                overlay_filename = f"{sop_uid}_overlay.png"
                sop_image_filename = f"{sop_uid}.png"
                coco_json_filename = f"{sop_uid}.json"

                cv2.imwrite(os.path.join(mask_images_dir, mask_filename), mask_image)
                cv2.imwrite(os.path.join(overlay_images_dir, overlay_filename), overlay_image)
                cv2.imwrite(os.path.join(sop_images_dir, sop_image_filename), image)

                # 複製 SOP Image 檔案到最外層目錄
                ensure_directory(root_sop_dir)
                shutil.copy(os.path.join(sop_images_dir, sop_image_filename), root_sop_dir)

                annotations = []
                for idx, nodule in enumerate(slice_data["nodules"], 1):
                    annotations.append({
                        "id": idx,
                        "image_id": 1,
                        "category_id": 1,
                        "segmentation": [nodule["segmentation"]],
                        "bbox": nodule["bbox"],
                        "area": nodule["area"],
                        "iscrowd": 0
                    })

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

                with open(os.path.join(sop_images_dir, coco_json_filename), 'w') as f:
                    json.dump(coco_data, f, indent=4)

                shutil.copy(os.path.join(sop_images_dir, coco_json_filename), root_sop_dir)

                # 寫入 CSV 檔案，加入 nodule 編號
                nodule_no = i + 1
                csv_writer.writerow([patient_id, nodule_no, sop_uid])

            except Exception as e:
                tqdm.write(f"Error processing DICOM {dcm_file}: {e}")

def main():
    root_dir = '.'
    root_sop_dir = './root_sop_images'  # 定義最外層統一資料夾
    ensure_directory(root_sop_dir)

    csv_filename = "patient_sop_uids.csv"
    with open(csv_filename, mode='w', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["Patient ID", "Nodule No", "SOP UID"])

        xml_files = []
        print("Scanning for XML files...")
        for dirpath, _, filenames in os.walk(root_dir):
            for file in filenames:
                if file.endswith('.xml'):
                    xml_files.append(os.path.join(dirpath, file))

        total_files = len(xml_files)
        print(f"Found {total_files} XML files to process")

        for i, xml_path in enumerate(tqdm(xml_files, desc="Processing XML files", unit="file")):
            output_dir = os.path.join(os.path.dirname(xml_path), os.path.splitext(os.path.basename(xml_path))[0])
            ensure_directory(output_dir)

            patient_id = f"LIDC-IDRI-{os.path.basename(xml_path).split('.')[0]}"  # 病人編號格式

            tqdm.write(f"\nProcessing XML {i+1}/{total_files}: {xml_path}")
            try:
                start_time = datetime.now()
                slices = parse_xml_to_coco_format(xml_path)
                export_roi_info_and_images(slices, xml_path, output_dir, root_sop_dir, csv_writer, patient_id)
                processing_time = datetime.now() - start_time
                tqdm.write(f"Completed processing {xml_path} in {processing_time}")
                tqdm.write(f"Found {len(slices)} unique slices with nodules")
            except Exception as e:
                tqdm.write(f"Error processing XML {xml_path}: {e}")

if __name__ == "__main__":
    main()
