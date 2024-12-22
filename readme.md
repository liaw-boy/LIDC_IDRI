# LIDC-IDRI Mark the Location of Lung Nodules
這份專案是用來標記肺結節所在位置，根據LIDC-IDRI資料庫中所提供的CT(.dcm)以及xml(有四位醫生針對切片的描述)，將座標值轉為COCO.json形式，並與原圖使用相同UID(用來辨識CT的唯一性)作為檔名，提供給roboflow標記
