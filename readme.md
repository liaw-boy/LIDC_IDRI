# LIDC-IDRI Mark the Location of Lung Nodules
這份專案是用來標記肺結節所在位置，根據LIDC-IDRI資料庫中所提供的CT(.dcm)以及xml(有四位醫生針對切片的描述)，將座標值轉為COCO.json形式，並與原圖使用相同UID(用來辨識CT的唯一性)作為檔名，提供給roboflow標記
文件還在慢慢編輯中.... 請耐心稍後\
資料庫來源：https://www.cancerimagingarchive.net/collection/lidc-idri/ \
資料標記&分配：Roboflow \
# 程式碼簡介
## movefile.py
從LIDC-IDRI下載Dataset時，我沒有只選擇CT下載，以至於同時下載了另一個Data，加上我們所需的資料需要打開兩個資料夾才到達，因此這個程式的作用是將較多的.dcm
檔案的資料夾內容物全部拖移到根目錄，並將其餘非必要的資料都刪除。
## annallcsv.py
每次寫程式都寫很多個版本，所以看到這個命名大概也知道是多個功能的合併版本，
