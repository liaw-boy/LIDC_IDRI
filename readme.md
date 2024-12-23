# LIDC-IDRI Mark the Location of Lung Nodules
這份專案是用來標記肺結節所在位置，根據LIDC-IDRI資料庫中所提供的CT(.dcm)以及xml(有四位醫生針對切片的描述)，將座標值轉為COCO.json形式，並與原圖使用相同UID(用來辨識CT的唯一性)作為檔名，提供給roboflow標記
文件還在慢慢編輯中.... 請耐心稍後\
資料庫來源：https://www.cancerimagingarchive.net/collection/lidc-idri/ \
資料標記&分配：https://universe.roboflow.com/lungcancerct/max_lung_img/dataset/1 
# 程式碼簡介
### movefile.py
從LIDC-IDRI下載Dataset時，我沒有只選擇CT下載，以至於同時下載了另一個Data，加上我們所需的資料需要打開兩個資料夾才到達，因此這個程式的作用是將較多的.dcm
檔案的資料夾內容物全部拖移到根目錄，並將其餘非必要的資料都刪除。 \
因為每次測試生成結果我都存在個別病人的建立一個以xml同檔名的資料夾中，每次的測試都會增減一些不同的內容物，而每當最終測試完都讓我的輸出塞了一堆垃圾，這個程式也只會刪除跟目錄底下的子目錄，不會刪到我們的原始資料。
### annallcsv.py
每次寫程式都寫很多個版本，所以看到這個命名大概也知道是多個功能的合併版本，我們的目的是提取xml中醫生給予病人所描繪的結節切片座標位置，轉換為COCO.json檔案，並同時將有病徵的切片以及.json命名為SOP_UID，使得上傳至Roboflow可以自動偵測相同檔名標記病徵區塊．
### CSV
紀錄這個SOP_UID(切片)是來自哪個病人
# Roboflow
