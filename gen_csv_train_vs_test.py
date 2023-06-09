import os
import glob
import pandas as pd

dir_img_RGB = r"/home/skm/SKM/WORK/ALL_CODE/WorkSpace_SKM_DucAnh/IMELE/data/contest/test_rgbs_base"
dir_img_height = r'/home/skm/SKM/WORK/ALL_CODE/WorkSpace_SKM_DucAnh/IMELE/data/contest/test_heights_adjust'
out_csv = r'/home/skm/SKM/WORK/ALL_CODE/WorkSpace_SKM_DucAnh/IMELE/dataset/test_DucAnh.csv'

remove_path = r'/home/skm/SKM/WORK/ALL_CODE/WorkSpace_SKM_DucAnh/IMELE/'

list_all_img = glob.glob(os.path.join(dir_img_RGB, "*.tif"))
list_ok_img = []
list_ok_img_height = []
check=True
for fp in list_all_img:
    fn_RGB = os.path.basename(fp)
    fn_ok_heght = fn_RGB.replace("_RGB_", "_height_")
    fn_ok_heght = fn_ok_heght.replace(".tif", ".png")
    
    fp_tmp_RGB = os.path.join(dir_img_RGB.replace(remove_path,''), fn_RGB)
    fp_tmp_height = os.path.join(dir_img_height.replace(remove_path,''), fn_ok_heght)
    if os.path.isfile(os.path.join(remove_path, fp_tmp_height)):
        list_ok_img.append(fp_tmp_RGB)
        list_ok_img_height.append(fp_tmp_height)
    else:
        print(f"khong co file: {os.path.join(remove_path, fp_tmp_height)}")
        check = False
        break
    
if check:
    rows = list(zip(list_ok_img, list_ok_img_height))
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False, header=False)
