import rasterio
import numpy as np
from scipy import signal



def read_win_and_chuyen_ve0_doc_vung_k_tot(img, window_size, bo_qua_pixel):
    # # Tạo một mảng numpy 2 chiều kích thước (100, 100) với các giá trị ngẫu nhiên từ 0 đến 255.
    # # arr = np.random.randint(low=0, high=9, size=(20, 16))
    # # arr = np.ones((20, 16))
    # # Thiết lập kích thước cửa sổ và khoảng cách giữa chúng.
    # window_size = 290
    # bo_qua_pixel = 20
    stride = window_size + bo_qua_pixel
    
    arr = img.copy()
    # Tạo một mảng boolean để chỉ định các vị trí cần được thay thế bằng giá trị 1.
    mask = np.zeros_like(arr, dtype=bool)

    # for i in range(0, arr.shape[0], stride):
    #     for j in range(0, arr.shape[1], stride):
    #         if i + stride >= arr.shape[0]:
    #             window_size_i = arr.shape[0] - i
    #         elif i == 0:
    #             window_size_i = window_size
    #         else:
    #             window_size_i = window_size - int(bo_qua_pixel/2)
                
    #         if j + stride >= arr.shape[1]:
    #             window_size_j = arr.shape[1] - j  
    #         elif j == 0:
    #             window_size_j = window_size
    #         else:
    #             window_size_j = window_size - int(bo_qua_pixel/2)
                
    #         mask[i:i+window_size_i, j:j+window_size_j] = True
    
    
    for i in range(int(bo_qua_pixel/2), arr.shape[0], stride):
        for j in range(int(bo_qua_pixel/2), arr.shape[1], stride):
            if i + stride >= arr.shape[0]:
                window_size_i = arr.shape[0] - i
            elif i == int(bo_qua_pixel/2) or i == 0:
                window_size_i = window_size + int(bo_qua_pixel/2)
                i = 0 
            else:
                # print("va",i)
                window_size_i = window_size
                
            if j + stride >= arr.shape[1]:
                window_size_j = arr.shape[1] - j  
            elif j == int(bo_qua_pixel/2) or j == 0 :
                window_size_j = window_size + int(bo_qua_pixel/2)
                j = 0
            else:
                window_size_j = window_size
                
            mask[i:i+window_size_i, j:j+window_size_j] = True
    
            
            
            # print(i,i+window_size_i, j,j+window_size_j)
    # for i in range(0, arr.shape[0] - window_size + 1, stride):
    #     for j in range(0, arr.shape[1] - window_size + 1, stride):
    #         mask[i:i+window_size, j:j+window_size] = True

    # Thay thế các giá trị của các cửa sổ 5x5 cách nhau 3 pixel bằng giá trị 1.
    arr[mask] = 0
    
    return arr, mask


fp_img_predict = r"/home/skm/SKM16/3D/3D/DataSkymap/reize_015/Rs_500_400/RS_400_Block0_skip_model_110_RGB_54366543_1681273066.7518811.tif"

with rasterio.open(fp_img_predict) as src:
    meta = src.meta
    img =  src.read()[0]


kernel = np.ones((5, 5))
# print(kernel)

400

# window_size = 220
# bo_qua_pixel = 80

# window_size_tin = 250
# bo_qua_pixel_tin = 50

window_size = 320
bo_qua_pixel = 80

window_size_tin = 350
bo_qua_pixel_tin = 50

arr, _ = read_win_and_chuyen_ve0_doc_vung_k_tot(img, window_size, bo_qua_pixel)
arr_pad = np.pad(arr, pad_width=((2, 2), (2, 2)), mode='edge')
mask_sua = np.divide(signal.convolve2d(arr_pad, kernel, mode='valid'), np.sum(kernel))
print(mask_sua.shape)

mask_sua, mask_bool = read_win_and_chuyen_ve0_doc_vung_k_tot(mask_sua, window_size_tin, bo_qua_pixel_tin)
img_ok, mask_bool = read_win_and_chuyen_ve0_doc_vung_k_tot(img, window_size_tin, bo_qua_pixel_tin)
mask_bool_01 = mask_bool*1
# # print(img_ok)

mask_ok = img*mask_bool_01 + mask_sua


out_file = r"/home/skm/SKM16/3D/3D/DataSkymap/reize_015/Rs_500_400/fix2.tif"
meta.update({'dtype':'float64'})
with rasterio.open(out_file, 'w', **meta) as dst:
    dst.write(np.array([mask_ok/1000]))


    
    
    