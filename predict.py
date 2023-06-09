import os
import re
import glob
import threading
import numpy as np
import concurrent.futures

import rasterio
from rasterio.windows import Window

import torch
import torch.nn.parallel
import loaddata_predict

from tqdm import tqdm
from models import modules, net, resnet, densenet, senet

IMAGENET_STATS = {'mean': [0.485, 0.456, 0.406],
                    'std': [0.229, 0.224, 0.225]}


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]  



def define_model(is_resnet, is_densenet, is_senet):
    if is_resnet:
        original_model = resnet.resnet50(pretrained = True)
        Encoder = modules.E_resnet(original_model) 
        model = net.model(Encoder, num_features=2048, block_channel = [256, 512, 1024, 2048])
    if is_densenet:
        original_model = densenet.densenet161(pretrained=True)
        Encoder = modules.E_densenet(original_model)
        model = net.model(Encoder, num_features=2208, block_channel = [192, 384, 1056, 2208])
    if is_senet:
        original_model = senet.senet154(pretrained=None)
        Encoder = modules.E_senet(original_model)
        model = net.model(Encoder, num_features=2048, block_channel = [256, 512, 1024, 2048])
        
        # original_model = senet.senet154(pretrained='imagenet')
        # Encoder = modules.E_senet(original_model)
        # model = net.model(Encoder, num_features=2048, block_channel = [256, 512, 1024, 2048])
    return model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def normalize(images, means, stds):
    for t, m, s in zip(images, means, stds):
        t.sub_(m).div_(s)
    return images.float()

def predict_win_thay(image_detect, model):
    model.to("cuda")
    image_detect = np.array(image_detect/255.0, dtype=np.float32).transpose(2,0,1)
    image_detect = np.array([image_detect])
    image_detect=torch.from_numpy(image_detect)
    image_detect=normalize(image_detect, IMAGENET_STATS['mean'], IMAGENET_STATS['std'])
    # test_loader = loaddata_predict.getTestingData(1,image_detect)
    image_detect = image_detect.to(device)
    # print(image_detect.shape, "z"*100)
    with torch.no_grad():
        output = model(image_detect)

    output = torch.nn.functional.interpolate(output,size=(440,440),mode='bilinear')
    output = output.detach().cpu().numpy()
    output = output*100
    output=output.squeeze()
    return output


def predict_win_oke(image_detect, model):
    # cai nay predict dk model pretrain"
    test_loader = loaddata_predict.getTestingData(1,image_detect)
    for i, sample_batched in enumerate(test_loader):
        image = sample_batched['image']
        image = image.cuda()
        output = model(image)
        output = torch.nn.functional.interpolate(output,size=(440,440),mode='bilinear')
        output = output.detach().cpu().numpy()
        
        output = output*100000
        output = np.clip(output, 0, 50000).astype(np.uint16)
        output=output.squeeze()
    return output


def predict_win_a(image_detect, model):
    """ham nay de predict model minh luyen ra"""
    test_loader = loaddata_predict.getTestingData_1(1,image_detect)
    for i, sample_batched in enumerate(test_loader):
        image = sample_batched['image']
        # import tifffile
        # import time
        # tifffile.imwrite(f'/home/skm/SKM16/3D/3D/xoa/image_{time.time()}.tiff', image.numpy())
        image = image.cuda()
        # print(image)
        # image = torch.autograd.Variable(image)
        output = model(image)
        output = torch.nn.functional.interpolate(output,size=(440,440),mode='bilinear')
        output = output.detach().cpu().numpy()
        output = output*100000
        output = np.clip(output, 0, 50000).astype(np.uint16)
        output=output.squeeze()
    return output


def predict_win(image_detect, model):
    """ham nay de predict custom"""
    test_loader = loaddata_predict.getTestingData_1(1,image_detect)
    for i, sample_batched in enumerate(test_loader):
        image = sample_batched['image']
        # import tifffile
        # import time
        # tifffile.imwrite(f'/home/skm/SKM16/3D/3D/xoa/image_{time.time()}.tiff', image.numpy())
        image = image.cuda()
        # print(image)
        # image = torch.autograd.Variable(image)
        output = model(image)
        output = torch.nn.functional.interpolate(output,size=(440,440),mode='bilinear')
        output = output.detach().cpu().numpy()
        output = output*100
        output=output.squeeze()
    return output


def get_quantile_schema(img):
    pass

def predict_all(model, path_image, path_predict, size=512, num_bands=3, crop_size=300):
    # fp_model='/home/skm/SKM/WORK/ALL_CODE/WorkSpace_SKM_DucAnh/IMELE/adjust_ducanh/adjust_ducanh_model_99.pth.tar'
    # path_image = "/home/skm/SKM16/3D/3D/data_dsm/RGB_54366543.tif"
    # path_predict = "/home/skm/SKM16/3D/3D/data_dsm/rs/a.tif"
    
    # model = define_model(is_resnet=False, is_densenet=False, is_senet=True)
    # state_dict = torch.load(fp_model)['state_dict']
    # model.load_state_dict(state_dict)
    # model.eval()
    # model.cuda()
    
    
    qt_scheme = get_quantile_schema(path_image)
    with rasterio.open(path_image) as raster:
        meta = raster.meta
        meta.update({'count': 1, 'nodata': 0, "dtype":"float32"})
        height, width = raster.height, raster.width
        input_size = size
        # stride_size = input_size - input_size //4
        # padding = int((input_size - stride_size) / 2)
        
        stride_size = crop_size
        padding = int((input_size - stride_size) / 2)

        list_coordinates = []
        for start_y in range(0, height, stride_size):
            for start_x in range(0, width, stride_size):
                x_off = start_x if start_x==0 else start_x - padding
                y_off = start_y if start_y==0 else start_y - padding

                end_x = min(start_x + stride_size + padding, width)
                end_y = min(start_y + stride_size + padding, height)

                x_count = end_x - x_off
                y_count = end_y - y_off
                list_coordinates.append(tuple([x_off, y_off, x_count, y_count, start_x, start_y]))
        with rasterio.open(path_predict,'w+', **meta, compress='lzw') as r:
            read_lock = threading.Lock()
            write_lock = threading.Lock()

            def process(coordinates):
                x_off, y_off, x_count, y_count, start_x, start_y = coordinates
                read_wd = Window(x_off, y_off, x_count, y_count)
                with read_lock:
                    values = raster.read(window=read_wd)[0:num_bands]

                image_detect = values.transpose(1,2,0)
                
                img_temp = np.zeros((input_size, input_size, image_detect.shape[2]))
                mask = np.pad(np.ones((stride_size, stride_size)), ((padding, padding),(padding, padding)))
                shape = (stride_size, stride_size)
                
                if y_count < input_size or x_count < input_size:
                    img_temp = np.zeros((input_size, input_size, image_detect.shape[2]))
                    mask = np.zeros((input_size, input_size))
                    if start_x == 0 and start_y == 0:
                        img_temp[(input_size - y_count):input_size, (input_size - x_count):input_size] = image_detect
                        mask[(input_size - y_count):input_size-padding, (input_size - x_count):input_size-padding] = 1
                        shape = (y_count-padding, x_count-padding)
                    elif start_x == 0:
                        img_temp[0:y_count, (input_size - x_count):input_size] = image_detect
                        if y_count == input_size:
                            mask[padding:y_count-padding, (input_size - x_count):input_size-padding] = 1
                            shape = (y_count-2*padding, x_count-padding)
                        else:
                            mask[padding:y_count, (input_size - x_count):input_size-padding] = 1
                            shape = (y_count-padding, x_count-padding)
                    elif start_y == 0:
                        img_temp[(input_size - y_count):input_size, 0:x_count] = image_detect
                        if x_count == input_size:
                            mask[(input_size - y_count):input_size-padding, padding:x_count-padding] = 1
                            shape = (y_count-padding, x_count-2*padding)
                        else:
                            mask[(input_size - y_count):input_size-padding, padding:x_count] = 1
                            shape = (y_count-padding, x_count-padding)
                    else:
                        img_temp[0:y_count, 0:x_count] = image_detect
                        mask[padding:y_count, padding:x_count] = 1
                        shape = (y_count-padding, x_count-padding)

                    image_detect = img_temp
                mask = (mask!=0)
                

                if np.count_nonzero(image_detect) > 0:
                    if len(np.unique(image_detect)) <= 2:
                        pass
                    else:
                        # print(image_detect,"x")
                        y_pred = predict_win(image_detect, model)
                        # print(y_pred.shape,"x")
                        a = int((y_pred.shape[0] - stride_size)/2)
                        b = y_pred.shape[0] - a
                        y_pred = np.array([y_pred[a:b, a:b]])
                        # print(y_pred.shape, a, b,"z")
                        with write_lock:
                            r.write(y_pred, window=Window(start_x, start_y, shape[1], shape[0]))
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                results = list(tqdm(executor.map(process, list_coordinates), total=len(list_coordinates)))


def main(model, fp_img, fp_out_fp, size_model, crop_size):
    # model.cuda()
    fn = os.path.basename(fp_out_fp)
    fp_out_dir = fp_out_fp.replace("/"+fn, "")
    os.makedirs(fp_out_dir, exist_ok=True)
    fp_out = fp_out_fp
    
    if fp_img != fp_out:
        predict_all(model, fp_img, fp_out, size=size_model, num_bands=3, crop_size = crop_size)

if __name__=="__main__":
    import time
    x = time.time()
    # fp_model='/home/skm/SKM/WORK/ALL_CODE/WorkSpace_SKM_DucAnh/IMELE/adjust_ducanh/adjust_ducanh_model_98.pth.tar'
    fp_model = '/home/skm/SKM/WORK/ALL_CODE/WORK/IMELE/pretrain/Block0_skip_model_110.pth.tar'
    # fp_img = "/home/skm/SKM16/3D/3D/Test_mau_chuan_9.tif"
    # dir_out = r"/home/skm/SKM16/3D/3D/test_3"
    
    fp_img = "/home/skm/SKM16/3D/3D/DataSkymap/reize_015/RGB_54366543_resize_015m.tif"
    fp_img = "/home/skm/SKM/WORK/ALL_CODE/WORK/IMELE/test/RGB_54366543.tif"
    dir_out = r"/home/skm/SKM16/3D/3D/DataSkymap/reize_015/Rs_500_400_goc"
    
    fn = os.path.basename(fp_img).replace('.tif','')
    os.makedirs(dir_out, exist_ok=True)
    size_model = 500
    crop_size = 400
    
    fn_model = os.path.basename(fp_model).replace('.pth.tar','')
    fp_out_dir = os.path.join(dir_out, f'RS_{crop_size}_{fn_model}_{fn}_{x}.tif')
    
    # os.remove(fp_out_dir)
    model = define_model(is_resnet=False, is_densenet=False, is_senet=True)
    model.to(device)
    state_dict = torch.load(fp_model)['state_dict']
    model.load_state_dict(state_dict, strict=False)
    
    main(model, fp_img, fp_out_dir, size_model, crop_size)