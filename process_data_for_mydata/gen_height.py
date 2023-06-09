import rasterio

fp_dsm = r"/home/skm/SKM16/3D/3D/DataSkymap/origin/DSM_53391530.tif"
fp_dtm = r"/home/skm/SKM16/3D/3D/DataSkymap/origin/DTM_53391530.tif"
fp_height = r"/home/skm/SKM16/3D/3D/DataSkymap/origin/Gen_height/height_53391530.tif"

def gen_height(fp_dsm, fp_dtm, fp_height):
    with rasterio.open(fp_dsm) as src_dsm:
        dsm = src_dsm.read()
        meta = src_dsm.meta

    with rasterio.open(fp_dtm) as src_dtm:
        dtm = src_dtm.read()

    height = dsm - dtm
    height[height<1] = 0
    with rasterio.open(fp_height, 'w', **meta) as dst:
        dst.write(height)

if __name__=="__main__":
    print('oke')
    gen_height(fp_dsm, fp_dtm, fp_height)