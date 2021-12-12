from osgeo import gdal
from osgeo import ogr
import numpy as np
import os
from tqdm import tqdm


def creat_distance_heatmap(raster_imagery,
                           vector_point_file,
                           distance_heatmap,
                           scale=10):
    '''
    Args:
     raster_imagery: raster imagery root (geoinformation is required)
     vector_point_file: the file root recording POIs (*.shp, *.csv, *.txt are supported)
     distance_heatmap: output file root
     scale: zooming scale
    '''
    dataset = gdal.Open(raster_imagery)
    adfGeoTransform = list(dataset.GetGeoTransform())
    adfGeoTransform[1] *= scale
    adfGeoTransform[5] *= scale
    adfProjection = dataset.GetProjection()
    nXsize = dataset.RasterXSize
    nYsize = dataset.RasterYSize
    R_matrix = np.array([[adfGeoTransform[1], adfGeoTransform[2]],
                         [adfGeoTransform[4], adfGeoTransform[5]]])
    b_matrix = np.array([adfGeoTransform[0], adfGeoTransform[3]])
    if vector_point_file[-3:] == 'shp':
        ds = ogr.Open(vector_point_file, 0)
        layer = ds.GetLayer()
        feature = layer.GetNextFeature()
        arrPOI = []
        while feature:
            geom = feature.GetGeometryRef()
            X = str(geom.GetX())
            Y = str(geom.GetY())
            arrPOI.append([X, Y])
            feature.Destroy()
            feature = layer.GetNextFeature()
        #清除DataSource缓存并关闭TXT文件
        ds.Destroy()
        arrPOI = np.array(arrPOI)
    elif vector_point_file[-3:] in ['txt', 'csv']:
        arrPOI = np.loadtxt(open(vector_point_file, 'r', encoding='utf-8'),
                            float,
                            delimiter=",",
                            skiprows=1,
                            usecols=(0, 1))
    else:
        print("Fial!")
        return 0
    min_distance_map = np.zeros((nYsize // scale, nXsize // scale))
    for i in tqdm(range(nXsize // scale)):
        # row = []
        for j in range(nYsize // scale):
            pixel = (R_matrix @ np.array([i, j]).T + b_matrix.T).T
            min_distance_map[j, i] = (((pixel - arrPOI)**2).sum(1)**0.5).min()

    driver = gdal.GetDriverByName("GTiff")
    if os.path.exists(distance_heatmap) and os.path.isfile(distance_heatmap):
        os.remove(distance_heatmap)
    output = driver.Create(distance_heatmap, nXsize // scale, nYsize // scale,
                           1, gdal.GDT_Float32)
    output.SetGeoTransform(adfGeoTransform)
    output.SetProjection(adfProjection)
    output.GetRasterBand(1).WriteArray(min_distance_map)
    del output


if __name__ == '__main__':
    pass
