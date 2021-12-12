# UDF4UFZ
This repository is the code of "A Unified Deep Learning Framework for Urban Functional Zone Extraction Based on Multi-source Heterogeneous Data"

## Dependency
- Pytorch
- TorchVision
- osgeo
- numpy

## Essential classes and function

### models.py
![Flowchart of the proposed unified deep learning framework for UFZ extraction](https://github.com/SalviaL/UDF4UFZ/blob/main/figures/fig_framework.png)
The models are in file "models.py". There are two essential classes in the file. the `complementary_fusion` and the `spatial_information_modeling`.
This two classes is corresponding to two parts in our paper.

### distance_heatmap.py
![Illustration of converting POIs into a hierarchical distance heatmap tensor](https://github.com/SalviaL/UDF4UFZ/blob/main/figures/fig_h_map.png)
The file contains the function of creating distance heatmap (`creat_distance_heatmap`). The inputs are
- a raster imagery with geoinformation, which implies the processing extend,
- a file contains POIs,
- the output root, and
- a zooming scale (default 10).
You can also create the distance heatmap by ArcGIS (in ArcToolbox/Spatial Analyst Tools/Distance/Euclidean Distance)

If any problem, please don't hesitate to contact us via salvial@csu.edu.cn.
