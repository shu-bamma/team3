---
description: 2.5 D
---

# Grid Maps

We have the point cloud as the first line of input in our pipeline and the final goal is to make a 3D occupancy grid of the mapped environment to deploy our motion planning algorithms on them.

Another popular approach is to build an elevation map of the environment, where each coordinate on the horizontal plane is associated with an elevation/height value ::::: 2.5D maps.

## **2.5D maps :** _An alternative to making 3D occpancy grids_ 

{% embed url="https://github.com/ANYbotics/grid\_map" caption="Simple Implementation with pcd files" %}

Features:

* **Multi-layered:** Developed for universal 2.5-dimensional grid mapping with support for any number of layers.
* **Efficient map re-positioning:** Data storage is implemented as two-dimensional circular buffer. This allows for non-destructive shifting of the map's position \(e.g. to follow the robot\) without copying data in memory.
* **Based on Eigen:** Grid map data is stored as [Eigen](http://eigen.tuxfamily.org/) data types. Users can apply available Eigen algorithms directly to the map data for versatile and efficient data manipulation.
* **Convenience functions:** Several helper methods allow for convenient and memory safe cell data access. For example, iterator functions for rectangular, circular, polygonal regions and lines are implemented.
* **ROS interface:** Grid maps can be directly converted to and from ROS message types such as PointCloud2, OccupancyGrid, GridCells, and our custom GridMap message. Conversion packages provide compatibility with [costmap\_2d](http://wiki.ros.org/costmap_2d), [PCL](http://pointclouds.org/), and [OctoMap](https://octomap.github.io/) data types.
* **OpenCV interface:** Grid maps can be seamlessly converted from and to [OpenCV](http://opencv.org/) image types to make use of the tools provided by [OpenCV](http://opencv.org/).
* **Visualizations:** The _grid\_map\_rviz\_plugin_ renders grid maps as 3d surface plots \(height maps\) in [RViz](http://wiki.ros.org/rviz). Additionally, the _grid\_map\_visualization_ package helps to visualize grid maps as point clouds, occupancy grids, grid cells etc.
* **Filters:** The _grid\_map\_filters_ provides are range of filters to process grid maps as a sequence of filters. Parsing of mathematical expressions allows to flexibly setup powerful computations such as thresholding, normal vectors, smoothening, variance, inpainting, and matrix kernel convolutions.

## In Depth Study 

{% file src=".gitbook/assets/universalgridmaplibrarywebsiteversion.pdf" caption="Grid\_maps paper" %}

**Section 2 :**  working with the library step by step through an example of a sinosoidal function   
                                                         **\( Highly recommended \)**

\*\*\*\*

\*\*\*\*

