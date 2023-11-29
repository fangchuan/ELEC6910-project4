# Project-4 Structure-from-Motion

## 0. Implementation details
1. feature extraction: SuperPoint
2. feature matching: SuperGlue and knn match(based on superpoint descriptors' similarity)
3. triangulation: DLT(self-implemented)
4. pose estimation: PnP(opencv)
5. bundle adjustment: g2o

## 1. Run the code:

1. `pip install -r requirement.txt`
2. `cd python`
3. reconstruct llff-fern:  `python main.py --config configs/llff_fern_superpoint_superglu.yaml`
4. Reconstruct llff-trex: `python main.py --config configs/llff_trex_superpoint_superglu.yaml`
5. Reconstruct templeRing: `python main.py --config configs/templering_superpoint_superglu.yaml`

6. If you want to visualize the resulting point clouds and camera pose, run `main.ipynb` by Jupyter.

## 2. Camera trajectory evaluation

1. rotation evaluation using evo_rpe:  evo_rpe tum xxx_gt_trajectory.txt xxx_est_trajectory.txt -asp -r angle_deg

   * Llff-fern:

     Rotation_rmse in degree: $1.77^\circ$

   * Llff-trex:

     Rotation_rmse in degree: $0.83^\circ$

2. translation evaluation using evo_rpe: evo_rpe tum xxx_gt_trajectory.txt xxx_est_trajectory.txt -asp -r trans_part

   * Llff-fern:

     Translation_rmse in meter: $0.074$

   * Llff-trex:

     Translation_rmse in meter: $0.078$

     

## 3. Result Analysis

1. Visualization results using ipvolume:

   the red/blue wireframes are cameras(the red one is the final camera pose), white points represent point clouds.

   * Llff-fern:

     ![image-20231129170250562](/Users/fc/Desktop/学习/ELEC6910/projects/project4/assets/image-20231129170250562.png)

   * Llff-trex:

     ![image-20231129170836912](/Users/fc/Desktop/学习/ELEC6910/projects/project4/assets/image-20231129170836912.png)

   * templeRing:

     ![image-20231129165915545](/Users/fc/Desktop/学习/ELEC6910/projects/project4/assets/image-20231129165915545.png)



2. Analysis:
   * Camera poses are heavily drifted on the templeRing data: Since we adopt a incremental method to estimate the camera pose and point clouds, the error will be accumulated. If a loop closure method is applied, the error will be reduced.
   * The sparse point cloud is still inferior to that of colmap: One of the reasons is that the feature extraction and matching is not accurate enough. If we use a more powerful feature extractor and matcher, the result will be better. Another reason is that my DLT triangulation method is not accurate enough, which will be improved to midpoint triangulation in the future.
   * The mechanism of keyframe selection is not perfect: Currently, I only select the keyframe based on a fix-number interval of input frames. If we use a more sophisticated method, the result should be improved.
   * Non-linear optimization should be applied after each step of triangulation and PnP.


## 4. Reference
[Hierarchical Localization at Large Scale with Superpoint Graphs](https://arxiv.org/pdf/2007.01578.pdf)
[Python-VO](https://github.com/Shiaoming/Python-VO)
[3D Reconstruction using Structure from Motion](https://github.com/harish-vnkt/structure-from-motion)