import g2o
import numpy as np

class BundleAdjustment(g2o.SparseOptimizer):
    def __init__(self, cam_K):
        super().__init__()
        solver = g2o.BlockSolverSE3(g2o.LinearSolverPCGSE3())
        solver = g2o.OptimizationAlgorithmLevenberg(solver)
        super().set_algorithm(solver)
        
        cam = g2o.CameraParameters(cam_K[0, 0], cam_K[:2, 2].ravel(), 0)
        cam.set_id(0)
        super().add_parameter(cam)
    
        self._fixed_cams = 0

    def optimize(self, max_iterations=10, verbose=True):
        if self._fixed_cams == 0:
            raise RuntimeError("At least one camera need to be fixed")
            
        super().initialize_optimization()
        super().set_verbose(verbose)
        super().optimize(max_iterations)
        

    def add_pose(self, pose_id, Rcw, tcw, fixed=False):
        # pose = g2o.SE3Quat(Tcw[:3, :3], Tcw[:3, 3])
        pose = g2o.SE3Quat(Rcw, tcw)

        v_se3 = g2o.VertexSE3Expmap()
        v_se3.set_id(pose_id * 2)   # internal id
        v_se3.set_estimate(pose)
        v_se3.set_fixed(fixed)
        
        if fixed:
            self._fixed_cams += 1
        
        super().add_vertex(v_se3) 

    def add_point(self, point_id, point, fixed=False, marginalized=True):
        v_p = g2o.VertexPointXYZ()
        v_p.set_id(point_id * 2 + 1)
        v_p.set_estimate(point)
        v_p.set_marginalized(marginalized)
        v_p.set_fixed(fixed)
        super().add_vertex(v_p)
        return v_p

    def add_edge(self, point_id, pose_id, 
            measurement,
            information=np.identity(2),
            robust_kernel=g2o.RobustKernelHuber(np.sqrt(5.991))):   # 95% CI

        edge = g2o.EdgeProjectXYZ2UV() # todo: EdgeSE3ProjectXYZ 
        edge.set_vertex(0, self.vertex(point_id * 2 + 1))
        edge.set_vertex(1, self.vertex(pose_id * 2))
        edge.set_measurement(measurement)   # projection
        edge.set_information(information)
        edge.set_parameter_id(0, 0)
        
        if robust_kernel is not None:
            edge.set_robust_kernel(robust_kernel)
        super().add_edge(edge)
        return edge

    def get_edge_costs(self):
        edges = self.edges()
        cost = 0
        for e in edges:
            e.compute_error()
            cost += np.linalg.norm(e.error())
        return cost
        
    def get_pose(self, pose_id):
        return self.vertex(pose_id * 2).estimate()

    def get_point(self, point_id):
        return self.vertex(point_id * 2 + 1).estimate()