from ROAR.agent_module.agent import Agent
from ROAR.control_module.pid_controller import PIDController
from pathlib import Path
from ROAR.perception_module.depth_to_pointcloud_detector import DepthToPointCloudDetector
from ROAR.utilities_module.data_structures_models import SensorsData
from ROAR.utilities_module.vehicle_models import Vehicle, VehicleControl
from ROAR.configurations.configuration import Configuration as AgentConfig
import numpy as np
from ROAR.utilities_module.data_structures_models import Transform, Location
from typing import Optional, List
import open3d as o3d
from collections import deque
from datetime import datetime
import math
import itertools

class Lidar106Agent(Agent):
    def __init__(self, vehicle: Vehicle, agent_settings: AgentConfig, **kwargs):
        super().__init__(vehicle, agent_settings, **kwargs)
        self.prev_steerings: deque = deque(maxlen=10)
        self.controller = PIDController(agent=self, steering_boundary=[-1, 1], throttle_boundary=[-1, 1])

        self.closeness_threshold = 0.4

        # # initialize open3d related content
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(width=500, height=500)
        self.pcd = o3d.geometry.PointCloud()
        # create a 3D coordinate system
        self.coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
        self.points_added = False
        
        self.depth_to_pcd = DepthToPointCloudDetector(agent=self)

        self.has_obstacle = False

    def run_step(self, sensors_data: SensorsData, vehicle: Vehicle) -> VehicleControl:
        super(Lidar106Agent, self).run_step(sensors_data, vehicle)
        if self.front_depth_camera.data is not None and self.front_rgb_camera.data is not None:        
            try :
                
                self.pcd = self.depth_to_pcd.run_in_series(self.front_depth_camera.data,
                                                            self.front_rgb_camera.data)

                folder_name = Path("./data/Lidar_pointcloud")
                folder_name.mkdir(parents=True, exist_ok=True)
                o3d.io.write_point_cloud((folder_name / f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.pcd").as_posix(),
                                            self.pcd, print_progress=True)
                
                self.non_blocking_pcd_visualization(pcd=self.pcd, should_center=True,
                                                    should_show_axis=True, axis_size=1)

                next_way = self.target_point(self.pcd)

                error = self.find_error_at(target_loc=next_way,
                                            vehicle=self.vehicle,
                                            error_scaling=[
                                                    (1.1, 0.1),
                                                    (1.07, 0.75),
                                                    (1.05, 0.8),
                                                    (1.03, 0.9),
                                                    (1.01, 0.95),
                                                    (1, 1)
                                            ])
        
                if error is None:
                    neutral = -90
                    incline = self.vehicle.transform.rotation.pitch - neutral
                    if incline < -10:
                        long_control = self.controller.long_pid_controller()
                        self.vehicle.control.throttle = long_control
                        return self.vehicle.control
                    else:
                        return self.execute_prev_command()

                self.kwargs["lat_error"] = error
                self.vehicle.control = self.controller.run_in_series(next_waypoint=next_way)
                self.prev_steerings.append(self.vehicle.control.steering)

                self.has_obstacle = False
                
                return self.vehicle.control
            except Exception as e:
                print(e)
        else: 
            return VehicleControl()
            

    @staticmethod
    def load_data(file_path: str) -> List[Transform]:
        waypoints = []
        f = Path(file_path).open('r')
        for line in f.readlines():
            x, y, z = line.split(",")
            x, y, z = float(x), float(y), float(z)
            l = Location(x=x, y=y, z=z)
            waypoints.append(Transform(location=l))
        return waypoints

    def filter_ground(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        # pcd = self.pointcloud_detector

        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        # height and distance filter
        # 0 -> left and right | 1 -> up and down | 2 = close and far
        points_of_interest = np.where((points[:, 1] < 0.3))
        points = points[points_of_interest]
        colors = colors[points_of_interest]
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        """
        ransac_dist_threshold: RANSAC distance threshold
        ransac_n: RANSAC starting number of points
        ransac_itr: RANSAC number of iterations
        """
        plane_model, inliers = pcd.segment_plane(distance_threshold = 0.01,
                                                 ransac_n = 3,
                                                 num_iterations = 100)

        pcd: o3d.geometry.PointCloud = pcd.select_by_index(inliers)
        pcd = pcd.voxel_down_sample(0.01)
        return pcd


    def filter_object(self, pcd: o3d.geometry.PointCloud, min_x, max_x, min_y, max_y) -> o3d.geometry.PointCloud:

        # points = np.asarray(pcd.points)
        # colors = np.asarray(pcd.colors)
        # # height and distance filter
        # # 0 -> left and right | 1 -> up and down | 2 = close and far
        # points_of_interest = np.where(np.logical_and(points[:, 0] > min_x, 
        #                                             points[:, 0] < max_x,
        #                                             points[:, 1] > min_y,
        #                                             points[:, 1] < max_y,
        #                                             points[:, 2] < -0.4))
        # points = points[points_of_interest]
        # colors = colors[points_of_interest]
        # pcd.points = o3d.utility.Vector3dVector(points)
        # pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # """
        # ransac_dist_threshold: RANSAC distance threshold
        # ransac_n: RANSAC starting number of points
        # ransac_itr: RANSAC number of iterations
        # """
        # plane_model, inliers = pcd.segment_plane(distance_threshold = 0.01,
        #                                          ransac_n = 3,
        #                                          num_iterations = 100)

        # pcd: o3d.geometry.PointCloud = pcd.select_by_index(inliers)
        # pcd = pcd.voxel_down_sample(0.01)
        # return pcd

        # # create bounding box:
        bounds = [[min_x, max_x], [min_y, max_y], [-1, -1.4]] # set the bounds
        bounding_box_points = list(itertools.product(*bounds)) # create limit points
        bounding_box = o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(bounding_box_points)) # create bounding box object

        return pcd.crop(bounding_box)
        

    """
    Obstacle Avoidance Using LiDAR (Depth Data)
    """

    def find_obs_pnts(self, pcd: o3d.geometry.PointCloud):

        try: 
            # the center of the geometry coordinates
            center = pcd.get_center()
            center_x, center_y, center_z = center[0], center[1], center[2]

            x_max = max(pcd.points, key=lambda x: x[0])[0]
            x_min = min(pcd.points, key=lambda x: x[0])[0]
            z_max = max(pcd.points, key=lambda x: x[2])[2]

            # get the left most center point of the obstacle
            obs_left_front_pnt = [x_min, center_y, z_max]

            # get the right most center point of the obstacle
            obs_right_front_pnt = [x_max, center_y, z_max]

            return obs_left_front_pnt, obs_right_front_pnt
        except ValueError:
            pass
    
    def find_fen_pnts(self, pcd: o3d.geometry.PointCloud):

        try:
            # the center of the geometry coordinates
            center = pcd.get_center()
            center_x, center_y, center_z = center[0], center[1], center[2]

            x_max = max(pcd.points, key=lambda x: x[0])[0]
            x_min = min(pcd.points, key=lambda x: x[0])[0]
            z_max = max(pcd.points, key=lambda x: x[2])[2]

            # get the left most center point of the obstacle
            fen_left_front_pnt = [x_min, center_y, z_max]

            # get the right most center point of the obstacle
            fen_right_front_pnt = [x_max, center_y, z_max]

            return fen_left_front_pnt, fen_right_front_pnt
        except ValueError:
            pass

    def target_point(self, pcd: o3d.geometry.PointCloud):

        try:

            # folder_name = Path("./data/Lidar_pointcloud")
            # self.pcd = o3d.io.read_point_cloud((folder_name / f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.pcd").as_posix(),
            #                                 print_progress=True)

            THRESHOLD = 0.075

            y_min = min(self.pcd.points, key=lambda x: x[1])[1]

            fen_pcd = self.filter_object(pcd, -1, 1, y_min + THRESHOLD, 0.1)
            obs_pcd = self.filter_object(pcd, -0.3, 0.3, 0.1, 0.2)

            obs_left, obs_right, = self.find_obs_pnts(obs_pcd)
            fen_left, fen_right = self.find_fen_pnts(fen_pcd)
            car_x, car_z = self.vehicle.transform.location.x, self.vehicle.transform.location.z

            px, py, pz = 0, 0, 0

            if abs(max(obs_left[2], obs_right[2]) - car_z) < self.closeness_threshold:
                self.has_obstacle = True

            # getting the maximum gaps between obstacles and fence
            if self.has_obstacle:
                if abs(fen_left[0] - obs_left[0]) > self.closeness_threshold and abs(fen_left[0] - obs_left[0]) >= np.abs(fen_right[0] - obs_right[0]):
                    px = (fen_left[0] + obs_left[0]) / 2
                    py = fen_left[1]
                    pz = (fen_left[2] + obs_left[2]) / 2

                if abs(fen_right[0] - obs_right[0]) > self.closeness_threshold and abs(fen_right[0] - obs_right[0]) > np.abs(fen_left[0] - obs_left[0]):
                    px = (fen_right[0] + obs_right[0]) / 2
                    py = fen_right[1]
                    pz = (fen_right[2] + obs_right[2]) / 2

            else:
                px = (fen_left[0] + fen_right[0]) / 2
                py = max(fen_left[1], fen_right[1])
                pz = max(fen_pcd.points, key=lambda x: x[2])[2]
                
            target_waypoints: Transform = [px, py, pz]

            return target_waypoints
        except ValueError:
            pass
        
    def find_error_at(self, target_loc, vehicle: Vehicle, error_scaling) -> Optional[float]:
        try:
            tx, tz = target_loc[0], target_loc[2]
            vx, vz = vehicle.transform.location.x, vehicle.transform.location.z

            dx = (tx - vx) ** 2
            dz = (tz - vz) ** 2
            error = math.sqrt(dx + dz)

            for e, scale in error_scaling:
                if abs(error) <= e:
                    error = error * scale
                    break
            print("final e: " + str(error))
            return error
        except ValueError:
            pass

    def execute_prev_command(self):
        # no lane found, execute the previous control with a decaying factor
        self.logger.info("Executing prev")

        if np.average(self.prev_steerings) < 0:
            self.vehicle.control.steering = -1
        else:
            self.vehicle.control.steering = 1
        # self.logger.info("Cannot see line, executing prev cmd")
        self.prev_steerings.append(self.vehicle.control.steering)
        self.vehicle.control.throttle = self.vehicle.control.throttle
        self.logger.info(f"No Lane found, executing discounted prev command: {self.vehicle.control}")
        return self.vehicle.control

    def non_blocking_pcd_visualization(self, pcd: o3d.geometry.PointCloud,
                                        should_center=False,
                                        should_show_axis=False,
                                        axis_size: float = 1):
                        
        """
        Real time point cloud visualization.

        Args:
            pcd: point cloud to be visualized
            should_center: true to always center the point cloud
            should_show_axis: true to show axis
            axis_size: adjust axis size

        Returns:
            None

        """
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        if should_center:
            points = points - np.mean(points, axis=0)

        if self.points_added is False:
            self.pcd = o3d.geometry.PointCloud()
            self.pcd.points = o3d.utility.Vector3dVector(points)
            self.pcd.colors = o3d.utility.Vector3dVector(colors)

            if should_show_axis:
                self.coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size,
                                                                                        origin=np.mean(points,
                                                                                                        axis=0))
                self.vis.add_geometry(self.coordinate_frame)
            self.vis.add_geometry(self.pcd)
            self.points_added = True
        else:
            # print(np.shape(np.vstack((np.asarray(self.pcd.points), points))))
            self.pcd.points = o3d.utility.Vector3dVector(points)
            self.pcd.colors = o3d.utility.Vector3dVector(colors)
            if should_show_axis:
                self.coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size,
                                                                                        origin=np.mean(points,
                                                                                                        axis=0))
                self.vis.update_geometry(self.coordinate_frame)
            self.vis.update_geometry(self.pcd)

        self.vis.poll_events()
        self.vis.update_renderer()
