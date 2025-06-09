#!/usr/bin/env python
import rospy
import tf
import numpy as np
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from collections import deque
import transforms3d as tf3


class PoseData:
    def __init__(self, pos, quat, stamp, frame_id):
        self.pos = pos
        self.quat = quat
        self.stamp = stamp
        self.frame_id = frame_id


class PoseManager:
    def __init__(self, maxlen=1):
        self.pose_queue = deque(maxlen=maxlen)
        rospy.Subscriber("/robot/pose", PoseStamped, self.pose_callback)

    def pose_callback(self, msg: PoseStamped):
        p = msg.pose.position
        q = msg.pose.orientation
        quat = [q.x, q.y, q.z, q.w]
        rospy.loginfo("Received pose: [%.2f, %.2f, %.2f] with orientation [%.2f, %.2f, %.2f, %.2f]",
                      p.x, p.y, p.z, q.x, q.y, q.z, q.w)

        pose = PoseData(
            pos=[p.x, p.y, p.z],
            quat=quat,
            stamp=msg.header.stamp,
            frame_id=msg.header.frame_id
        )
        self.pose_queue.append(pose)

    def get_latest_pose(self):
        return self.pose_queue[-1] if self.pose_queue else None


class PointCloudTransformer:
    def __init__(self, pose_manager):
        self.pose_manager = pose_manager
        self.points = np.zeros((0, 4))

    def transform_pointcloud(self, points, pose: PoseData, sensor_offset):
        tf_world2body = tf3.affines.compose(np.array(pose.pos), tf3.euler.quat2mat(pose.quat), np.ones(3))
        tf_body2sensor = tf3.affines.compose(np.array(sensor_offset), np.eye(3), np.ones(3))
        tf_world2sensor = np.dot(tf_world2body, tf_body2sensor)

        pt_homog = np.hstack((points, np.ones((points.shape[0], 1))))
        pt_world = np.dot(tf_world2sensor, pt_homog.T).T
        return pt_world[:, :4]

    def process_pc(self, msg, offset):
        pose = self.pose_manager.get_latest_pose()
        if pose is None:
            rospy.logwarn("No pose data available for pointcloud transformation.")
            return

        points = np.array(list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)))
        if points.shape[0] == 0:
            rospy.logwarn("Received empty pointcloud.")
            return

        transformed = self.transform_pointcloud(points, pose, offset)
        self.points = np.vstack((self.points, transformed))
        rospy.loginfo("Transformed %d points from sensor at offset %s", len(points), str(offset))

    def lidar_callback(self, msg):
        rospy.loginfo("Received LiDAR pointcloud: width = %d, height = %d", msg.width, msg.height)
        self.process_pc(msg, offset=[0.03, 0, 0.55])

    def shoulder_callback(self, msg):
        rospy.loginfo("Received Shoulder pointcloud: width = %d, height = %d", msg.width, msg.height)
        self.process_pc(msg, offset=[0.11, 0, 0.445])

    def torso_callback(self, msg):
        rospy.loginfo("Received Torso pointcloud: width = %d, height = %d", msg.width, msg.height)
        self.process_pc(msg, offset=[0.11, 0, -0.04])


if __name__ == '__main__':
    rospy.init_node('pointcloud_tf3_transformer_deploy', anonymous=True)

    pose_manager = PoseManager()
    transformer = PointCloudTransformer(pose_manager)

    rospy.Subscriber("/velodyne_points", PointCloud2, transformer.lidar_callback)
    rospy.Subscriber("/digit/shoulder_points", PointCloud2, transformer.shoulder_callback)
    rospy.Subscriber("/digit/torso_points", PointCloud2, transformer.torso_callback)

    rospy.loginfo("Digit real-world pointcloud transformer started.")
    rospy.spin()
