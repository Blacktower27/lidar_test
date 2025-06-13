#!/usr/bin/env python
import rospy
import numpy as np
import tf
from geometry_msgs.msg import PoseStamped, TransformStamped
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from collections import deque
import transforms3d as tf3
import std_msgs.msg
import tf2_ros

import message_filters


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
        rospy.loginfo_throttle(1.0, "Received pose: [%.2f, %.2f, %.2f] orientation: [%.2f, %.2f, %.2f, %.2f]",
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


class SynchronizedPointCloudProcessor:
    def __init__(self, pose_manager):
        self.pose_manager = pose_manager
        self.pc_pub = rospy.Publisher("/digit/global_points", PointCloud2, queue_size=1)
        self.points = np.empty((0, 4))

        self.offsets = {
            'lidar':    [0.02, 0.0, 0.49],
            'shoulder': [0.093981, 0.0225, 0.426449],
            'torso':    [0.0305, 0.025, -0.03268]
        }

        lidar_sub    = message_filters.Subscriber("/velodyne_points", PointCloud2)
        shoulder_sub = message_filters.Subscriber("/digit/shoulder_points", PointCloud2)
        torso_sub    = message_filters.Subscriber("/digit/torso_points", PointCloud2)

        ats = message_filters.ApproximateTimeSynchronizer(
            [lidar_sub, shoulder_sub, torso_sub],
            queue_size=10, slop=0.05)
        ats.registerCallback(self.synced_callback)

    def synced_callback(self, lidar_msg, shoulder_msg, torso_msg):
        pose = self.pose_manager.get_latest_pose()
        q_ros = pose.quat  # [x, y, z, w]
        q_tf3 = [q_ros[3], q_ros[0], q_ros[1], q_ros[2]]  # [w, x, y, z] for transforms3d
        R = tf3.euler.quat2mat(q_tf3)
        tf_World2Body = tf3.affines.compose(np.array(pose.pos), R, np.ones(3))

        # LiDAR
        lidar_points = np.array(list(pc2.read_points(lidar_msg, field_names=("x", "y", "z"), skip_nans=True)))
        tf_Body2LiDAR = tf3.affines.compose(np.array(self.offsets["lidar"]), np.eye(3), np.ones(3))
        tf_World2LiDAR = np.dot(tf_World2Body, tf_Body2LiDAR)
        lidar_homog = np.hstack((lidar_points, np.ones((lidar_points.shape[0], 1))))
        lidar_world = np.dot(tf_World2LiDAR, lidar_homog.T).T

        # Shoulder
        shoulder_points = np.array(list(pc2.read_points(shoulder_msg, field_names=("x", "y", "z"), skip_nans=True)))
        shoulder_offset = np.array(self.offsets["shoulder"])
        shoulder_euler_deg = [0, -45, 0]
        shoulder_rpy = np.radians(shoulder_euler_deg)
        shoulder_rot = tf3.euler.euler2mat(*shoulder_rpy)
        tf_Body2Shoulder = tf3.affines.compose(shoulder_offset, shoulder_rot, np.ones(3))
        tf_World2Shoulder = np.dot(tf_World2Body, tf_Body2Shoulder)
        shoulder_homog = np.hstack((shoulder_points, np.ones((shoulder_points.shape[0], 1))))
        shoulder_world = np.dot(tf_World2Shoulder, shoulder_homog.T).T

        # Torso
        torso_points = np.array(list(pc2.read_points(torso_msg, field_names=("x", "y", "z"), skip_nans=True)))
        torso_offset = np.array(self.offsets["torso"])
        torso_euler_deg = [0, -45, 0]
        torso_rpy = np.radians(torso_euler_deg)
        torso_rot = tf3.euler.euler2mat(*torso_rpy)
        tf_Body2Torso = tf3.affines.compose(torso_offset, torso_rot, np.ones(3))
        tf_World2Torso = np.dot(tf_World2Body, tf_Body2Torso)
        torso_homog = np.hstack((torso_points, np.ones((torso_points.shape[0], 1))))
        torso_world = np.dot(tf_World2Torso, torso_homog.T).T

        self.points = np.vstack((self.points, lidar_world[:, :4], shoulder_world[:, :4], torso_world[:, :4]))

        tf_Body2World = np.linalg.inv(tf_World2Body)
        homog_body = np.dot(tf_Body2World, self.points.T).T

        points_out = homog_body[:, :3].tolist()
        header = std_msgs.msg.Header()
        header.stamp = pose.stamp
        header.frame_id = "base_footprint"
        cloud = pc2.create_cloud_xyz32(header, points_out)
        self.pc_pub.publish(cloud)

        self.points = np.empty((0, 4))


class RobotTFBroadcaster:
    def __init__(self, pose_manager):
        self.pose_manager = pose_manager
        self.br = tf2_ros.TransformBroadcaster()
        self.timer = rospy.Timer(rospy.Duration(0.02), self.broadcast_tf)  # 50Hz

    def broadcast_tf(self, event):
        pose = self.pose_manager.get_latest_pose()
        if pose is None:
            return

        # odom -> base_footprint
        tf_base = TransformStamped()
        tf_base.header.stamp = rospy.Time.now()
        tf_base.header.frame_id = "odom"
        tf_base.child_frame_id = "base_footprint"
        tf_base.transform.translation.x = pose.pos[0]
        tf_base.transform.translation.y = pose.pos[1]
        tf_base.transform.translation.z = pose.pos[2]
        q = pose.quat  # Already in ROS order [x, y, z, w]
        tf_base.transform.rotation.x = q[0]
        tf_base.transform.rotation.y = q[1]
        tf_base.transform.rotation.z = q[2]
        tf_base.transform.rotation.w = q[3]

        tf_proj = TransformStamped()
        tf_proj.header.stamp = tf_base.header.stamp
        tf_proj.header.frame_id = "odom"
        tf_proj.child_frame_id = "base_projected"
        tf_proj.transform.translation.x = pose.pos[0]
        tf_proj.transform.translation.y = pose.pos[1]
        tf_proj.transform.translation.z = 0.0
        tf_proj.transform.rotation.x = 0.0
        tf_proj.transform.rotation.y = 0.0
        tf_proj.transform.rotation.z = 0.0
        tf_proj.transform.rotation.w = 1.0

        self.br.sendTransform(tf_base)
        self.br.sendTransform(tf_proj)

if __name__ == '__main__':
    rospy.init_node('testing_node', anonymous=True)

    pose_manager = PoseManager()
    processor = SynchronizedPointCloudProcessor(pose_manager)
    tf_broadcaster = RobotTFBroadcaster(pose_manager)

    rospy.loginfo("Digit pointcloud synchronizer with message_filters started.")
    rospy.spin()
