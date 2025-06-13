#!/usr/bin/env python3
# Copyright (c) Agility Robotics

import asyncio
import json
import select
import socket
import sys
import time
import numpy as np
import websockets
import rospy
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from rosgraph_msgs.msg import Clock
import struct
import psutil
import os
import threading
from enum import Enum
from typing import Tuple, Union

cores_to_use = [8]
pid = os.getpid()
os.sched_setaffinity(pid, cores_to_use)
print(f"Process {pid} set affinity to cores: {cores_to_use}")

ip = "10.10.1.1"
flow_control = 'framerate'
close_all = False

class StreamType(Enum):
    RGB8 = "RGB8"
    Gray8 = "Gray8"
    Depth16 = "Depth16"
    XYZ = "XYZ"
    XYZI = "XYZI",
    XYZIRT = "XYZIRT"

class ReadState(Enum):
    FIND_JSON = 'FIND_JSON'
    READ_DATA = 'READ_DATA'
    PROCESS = 'PROCESS'

def check_response_msg(msg, expected: str) -> None:
    if msg[0] != expected:
        raise ValueError(f"Response must be '{expected}'. Got: {msg[0]}")

async def start_stream(name: str) -> int:
    async with websockets.connect(f'ws://{ip}:8080', subprotocols=['json-v1-agility']) as ws:
        await ws.send(json.dumps(['perception-stream-start', {
            'stream': name,
            'flow-control': flow_control}]))
        msg_raw = await ws.recv()
        print(f"[debug] message:{msg_raw}")
        msg = json.loads(msg_raw)
    check_response_msg(msg, 'perception-stream-response')
    port = msg[1]['port']
    print(f"Successfully started {name} stream at port {port}")
    return port

def find_json(data: bytearray) -> Union[Tuple[str, bytearray], Tuple[None, bytearray]]:
    text = data.decode('latin-1')
    start = text.find('["perception-stream-frame"')
    count = 0
    for i, ch in enumerate(text[start:]):
        if ch == '[':
            count += 1
        elif ch == ']':
            count -= 1
            if count == 0:
                end = start + i + 1
                return text[start:end], data[end:]
    return None, data

class FrameInfo:
    IMG_TYPES = [StreamType.RGB8, StreamType.Gray8, StreamType.Depth16]
    PT_CLOUD_TYPES = [StreamType.XYZ, StreamType.XYZI, StreamType.XYZIRT]

    def __init__(self, json_msg):
        for k in json_msg[1]:
            setattr(self, k.replace('-', '_'),  json_msg[1][k])
        self.format = StreamType(self.format)
        if self.format == StreamType.RGB8 or self.format == StreamType.Gray8:
            self.channels = 3 if self.format == StreamType.RGB8 else 1
            self.bit_depth = np.uint8
        elif self.format == StreamType.Depth16:
            self.channels = 1
            self.bit_depth = np.uint16
        elif self.format == StreamType.XYZ:
            self.channels = 3
            self.bit_depth = np.float32
        elif self.format == StreamType.XYZI:
            self.channels = 4
            self.bit_depth = np.float32
        elif self.format == StreamType.XYZIRT:
            self.channels = 6
            self.bit_depth = np.float32
        else:
            raise ValueError(f"Unsupported format: {self.format}")

        bytes_per_channel = int(np.dtype(self.bit_depth).itemsize)
        if self.format in FrameInfo.IMG_TYPES:
            self.size = self.height * self.width * self.channels * bytes_per_channel
        elif self.format in FrameInfo.PT_CLOUD_TYPES:
            self.size = self.size * self.channels * bytes_per_channel

def numpy_to_pointcloud2(points, frame_id='sensor'):
    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = frame_id
    fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1)
    ]
    point_step = 12
    width = points.shape[0]
    point_cloud_msg = PointCloud2()
    point_cloud_msg.header = header
    point_cloud_msg.height = 1
    point_cloud_msg.width = width
    point_cloud_msg.is_dense = True
    point_cloud_msg.point_step = point_step
    point_cloud_msg.row_step = point_step * width
    point_cloud_msg.fields = fields
    point_cloud_msg.is_bigendian = False
    point_cloud_data = [struct.pack('fff', *p[:3]) for p in points]
    point_cloud_msg.data = b''.join(point_cloud_data)
    return point_cloud_msg

def process_stream(name: str, port: int, pub: rospy.Publisher):
    global close_all
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((ip, port))
    sock.setblocking(0)

    data = bytearray()
    state = ReadState.FIND_JSON
    read_chunk_size = 8192
    frames_received = 0
    start_time = time.time()

    try:
        while not close_all:
            readable, writable, _ = select.select([sock], [sock], [], 1)
            if writable and flow_control == 'framerate':
                if time.time() - start_time > 1.0:
                    start_time = time.time()
                    writable[0].send((frames_received).to_bytes(1, byteorder='big'))
                    frames_received = 0
            if readable:
                data += readable[0].recv(read_chunk_size)
                if state == ReadState.FIND_JSON:
                    json_msg, data = find_json(data)
                    if json_msg:
                        frame_info = FrameInfo(json.loads(json_msg))
                        state = ReadState.READ_DATA
                elif state == ReadState.READ_DATA:
                    if frame_info.size <= len(data):
                        state = ReadState.PROCESS
                    elif frame_info.size - len(data) < read_chunk_size:
                        read_chunk_size = frame_info.size - len(data)
                elif state == ReadState.PROCESS:
                    buffer = data[:frame_info.size]
                    data = data[frame_info.size:]
                    state = ReadState.FIND_JSON
                    read_chunk_size = 8192
                    frames_received += 1

                    arr = np.frombuffer(buffer, dtype=frame_info.bit_depth)
                    cloud = arr.reshape((-1, frame_info.channels))
                    cloud_xyz = cloud[:, :3]
                    cloud_xyz = np.append(cloud_xyz, np.ones([len(cloud_xyz), 1]), axis=1)
                    cloud_xyz = cloud_xyz.dot(frame_info.T_base_to_stream)
                    msg = numpy_to_pointcloud2(cloud_xyz, frame_id=name)
                    pub.publish(msg)

                    clock_msg = Clock()
                    clock_msg.clock = rospy.Time.now()
                    clock_pub.publish(clock_msg)
    except Exception as e:
        rospy.logerr(f"[{name}] error: {e}")
        close_all = True
        sys.exit()

if __name__ == '__main__':
    rospy.init_node('digit_multistream_pointcloud', anonymous=True)
    lidar_pub = rospy.Publisher('/velodyne_points', PointCloud2, queue_size=10)
    shoulder_pub = rospy.Publisher('/digit/shoulder_points', PointCloud2, queue_size=10)
    torso_pub = rospy.Publisher('/digit/torso_points', PointCloud2, queue_size=10)
    clock_pub = rospy.Publisher('/clock', Clock, queue_size=10)

    pub_dict = {
        'upper-velodyne-vlp16/depth/points': lidar_pub,
        'forward-chest-realsense-d435/depth/points': shoulder_pub,
        'forward-pelvis-realsense-d430/depth/points': torso_pub
    }

    stream_names = ['upper-velodyne-vlp16/depth/points', 
                    'forward-chest-realsense-d435/depth/points',
                    'forward-pelvis-realsense-d430/depth/points']
    try:
        for name in stream_names:
            port = asyncio.get_event_loop().run_until_complete(start_stream(name))
            t = threading.Thread(target=process_stream, args=(name, port, pub_dict[name]))
            t.start()
    except Exception as e:
        close_all = True
        print(f"Startup error: {e}")
        sys.exit()
