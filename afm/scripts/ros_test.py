#!/home/ubuntu20-jrl/anaconda3/envs/afm/bin/python
import rospy
from config import cfg
from modeling.afm import AFM
from std_msgs.msg import String
from sensor_msgs.msg import Image
# from cv_bridge import CvBridge
from afm.msg import lines2d
import cv2
import numpy as np
import os


def image_msg_to_numpy(msg):
    """
    将ROS的sensor_msgs/Image消息转换为NumPy数组，手动解析图像数据而不使用cv_bridge。
    """
    # 获取图像的宽度和高度
    width = msg.width
    height = msg.height
    encoding = msg.encoding

    # 判断图像的编码格式
    if encoding == 'bgr8':  # RGB图像
        np_arr = np.frombuffer(msg.data, dtype=np.uint8).reshape((height, width, 3))
    elif encoding == 'mono8':  # 灰度图像
        np_arr = np.frombuffer(msg.data, dtype=np.uint8).reshape((height, width))
    elif encoding == 'rgb8':  # RGB格式
        np_arr = np.frombuffer(msg.data, dtype=np.uint8).reshape((height, width, 3))
        np_arr = cv2.cvtColor(np_arr, cv2.COLOR_RGB2BGR)
    else:
        rospy.logerr(f"Unsupported encoding: {encoding}")
        return None

    return np_arr
class Nodo(object):
    def __init__(self, cfg):
        # Params
        self.image = None
        self.header = None
        self.pub_img_set = False 
        self.image_topic = cfg.image_topic

        self.system = AFM(cfg)
        self.system.model.eval()
        self.system.load_weight_by_epoch(-1)
        # Node cycle rate (in Hz).
        self.loop_rate = rospy.Rate(100)

        # Publishers
        self.pub = rospy.Publisher('Lines2d', lines2d, queue_size=1000)
        if self.pub_img_set:
            self.pub_image = rospy.Publisher('feature_image', Image, queue_size=1000)

        # Subscribers
        rospy.Subscriber(self.image_topic, Image, self.callback)

        # camera parameters
        self.mtx = np.array([[cfg.projection_parameters.fx, 0, cfg.projection_parameters.cx],
                             [0, cfg.projection_parameters.fy, cfg.projection_parameters.cy],
                             [0, 0, 1]])
        self.dist = np.array([cfg.distortion_parameters.k1, cfg.distortion_parameters.k2, cfg.distortion_parameters.p1, cfg.distortion_parameters.p2])
        self.width = cfg.width
        self.height = cfg.height
        self.newmtx, self.validpixROI = cv2.getOptimalNewCameraMatrix(self.mtx, self.dist, (self.width, self.height), 0, (self.width, self.height))

    def callback(self, msg):
        self.header = msg.header
        # 使用手动转换函数，而不是cv_bridge
        self.image = image_msg_to_numpy(msg)

    def start(self, cfg):
        pre_msg_time = rospy.Time(0)
        while not rospy.is_shutdown():
            if self.image is not None:
                if len(self.image.shape) == 2:  # gray image
                    img = cv2.cvtColor(self.image.copy(), cv2.COLOR_GRAY2BGR)
                else:
                    img = self.image.copy()

                msg_time = self.header.stamp
                if msg_time > pre_msg_time:
                    pre_msg_time = msg_time
                    # dst_img=cv2.undistort(img, self.mtx, self.dist, newCameraMatrix=self.newmtx)
                    dst_img = img
                    feats = self.system.detect(dst_img, cfg)
                    lines2d_msg = lines2d(
                        header=self.header, startx=feats[:, 0], starty=feats[:, 1], endx=feats[:, 2], endy=feats[:, 3])
                    self.pub.publish(lines2d_msg)

                    if self.pub_img_set:
                        feat_imge = dst_img.copy()
                        for i in range(feats.shape[0]):
                            cv2.line(feat_imge, (feats[i, 0], feats[i, 1]),
                                     (feats[i, 2], feats[i, 3]), (0, 0, 255), 2)
                        # 手动将OpenCV图像转换为ROS Image消息
                        image_msg = Image()
                        image_msg.header = self.header
                        image_msg.height, image_msg.width, _ = feat_imge.shape
                        image_msg.encoding = "bgr8"
                        image_msg.data = feat_imge.tobytes()
                        self.pub_image.publish(image_msg)

            self.loop_rate.sleep()



if __name__ == "__main__":

    rospy.init_node('afm')

    config_file = rospy.get_param('~config_file')
    img_file = rospy.get_param('~image')
    gpu = rospy.get_param('~gpu')

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

    cfg.merge_from_file(config_file)
    # print(cfg)
    my_node = Nodo(cfg)
    my_node.start(cfg)
