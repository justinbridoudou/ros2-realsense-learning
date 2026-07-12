import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image


def main():
    rclpy.init()

    node = Node("realsense_color_publisher")
    publisher = node.create_publisher(
        Image,
        "/camera/color/image_raw",
        10,
    )

    bridge = CvBridge()

    camera = cv2.VideoCapture("/dev/video4", cv2.CAP_V4L2)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camera.set(cv2.CAP_PROP_FPS, 30)

    if not camera.isOpened():
        node.get_logger().error("Impossible d'ouvrir /dev/video4")
        node.destroy_node()
        rclpy.shutdown()
        return

    node.get_logger().info(
        "RealSense active : publication sur /camera/color/image_raw"
    )
    node.get_logger().info("Appuie sur q dans l'image pour arrêter.")

    try:
        while rclpy.ok():
            success, frame = camera.read()

            if not success:
                node.get_logger().warning("Image non reçue")
                continue

            message = bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            message.header.stamp = node.get_clock().now().to_msg()
            message.header.frame_id = "camera_color_optical_frame"

            publisher.publish(message)

            cv2.imshow("RealSense D435 - ROS2", frame)

            rclpy.spin_once(node, timeout_sec=0)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
