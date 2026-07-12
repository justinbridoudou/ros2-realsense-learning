import subprocess
import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from ultralytics import YOLO


class ObjectRecognizer(Node):
    def __init__(self):
        super().__init__("object_recognizer")

        self.publisher = self.create_publisher(
            String,
            "/vision/detected_object",
            10,
        )

        self.subscription = self.create_subscription(
            Image,
            "/camera/color/image_raw",
            self.image_callback,
            1,
        )

        self.get_logger().info("Loading YOLO model...")
        self.model = YOLO("yolo11n.pt")

        self.frame_counter = 0
        self.last_label = ""
        self.last_announcement = 0.0

        self.get_logger().info(
            "Ready. Show an object in the center of the camera."
        )

    def speak(self, label):
        try:
            subprocess.Popen(
                ["espeak-ng", "-v", "en", f"I see a {label}"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            pass

    def image_callback(self, message):
        self.frame_counter += 1

        # Analyse environ deux images par seconde.
        if self.frame_counter % 15 != 0:
            return

        try:
            frame = np.frombuffer(
                message.data,
                dtype=np.uint8,
            ).reshape(
                message.height,
                message.width,
                3,
            )
        except Exception as error:
            self.get_logger().error(f"Image conversion error: {error}")
            return

        results = self.model.predict(
            source=frame,
            imgsz=416,
            conf=0.35,
            device="cpu",
            verbose=False,
        )

        boxes = results[0].boxes

        if boxes is None or len(boxes) == 0:
            return

        objects = []

        for box in boxes:
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            label = self.model.names[class_id]

            # Évite qu'il dise toujours "person".
            if label == "person":
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            area = (x2 - x1) * (y2 - y1)

            objects.append(
                {
                    "label": label,
                    "confidence": confidence,
                    "area": area,
                }
            )

        if not objects:
            return

        # Privilégie l'objet le mieux reconnu et le plus visible.
        best = max(
            objects,
            key=lambda item: item["confidence"] + min(
                item["area"] / 150000,
                0.3,
            ),
        )

        label = best["label"]
        confidence = round(best["confidence"] * 100)

        ros_message = String()
        ros_message.data = f"{label} ({confidence}%)"
        self.publisher.publish(ros_message)

        now = time.time()

        if label != self.last_label or now - self.last_announcement > 4:
            self.get_logger().info(
                f"I see: {label} — confidence {confidence}%"
            )

            self.speak(label)

            self.last_label = label
            self.last_announcement = now


def main():
    rclpy.init()
    node = ObjectRecognizer()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
