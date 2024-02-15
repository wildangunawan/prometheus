import numpy as np
import torch, torch.nn, torchvision
import torchvision.transforms as transforms
from torchvision.ops import masks_to_boxes
from torchvision.models.segmentation.lraspp import LRASPPHead
import pyrealsense2 as rs
from pymycobot import MyCobotSocket
import time
from math import tan, radians, degrees, asin, sin, cos, sqrt
from dotenv import load_dotenv

load_dotenv()
robot = MyCobotSocket(os.getenv("MYCOBOT_IP"), 9000)
robot.set_fresh_mode(1)

pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(
    stream_type=rs.stream.depth,
    width=480,
    height=270,
    format=rs.format.z16,
    framerate=30,
)
config.enable_stream(stream_type=rs.stream.color, width=1920, height=1080, framerate=30)

pipeline.start(config)

# Skip 5 first frames to give the Auto-Exposure time to adjust
for _ in range(5):
    pipeline.wait_for_frames()


class Model(torch.nn.Module):
    def __init__(self, freeze_backbone: bool = True):
        super().__init__()

        # Layers
        self.mobile = torchvision.models.segmentation.lraspp_mobilenet_v3_large(
            weights=torchvision.models.segmentation.LRASPP_MobileNet_V3_Large_Weights.DEFAULT
        )
        self.mobile.classifier = LRASPPHead(
            low_channels=40, high_channels=960, num_classes=3, inter_channels=128
        )
        self.softmax = torch.nn.Softmax(dim=1)

        # Freeze backbone
        if freeze_backbone:
            # Freeze all backbone
            for param in self.mobile.backbone.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.mobile.backbone.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.softmax(self.mobile(x)["out"])


def load_model() -> torch.nn.Module:
    model = torch.load(os.getenv("CV_MODEL_PATH"))
    model.eval().to("cuda")

    return model


def prep_image(img: torch.Tensor) -> torch.Tensor:
    # Resize and center crop
    img = torch.nn.Sequential(
        transforms.Resize(
            540, interpolation=transforms.InterpolationMode.BILINEAR, antialias=None
        ),
    )(img)

    # Normalize
    return img


def create_mask(model: torch.nn.Module, img: torch.Tensor) -> torch.Tensor:
    predictions = model(img.unsqueeze(0))

    return predictions.argmax(1) == 2  # nasi


def calculate_meter_from_pixel(depth_frame, x, y):
    # Calculate depth
    depth = depth_frame.get_distance(int(x / 4 / 1.261), int(y / 4 / 1.381))
    dx = x - 1920 / 2
    dy = y - 1080 / 2

    if depth > 1:
        return 0, 0, 0

    # FOV camera: https://www.intelrealsense.com/depth-camera-d435/
    Rx = tan(radians(69 / 2)) * depth
    Ry = tan(radians(42 / 2)) * depth

    # Frame width and height
    x = dx * Rx / (1920 / 2)
    y = dy * Ry / (1080 / 2)

    # depth: forward, x: horizontal, y: vertical
    return depth, x, y


def calculate_joint_1_correction(current_angle, depth, horizontal):
    horizontal -= 0.04  # 4cm offset between RGB camera and center of EoF

    return current_angle + round(degrees(asin(horizontal / depth))) * -1


def calculate_x_y(depth, current_coords, current_angles):
    print(type(depth), type(current_coords[2]))
    print(depth)

    distance_forward = (
        sqrt(abs((depth * 1000) ** 2 - current_coords[2] ** 2)) - 130
    )  # 150 panjang sendok
    print(distance_forward)

    # Calculate x and y
    x = distance_forward * cos(radians(current_angles[0]))
    y = distance_forward * sin(radians(current_angles[0]))

    return current_coords[0] + x, current_coords[1] + y


def move_till_coords(coords, speed=30, mode=1):
    t = time.time()
    robot.send_coords(coords, speed, mode)

    while time.time() - t < 3:
        # Check position
        current_coords = robot.get_coords()
        print(current_coords)
        print(coords)

        if current_coords == []:
            continue

        # Check for x and y axis only due to
        # z not being accurate and reliable
        if (
            abs(current_coords[1] - coords[0]) <= 5
            and abs(current_coords[2] - coords[1]) <= 5
        ):
            break

        time.sleep(0.5)


# Load model
model = load_model()

# 1. Ke posisi untuk ambil nasi
robot.sync_send_angles([-90, 0, -100, 40, 0, 0], 50)

time.sleep(2)


# 2. Ke posisi nasi
while True:
    frames = pipeline.wait_for_frames()
    rgb_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()

    color = torch.tensor(np.asanyarray(rgb_frame.get_data()))

    # Get image
    raw_img = prep_image(color.permute(2, 0, 1).type(torch.uint8))

    img = torch.nn.Sequential(
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    )(raw_img / 255)

    mask = create_mask(model, img.to("cuda"))

    # Find till we get a True value
    if True in mask.unique():
        # Turn mask into boxes
        boxes = masks_to_boxes(mask)

        x1, y1, x2, y2 = boxes[0][0], boxes[0][1], boxes[0][2], boxes[0][3]

        # * 2 to convert to 1920x1080 from 960x540
        x, y = (x2 + x1) * 2 / 2, (y2 + y1) * 2 / 2

        (
            # deviation in the axis, karena kita -90 derajat
            # maka gerak majunya ada di y axis bukan x axis
            depth,  # depth
            delta_horizontal,  # - left, + right
            delta_vertical,  # - down, + up
        ) = calculate_meter_from_pixel(
            depth_frame,
            x.item(),
            y.item(),
        )

        if depth != 0:
            break

print(depth, delta_horizontal, delta_vertical)

# Do angle correction
current_angles = []
while current_angles == []:
    current_angles = robot.get_angles()
    time.sleep(0.1)

new_angles = current_angles.copy()
new_angles[0] = calculate_joint_1_correction(new_angles[0], depth, delta_horizontal)

robot.sync_send_angles(new_angles, 30)
time.sleep(2)

if abs(delta_horizontal) < 1:
    current_coordinates, current_angles = [], []
    while current_coordinates == [] or current_angles == []:
        current_coordinates = robot.get_coords()
        current_angles = robot.get_angles()

    print("Current coordinates: ", current_coordinates)
    print("Current angles: ", current_angles)

    # Calculate distance forward
    to_do_x_axis, to_do_y_axis = calculate_x_y(
        depth, current_coordinates, current_angles
    )

    print(to_do_x_axis, to_do_y_axis)

    to_do_x_axis = min(max(to_do_x_axis, -280), 280)
    to_do_y_axis = min(max(to_do_y_axis, -255), -200)

    print(
        f"Moving to x: {to_do_x_axis} y: {to_do_y_axis}",
    )

    move_till_coords(
        [
            int(to_do_x_axis),
            int(to_do_y_axis),
            int(current_coordinates[2]),
            int(current_coordinates[3]),
            int(current_coordinates[4]),
            int(current_coordinates[5]),
        ],
    )

# 2.5 Berdirikan kepalanya
robot.send_coord(4, -120, 30)
time.sleep(3)

# 3. Get current coordinates
current_coordinates = []
while current_coordinates == []:
    current_coordinates = robot.get_coords()
    time.sleep(0.1)
height = current_coordinates[2]

# 5. Set angle 5 dan 6 ke -20 dan -20
robot.get_angles()
current_angles = []
while current_angles == []:
    current_angles = robot.get_angles()
    time.sleep(0.1)

robot.sync_send_angles(
    [
        current_angles[0],
        current_angles[1],
        current_angles[2],
        current_angles[3],
        current_angles[4] - 30,
        current_angles[5] - 30,
    ],
    30,
)
time.sleep(2)

# 4. Descend to nasi
if to_do_y_axis >= -215 and to_do_y_axis <= -200:
    robot.send_coord(3, 65, 30)
else:
    robot.send_coord(3, 65, 30)

# 5. Set angle 5 dan 6 ke -20 dan -20
time.sleep(2)

past_angle = current_angles
current_angles = []
while current_angles == [] or current_angles == past_angle:
    current_angles = robot.get_angles()
    time.sleep(0.1)

# 6. Set angle 5 dan 6 ke 30 dan 20
print(current_angles)
robot.sync_send_angles(
    [
        current_angles[0],
        current_angles[1],
        current_angles[2],
        current_angles[3],
        current_angles[4] + 30 + 10,
        current_angles[5] + 10,
    ],
    30,
)
print("6 done")

time.sleep(2)

# 7. Naik dari nasi
# 3. Get current coordinates
current_coordinates = []
while current_coordinates == []:
    current_coordinates = robot.get_coords()
    time.sleep(0.1)

robot.sync_send_coords(
    [
        current_coordinates[0],
        current_coordinates[1],
        200,
        -90,
        current_coordinates[4],
        current_coordinates[5],
    ],
    30,
    1,
)
print("7 done")

# 8. Kembali ke posisi standby
robot.sync_send_angles([45, 0, -90, 90, -45, 0], 40)
