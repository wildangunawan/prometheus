import numpy as np
import torch, torch.nn, torchvision
import torchvision.transforms as transforms
from torchvision.utils import draw_segmentation_masks, draw_bounding_boxes
from torchvision.ops import masks_to_boxes
from torchvision.models.segmentation.lraspp import LRASPPHead
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pyrealsense2 as rs
from pymycobot import MyCobotSocket
import time
from math import tan, radians
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

    return predictions.argmax(1) == 1  # wajah


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

# Create figure and put black image in it
fig = plt.figure()
im = plt.imshow(torch.zeros(1080, 1920, 3))

last_coord_sent_timestamp = time.time()


def update_figure(*_):
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

    global last_coord_sent_timestamp
    if True in mask.unique() and last_coord_sent_timestamp + 2 < time.time():
        boxes = masks_to_boxes(mask)

        x1, y1, x2, y2 = boxes[0][0], boxes[0][1], boxes[0][2], boxes[0][3]

        # * 2 to convert to 1920x1080 from 960x540
        x, y = (x2 + x1) * 2 / 2, (y2 + y1) * 2 / 2

        (
            deviation_x,
            deviation_y,  # + left, - right
            deviation_z,  # - up, + down
        ) = calculate_meter_from_pixel(
            depth_frame,
            x.item(),
            y.item(),
        )

        print(deviation_x, deviation_y, deviation_z)

        if abs(deviation_x) < 1 and (
            abs(deviation_y) > 0.040 or abs(deviation_z) > 0.035
        ):
            current_coordinates = robot.get_coords()
            print("Current coordinates: ", current_coordinates)

            if current_coordinates != []:
                to_do_x_axis = current_coordinates[0] + (deviation_x * 1000) - 200
                to_do_y_axis = current_coordinates[1] - (deviation_y * 1000) + 40
                to_do_z_axis = (
                    current_coordinates[2] - (deviation_z * 1000) + 35
                )  # 5cm camera offset

                to_do_x_axis = min(max(to_do_x_axis, 109), 231)
                to_do_y_axis = min(max(to_do_y_axis, -140), 140)
                to_do_z_axis = min(max(to_do_z_axis, 233), 360)

                print(
                    f"Moving to x: {to_do_x_axis} y: {to_do_y_axis} z: {to_do_z_axis}",
                )

                move_till_coords(
                    [
                        int(to_do_x_axis),
                        int(to_do_y_axis),
                        int(to_do_z_axis),
                        int(-90),
                        int(current_coordinates[4]),
                        int(current_coordinates[5]),
                    ],
                )

                last_coord_sent_timestamp = time.time()

    # print(depth)
    to_be_displayed = draw_segmentation_masks(raw_img, mask, alpha=0.5, colors="blue")

    # Update image
    im.set_data(to_be_displayed.permute(1, 2, 0))

    return (im,)


ani = animation.FuncAnimation(fig, update_figure, interval=50, blit=True)
plt.show()
