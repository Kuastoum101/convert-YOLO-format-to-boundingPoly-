from typing import Callable, Collection, Optional, Tuple, Union, cast, TypeVar, Dict

import cv2
import numpy as np


class Point:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __str__(self) -> str:
        return f"Point(x={self.x}, y={self.y})"


class Size:
    def __init__(self, width: float, height: float):
        self.width = width
        self.height = height

    def __str__(self) -> str:
        return f"Size(w={self.width}, h={self.height})"


TupleFloat4 = Tuple[float, float, float, float]
TupleFloat3 = Tuple[float, float, float]
TupleInt4 = Tuple[int, int, int, int]
TupleInt3 = Tuple[int, int, int]

TensorType = TypeVar("TensorType")

TensorsTuple = Tuple[TensorType, ...]

PreprocessingFunction = Callable[[np.ndarray], np.ndarray]


class Rect:
    def __init__(self, top_left: Point, bottom_right: Point) -> None:
        self.top_left = top_left
        self.bottom_right = bottom_right

    def __str__(self) -> str:
        return f"Rect(x={self.top_left.x}, y={self.top_left.y}, "\
            f"w={self.width}, h={self.height})"

    def __repr__(self) -> str:
        return str(self)

    @property
    def width(self) -> float:
        return self.bottom_right.x - self.top_left.x

    @property
    def height(self) -> float:
        return self.bottom_right.y - self.top_left.y

    @classmethod
    def create_from_point_and_size(cls, top_left: Point, size: Size):
        return cls(top_left, Point(top_left.x + size.width,
                                   top_left.x + size.height))

    @classmethod
    def create_from_array(cls, box: np.ndarray,
                          rescale_from: Tuple[int, int],
                          rescale_to: Tuple[int, int]):
        box_lt_x, box_lt_y, box_rb_x, box_rb_y = box

        from_height, from_width = rescale_from
        to_height, to_width = rescale_to
        scale_x, scale_y = from_width / to_width, from_height / to_height

        dw = 0.5 * (from_width - scale_x * to_width)
        dh = 0.5 * (from_height - scale_y * to_height)
        return cls(Point((box_lt_x - dw) / scale_x, (box_lt_y - dh) / scale_y),
                   Point((box_rb_x - dw) / scale_x, (box_rb_y - dh) / scale_y))

    @classmethod
    def create_from_collection(cls, box: Collection[float]):
        assert len(box) == 4, \
            "Can't create Rect from collection representing box with wrong size. "\
            f"Expected 4. Got: {box}"
        box_lt_x, box_lt_y, box_rb_x, box_rb_y = box
        return cls(Point(box_lt_x, box_lt_y), Point(box_rb_x, box_rb_y))


    def to_xywh(self, convert_to_integers: bool = False) -> Union[TupleFloat4, TupleInt4]:
        xywh = (self.top_left.x, self.top_left.y, self.width, self.height)
        if convert_to_integers:
            xywh = cast(TupleFloat4, tuple(map(int, xywh)))
        return xywh

    def to_xyxy(self, convert_to_integers: bool = False) -> Union[TupleFloat4, TupleInt4]:
        xyxy = (self.top_left.x, self.top_left.y, self.bottom_right.x,
                self.bottom_right.y)
        if convert_to_integers:
            xyxy = cast(TupleFloat4, tuple(map(int, xyxy)))
        return xyxy

    def clip_to_size(self, size: Size) -> "Rect":
        return Rect(
            Point(min(size.width, max(0, self.top_left.x)),
                  min(size.height, max(0, self.top_left.y))),
            Point(min(size.width, max(0, self.bottom_right.x)),
                  min(size.height, max(0, self.bottom_right.y)))
        )


class Detection:
    """Class representing result of detection model"""

    def __init__(self, bounding_box: Rect, score: float, class_id: int, class_label: str):
        self.bounding_box = bounding_box
        self.score = score
        self.class_id = class_id
        self.class_label = class_label

    def __str__(self) -> str:
        return f"Detection(class={self.class_id}, label={self.class_label}, score={self.score}, bb={self.bounding_box})"

    def __repr__(self) -> str:
        return str(self)

    def to_json(self) -> Dict:
        return {
            "class_id": self.class_id,
            "class_label": self.class_label,
            "score": self.score,
            "bounding_box": self.bounding_box.to_xyxy(),
        }

    def draw_on_image(self, image: np.ndarray,
                      colors: Tuple[Tuple[int, int, int]]) -> np.ndarray:
        image_size = Size(*reversed(image.shape[:2]))
        object_bounding_box = cast(TupleInt4, self.bounding_box.clip_to_size(image_size).to_xywh(
            convert_to_integers=True
        ))

        class_color = colors[self.class_id]
        image = cv2.rectangle(image, object_bounding_box, class_color, thickness=3)

        label = f"{self.class_label} {self.score:.2f}"
        font_face = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 1
        font_scale = min(image.shape[1] / image.shape[0], image.shape[0] / image.shape[1])
        label_size, baseline = cv2.getTextSize(label, font_face, font_scale, thickness)

        # Label is out of image
        if object_bounding_box[1] - label_size[1] >= 0:
            label_origin = (object_bounding_box[0],
                            object_bounding_box[1] - baseline // 2)
        else:
            label_origin = (object_bounding_box[0],
                            object_bounding_box[1] + 2 * baseline)

        image = cv2.rectangle(image, (label_origin[0], label_origin[1] + baseline),
                              (label_origin[0] + label_size[0], label_origin[1] - label_size[1]),
                              class_color, cv2.FILLED)

        image = cv2.putText(image, label, label_origin,
                            font_face, font_scale, (0, 0, 0, 0), thickness)
        return image

