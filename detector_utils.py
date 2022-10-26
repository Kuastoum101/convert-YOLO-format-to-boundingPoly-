import abc
from typing import Sequence, Callable, Optional

import json
from functools import partial

from typing import List, Tuple

import torchvision
import torch
import torch.nn.functional as F



from types_centernet import PreprocessingFunction, Detection
from typing import Dict, Union, Any
from types_centernet import TupleInt4, TensorsTuple, TupleInt3, Rect
import pynvml
from contextlib import contextmanager

import albumentations
import albumentations.augmentations.geometric.functional as GF
import functools
from pathlib import Path
import numpy as np
import cv2

def pop_existing_file_path(dictionary: Dict, key: Any,
                           error_message: Optional[str] = None) -> Path:
    value = pop_existing(dictionary, key, error_message)
    path = Path(value)
    assert path.is_file(), \
        f"{key}: '{path}' doesn't reference an existing file. " \
        f"{_optional_error_message(error_message)}"
    return path


def _optional_error_message(error_message: Optional[str] = None) -> str:
    return error_message if error_message is not None else ""


def pop_existing(dictionary: Dict, key: Any,
                 error_message: Optional[str] = None) -> Any:
    value = dictionary.pop(key, None)
    assert value is not None, \
        f"Can't find required key: '{key}' in {dictionary}. " \
        f"{_optional_error_message(error_message)}"
    return value


def init_cuda_context():
    pynvml.nvmlInit()


def shutdown_cuda_context():
    pynvml.nvmlShutdown()


@contextmanager
def cuda_context():
    init_cuda_context()
    try:
        yield
    finally:
        shutdown_cuda_context()


def get_visible_gpu_devices() -> Tuple[int, ...]:
    with cuda_context():
        return tuple(i for i in range(pynvml.nvmlDeviceGetCount()))


def assert_gpu_devices_are_available(gpu_devices: Tuple[int, ...]):
    visible_gpu_devices = get_visible_gpu_devices()
    assert len(visible_gpu_devices) >= len(gpu_devices), \
        f"Number of visible GPU devices {visible_gpu_devices} " \
        " should be greater or equal to the number of specified " \
        f"gpu devices: {gpu_devices}"

    specified_gpu_indices = set(gpu_devices)
    assert specified_gpu_indices.issubset(visible_gpu_devices), \
        f"Not all specified indices of GPU devices: {gpu_devices} are available. " \
        f"Available devices: {visible_gpu_devices}"


class ModelConfiguration:
    def __init__(self, model_name: str, model_definition_name: str,
                 model_weights_name: str, class_labels_name: str) -> None:
        self.model_name = model_name
        self.model_definition_name = model_definition_name
        self.model_weights_name = model_weights_name
        self.class_labels_name = class_labels_name

    def __str__(self) -> str:
        return f"ModelConfiguration(model_name={self.model_name}, " \
               f"model_definition_name={self.model_definition_name}, " \
               f"model_weights_name={self.model_weights_name})"

    def __repr__(self) -> str:
        return str(self)


class Configuration(abc.ABC):
    @classmethod
    def parse(cls, json_config: Dict[str, Any]):
        instance = cls._parse(json_config)
        assert len(json_config) == 0, \
            f"Unexpected keys are met while trying to instantiate {type(instance).__name__}.\n" \
            f"Unexpected keys with their values:\n{json_config.items()}."
        return instance

    @abc.abstractclassmethod
    def _parse(cls, json_config: Dict[str, Any]):
        pass

    @abc.abstractmethod
    def serialize(self) -> Dict[str, Any]:
        pass

    def create_required_directories(self):
        pass


class ModelDefinition(abc.ABC):
    def __init__(self, model_definition_path: Path) -> None:
        self.loaded_from_path = model_definition_path

    @abc.abstractproperty
    def model_name(self) -> str:
        pass

    @abc.abstractproperty
    def input_shape(self) -> TupleInt3:
        pass

    @abc.abstractmethod
    def create_preprocessing(self) -> PreprocessingFunction:
        pass

    @classmethod
    def parse(cls, model_name: str, model_definition_path: Path):
        # model_name = f"{model_name}.model_definition"
        # try:
        #     model_definition_module = importlib.import_module(model_name,
        #         package=__package__
        #     )
        # except ModuleNotFoundError as e:
        #     raise NotImplementedError(
        #         f"Can't find module for '{model_name}' model."
        #     ) from e
        #
        # model_definition_class = getattr(model_definition_module,
        #                                  f"{model_name.capitalize()}ModelDefinition",
        #                                  None)
        # if model_definition_class is None:
        #     raise NotImplementedError(
        #         f"Can't find model definition for the specified name: {model_name}. "
        #         f"Please implement {model_name.capitalize()}ModelDefinition class as a derived class of {cls.__name__}."
        #     )
        model_definition_class = CenternetModelDefinition
        return model_definition_class._parse(model_definition_path)

    @abc.abstractclassmethod
    def _parse(cls, model_definition_path: Path):
        pass


class Model(abc.ABC):
    @abc.abstractproperty
    def gpu_device(self) -> int:
        pass

    @abc.abstractproperty
    def input_shape(self) -> TupleInt4:
        pass

    @abc.abstractmethod
    def __call__(self, model_input: np.ndarray) -> Union[TensorsTuple,
                                                         Dict[str, TensorsTuple]]:
        pass


class DetectorConfiguration(Configuration):
    class Keys:
        NUMBER_OF_DETECTORS = "number_of_detectors"
        GPU_DEVICES = "gpu_devices"
        MODEL_NAME = "model_name"
        BATCH_SIZE = "batch_size"
        SCORE_THRESHOLD = "score_threshold"
        IOU_THRESHOLD = "iou_threshold"
        MAX_BOXES_PER_CLASS = "max_boxes_per_class"
        MAX_DETECTIONS_PER_IMAGE = "max_detections_per_image"
        RESTRICT_MEMORY_USAGE = "restrict_memory_usage"
        MODEL_DEFINITION_PATH = "model_definition_path"
        MODEL_WEIGHTS_PATH = "model_weights_path"
        CLASS_LABELS_PATH = "class_labels_path"
        CLASS_LABELS = "class_labels"

    def __init__(self, number_of_detectors: int, gpu_devices: Tuple[int, ...],
                 batch_size: int,
                 score_threshold: float, iou_threshold: float,
                 max_boxes_per_class: int, max_detections_per_image: int,
                 class_labels: Tuple[str, ...],
                 model_definition: ModelDefinition,
                 model_weights_path: Optional[Path] = None,
                 restrict_memory_usage: bool = False) -> None:
        self.number_of_detectors = number_of_detectors
        self.restrict_memory_usage = restrict_memory_usage
        self.model_weights_path = model_weights_path
        self.batch_size = batch_size
        self.gpu_devices = gpu_devices
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold
        self.max_boxes_per_class = max_boxes_per_class
        self.max_detections_per_image = max_detections_per_image
        self.model_definition = model_definition
        self.class_labels = class_labels

    def create_preprocessing_function(self) -> PreprocessingFunction:
        return self.model_definition.create_preprocessing()

    @classmethod
    def _parse(cls, json_config: Dict[str, Any]):
        number_of_detectors = json_config.pop(cls.Keys.NUMBER_OF_DETECTORS, 1)
        batch_size = json_config.pop(cls.Keys.BATCH_SIZE, 1)
        gpu_devices = json_config.pop(cls.Keys.GPU_DEVICES, (0,))
        assert_gpu_devices_are_available(gpu_devices)

        score_threshold = json_config.pop(cls.Keys.SCORE_THRESHOLD, 0.5)
        iou_threshold = json_config.pop(cls.Keys.IOU_THRESHOLD, 0.35)
        max_boxes_per_class = json_config.pop(cls.Keys.MAX_BOXES_PER_CLASS, 20)
        max_detections_per_image = json_config.pop(cls.Keys.MAX_DETECTIONS_PER_IMAGE, 40)

        restrict_memory_usage = json_config.pop(cls.Keys.RESTRICT_MEMORY_USAGE, False)

        model_name = pop_existing(
            json_config, cls.Keys.MODEL_NAME,
            "Configuration file should specify one of the known model names"
        )

        model_definition_path = pop_existing_file_path(json_config,
                                                       cls.Keys.MODEL_DEFINITION_PATH)
        model_definition = ModelDefinition.parse(model_name, model_definition_path)

        model_weights_path = pop_existing_file_path(json_config,
                                                    cls.Keys.MODEL_WEIGHTS_PATH)
        class_labels_path = pop_existing_file_path(json_config,
                                                   cls.Keys.CLASS_LABELS_PATH)
        with class_labels_path.open("r") as fp:
            class_labels = tuple(map(str.strip, fp.readlines()))
        return cls(number_of_detectors, gpu_devices, batch_size,
                   score_threshold, iou_threshold,
                   max_boxes_per_class, max_detections_per_image, class_labels,
                   model_definition,
                   model_weights_path=model_weights_path,
                   restrict_memory_usage=restrict_memory_usage)

    def serialize(self) -> Dict[str, Any]:
        return {
            self.Keys.NUMBER_OF_DETECTORS: self.number_of_detectors,
            self.Keys.GPU_DEVICES: self.gpu_devices,
            self.Keys.MODEL_NAME: self.model_definition.model_name,
            self.Keys.BATCH_SIZE: self.batch_size,
            self.Keys.SCORE_THRESHOLD: self.score_threshold,
            self.Keys.IOU_THRESHOLD: self.iou_threshold,
            self.Keys.MAX_BOXES_PER_CLASS: self.max_boxes_per_class,
            self.Keys.MAX_DETECTIONS_PER_IMAGE: self.max_detections_per_image,
            self.Keys.MODEL_DEFINITION_PATH: str(self.model_definition.loaded_from_path),
            self.Keys.MODEL_WEIGHTS_PATH: str(self.model_weights_path),
            self.Keys.CLASS_LABELS: self.class_labels
        }


def create_detector_config(model_config: ModelConfiguration, model_data_path: Path,
                           score_threshold: float, iou_threshold: float,
                           batch_size=1) -> DetectorConfiguration:
    # class_labels_path = model_data_path / model_config.class_labels_name
    # if not class_labels_path.is_file():
    #     raise FileNotFoundError(
    #         f"Class labels file: '{class_labels_path}' doesn't reference existing file"
    #     )
    # with class_labels_path.open("r") as fp:
    #     class_labels = tuple(map(str.strip, fp.readlines()))
    class_labels = model_config.class_labels_name
    path = model_data_path / model_config.model_definition_name
    model_definition = ModelDefinition.parse(model_config.model_name, path)
    return DetectorConfiguration(
        number_of_detectors=1,
        gpu_devices=(0,),
        batch_size=batch_size,
        score_threshold=score_threshold,
        iou_threshold=iou_threshold,
        max_boxes_per_class=20,
        max_detections_per_image=40,
        class_labels=class_labels,
        model_definition=model_definition,
        model_weights_path=model_data_path / model_config.model_weights_name
    )


class Detector(abc.ABC):
    def __init__(self, detector_model_preprocessing: PreprocessingFunction,
                 detector_model: Model, detector_class_labels: Sequence[str]) -> None:
        self.__preprocess = detector_model_preprocessing
        self.__model = detector_model
        self.__class_labels = detector_class_labels

    @property
    def gpu_device(self) -> int:
        return self.__model.gpu_device

    @property
    def input_shape(self) -> TupleInt4:
        return self.__model.input_shape

    @property
    def class_labels(self) -> Sequence[str]:
        return self.__class_labels

    def detect(self, images: np.ndarray) -> Tuple[Tuple[Detection, ...], ...]:
        # Adding batch dimension if it is absent
        images = np.asarray(images)
        if images.ndim == 3:
            images = np.expand_dims(images, 0)
        preprocessed_images = self.__preprocess(images)
        return self.__detect(preprocessed_images, images.shape)

    def detect_preprocessed(self, images: np.ndarray,
                            original_images_shape: TupleInt4) -> Tuple[Tuple[Detection, ...], ...]:
        images = np.asarray(images)
        if images.ndim == 3:
            images = np.expand_dims(images, 0)
        return self.__detect(images, original_images_shape)

    def __detect(self, model_input: np.ndarray,
                 original_input_shape: TupleInt4) -> Tuple[Tuple[Detection, ...], ...]:
        raw_detector_output = self.__model(model_input)  # type: ignore
        return self._decode(raw_detector_output, original_input_shape)

    @abc.abstractmethod
    def _decode(self, raw_detector_output: Union[TensorsTuple, Dict[str, TensorsTuple]],
                input_shape: TupleInt4) -> Tuple[Tuple[Detection, ...], ...]:
        pass

    @abc.abstractclassmethod
    def _create(cls, model_definition: ModelDefinition,
                gpu_device: int,
                score_threshold: float, iou_threshold: float,
                max_boxes_per_class: int, max_detections_per_image: int,
                batch_size: int, weights_file_path: Path,
                class_labels: Sequence[str],
                gpu_memory_limit_in_bytes: Optional[int] = None):
        pass

    @abc.abstractmethod
    def profile_peak_memory_usage(self, temporary_files_directory: Path) -> int:
        pass

    @classmethod
    def create(cls, config: DetectorConfiguration, gpu_device: int = 0,
               gpu_memory_limit_in_bytes: Optional[int] = None):
        model_name = config.model_definition.model_name
        # try:
        #     detector_module = importlib.import_module(
        #         f".{model_name}.detector",
        #         package=__package__
        #     )
        # except ModuleNotFoundError as e:
        #     raise NotImplementedError(
        #         f"Can't find module for '{model_name}'' model detector."
        #     ) from e

        # detector_class = getattr(detector_module,
        #                          f"{model_name.capitalize()}Detector",
        #                          None)
        detector_class = CenternetDetector
        if detector_class is None:
            raise NotImplementedError(
                f"Can't find detector implementation for the specified name: {model_name}. "
                f"Please implement {model_name.capitalize()}Detector class as a derived class of {cls.__name__}."
            )
        return detector_class._create(config.model_definition,
                                      gpu_device=gpu_device,
                                      score_threshold=config.score_threshold,
                                      iou_threshold=config.iou_threshold,
                                      max_boxes_per_class=config.max_boxes_per_class,
                                      max_detections_per_image=config.max_detections_per_image,
                                      batch_size=config.batch_size,
                                      weights_file_path=config.model_weights_path,
                                      class_labels=config.class_labels,
                                      gpu_memory_limit_in_bytes=gpu_memory_limit_in_bytes)


class ImageApplicator:
    def __init__(self, augmentation: albumentations.BasicTransform) -> None:
        self.augmentation = augmentation

    def __call__(self, img: np.ndarray) -> np.ndarray:
        return self.augmentation(image=img)["image"]


class CustomImageTranspose(albumentations.ImageOnlyTransform):
    def __init__(self, shape):
        super(CustomImageTranspose, self).__init__(True, 1.0)
        self.new_shape = shape

    def apply(self, image, **params):
        return image.transpose(self.new_shape)


class DivisibleResolution(albumentations.DualTransform):
    """Resize height and width to be divisible by value.

    Args:
        div_value (int): height and width will be divisible by this value
        interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        allow_upscale (bool): Allow upscale image for make is size divisible.
            For ``div_value=32`` and ``allow_upscale=False`` it will resize
            511x511 to 480x480. The same input and parameters, but with
            ``allow_upscale=False`` will resize input image to 512x512.
        p (float): probability of applying the transform. Default: 1.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """

    def __init__(self, div_value, interpolation=cv2.INTER_LINEAR, allow_upscale=False, always_apply=False, p=1):
        super(DivisibleResolution, self).__init__(always_apply, p)
        self.div_value = div_value
        self.allow_upscale = allow_upscale
        self.interpolation = interpolation

    def apply(self, img, interpolation=cv2.INTER_LINEAR, **params):
        height = img.shape[0]
        width = img.shape[1]
        if self.allow_upscale:
            divisible_height = round(height / self.div_value) * self.div_value
            divisible_width = round(width / self.div_value) * self.div_value
        else:
            divisible_height = height - height % self.div_value
            divisible_width = width - width % self.div_value

        return GF.resize(img, height=divisible_height, width=divisible_width,
                         interpolation=interpolation)

    def apply_to_bbox(self, bbox, **params):
        # Bounding box coordinates are scale invariant
        return bbox

    def apply_to_keypoint(self, keypoint, **params):
        height = params["rows"]
        width = params["cols"]
        divisible_height = height - height % self.div_value
        divisible_width = width - width % self.div_value
        scale_x = divisible_width / width
        scale_y = divisible_height / height
        return GF.keypoint_scale(keypoint, scale_x, scale_y)

    def get_transform_init_args_names(self):
        return ("value", "interpolation")


def create_augmentation(augmentation_name: str, **augmentation_params) -> albumentations.BasicTransform:
    if augmentation_name == "CustomImageTranspose":
        augmentation_cls = CustomImageTranspose
    elif augmentation_name == "DivisibleResolution":
        augmentation_cls = DivisibleResolution
    elif augmentation_name in dir(albumentations):
        augmentation_cls = getattr(albumentations, augmentation_name)
    else:
        assert False, f"Invalid augmentation name: {augmentation_name}. " \
                      "Can't find augmentation neither in augmentations package nor " \
                      "in custom augmentations list"
    return augmentation_cls(**augmentation_params)


def create_composite_augmentation(augmentations_definition: List[Dict]) -> albumentations.BasicTransform:
    augmentations: List[albumentations.BasicTransform] = []
    for augmentation_definition in augmentations_definition:
        assert len(augmentation_definition) == 1, \
            "More than 1 augmentation shouldn't be present in 1 item from " \
            "augmentation definitions list"
        augmentation_name, augmentation_params = next(iter(augmentation_definition.items()))
        augmentations.append(
            create_augmentation(augmentation_name, **augmentation_params)
        )
    return albumentations.Compose(augmentations, p=1)


def create_image_augmentation_applicator(augmentations_definitions: List[Dict]) -> Callable[[np.ndarray], np.ndarray]:
    return ImageApplicator(create_composite_augmentation(augmentations_definitions))


def batch_apply(src: np.ndarray,
                augmentation: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    src = np.asarray(src)
    if src.ndim == 3:
        src = np.expand_dims(src, 0)
    else:
        assert src.ndim == 4, "Centernet input should have 4 dimensions: NHWC"
    preprocessed = tuple(
        augmentation(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        for img in src
    )
    return np.asarray(preprocessed)


class CenternetModelDefinition(ModelDefinition):
    def __init__(self, model_definition_path: Path,
                 augmentations_definition: List[Dict],
                 input_shape: TupleInt3) -> None:
        super().__init__(model_definition_path)
        self.__augmentations_definition = augmentations_definition
        self.__input_shape = input_shape

    @property
    def model_name(self) -> str:
        return "centernet"

    @property
    def input_shape(self) -> TupleInt3:
        return self.__input_shape

    def create_preprocessing(self) -> PreprocessingFunction:
        return partial(batch_apply,
                       augmentation=create_image_augmentation_applicator(self.__augmentations_definition))

    @classmethod
    def _parse(cls, model_definition_path: Path):
        with model_definition_path.open("r") as model_definition_file:
            model_definition_config = json.load(model_definition_file)

        augmentations = model_definition_config.get("preprocessing")
        assert augmentations is not None, \
            "Can't find preprocessing definition. Add required augmentations to preprocessing list"
        input_shape = model_definition_config.get("input_shape")
        assert input_shape is not None, \
            "Can't find input_shape parameter in CenterNet configuration"
        assert len(input_shape) == 3, \
            "Input shape should be in format HWC: (H)eight(W)idth(C)hannels. " \
            f"Got: {input_shape}"
        return cls(model_definition_path, augmentations, input_shape)


class CenternetModel(Model):
    def __init__(self, model_impl: torch.jit.ScriptModule,
                 input_shape: TupleInt4,
                 gpu_device: int,
                 gpu_memory_limit_in_bytes: Optional[int] = None) -> None:
        self.__gpu_device = torch.device("cuda", gpu_device)
        self.__model = model_impl
        self.__model.eval()
        self.__model.to(device=self.__gpu_device)
        self.__input_shape = input_shape
        if gpu_memory_limit_in_bytes is not None:
            total_memory_in_bytes = torch.cuda.get_device_properties(self.__gpu_device).total_memory
            gpu_memory_fraction = gpu_memory_limit_in_bytes / total_memory_in_bytes
            torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction, self.__gpu_device)

    @property
    def gpu_device(self) -> int:
        return self.__gpu_device.index

    @property
    def input_shape(self) -> TupleInt4:
        return self.__input_shape

    def __call__(self, model_input: np.ndarray) -> Dict[str, TensorsTuple]:
        with torch.no_grad():
            input_tensor = torch.tensor(model_input, device=self.__gpu_device)
            return self.__model(input_tensor)

    @classmethod
    def instantiate(cls, model_definition: CenternetModelDefinition,
                    gpu_device: int, batch_size: int,
                    weights_file_path: Path,
                    gpu_memory_limit_in_bytes: Optional[int] = None):
        model = torch.jit.load(str(weights_file_path))
        return cls(model, (batch_size, *model_definition.input_shape),
                   gpu_device, gpu_memory_limit_in_bytes)


@functools.lru_cache()
def make_grid(h: int, w: int, device: torch.device) -> torch.Tensor:
    """Return grid of point coordinates (x, y) for each point within h * w

    Output shape: [h, w, 2]
    """
    x = torch.linspace(0, w - 1, w, device=device)
    y = torch.linspace(0, h - 1, h, device=device)
    grid_y, grid_x = torch.meshgrid(y, x)
    return torch.stack([grid_x, grid_y], dim=0)


def nms(heatmap_tensor: torch.Tensor, kernel_size=3):
    pad = (kernel_size - 1) // 2

    heatmap_max = F.max_pool2d(heatmap_tensor[None], (kernel_size, kernel_size), stride=1,
                               padding=pad)[0]
    return heatmap_max == heatmap_tensor


class CenternetDetector(Detector):
    def __init__(self, detector_model_preprocessing: PreprocessingFunction,
                 detector_model: Model, score_threshold: float, iou_threshold: float,
                 max_boxes_per_class: int, max_detections_per_image: int,
                 class_labels: Sequence[str]) -> None:
        super().__init__(detector_model_preprocessing, detector_model, class_labels)
        self.__score_threshold = score_threshold
        self.__iou_threshold = iou_threshold
        self.__max_boxes_per_class = max_boxes_per_class
        self.__max_detections_per_image = max_detections_per_image

    def _decode(self, raw_detector_output: Dict[str, torch.Tensor],
                input_shape: TupleInt4) -> Tuple[Tuple[Detection, ...], ...]:
        with torch.no_grad():
            batch_heatmap_tensor = raw_detector_output["hm"].sigmoid_()
            batch_wh_tensor = raw_detector_output["wh"]
            batch_regression_tensor = raw_detector_output["reg"]

            batch_detections = []
            for img_heatmap_tensor, img_wh_tensor, img_regression_tensor in zip(batch_heatmap_tensor,
                                                                                batch_wh_tensor,
                                                                                batch_regression_tensor):
                batch_detections.append(self.__to_detections(
                    img_heatmap_tensor, img_wh_tensor, img_regression_tensor, input_shape
                ))
            return tuple(batch_detections)

    def __parse_network_output(self, heatmap_tensor: torch.Tensor, wh_tensor: torch.Tensor,
                               regression_tensor: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        _, heatmap_height, heatmap_width = heatmap_tensor.shape  # num_classes, height, width
        # grid of points centers (x, y)
        grid = make_grid(heatmap_height, heatmap_width, device=heatmap_tensor.device)

        # Keep shape: [num_classes, height, width]
        keep = nms(heatmap_tensor)
        # Suppress everything else
        heatmap_tensor[~keep] = 0

        # Select points to keep
        keep_pos, _ = keep.max(dim=0)
        heatmap_tensor = heatmap_tensor[:, keep_pos]
        regression_tensor = regression_tensor[:, keep_pos]
        wh_tensor = wh_tensor[:, keep_pos]
        grid = grid[:, keep_pos]
        # heatmap shape: [num_classes, num_points_to_keep]
        scores, classes = heatmap_tensor.max(dim=0)

        # Select best max_detections_per_image
        max_detections = min(self.__max_detections_per_image, len(scores))
        top_scores, top_indices = torch.topk(scores, max_detections)

        # Choose corresponding classes, regression value, width/height and grid centers
        classes = classes[top_indices]
        regression_tensor = regression_tensor[:, top_indices]
        wh_tensor = wh_tensor[:, top_indices]
        grid = grid[:, top_indices]

        # Calculate bounding boxes
        centers = grid + regression_tensor
        half_wh_tensor = 0.5 * wh_tensor
        bounding_boxes = torch.cat((centers - half_wh_tensor, centers + half_wh_tensor),
                                   dim=0).transpose(1, 0)

        # Filter top_scores by score threshold
        threshold_indices = top_scores > self.__score_threshold
        top_scores = top_scores[threshold_indices]
        # We need to increment all classes, because class __background__ starts at 0
        classes = classes[threshold_indices] + 1
        bounding_boxes = bounding_boxes[threshold_indices]

        # Perform NMS of filtered boxes
        nms_indices = torchvision.ops.nms(bounding_boxes, top_scores, self.__iou_threshold)
        bounding_boxes = bounding_boxes[nms_indices].cpu().numpy()
        top_scores = top_scores[nms_indices].cpu().numpy()
        classes = classes[nms_indices].cpu().numpy()

        return bounding_boxes, top_scores, classes

    def __to_detections(self, heatmap_tensor: torch.Tensor, wh_tensor: torch.Tensor,
                        regression_tensor: torch.Tensor, input_shape: TupleInt4) -> Tuple[Detection, ...]:
        bounding_boxes, scores, classes = self.__parse_network_output(heatmap_tensor, wh_tensor,
                                                                      regression_tensor)
        detections: List[Detection] = []
        for bb, score, class_id in zip(bounding_boxes, scores, classes):
            detections.append(
                Detection(
                    Rect.create_from_array(bb,
                                           rescale_from=heatmap_tensor.shape[1:],
                                           rescale_to=input_shape[1:3]),
                    float(score),
                    int(class_id),
                    self.class_labels[int(class_id)]
                )
            )
        return tuple(detections)

    def profile_peak_memory_usage(self, temporary_files_directory: Path) -> int:
        torch.cuda.reset_peak_memory_stats(self.gpu_device)
        _ = self.detect_preprocessed(
            np.zeros(self.input_shape, dtype=np.float32),
            self.input_shape
        )
        memory_stats = torch.cuda.memory_stats(self.gpu_device)
        return 1.5 * memory_stats["allocated_bytes.all.peak"]

    @classmethod
    def _create(cls, model_definition: CenternetModelDefinition,
                gpu_device: int,
                score_threshold: float, iou_threshold: float,
                max_boxes_per_class: int, max_detections_per_image: int,
                batch_size: int, weights_file_path: Path,
                class_labels: Sequence[str],
                gpu_memory_limit_in_bytes: Optional[int] = None):
        assert isinstance(model_definition, CenternetModelDefinition), \
            f"Can't create CenternetDetector from model definition of other type: {type(model_definition)}"
        preprocessing = model_definition.create_preprocessing()
        model = CenternetModel.instantiate(model_definition, gpu_device, batch_size,
                                           weights_file_path, gpu_memory_limit_in_bytes)
        return cls(preprocessing, model, score_threshold, iou_threshold,
                   max_boxes_per_class, max_detections_per_image, class_labels)

