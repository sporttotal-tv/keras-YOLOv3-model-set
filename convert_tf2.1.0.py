import tensorflow as tf
from pathlib import Path
import sys
sys.path.append('./keras-YOLOv3-model-set')
from common.data_utils import preprocess_image
import numpy as np
from PIL import Image
from train import get_anchors
from yolo3.model import get_yolo3_model

anchors_path = './keras-YOLOv3-model-set/configs/tiny_yolo3_anchors.txt'
model_type = 'tiny_yolo3_darknet_thin'
anchors = get_anchors(anchors_path)


train_dir = Path('/notebooks/ceph-data/sporttotal-ml/datasets/basketball_1/')
test_dir = Path('/notebooks/ceph-data/sporttotal-ml/datasets/Jgor_test/')
images_paths = sorted((train_dir/'images').glob('*.jp*g')) + sorted(test_dir.glob('*.jp*g'))
images_paths_str = [str(x) for x in images_paths]

images_paths = sorted((train_dir/'images').glob('*.jp*g')) + sorted(test_dir.glob('*.jp*g'))
images_paths_str = [str(x) for x in images_paths]

images_paths_str = [str(x) for x in images_paths]

import sys
sys.path.append('./keras-YOLOv3-model-set')
from common.data_utils import preprocess_image
import numpy as np
from PIL import Image

def process_path(file_path):
    img = tf.io.read_file(file_path)
    return img

def decode_img(img):
    # Convert the compressed string to a 3D uint8 tensor
    img = tf.io.decode_jpeg(img, channels=3)
    # Resize the image to the desired size
    return img

def rezize_img(img):
    img = Image.fromarray(img)
    img = preprocess_image(img, model_input_shape=(768, 768))
    img = np.array(img)
    return img


ds = tf.data.Dataset.from_tensor_slices(images_paths_str)
ds = ds.map(process_path)
ds = ds.map(decode_img)
ds = ds.map(
    lambda img: tf.numpy_function(rezize_img, inp=[img], Tout=tf.float32),
)

num_anchors = len(anchors)
num_feature_layers = num_anchors//3
num_classes = 2
keras_model, _ = get_yolo3_model(model_type, num_feature_layers, num_anchors, num_classes, input_shape=(768, 768, 3))
keras_model.summary()
keras_model.load_weights('yolov3-tiny-relu-thin-infer.weights')

from tqdm import tqdm

def representative_dataset():
    for img in tqdm(ds):
        yield [img]

converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.float32
converter.inference_output_type = tf.float32

tflite_quant_model = converter.convert()

quantized_tflite_file = Path("yolov3-tiny-relu-thin-infer-tf2.1.0.tflite")
quantized_tflite_file.write_bytes(tflite_quant_model)