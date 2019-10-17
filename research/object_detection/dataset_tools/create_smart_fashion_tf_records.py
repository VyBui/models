r"""Convert raw Smart Fashion dataset to TFRecord for module 2.
Example usage:
    python baseline_models/dataset/create_module_2_tf_records.py \
        --data_dir=/home/user/dataset_cropped \
        --year=VOC2012 \
        --sub_dataset_name=bodyparts \
        --output_path=/home/user/smart_fashion/module2/dataset_cropped/smart_fashion_bodyparts.record
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib2
import io
import os

import PIL.Image
import tensorflow as tf

from object_detection.utils import dataset_util

flags = tf.app.flags
tf.flags.DEFINE_boolean('include_masks', False,
                        'Whether to include instance segmentations masks '
                        '(PNG encoded) in the result. default: False.')
tf.flags.DEFINE_string('train_image_dir', '/home/vybt/dataset_cropped/train/bodyparts',
                       'Training image directory.')
tf.flags.DEFINE_string('test_image_dir', '/home/vybt/dataset_cropped/test/bodyparts',
                       'Test image directory.')
tf.flags.DEFINE_string('output_dir', '/tmp/smart_fashion', 'Output data directory.')

FLAGS = flags.FLAGS

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

path_to_train_body_parts = '/home/vybt/dataset_cropped/train/bodyparts'
path_to_test_body_parts = '/home/vybt/dataset_cropped/test/bodyparts'


def get_image_files_in_folder(path_to_folder):
    """
    Get all file from a particular folder
    
    :param path_to_folder:
    :return:
    """
    original_images = []

    for path in os.listdir(path_to_folder):
        full_path = os.path.join(path_to_folder, path)
    
        for file in os.listdir(full_path):
            absolutely_path = os.path.join(full_path, file)
    
            if os.path.isfile(absolutely_path):
                original_images.append(absolutely_path)
                
    return original_images


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def create_tf_example(full_path):
    """Converts image to a tf.Example proto.
        Args:
        image: dict with keys:
          [u'license', u'file_name', u'coco_url', u'height', u'width',
          u'date_captured', u'flickr_url', u'id']
        image_dir: directory containing the image files.
    Returns:
        example: The converted tf.Example
        num_annotations_skipped: Number of (invalid) annotations that were ignored.
    Raises:
        ValueError: if the image pointed to by data['filename'] is not a valid JPEG
    """
    image_parameters = full_path.split("/")
    
    get_body_part_image_path = image_parameters[-1]
    get_body_part = get_body_part_image_path.split(".")[0]

    with tf.io.gfile.GFile(full_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    image_raw = image.tobytes()

    feature_dict = {
        'image/height':
            dataset_util.int64_feature(320),
        'image/width':
            dataset_util.int64_feature(320),
        'image/to_string':
            dataset_util.bytes_feature(image_raw),
        'image/body_part':
            dataset_util.int64_feature(int(get_body_part)),
        'image/full_path':
            tf.train.Feature(
                bytes_list=tf.train.BytesList(
                value=[m.encode('utf-8') for m in full_path])
        )
    }

    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))

    return example


def open_sharded_output_tfrecords(exit_stack, base_path, num_shards):
  """Opens all TFRecord shards for writing and adds them to an exit stack.

  Args:
    exit_stack: A context2.ExitStack used to automatically closed the TFRecords
      opened in this function.
    base_path: The base path for all shards
    num_shards: The number of shards

  Returns:
    The list of opened TFRecords. Position k in the list corresponds to shard k.
  """
  tf_record_output_filenames = [
      '{}-{:05d}-of-{:05d}'.format(base_path, idx, num_shards)
      for idx in range(num_shards)
  ]

  tfrecords = [
      exit_stack.enter_context(tf.python_io.TFRecordWriter(file_name))
      for file_name in tf_record_output_filenames
  ]

  return tfrecords


def _create_tf_record_from_fashion_dateset(image_dir, output_path, num_shards):
    """
    Load fashion dataset and converts to tf.Record format
    :param image_dir:
    :param num_shards:
    :return:
    """
    with contextlib2.ExitStack() as tf_record_close_stack:
        output_tfrecords = open_sharded_output_tfrecords(
            tf_record_close_stack, output_path, num_shards)

    images = get_image_files_in_folder(image_dir)
    
    for idx, image in enumerate(images):
        if idx % 100 == 0:
            tf.logging.info('On image %d of %d', idx, len(images))
        tf_example = create_tf_example(image)

        shard_idx = idx % num_shards
        print(shard_idx)
        output_tfrecords[shard_idx].write(tf_example.SerializeToString())
        output_tfrecords[shard_idx].close()
        tf.logging.info('shard_idx %d', shard_idx)
        
    tf.logging.info('Finished writing to: %s', output_path)


def main(_):
    assert FLAGS.train_image_dir, '`train_image_dir` missing.'
    assert FLAGS.test_image_dir, '`test_image_dir` missing.'
    
    if not tf.io.gfile.isdir(FLAGS.output_dir):
        tf.gfile.MakeDirs(FLAGS.output_dir)
    
    train_output_path = os.path.join(FLAGS.output_dir, 'body_parts_train.record')
    testdev_output_path = os.path.join(FLAGS.output_dir, 'body_parts_testdev.record')

    _create_tf_record_from_fashion_dateset(
        FLAGS.train_image_dir,
        train_output_path,
        num_shards=60
    )

    _create_tf_record_from_fashion_dateset(
        FLAGS.test_image_dir,
        testdev_output_path,
        num_shards=60
    )

if __name__ == '__main__':
  tf.compat.v1.app.run()
