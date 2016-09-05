"""Different methods for manipulating data"""

import tensorflow as tf
import random
import os.path
from pathlib import Path
base_dir = os.path.dirname(__file__)

def _find_image_files(data_dir, labels_file):
  """Build a list of all images files and labels in the data set.
  Args:
    data_dir: string, path to the root directory of images.
      Assumes that the image data set resides in JPEG files located in
      the following directory structure.
        data_dir/dog/another-image.JPEG
        data_dir/dog/my-image.jpg
      where 'dog' is the label associated with these images.
    labels_file: string, path to the labels file.
      The list of valid labels are held in this file. Assumes that the file
      contains entries as such:
        dog
        cat
        flower
      where each line corresponds to a label. We map each label contained in
      the file to an integer starting with the integer 0 corresponding to the
      label contained in the first line.
  Returns:
    filenames: list of strings; each string is a path to an image file.
    texts: list of strings; each string is the class, e.g. 'dog'
    labels: list of integer; each integer identifies the ground truth.
  """
  print('Determining list of input files and labels from %s.' % data_dir)
  unique_labels = [l.strip() for l in tf.gfile.FastGFile(
      labels_file, 'r').readlines()]

  labels = []
  filenames = []
  texts = []

  # Leave label index 0 empty as a background class.
  label_index = 1

  # Construct the list of JPEG files and labels.
  for text in unique_labels:
    jpeg_file_path = '%s/%s/*' % (data_dir, text)
    matching_files = tf.gfile.Glob(jpeg_file_path)

    labels.extend([label_index] * len(matching_files))
    texts.extend([text] * len(matching_files))
    filenames.extend(matching_files)

    if not label_index % 100:
      print('Finished finding files in %d of %d classes.' % (
          label_index, len(labels)))
    label_index += 1

  # Shuffle the ordering of all image files in order to guarantee
  # random ordering of the images with respect to label in the
  # saved TFRecord files. Make the randomization repeatable.
  shuffled_index = list(range(len(filenames)))
  random.seed(12345)
  random.shuffle(shuffled_index)

  filenames = [filenames[i] for i in shuffled_index]
  texts = [texts[i] for i in shuffled_index]
  labels = [labels[i] for i in shuffled_index]

  print('Found %d JPEG files across %d labels inside %s.' %
        (len(filenames), len(unique_labels), data_dir))
  return filenames, texts, labels, unique_labels
  
def get_unique_labels(labels_file):
    unique_labels = [l.strip() for l in tf.gfile.FastGFile(
    labels_file, 'r').readlines()]
                     
    return unique_labels
    
def get_absolute_paths():
    # 1.) Define Paths
    BASE_DIR = str(Path(base_dir).parent)
    DATA_DIR = os.path.join(BASE_DIR,'data')
    CLASSIFIER_DIR = os.path.join(DATA_DIR,'classifier/clf.pkl')
    INCEPTION_MODEL_DIR = os.path.join(DATA_DIR,'inceptionmodel','classify_image_graph_def.pb')
    VGG_FACE_MODEL_DIR = os.path.join(DATA_DIR,'vggfacemodel','vggface16.tfmodel')
    FULL_IMAGES_DIR = os.path.join(DATA_DIR,'images/full')
    FULL_IMAGES_LABELS_DIR = os.path.join(FULL_IMAGES_DIR,'labels')
    FACE_DETECTED_IMAGES_DIR = os.path.join(DATA_DIR,'images/facedetected')
    FACE_DETECTED_IMAGES_LABELS_DIR = os.path.join(FACE_DETECTED_IMAGES_DIR,'labels')
    #------------------------------(.1)
    
    # 2.) Return
    return DATA_DIR,CLASSIFIER_DIR,INCEPTION_MODEL_DIR,VGG_FACE_MODEL_DIR,FULL_IMAGES_DIR,FULL_IMAGES_LABELS_DIR,FACE_DETECTED_IMAGES_DIR,FACE_DETECTED_IMAGES_LABELS_DIR