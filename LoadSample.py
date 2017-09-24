# LoadSample.py

import numpy as np
from DataSet import DataSet
from tensorflow.python.framework import dtypes

def _read32(bytestream):
  dt = np.dtype(np.uint32).newbyteorder('>')
  return np.frombuffer(bytestream.read(4), dtype=dt)[0]

def dense_to_one_hot(labels_dense, num_classes):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = np.arange(num_labels) * num_classes
  labels_one_hot = np.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot



def LoadSample(fname_image,fname_label,validate=False) :

  # read image file and the corresponding label file
  # one_hot = True
  # num_classes = 10

  one_hot = True
  num_classes = 10

  with open(fname_image,'rb') as f :
    magic = _read32(f)
    num_images = _read32(f)
    rows = _read32(f)
    cols = _read32(f)
    print(fname_image," : ",magic,num_images,rows,cols)
    buf = f.read(rows * cols * num_images)
    image = np.frombuffer(buf, dtype=np.uint8)
    image = image.reshape(num_images, rows, cols, 1)

  with open(fname_label,'rb') as f :
    magic = _read32(f)
    if magic != 2049:
      raise ValueError('Invalid magic number %d in MNIST label file: %s' % (magic, f.name))
    num_items = _read32(f)
    print(fname_label," : ",magic,num_items)
    buf = f.read(num_items)
    label = np.frombuffer(buf, dtype=np.uint8)

    if one_hot:
      label = dense_to_one_hot(label, num_classes)

  if validate :
    train = DataSet(image[0:int(num_images/2),:,:],label[0:int(num_items/2),:])
    valid = DataSet(image[int(num_images/2):num_images,:,:],label[int(num_items/2):num_items,:])
    return train, valid
  else :
    return DataSet(image,label)
