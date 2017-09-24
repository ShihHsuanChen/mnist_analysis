# Model.py

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import dtypes

class Model() :
  
  # declare variables
  image   = None
  label   = None
  dropout = None
  weight = {"conv1": None, "conv2": None, "fc1": None, "fc2": None}
  bias   = {"conv1": None, "conv2": None, "fc1": None, "fc2": None}
  out    = {"conv1": None, "conv2": None, "fc1": None, "fc2": None}
  plout  = {"pool1": None, "pool2": None}
  label_out     = None
  cross_entropy = None
  accuracy      = None

  def __init__(self,name) :
    self.name = name

  def add_conv(self, name, tensor, size_in, size_out) :
    self.weight[name] = tf.Variable( tf.truncated_normal([5,5,size_in,size_out], stddev=0.1), name="w_"+name )
    self.bias[name]   = tf.Variable( tf.constant(0.1, shape=[size_out]), name="b_"+name)
    return tf.nn.relu( tf.nn.conv2d(tensor, self.weight[name], strides=[1,1,1,1],padding="SAME") + self.bias[name] )

  def add_pool(self, name, tensor) :
    return tf.nn.max_pool( tensor, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME", name="pool_"+name)

  def add_full_connect(self, name, tensor, size_in, size_out) :
    self.weight[name] = tf.Variable( tf.truncated_normal([size_in,size_out],stddev=0.1), name="w_"+name )
    self.bias[name]   = tf.Variable( tf.constant(0.1, shape=[size_out]), name="b_"+name )

    return tf.nn.relu( tf.matmul(tensor, self.weight[name]) + self.bias[name])
    

  def build(self) :
    """
    1. Reshape to use within a convolutional neural net. 
       Last dimension is for "features" - there is only one here, since images are grayscale 
       -- it would be 3 for an RGB image, 4 for RGBA, etc.
    2. First convolutional layer - maps one grayscale image to 32 feature maps.
    3. Pooling layer 
       -- downsamples by 2X.
    4. Second convolutional layer 
       -- maps 32 feature maps to 64.
    5. Second pooling layer.
    6. Fully connected layer 1 
       -- after 2 round of downsampling, our 28x28 image is down to 7x7x64 feature maps -- maps this to 1024 features.
    7. Dropout 
       -- controls the complexity of the model, prevents co-adaptation of features.
    # Map the 1024 features to 10 classes, one for each digit
    """
    self.image   = tf.placeholder(tf.float32,[None,784]) # input image
    self.label   = tf.placeholder(tf.float32,[None,10])  # output label
    self.dropout = tf.placeholder(tf.float32)

    im_reshape = tf.reshape(self.image,[-1,28,28,1])
    self.out["conv1"]   = self.add_conv("conv1",im_reshape,1,32)
    self.plout["pool1"] = self.add_pool("pool1",self.out["conv1"])
    self.out["conv2"]   = self.add_conv("conv2",self.plout["pool1"],32,64)
    self.plout["pool2"] = self.add_pool("pool2",self.out["conv2"])

    plout2_flat         = tf.reshape(self.plout["pool2"],[-1,7*7*64])

    self.out["fc1"]     = self.add_full_connect("fc1",plout2_flat,7*7*64,1024)
    fc1_drop            = tf.nn.dropout(self.out["fc1"], self.dropout)

    self.out["fc2"]     = self.add_full_connect("fc2",fc1_drop,1024,10)

    self.label_out      = self.out["fc2"]

  def define_loss(self) :
    self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.label,logits=self.label_out)
    self.cross_entropy = tf.reduce_mean(self.cross_entropy)

    correct_prediction = tf.equal(tf.argmax(self.label_out,1), tf.argmax(self.label,1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)

    self.accuracy = tf.reduce_mean(correct_prediction)

  def Get_initializer(self) :
    #tmp_list = []
    #for i in ["conv1","conv2","fc1","fc2"] :
    #  tmp_list.append(self.weight[i])
    #  tmp_list.append(self.bias[i])

    tmp_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "")
    return tf.variables_initializer(tmp_list, name="init")
    #tmp = tf.global_variables()
    #for i in tmp : print(i)
    #for i in tmp_list : print(i)
    #return tf.global_variables_initializer()

  def Get_feed_dict(self,image,label=[],dropout=1) :
    if len(label)==0 :
      return {self.image: image, self.dropout: 1}
    else :
      return {self.image: image, self.label: label, self.dropout: dropout}

  def Get_train_step(self) :
    step = tf.train.AdamOptimizer(1e-4).minimize(self.cross_entropy)
    #step = tf.train.GradientDescentOptimizer(0.1).minimize(self.cross_entropy)
    tf.add_to_collection("train_step", step)
    return step

  def Get_Saver(self) :
    #tmp_list = []
    #for i in ["conv1","conv2","fc1","fc2"] :
    #  tmp_list.append(self.weight[i])
    #  tmp_list.append(self.bias[i])
    tmp_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "")
    
    return tf.train.Saver(tmp_list,max_to_keep=1)

  def Restore(self,sess,model_dir) :
    #saver = tf.train.import_meta_graph(model_dir+"Model_out.meta")
    saver = self.Get_Saver()
    saver.restore( sess, tf.train.latest_checkpoint(model_dir) )
