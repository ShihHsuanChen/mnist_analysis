# train.py

from sys import exit
import tensorflow as tf
import numpy as np
from LoadSample import LoadSample
from tensorflow.contrib.learn.python.learn.datasets import base
import matplotlib.pyplot as plt
from Model import Model

def main() :

  Ntrain = 20000

  # Input data from file
  fname_train_image = "./data/train-images.idx3-ubyte"
  fname_train_label = "./data/train-labels.idx1-ubyte"

  fname_test_image = "./data/t10k-images.idx3-ubyte"
  fname_test_label = "./data/t10k-labels.idx1-ubyte"

  train_sample, valid_sample = LoadSample(fname_train_image,fname_train_label,validate=True)
  test_sample  = LoadSample(fname_test_image,fname_test_label)

  print(train_sample.images.shape)

  while input("q: quit, other : continue ")!='q' :
    i = np.random.randint(train_sample.images.shape[0])
    arr = train_sample.images[i,:]
    lab = train_sample.labels[i,:]
    for j in range(10) :
      if lab[j]==1 : print("Number = ", j)
    arr = 255*np.ones([28,28]) - arr.reshape(28,28)
    plt.gray()
    plt.imshow(arr)
    plt.show()
    i += 1

  mnist = base.Datasets(train=train_sample, validation=valid_sample, test=test_sample)

  model = Model("model")

  model.build()

  model.define_loss()
  
  train_step = model.Get_train_step()


  ## save graph
  saver = model.Get_Saver()
  graph_location = "./logs/"
  sum_accuracy = tf.summary.scalar("accuracy",model.accuracy)
  sum_xentropy = tf.summary.scalar("cross entropy",model.cross_entropy)
  train_writer = tf.summary.FileWriter(graph_location)
  train_writer.add_graph(tf.get_default_graph())
  print('Saving graph to: %s' % graph_location)
  ##

  with tf.Session() as sess :
    init = model.Get_initializer()
    sess.run( init )
    for i in range(Ntrain) :
      batch = mnist.train.next_batch(50)
      if i % 100 == 0 :
        feed_dict = model.Get_feed_dict(batch[0],batch[1],1)
        train_accuracy   = model.accuracy.eval(feed_dict=feed_dict)
        tmp_sum_accuracy = sum_accuracy.eval(feed_dict=feed_dict)
        tmp_sum_xentropy = sum_xentropy.eval(feed_dict=feed_dict)
        train_writer.add_summary(tmp_sum_accuracy,i)
        train_writer.add_summary(tmp_sum_xentropy,i)
        print('step %d, training accuracy %g' % (i, train_accuracy))

      feed_dict = model.Get_feed_dict(batch[0],batch[1],0.5)
      train_step.run(feed_dict=feed_dict) # dropout 50% neurons within 1024->10 layer

    feed_dict = model.Get_feed_dict(mnist.test.images, mnist.test.labels, 1)
    print('test accuracy %g' % model.accuracy.eval(feed_dict=feed_dict))

    savePath = saver.save(sess, "./out/Model_out")

  train_writer.flush()
  train_writer.close()

if __name__=="__main__" :
  main()
