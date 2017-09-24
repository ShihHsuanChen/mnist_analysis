# DrawFilter.py

from sys import exit
import tensorflow as tf
import numpy as np
from LoadSample import LoadSample
from tensorflow.contrib.learn.python.learn.datasets import base
import matplotlib.pyplot as plt
from Model import Model

def main() :

  fname_test_image = "./data/t10k-images.idx3-ubyte"
  fname_test_label = "./data/t10k-labels.idx1-ubyte"

  test_sample  = LoadSample(fname_test_image,fname_test_label)
  print(test_sample.images.shape)

  model = Model("model")

  model.build()

  model.define_loss()

  train_step = model.Get_train_step()

  # restore
  modeldir = "./out/" 

  with tf.Session() as sess :

    model.Restore(sess,modeldir)

    W_conv1 = sess.run(model.weight["conv1"])
    W_conv2 = sess.run(model.weight["conv2"])
    b_conv1 = sess.run(model.bias["conv1"])
    b_conv2 = sess.run(model.bias["conv2"])

    x = tf.cast( np.array( [128.]*28*28 ).reshape([1,-1]), tf.float32 );
    x = tf.reshape(x,[-1,28,28,1])
    filter_1 = tf.nn.relu( tf.nn.conv2d(x, W_conv1, strides=[1,1,1,1],padding="SAME") + b_conv1 )
    pool_1   = tf.nn.max_pool( filter_1, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")
    print("filter_1: ",filter_1.shape)
    print("pool_1  : ",pool_1.shape)
    print("Finish loading model\n")
   
    # convolution 1 
    image_1 = np.zeros([32,5,5])
    for i in range(32) :
      tmp_image = np.zeros([5,5])

      for im in range(100) :
        tmp_pict = test_sample.images[im,:].reshape([28,28])
        for ix in range(2,26) :
          for iy in range(2,26) :
            tmp_weight = 0

            for ipx in range(-2,3) :
              for ipy in range(-2,3) :
                tmp_weight = tmp_weight + W_conv1[ipx,ipy,0,i] * tmp_pict[ix+ipx,iy+ipy]

            if tmp_weight + b_conv1[i] <= 0: tmp_weight = 0
            else : tmp_weight = tmp_weight + b_conv1[i]

            tmp_image = tmp_image + tmp_pict[ix-2:ix+3,iy-2:iy+3] * tmp_weight

      max_pt = np.max(tmp_image)
      min_pt = np.min(tmp_image)
      print("conv1 filter %d:  max: %.2f  min: %.2f" % (i,max_pt,min_pt) )
      scale = 1

      if max_pt <= 0 :
        scale = -1./min_pt
      else :
        scale = 1./max_pt

      tmp_image = tmp_image * scale * 128 + 128
      tmp_image = tf.cast( tf.reshape(tmp_image,[5,5]), tf.int32 )
      image_1[i,:,:] = sess.run(tmp_image)

      fig = plt.imshow(image_1[i,:,:], interpolation='none', cmap=plt.cm.gray_r)
      plt.colorbar()
      plt.savefig("filter/filter1_"+str(i)+".png")
      plt.clf()
      
    # convolution 2
    image_2 = np.zeros([64,14,14])

    for im in range(100) :
      tmp_pict = test_sample.images[im,:].reshape([28,28])
      x = tf.reshape(tmp_pict,[-1,28,28,1])
      tmp_conv1 = sess.run( tf.nn.relu( tf.nn.conv2d(x, W_conv1, strides=[1,1,1,1],padding="SAME") + b_conv1 ) )
      tmp_conv1 = tmp_conv1.reshape([28,28,32])

      pool_val = np.zeros([32,14,14])
      pool_x   = np.zeros([32,14,14])
      pool_y   = np.zeros([32,14,14])
      # get max_pool
      for ifilter in range(32) :
        
        for ipoolx in range(14) :
          for ipooly in range(14) :
            tmp_pool_val = -9999
            tmp_pool_x   = 0
            tmp_pool_y   = 0

            for tmp_i in range(2) :
              for tmp_j in range(2) :
                tmp_val = tmp_conv1[ipoolx*2+tmp_i,ipooly*2+tmp_j,ifilter]
                if tmp_val > tmp_pool_val :
                  tmp_pool_val = tmp_val
                  tmp_pool_x   = tmp_i
                  tmp_pool_y   = tmp_j

            pool_val[ifilter,ipoolx,ipooly] = tmp_pool_val
            pool_x  [ifilter,ipoolx,ipooly] = tmp_pool_x
            pool_y  [ifilter,ipoolx,ipooly] = tmp_pool_y

      print("make pool: image ",im)
      # convolution 2
      for ix in range(2,12) :
        for iy in range(2,12) :
          tmp_weight = np.zeros([32,64])
          tmp_sum = 0

          for i in range(64) :
            tmp_image = np.zeros([14,14])

            for ifilter in range(32) :
              Wx = np.sum( (W_conv2[:,:,ifilter,i] * pool_val[ifilter,ix+-2:ix+3,iy-2:iy+3]) )
              tmp_sum += Wx
              tmp_weight[ifilter,i] += Wx

            if tmp_sum + b_conv2[i] <= 0 : continue


            for ipx in range(-2,3) :
              for ipy in range(-2,3) :
                for ifilter in range(32) : 

                  tmp_px = int( 4+ipx*2+pool_x[ifilter,ix+ipx,iy+ipy] )
                  tmp_py = int( 4+ipy*2+pool_y[ifilter,ix+ipx,iy+ipy] )
                  tmp_image[tmp_px:tmp_px+5,tmp_py:tmp_py+5] += image_1[ifilter,:,:].astype(float) * tmp_weight[ifilter,i]

            image_2[i,:,:] += tmp_image

    for i in range(64) :
      tmp_image = image_2[i,:,:]
      max_pt = np.max(tmp_image)
      min_pt = np.min(tmp_image)
      print("conv2 filter %d:  max: %.2f  min: %.2f" % (i,max_pt,min_pt) )
      scale = 1

      if max_pt < 0 :
        scale = -1./min_pt
      elif min_pt > 0 :
        scale = 1./max_pt
      else : continue

      tmp_image = tmp_image * scale * 128 + 128
      tmp_image = tf.cast( tmp_image, tf.int32 )
      image_2[i,:,:] = sess.run(tmp_image)

      fig = plt.imshow(image_2[i,:,:], interpolation='none', cmap=plt.cm.gray_r)
      plt.colorbar()
      plt.savefig("filter/filter2_"+str(i)+".png")
      plt.clf()


if __name__=="__main__" :
  main()
