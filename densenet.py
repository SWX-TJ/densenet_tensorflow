import tensorflow as tf
import tensorlayer as tl
import os
import datetime
import time
import numpy as np
from commonset import load_ckpt
def Compositeblock(Input,OutputChannel,is_training = True,name='compositebock'):
    with tf.variable_scope(name):
        net = tl.layers.BatchNormLayer(Input,decay=0.99, act=tf.nn.relu, is_train=is_training, name='batch_norm_2')
        net = tl.layers.Conv2dLayer(net,
                                    act=tf.identity,
                                    shape=(3, 3,net.outputs.get_shape()[-1],OutputChannel),
                                    strides=(1, 1, 1, 1),
                                    W_init=tf.truncated_normal_initializer(stddev=0.01),
                                    b_init=None,
                                    padding='SAME',
                                    name="conv_2")
        return net
def DenseBlock(network,n_layer,growth,reuse=False,name='denseblock'):
    with tf.variable_scope(name, reuse=reuse):
        for i in range(n_layer):
            layerblock = Compositeblock(network,growth,True,name="layerblock_"+str(i))
            network = tl.layers.ConcatLayer([network,layerblock],3,name="concat_"+str(i))
        return network
def transitionLayer(network,output_channel,is_training= True,name = "trainsition_layer"):
    with tf.variable_scope(name):
        net = tl.layers.BatchNormLayer(network,decay=0.99,act=tf.nn.relu,is_train=is_training,name='batch_norm_1')
        net = tl.layers.Conv2dLayer(net,
                                    act=tf.identity,
                                    shape=(1, 1, net.outputs.get_shape()[-1],output_channel),
                                    strides=(1, 1, 1, 1),
                                    W_init=tf.truncated_normal_initializer(stddev=0.01),
                                    b_init=None,
                                    padding='SAME',
                                    name="conv_1_1")
        net = tl.layers.PoolLayer(net,ksize =(1,2,2,1),strides=(1,2,2,1),padding="SAME",pool=tf.nn.avg_pool,name='poollayer')
        return net

def DenseNet(Input,block_L=100,growth_rate=32,compress_rate=0.5,is_training=True,reuse= False):
    L = int((block_L-4)/3)
    def reduce_dim(input_feature):
        return int(int(input_feature.get_shape().as_list()[-1]) * compress_rate)
    InputLayer = tl.layers.InputLayer(Input, name='InputLayer')  # (-1,640,640,3)
    with tf.variable_scope("dense_net", reuse=reuse):
        network = tl.layers.Conv2dLayer(InputLayer, act=tf.identity,
                                        shape=(3, 3, 3, 16),
                                        strides=(1, 1, 1, 1),
                                        W_init=tf.truncated_normal_initializer(stddev=0.01),
                                        b_init=None,
                                        padding='SAME', name='conv1_1')
        network = DenseBlock(network,L,growth_rate, reuse, name='deneseblock_1')
        network = transitionLayer(network, reduce_dim(network.outputs), is_training, name="transitionlayer_1")
        network = DenseBlock(network, L, growth_rate, reuse, name='deneseblock_2')
        network = transitionLayer(network, reduce_dim(network.outputs), is_training, name="transitionlayer_2")
        network = DenseBlock(network, L, growth_rate, reuse, name='deneseblock_3')
        #network = transitionLayer(network, reduce_dim(network.outputs), is_training, name="transitionlayer_3")
        network = tl.layers.BatchNormLayer(network,decay=0.99, act=tf.nn.relu, is_train=is_training, name='batch_norm_1')
        network = tl.layers.GlobalMeanPool2d(network,name="goal_pool")
        #network = tl.layers.FlattenLayer(network, name="flatten")
        network = tl.layers.DenseLayer(network, n_units=10, act=tf.identity, name="fc")
    return network

def distort_fn(x, is_train=False):
    #x = tl.prepro.crop(x, 24, 24, is_random=is_train)
    # if is_train:
    #     x = tl.prepro.flip_axis(x, axis=1, is_random=True)
    #    # x = tl.prepro.brightness(x, gamma=0.1, gain=1, is_random=True)
    x = x/256.0
    return x


if __name__ == '__main__':
    start_learning_rate = 0.001
    decay_rate = 0.9
    decay_steps = 1
    global_steps = tf.Variable(0, trainable=False)
    batchsize = 64
    train_epoch = 150
    project_current_dir = os.getcwd()
    curdata = datetime.datetime.now()  # current year,month,day
    model_dir = project_current_dir + "/dense_model/"
    X_train, y_train, X_test, y_test = tl.files.load_cifar10_dataset(shape=(-1, 32, 32, 3))
    X_train = distort_fn(X_train)
    X_test= distort_fn(X_test)
    sess = tf.InteractiveSession()
    train_x = tf.placeholder(dtype=tf.float32, shape=[None, 32, 32, 3], name='train_x')
    tf.summary.image('train_image', train_x)
    test_x = tf.placeholder(dtype=tf.float32, shape=[None, 32, 32, 3], name='test_x')
    real_y = tf.placeholder(dtype=tf.int64, shape=[None], name='real_y')
    test_y = tf.placeholder(dtype=tf.int64, shape=[None], name='test_y')
    net = DenseNet(train_x, 40, 12, 1, True, False)
    net_test = DenseNet(test_x, 40, 12, 1, False, True)
    with tf.name_scope("loss"):
        l2 = 0
        for w in tl.layers.get_variables_with_name('W', train_only=True, printable=False):
            print(w.name)
            l2 += tf.contrib.layers.l2_regularizer(1e-4)(w)
        cost = tl.cost.cross_entropy(net.outputs, real_y, name="train_loss") + l2
        tf.summary.scalar('loss', cost)
        tf.summary.scalar('l2_loss', l2)
        test_loss = tl.cost.cross_entropy(net_test.outputs, test_y, name="test_loss")
    with tf.name_scope("train"):
        train_params = net.all_params
        train_op = tf.train.MomentumOptimizer(momentum=0.9, learning_rate=start_learning_rate).minimize(cost)
        # with tf.Session(graph=g) as sess:
    tl.files.exists_or_mkdir(model_dir)
    merged = tf.summary.merge_all()
    trainwrite = tf.summary.FileWriter("logs/", sess.graph)
    sess.run(tf.global_variables_initializer())
    load_ckpt(sess=sess, save_dir=model_dir, var_list=net.all_params, printable=True)
    tensorboard_idx = 0
    saver = tf.train.Saver(max_to_keep=1)
    for epoch in range(train_epoch):
        idx = 0
        start_time = time.time()
        for batch_x, batch_y in tl.iterate.minibatches(X_train, y_train, batchsize, shuffle=True):
            sess.run(train_op, feed_dict={train_x: batch_x, real_y: batch_y})
            if idx % 10 == 0:
                print_loss = sess.run(cost, feed_dict={train_x: batch_x, real_y: batch_y})
                merge = sess.run(merged, feed_dict={train_x: batch_x, real_y: batch_y})
                trainwrite.add_summary(merge, tensorboard_idx)
                test_lossa = 0
                test_iter = 0
                for batch_test_x, batch_test_y in tl.iterate.minibatches(X_test, y_test, batchsize,shuffle=True):
                    #test_batch_x = tl.prepro.threading_data(batch_test_x, fn=distort_fn, is_train=False)
                    print_test_loss = sess.run(test_loss,feed_dict={test_x: batch_test_x, test_y: batch_test_y})
                    test_lossa = test_lossa + print_test_loss
                    test_iter = test_iter + 1
                print("idx:%d,epoch:%d,loss:%.4f,test_loss:%.4f,time:%.4f" % (
                    idx, epoch, print_loss, test_lossa/test_iter, time.time() - start_time))
                start_time = time.time()
                saver.save(sess, os.path.join(model_dir,'model.ckpt'), global_step=idx)
                #tl.files.save_ckpt(sess=sess, mode_name='model.ckpt', save_dir=model_dir, printable=False)
            idx = idx + 1
            sess.graph.finalize()
            tensorboard_idx = tensorboard_idx + 1






