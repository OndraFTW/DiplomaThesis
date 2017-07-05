#projekt: Paralelní trénování hlubokých neuronových sítí
#Popis: Síť SqueezeNet
#Autor: Ondřej Šlampa, xslamp01@stud.fit.vutbr.cz

import os
import numpy as np
import tensorflow as tf
from abstract_network import AbstractNetwork
from caltech256_network import Caltech256Network

class Network(AbstractNetwork,Caltech256Network):
    def __init__(self, datadir, servers):
        self.datadir=datadir
        self.set_servers(servers)
        
        self.EXAMPLE_SIZE=224
        self.BATCH_SIZE=96
        self.BATCH_NO=128*128
        self.CLASSES_NO=256
        
        self.model=None
        self.cost=None
        self.accuracy=None
        self.optimizer=None
        self.training_operation=None
        
        self.summaries=[]
    
    def get_cost(self):
        if self.cost is None:
            with tf.device("/gpu:0"):
                with tf.name_scope("Cost") as scope:
                    self.cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.model,labels=self.y))
                    self.summaries.append(tf.summary.scalar("cost", self.cost))
        return self.cost
    
    def get_optimizer(self):
        if self.optimizer is None:
            with tf.device("/gpu:0"):
                with tf.name_scope("Optimizer") as scope:
                    self.optimizer=tf.train.AdamOptimizer(learning_rate=8.0*0.001)
        return self.optimizer
    
    def set_optimizer(self, opt):
        self.optimizer=opt
    
    def get_training_operation(self, global_step=None):
        if self.training_operation is None:
            with tf.device("/gpu:0"):
                with tf.name_scope("Trainer") as scope:
                    if global_step is None:
                        self.training_operation=self.get_optimizer().minimize(self.get_cost())
                    else:
                        self.training_operation=self.get_optimizer().minimize(self.get_cost(), global_step=global_step)
        return self.training_operation
    
    def get_accuracy(self):
        if self.accuracy is None:
            with tf.name_scope("Accuracy") as scope:
                correct=tf.equal(tf.argmax(self.get_model(),1), tf.argmax(self.y,1))
                self.accuracy=tf.reduce_mean(tf.cast(correct, tf.float32))
                self.summaries.append(tf.summary.scalar("accuracy", self.accuracy))
        return self.accuracy

    def add_fire(self, in_model, squeeze_size, expand1_size, expand3_size):
        in_size=in_model.get_shape().as_list()[-1]
        squeeze_kernel=self.weight_variable([1, 1, in_size, squeeze_size], "squeeze_kernel")
        expand1_kernel=self.weight_variable([1, 1, squeeze_size, expand1_size], "expand1_kernel")
        expand3_kernel=self.weight_variable([3, 3, squeeze_size, expand3_size], "expand3_kernel")
        
        squeezed=tf.nn.relu(tf.nn.conv2d(in_model, squeeze_kernel, strides=[1, 1, 1, 1],padding="SAME"))
        expanded1=tf.nn.conv2d(squeezed, expand1_kernel, strides=[1, 1, 1, 1],padding="SAME")
        expanded3=tf.nn.conv2d(squeezed, expand3_kernel, strides=[1, 1, 1, 1],padding="SAME")
        
        out_model=tf.nn.relu(tf.concat([expanded1,expanded3],3))
        return out_model

    def get_model(self):
        if(not self.model is None):
            return self.model
        
        with tf.device("/gpu:0"):
            self.x=tf.placeholder(tf.float32,[None, 224, 224, 3], name="x")
            self.y=tf.placeholder(tf.float32,[None, self.CLASSES_NO], name="y")
        
            with tf.name_scope("Network") as scope:
                
                with tf.name_scope("layer1") as scope:
                    cl1_weights=self.weight_variable([7, 7, 3, 96], "kernel_1")
                    cl1_conv=tf.nn.relu(tf.nn.conv2d(self.x, cl1_weights, strides=[1, 2, 2, 1],padding="SAME"))
                    cl1_pool=tf.nn.max_pool(cl1_conv, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],padding="SAME")
                
                model=cl1_pool
                with tf.name_scope("layer2") as scope:
                    model=self.add_fire(model,16,64,64)
                
                with tf.name_scope("layer3") as scope:
                    model=self.add_fire(model,16,64,64)
                
                with tf.name_scope("layer4") as scope:
                    model=self.add_fire(model,32,128,128)
                    model=tf.nn.max_pool(model, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],padding="SAME")
                
                with tf.name_scope("layer5") as scope:
                    model=self.add_fire(model,32,128,128)
                
                with tf.name_scope("layer6") as scope:
                    model=self.add_fire(model,48,192,192)
                
                with tf.name_scope("layer7") as scope:
                    model=self.add_fire(model,48,192,192)
                
                with tf.name_scope("layer8") as scope:
                    model=self.add_fire(model,64,256,256)
                    model=tf.nn.max_pool(model, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],padding="SAME")
                
                with tf.name_scope("layer9") as scope:
                    model=self.add_fire(model,64,256,256)
                
                with tf.name_scope("layer10") as scope:
                    model=tf.nn.dropout(model, 0.5)
                    cl10_weights=self.weight_variable([1, 1, 512, self.CLASSES_NO], "kernel_10")
                    model=tf.nn.relu(tf.nn.conv2d(model, cl10_weights, strides=[1, 1, 1, 1],padding="SAME"))
                    model=tf.nn.avg_pool(model, ksize=[1, 14, 14, 1], strides=[1, 1, 1, 1],padding="VALID")
                    model=tf.reshape(model, [-1, self.CLASSES_NO])
                
        self.saver=tf.train.Saver(self.weights)
        self.model=model
        return self.model
