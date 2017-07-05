#projekt: Paralelní trénování hlubokých neuronových sítí
#Popis: Síť Resnet34
#Autor: Ondřej Šlampa, xslamp01@stud.fit.vutbr.cz

import os
import numpy as np
import tensorflow as tf
from abstract_network import AbstractNetwork
from caltech256_network import Caltech256Network
from tensorflow.python.client import timeline

class Network(AbstractNetwork,Caltech256Network):
    def __init__(self, datadir, servers):
        self.datadir=datadir
        self.set_servers(servers)
        
        self.EXAMPLE_SIZE=224
        self.BATCH_SIZE=64
        self.BATCH_NO=128
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
        return self.cost
    
    def get_optimizer(self):
        if self.optimizer is None:
            with tf.device("/gpu:0"):
                with tf.name_scope("Optimizer") as scope:
                    self.optimizer=tf.train.AdamOptimizer()
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

    def add_block(self, in_model, step):
        in_size=in_model.get_shape().as_list()[-1]
        
        if step:
            out_size=2*in_size
        else:
            out_size=in_size
        
        l1_kernel=self.weight_variable([3, 3, in_size, out_size], "kernel_1")
        l2_kernel=self.weight_variable([3, 3, out_size, out_size], "kernel_2")
        sc_kernel=self.weight_variable([1, 1, in_size, out_size], "sc_kernel")
        
        if step:
            first_strides=[1, 2, 2, 1]
        else:
            first_strides=[1, 1, 1, 1]
        
        model=tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv2d(in_model, l1_kernel, strides=first_strides,padding="SAME"),fused=True))
        model=tf.contrib.layers.batch_norm(tf.nn.conv2d(model, l2_kernel, strides=[1, 1, 1, 1],padding="SAME"),fused=True)
        
        shortcut=tf.nn.conv2d(in_model, sc_kernel, strides=first_strides,padding="SAME")
        
        out_model=tf.nn.relu(tf.add(model,shortcut))
        return out_model

    def get_model(self):
        if(not self.model is None):
            return self.model
        
        with tf.device("/gpu:0"):
            self.x=tf.placeholder(tf.float32,[None, 224, 224, 3], name="x")
            self.y=tf.placeholder(tf.float32,[None, self.CLASSES_NO], name="y")
            
            with tf.name_scope("Network") as scope:
                
                with tf.name_scope("layer1") as scope:
                    cl11_weights=self.weight_variable([7, 7, 3, 64], "kernel_1")
                    model=tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv2d(self.x, cl11_weights, strides=[1, 2, 2, 1],padding="SAME"),fused=True))
                    model=tf.nn.max_pool(model, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding="SAME")
                
                with tf.name_scope("layer2") as scope:
                    model=self.add_block(model,False)
                
                with tf.name_scope("layer3") as scope:
                    model=self.add_block(model,False)
                
                with tf.name_scope("layer4") as scope:
                    model=self.add_block(model,False)
                
                with tf.name_scope("layer5") as scope:
                    model=self.add_block(model,True)
                
                with tf.name_scope("layer6") as scope:
                    model=self.add_block(model,False)
                
                with tf.name_scope("layer7") as scope:
                    model=self.add_block(model,False)
                
                with tf.name_scope("layer8") as scope:
                    model=self.add_block(model,False)
                
                with tf.name_scope("layer9") as scope:
                    model=self.add_block(model,True)
                
                with tf.name_scope("layer10") as scope:
                    model=self.add_block(model,False)
                
                with tf.name_scope("layer11") as scope:
                    model=self.add_block(model,False)
                
                with tf.name_scope("layer12") as scope:
                    model=self.add_block(model,False)
                
                with tf.name_scope("layer13") as scope:
                    model=self.add_block(model,False)
                
                with tf.name_scope("layer14") as scope:
                    model=self.add_block(model,False)
                
                with tf.name_scope("layer15") as scope:
                    model=self.add_block(model,True)
                
                with tf.name_scope("layer16") as scope:
                    model=self.add_block(model,False)
                
                with tf.name_scope("layer17") as scope:
                    model=self.add_block(model,False)
                
                with tf.name_scope("layer18") as scope:
                    model=tf.nn.avg_pool(model, ksize=[1, 7, 7, 1], strides=[1, 1, 1, 1],padding="VALID")
                    fl1_weights=self.weight_variable([512,self.CLASSES_NO], "weights")
                    model=tf.reshape(model, [-1, 512])
                    model=tf.nn.relu(tf.matmul(model, fl1_weights))
        
        self.saver=tf.train.Saver(self.weights)
        self.model=model
        return self.model
