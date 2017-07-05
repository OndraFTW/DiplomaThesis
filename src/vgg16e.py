#projekt: Paralelní trénování hlubokých neuronových sítí
#Popis: Síť VGG16E
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
        self.BATCH_SIZE=16
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
            with tf.name_scope("Cost") as scope:
                self.cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.model,labels=self.y))
        return self.cost
    
    def get_optimizer(self):
        if self.optimizer is None:
            with tf.name_scope("Optimizer") as scope:
                self.optimizer=tf.train.AdamOptimizer()
        return self.optimizer
    
    def set_optimizer(self, opt):
        self.optimizer=opt
    
    def get_training_operation(self, global_step=None):
        if self.training_operation is None:
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

    def get_model(self):
        if(not self.model is None):
            return self.model
        
        self.x=tf.placeholder(tf.float32,[None, 224, 224, 3], name="x")
        self.y=tf.placeholder(tf.float32,[None, self.CLASSES_NO], name="y")
        
        with tf.name_scope("Network") as scope:
            
            with tf.name_scope("layer1") as scope:
                cl11_weights=self.weight_variable([3, 3, 3, 64], "kernel_1")
                cl12_weights=self.weight_variable([3, 3, 64, 64], "kernel_2")
                cl11_conv=tf.nn.relu(self.conv2d(self.x, cl11_weights))
                cl12_conv=tf.nn.relu(self.conv2d(cl11_conv, cl12_weights))
                cl1_pool=self.max_pool_2x2(cl12_conv)
            
            with tf.name_scope("layer2") as scope:
                cl21_weights=self.weight_variable([3, 3, 64, 128], "kernel_1")
                cl22_weights=self.weight_variable([3, 3, 128, 128], "kernel_2")
                cl21_conv=tf.nn.relu(self.conv2d(cl1_pool, cl21_weights))
                cl22_conv=tf.nn.relu(self.conv2d(cl21_conv, cl22_weights))
                cl2_pool=self.max_pool_2x2(cl22_conv)
            
            with tf.name_scope("layer3") as scope:
                cl31_weights=self.weight_variable([3, 3, 128, 256], "kernel_1")
                cl32_weights=self.weight_variable([3, 3, 256, 256], "kernel_2")
                cl33_weights=self.weight_variable([3, 3, 256, 256], "kernel_3")
                cl34_weights=self.weight_variable([3, 3, 256, 256], "kernel_4")
                cl31_conv=tf.nn.relu(self.conv2d(cl2_pool, cl31_weights))
                cl32_conv=tf.nn.relu(self.conv2d(cl31_conv, cl32_weights))
                cl33_conv=tf.nn.relu(self.conv2d(cl32_conv, cl33_weights))
                cl34_conv=tf.nn.relu(self.conv2d(cl33_conv, cl34_weights))
                cl3_pool=self.max_pool_2x2(cl34_conv)
            
            with tf.name_scope("layer4") as scope:
                cl41_weights=self.weight_variable([3, 3, 256, 512], "kernel_1")
                cl42_weights=self.weight_variable([3, 3, 512, 512], "kernel_2")
                cl43_weights=self.weight_variable([3, 3, 512, 512], "kernel_3")
                cl44_weights=self.weight_variable([3, 3, 512, 512], "kernel_4")
                cl41_conv=tf.nn.relu(self.conv2d(cl3_pool, cl41_weights))
                cl42_conv=tf.nn.relu(self.conv2d(cl41_conv, cl42_weights))
                cl43_conv=tf.nn.relu(self.conv2d(cl42_conv, cl43_weights))
                cl44_conv=tf.nn.relu(self.conv2d(cl43_conv, cl44_weights))
                cl4_pool=self.max_pool_2x2(cl44_conv)
            
            with tf.name_scope("layer5") as scope:
                cl51_weights=self.weight_variable([3, 3, 512, 512], "kernel_1")
                cl52_weights=self.weight_variable([3, 3, 512, 512], "kernel_2")
                cl53_weights=self.weight_variable([3, 3, 512, 512], "kernel_3")
                cl54_weights=self.weight_variable([3, 3, 512, 512], "kernel_4")
                cl51_conv=tf.nn.relu(self.conv2d(cl4_pool, cl51_weights))
                cl52_conv=tf.nn.relu(self.conv2d(cl51_conv, cl52_weights))
                cl53_conv=tf.nn.relu(self.conv2d(cl52_conv, cl53_weights))
                cl54_conv=tf.nn.relu(self.conv2d(cl53_conv, cl54_weights))
                cl5_pool=self.max_pool_2x2(cl54_conv)
                
                cl5_flat_len=7*7*512
                cl5_flat=tf.reshape(cl5_pool, [-1, cl5_flat_len])
            
            with tf.name_scope("layer6") as scope:
                number_of_slides=8
                fl1_weight_slices=[self.weight_variable([cl5_flat_len//number_of_slides,4096], "weight_{0}".format(i)) for i in range(number_of_slides)]
                fl1_weights=tf.concat(fl1_weight_slices,0)
                fl1_out=tf.nn.relu(tf.matmul(cl5_flat, fl1_weights))
            
            with tf.name_scope("layer7") as scope:
                fl2_weights=self.weight_variable([4096,4096], "weights")
                fl2_out=tf.nn.relu(tf.matmul(fl1_out, fl2_weights))
            
            with tf.name_scope("layer8") as scope:
                fl3_weights=self.weight_variable([4096,self.CLASSES_NO], "weights")
                fl3_out=tf.matmul(fl2_out, fl3_weights)
        
        self.saver=tf.train.Saver(self.weights)
        self.model=fl3_out
        return self.model
