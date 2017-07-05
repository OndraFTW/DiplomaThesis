#projekt: Paralelní trénování hlubokých neuronových sítí
#Popis: Funkčnost sdílená všemi neuronovými sítěmi
#Autor: Ondřej Šlampa, xslamp01@stud.fit.vutbr.cz

import tensorflow as tf
from operator import mul
from operator import itemgetter
from functools import reduce

class AbstractNetwork:
    def set_servers(self, servers):
        self.servers=[[server,0] for server in servers]
        self.weights=list()
    
    def weight_variable(self, shape, name):
        self.servers.sort(key=itemgetter(1))
        self.servers[0][1]+=reduce(mul,shape,1)
        with tf.device(self.servers[0][0]+"/gpu:0"):
            #print("weight added to: {0} {1}".format(self.servers[0][0],self.servers[0][1]))
            initial=tf.Variable(tf.truncated_normal(shape,dtype=tf.float32),dtype=tf.float32,name=name)
        self.weights.append(initial)
        return initial
    
    def conv2d(self, x, W, padding="SAME"):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding=padding)

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    
