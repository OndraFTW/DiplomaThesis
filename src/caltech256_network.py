#projekt: Paralelní trénování hlubokých neuronových sítí
#Popis: Funkčnost sdílená všemi neuronovými sítěmi trénovatelnými množinou Caltech256
#Autor: Ondřej Šlampa, xslamp01@stud.fit.vutbr.cz

import os
import numpy as np
import tensorflow as tf

class Caltech256Network:
    def get_train_dict_ops(self):
        return [self.image_batch,self.label_batch]
    
    def get_train_dict(self,n,sess):
        [images,labels]=sess.run([self.image_batch,self.label_batch])
        return {self.x:images,self.y:labels}
    
    def get_test_dict(self, sess):
        [images,labels]=sess.run([self.test_image_batch,self.test_label_batch])
        return {self.x:images,self.y:labels}
    
    def prepare_data(self):
        with tf.device("/cpu:0"):
            classes=[]
            for i in range(self.CLASSES_NO):
                z=[0]*self.CLASSES_NO
                z[i]=1
                classes.append(tf.constant(z))
            
            filenames=[]
            labels=[]
            test_filenames=[]
            test_labels=[]
            file_counter=0
            i=0
            for traindir in sorted(os.listdir(self.datadir)):
                cls=int(traindir.split(".")[0])-1
                label=classes[cls]
                for filename in sorted(os.listdir(self.datadir+"/"+traindir)):
                    file_counter+=1
                    full_path=os.path.join(os.path.join(self.datadir,traindir),filename)
                    if file_counter%100==0:
                        test_filenames.append(full_path)
                        test_labels.append(label)
                    else:
                        filenames.append(full_path)
                        labels.append(label)
            
            self.EXAMPLES_NO=len(filenames)
            
            train_file_queue=tf.train.slice_input_producer([filenames,labels],shuffle=True)
            train_file_contents=tf.read_file(train_file_queue[0])
            train_images=tf.image.decode_jpeg(train_file_contents,channels=3)
            train_images.set_shape([self.EXAMPLE_SIZE,self.EXAMPLE_SIZE,3])
            train_labels=train_file_queue[1]
            self.image_batch,self.label_batch=tf.train.batch([train_images, train_labels],batch_size=self.BATCH_SIZE,num_threads=4,capacity=self.BATCH_SIZE)
             
            test_file_queue=tf.train.slice_input_producer([test_filenames,test_labels],shuffle=False)
            test_file_contents=tf.read_file(test_file_queue[0])
            test_images=tf.image.decode_jpeg(test_file_contents,channels=3)
            test_images.set_shape([self.EXAMPLE_SIZE,self.EXAMPLE_SIZE,3])
            test_labels=test_file_queue[1]
            self.test_image_batch,self.test_label_batch=tf.train.batch([test_images, test_labels],batch_size=len(test_filenames))
            
