#projekt: Paralelní trénování hlubokých neuronových sítí
#Popis: Synchronizační prvek bariéra
#Autor: Ondřej Šlampa, xslamp01@stud.fit.vutbr.cz

import tensorflow as tf

class Barrier:
    def __init__(self,tasks,index,name):
        self.tasks=tasks
        self.index=index
        self.enque_list=[]
        self.deque_list=[]
        for (i, task) in enumerate(tasks):
            queue_name=name+str(i)
            with tf.device(task):
                queue=tf.FIFOQueue(len(tasks),[tf.int32],name=queue_name,shared_name=queue_name)
            TASK_INDEX=tf.constant(index, name="TASK_INDEX_"+str(index))
            enque=queue.enqueue([TASK_INDEX])
            deque=queue.dequeue()
            self.enque_list.append(enque)
            self.deque_list.append(deque)
    
    def wait(self,sess):
        for _ in self.tasks:
            sess.run(self.enque_list[self.index])
        for i in range(len(self.tasks)):
            sess.run(self.deque_list[i])
