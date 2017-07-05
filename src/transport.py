#projekt: Paralelní trénování hlubokých neuronových sítí
#Popis: Měření délky přenosu
#Autor: Ondřej Šlampa, xslamp01@stud.fit.vutbr.cz

import os
import sys
import logging
import argparse
import statistics
import tensorflow as tf
from timeit import default_timer as timer
from importlib.machinery import SourceFileLoader

DEFAULT_LOG_DIR="./logs"

def main():
    parser=argparse.ArgumentParser(description="Distributed neural network weights transporter.")
    parser.add_argument("--logdir",type=str,help="directory for logs")
    parser.add_argument("--my-address",type=str,required=True,help="my address")
    parser.add_argument("--other-address",type=str,required=True,help="other computer address")
    parser.add_argument("--my-role",type=str,required=True,choices=["ps","worker"],help="my role")
    parser.add_argument("--net",type=str,required=True,help="network")
    args=parser.parse_args()
    
    if args.my_role=="ps":
        ps_hosts=[args.my_address+":4000"]
        worker_hosts=[args.other_address+":4001"]
    else:
        ps_hosts=[args.other_address+":4000"]
        worker_hosts=[args.my_address+":4001"]
    
    cluster=tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
    server=tf.train.Server(cluster, job_name=args.my_role, task_index=0)
    ps_tasks=["/job:ps/task:0"]
    worker_tasks=["/job:worker/task:0"]
    all_tasks=ps_tasks+worker_tasks
    task_index=0
    datadir=""
    
    logger=logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logdir=args.logdir
    if logdir:
        if not os.path.exists(logdir):
            os.mkdir(logdir)
        logfile=logdir+"/"+args.my_role+str(task_index)+".log"
        out_hdlr=logging.FileHandler(logfile)
    else:
        logdir=DEFAULT_LOG_DIR+"/"+args.my_role+str(task_index)
        if not os.path.exists(logdir):
            os.mkdir(logdir)
        out_hdlr=logging.StreamHandler(sys.stdout)
    out_hdlr.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
    out_hdlr.setLevel(logging.DEBUG)
    logger.addHandler(out_hdlr)
    
    logger.info("Logging started.")
    
    logger.info("Netfile: {0}.".format(args.net))
    logger.info("Ps hosts: {0}.".format(ps_hosts))
    logger.info("Worker hosts: {0}.".format(worker_hosts))
    logger.info("Role: {0}.".format(args.my_role))
    
    enque_session_start_list=[]
    deque_session_start_list=[]
    enque_worker_list=[]
    deque_worker_list=[]
    
    for (i,task) in enumerate(all_tasks):
        name_start="session_start_queue_"+str(i)
        with tf.device(task):
            queue_start=tf.FIFOQueue(len(all_tasks),[tf.int32],name=name_start,shared_name=name_start)
        TASK_INDEX=tf.constant(task_index, name="TASK_INDEX_"+str(task_index))
        with tf.device("/cpu:0"):
            enque_session_start=queue_start.enqueue([TASK_INDEX])
            deque_session_start=queue_start.dequeue()
            enque_session_start_list.append(enque_session_start)
            deque_session_start_list.append(deque_session_start)
    
    for (i,ps_task) in enumerate(ps_tasks):        
        name="worker_end_queue_"+str(i)
        with tf.device(ps_task):
            queue=tf.FIFOQueue(len(worker_hosts),[tf.int32],name=name,shared_name=name)
        TASK_INDEX=tf.constant(task_index, name="TASK_INDEX_"+str(task_index))
        with tf.device("/cpu:0"):
            enque_worker=queue.enqueue([TASK_INDEX])
            deque_worker=queue.dequeue()
            enque_worker_list.append(enque_worker)
            deque_worker_list.append(deque_worker)
    
    if args.my_role=="ps":
        node_index=0
    else:
        node_index=1
    
    logger.info("Loading net.")
    with tf.device("/cpu:0"): 
        net_module=SourceFileLoader("Module", args.net).load_module()
        net=net_module.Network(datadir,ps_tasks)
        model=net.get_model()
        accuracy=net.get_accuracy()
        global_step=tf.Variable(0,name="global_step",trainable=False)
        training_operation=net.get_training_operation(global_step)
        init_op=tf.global_variables_initializer()
    
    logger.info("Create session.")
    t=server.target
    logger.info(t)
    is_chief=(args.my_role=="worker")
    logger.info("Chief {0}.".format(is_chief))
    
    logger.info("Starting session.")
    with tf.Session(t,config=tf.ConfigProto(allow_soft_placement=True,intra_op_parallelism_threads=0,inter_op_parallelism_threads=0)) as sess:
        logger.info("Started session.")
        
        if is_chief:
            sess.run(init_op)
        
        logger.info("Session sync start.")
        for _ in all_tasks:
            sess.run(enque_session_start_list[node_index])
        for i in range(len(all_tasks)):
            sess.run(deque_session_start_list[i])
        logger.info("Session sync end.")
        
        if args.my_role=="ps":
            logger.info("Worker token collecting started.")
            start_time=timer()
            for _ in worker_hosts:
                index=sess.run(deque_worker_list[task_index])
                logger.info("Worker {0} token collected.".format(index))
            end_time=timer()
            logger.info("Worker token collecting stopped.")
        else:
            step=0
            load_time_list=[]
            start_time=timer()
            while step<net.BATCH_NO:
                step_load1=timer()
                w=sess.run(net.weights)
                step_load2=timer()
                load_time=step_load2-step_load1
                logger.info("Weight load time: {0}".format(load_time))
                load_time_list.append(load_time)
                step+=1
            end_time=timer()
            logger.info("Complete time: {0}".format(end_time-start_time))
            logger.info("Load stats: mean: {0} stdev: {1}".format(statistics.mean(load_time_list), statistics.pstdev(load_time_list)))
            
            logger.info("End sync start.")
            for i in range(len(ps_tasks)):
                sess.run(enque_worker_list[i])
            logger.info("End sync end.")

if __name__ == "__main__":
    main()
