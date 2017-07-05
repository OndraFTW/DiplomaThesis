#projekt: Paralelní trénování hlubokých neuronových sítí
#Popis: Distribuované trénování
#Autor: Ondřej Šlampa, xslamp01@stud.fit.vutbr.cz

import os
import sys
import time
import socket
import argparse
import tensorflow as tf
import logging
import statistics
import pickle
from barrier import Barrier
from timeit import default_timer as timer
from importlib.machinery import SourceFileLoader
from tensorflow.python.client import timeline

tf.logging.set_verbosity(tf.logging.DEBUG)

DEFAULT_NODE_FILE="./nodefile.txt"
DEFAULT_TMP_DIR="/tmp"
FIRST_PORT=4000
MERGED_PORT=5050
DEFAULT_LOG_DIR="./logs"

def parse_args():
    parser=argparse.ArgumentParser(description="Distributed neural network trainer.")
    parser.add_argument("--nodefile",type=str,default=DEFAULT_NODE_FILE,help="nodefile file")
    parser.add_argument("--datadir",type=str,default=DEFAULT_TMP_DIR,help="directory with data files")
    parser.add_argument("--logdir",type=str,help="directory for logs")
    parser.add_argument("--net",type=str,required=True,help="network")
    parser.add_argument("--weights",type=str,help="file with saved weights")
    parser.add_argument("--index",type=int,help="node index in nodefile")
    parser.add_argument("--sync",type=str,default="none",choices=["none","join","split"],help="synchronize workers (defaults to none)")
    parser.add_argument("--server",type=str,default="central",choices=["central","distributed"],help="type of parameter server (defaults to central)")
    parser.add_argument("--accuracy",action="store_true",default=False,help="compute accuracy (defaults to False)")
    parser.add_argument("--profile",action="store_true",default=False,help="profile every 100th batch (defaults to False)")
    args=parser.parse_args()
    
    nodefile_file=open(args.nodefile)
    nodefile=[]
    for (i, node) in enumerate(nodefile_file):
        node=node.strip()
        if ":" in node:
            (address, port)=node.split(":")
            nodefile.append((address, port))
        else:
            nodefile.append((node, str(FIRST_PORT+i)))
    
    index=args.index
    
    if index==0:
        role="ps"
    else:
        role="worker"
        index-=1
    nodefile=[":".join(node) for node in nodefile]
    ps_hosts=[nodefile[0]]
    worker_hosts=nodefile[1:]
    
    return (ps_hosts, worker_hosts, args.server, role, index, args.sync, args.datadir, args.logdir, args.net, args.weights, args.accuracy, args.profile)

def send_object(s,o):
    po=pickle.dumps(o,-1)
    size=bytes("{0:64}".format(len(po)),encoding="UTF-8")
    s.sendall(size)
    s.sendall(po)

def recv_object(s):
    to_receive=int(s.recv(64))
    received=[]
    while to_receive>0:
        r=s.recv(to_receive)
        received.append(r)
        to_receive-=len(r)
    return pickle.loads(b"".join(received))

def main():
    (ps_hosts, worker_hosts, server_type, job_name, task_index, sync, datadir, logdir, netfile, weightsfile, do_accuracy, do_profile)=parse_args()
    cluster=tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
    configuration=tf.ConfigProto(allow_soft_placement=True,intra_op_parallelism_threads=4,inter_op_parallelism_threads=4)
    server=tf.train.Server(cluster,job_name=job_name,task_index=task_index,config=configuration)
    ps_tasks=["/job:ps/task:0"]
    worker_tasks=["/job:worker/task:{0}".format(i) for i in range(len(worker_hosts))]
    all_tasks=ps_tasks+worker_tasks
    
    if server_type=="distributed":
        server_tasks=worker_tasks
    else:
        server_tasks=ps_tasks
    
    if job_name=="ps":
        node_index=task_index
    else:
        node_index=task_index+1
    
    logger=logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    if logdir:
        if not os.path.exists(logdir):
            os.mkdir(logdir)
        logfile=logdir+"/"+job_name+str(task_index)+".log"
        out_hdlr=logging.FileHandler(logfile)
    else:
        logdir=DEFAULT_LOG_DIR+"/"+job_name+str(task_index)
        if not os.path.exists(logdir):
            os.mkdir(logdir)
        out_hdlr=logging.StreamHandler(sys.stdout)
    out_hdlr.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
    out_hdlr.setLevel(logging.DEBUG)
    logger.addHandler(out_hdlr)
    
    logger.info("Logging started.")
    
    logger.info("Netfile: {0}.".format(netfile))
    logger.info("Ps hosts: {0}.".format(ps_hosts))
    logger.info("Worker hosts: {0}.".format(worker_hosts))
    logger.info("Server type: {0}.".format(server_type))
    logger.info("Synchronization: {0}.".format(sync))
    
    session_start=Barrier(all_tasks,node_index,"session_start")
    session_end=Barrier(all_tasks,node_index,"session_end")
    training_start=Barrier(all_tasks,node_index,"training_start")
    training_end=Barrier(all_tasks,node_index,"training_end")
    
    if job_name=="ps":
        net_module=SourceFileLoader("Module", netfile).load_module()
        net=net_module.Network(datadir,server_tasks)
        model=net.get_model()
        logger.info("Starting session.")
        with tf.Session(server.target,config=configuration) as sess:
            logger.info("Started session.")
            
            logger.info("Session start sync start.")
            session_start.wait(sess)
            logger.info("Session start sync end.")
            
            if weightsfile:
                net.saver.restore(sess,weightsfile)
            
            logger.info("Training start sync start.")
            training_start.wait(sess)
            logger.info("Training start sync end.")
            
            logger.info("Training end sync start.")
            start_time=timer()
            training_end.wait(sess)
            end_time=timer()
            logger.info("Training end sync end.")
            
            if task_index==0:
                net.saver.save(sess,logdir+"/model")
            
            logger.info("Session end sync start.")
            session_end.wait(sess)
            logger.info("Session end sync end.")
        logger.info("Computation time: {0}.".format(end_time-start_time))
        print("{0} {1}".format(0,end_time-start_time),file=open(logdir+"/stats.txt","w"))
    elif job_name=="worker":
        logger.info("Worker {0} started.".format(task_index))
        is_chief=(task_index==0)
        do_merged=(sync=="join" or sync=="split")
        
        with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d"%task_index,cluster=cluster)):
            net_module=SourceFileLoader("Module", netfile).load_module()
            net=net_module.Network(datadir,server_tasks)
            model=net.get_model()
            
            global_step=tf.Variable(0,name="global_step",trainable=False)
            accuracy=net.get_accuracy()
            
            if sync!="none":
                if sync=="split":
                    net.BATCH_SIZE//=len(worker_hosts)
                else:
                    net.BATCH_NO//=len(worker_hosts)
                aggregate=len(worker_hosts)
                with tf.device("/gpu:0"):
                    net.set_optimizer(tf.train.SyncReplicasOptimizer(net.get_optimizer(), 
                                    replicas_to_aggregate=aggregate,
                                    total_num_replicas=len(worker_hosts)
                                    ))
            
            net.prepare_data()
            
            cost=net.get_cost()
            training_operation=net.get_training_operation(global_step)
            
            init_op=tf.global_variables_initializer()
            merged=tf.summary.merge(net.summaries)
        
        if sync!="none" and is_chief:
            optimizer=net.get_optimizer()
            init_token_op=optimizer.get_init_tokens_op()
            chief_queue_runner=optimizer.get_chief_queue_runner()
        
        logger.info("Set up Supervisor.")
        sv=tf.train.Supervisor(is_chief=is_chief,
                         logdir=logdir,
                         saver=None,
                         init_op=init_op,
                         summary_op=None,
                         global_step=global_step
                         )
        
        if do_merged:
            logger.info("Started socket creation.")
            my_socket=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            my_socket.setblocking(True)
            if  is_chief:
                my_socket.bind((worker_hosts[0].split(":")[0], MERGED_PORT))
                my_socket.listen(1)
                connection_list=[]
                for _ in worker_hosts[1:]:
                    (conn,addr)=my_socket.accept()
                    connection_list.append(conn)
            else:
                time.sleep(5)
                my_socket.connect((worker_hosts[0].split(":")[0], MERGED_PORT))
        logger.info("Ended socket creation.")
        
        cost_file=open(logdir+"/loss.csv","w")
        
        logger.info("Create session.")
        t=server.target
        logger.info(t)
        logger.info("Chief {0}.".format(is_chief))
        logger.info("Starting session.")
        with sv.managed_session(t,config=configuration) as sess:
            logger.info("Started session.")
            
            logger.info("Session start sync start.")
            session_start.wait(sess)
            logger.info("Session start sync end.")
            
            summaries_writer=tf.summary.FileWriter(logdir,sess.graph)
            batch_counter=0
            step=0
            comp_time_list=[]
            total_time_list=[]
            
            if sync!="none" and is_chief:
                sv.start_queue_runners(sess, [chief_queue_runner])
                sess.run(init_token_op)
            
            current_batch=[]
            next_batch_data=sess.run(net.get_train_dict_ops())
            
            logger.info("Training start sync start.")
            training_start.wait(sess)
            logger.info("Training start sync end.")
            
            wait_sync=False
            
            logger.info("Started gradients computations.")
            start_time=timer()
            while not sv.should_stop() and step<net.BATCH_NO:
                step_start_time=timer()
                dict_number=batch_counter*len(worker_hosts)+task_index
                logger.info("Started step.")
                profile_step=do_profile and step%1000==5
                if profile_step:
                    run_options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata=tf.RunMetadata()
                else:
                    run_options=None
                    run_metadata=None
                if wait_sync:
                    logger.info("Waiting for turn.")
                    time.sleep(task_index*0.811119201914039)
                logger.info("Computing step.")
                current_batch={net.x:next_batch_data[0],net.y:next_batch_data[1]}
                step_comp_time1=timer()
                [next_batch_data,_,step,batch_cost]=sess.run([net.get_train_dict_ops(),training_operation,global_step,cost],feed_dict=current_batch,options=run_options,run_metadata=run_metadata)
                step_comp_time2=timer()
                if profile_step:
                    step_stats=run_metadata.step_stats
                    tl=timeline.Timeline(step_stats,tf.get_default_graph())
                    ctf=tl.generate_chrome_trace_format(True,True)
                    summaries_writer.add_run_metadata(run_metadata,"step:{0}".format(step))
                    with open(logdir+"/timeline_step{0}.json".format(step), "w") as f:
                        f.write(ctf)
                    logger.info("Profiled as timeline_step{0}.json.".format(step))
                    if do_merged:
                        if is_chief:
                            for c in connection_list:
                                received_stats=recv_object(c)
                                step_stats.MergeFrom(received_stats)
                            tl=timeline.Timeline(step_stats,tf.get_default_graph())
                            ctf=tl.generate_chrome_trace_format(True,True)
                            with open(logdir+"/timeline_step{0}_merged.json".format(step), "w") as f:
                                f.write(ctf)
                            logger.info("Profiled as timeline_step{0}_merged.json.".format(step))
                        else:
                            send_object(my_socket,step_stats)
                batch_counter+=1;
                step_end_time=timer()
                logger.info("Cost: {0}".format(batch_cost))
                print(str(batch_cost),end=",",file=cost_file)
                logger.info("Times: pre:{0:.6f} comp:{1:.6f} post:{2:.6f} total:{3:.6f}.".format(step_comp_time1-step_start_time, step_comp_time2-step_comp_time1, step_end_time-step_comp_time2, step_end_time-step_start_time))
                comp_time_list.append(step_comp_time2-step_comp_time1)
                total_time_list.append(step_end_time-step_start_time)
                logger.info("Ended step {0}.".format(step))
            end_time=timer()
            
            logger.info("Training end sync start.")
            training_end.wait(sess)
            logger.info("Training end sync end.")
            
            if is_chief and do_accuracy:
                logger.info("Computing accuracy.")
                accuracy_value=sess.run(accuracy,feed_dict=net.get_test_dict(sess))
                logger.info("Accuracy: {0}.".format(accuracy_value))
            
            logger.info("Session end sync start.")
            session_end.wait(sess)
            logger.info("Session end sync end.")
        logger.info("Stopping supervisor.")
        sv.request_stop()
        if do_merged:
            my_socket.close()
        logger.info("Worker {0} stopped after {1} batches.".format(task_index,batch_counter))
        logger.info("Computation time: {0}.".format(end_time-start_time))
        logger.info("Comp stats: mean: {0} stdev: {1}".format(statistics.mean(comp_time_list), statistics.pstdev(comp_time_list)))
        logger.info("Total stats: mean: {0} stdev: {1}".format(statistics.mean(total_time_list), statistics.pstdev(total_time_list)))
        logging.shutdown()
        cost_file.close()
        print("{0} {1}".format(batch_counter,end_time-start_time),file=open(logdir+"/stats.txt","w"))

if __name__ == "__main__":
    main()

