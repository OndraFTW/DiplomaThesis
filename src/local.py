#projekt: Paralelní trénování hlubokých neuronových sítí
#Popis: Lokální trénování
#Autor: Ondřej Šlampa, xslamp01@stud.fit.vutbr.cz

import os
import sys
import argparse
import tensorflow as tf
import logging
import statistics
from timeit import default_timer as timer
from importlib.machinery import SourceFileLoader
from tensorflow.python.client import timeline

tf.logging.set_verbosity(tf.logging.DEBUG)

DEFAULT_TMP_DIR="/tmp"
FIRST_PORT=4000
DEFAULT_LOG_DIR="./logs"

def parse_args():
    parser=argparse.ArgumentParser(description="Distributed neural network trainer.")
    parser.add_argument("--datadir",type=str,default=DEFAULT_TMP_DIR,help="directory with data files")
    parser.add_argument("--logdir",type=str,help="directory for logs")
    parser.add_argument("--weights",type=str,help="file with saved weights")
    parser.add_argument("--net",type=str,required=True,help="network")
    parser.add_argument("--accuracy",type=bool,default=False,help="compute accuracy (defaults to False)")
    parser.add_argument("--profile",type=bool,default=False,help="profile every 100th batch (defaults to False)")
    args=parser.parse_args()
    
    return (args.datadir, args.logdir, args.weights, args.net, args.accuracy, args.profile)

def main():
    (datadir, logdir, weightsfile, netfile, do_accuaracy, do_profile)=parse_args()
    
    logger=logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    if logdir:
        logfile=logdir+"/logger.txt"
        if not os.path.exists(logdir):
            os.mkdir(logdir)
        out_hdlr=logging.FileHandler(logfile)
    else:
        logdir=DEFAULT_LOG_DIR
        if not os.path.exists(logdir):
            os.mkdir(logdir)
        out_hdlr=logging.StreamHandler(sys.stdout)
    out_hdlr.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    out_hdlr.setLevel(logging.INFO)
    logger.addHandler(out_hdlr)
    
    logger.info("Logging started.")
    logger.info("Netfile: {0}.".format(netfile))
    
    logger.info("Loading net.")
    net_module=SourceFileLoader("Module", netfile).load_module()
    net=net_module.Network(datadir,[""])
    model=net.get_model()
    accuracy=net.get_accuracy()
    training_operation=net.get_training_operation()
    merged=tf.summary.merge(net.summaries)
    
    net.prepare_data()
    
    logger.info("Setting up sesion.")
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,intra_op_parallelism_threads=4,inter_op_parallelism_threads=4)) as sess:
        start_time=timer()
        logger.info("Initializing variables.")
        sess.run(tf.global_variables_initializer())
        if weightsfile:
            net.saver.restore(sess,weightsfile)
        logger.info("Starting coordinator.")
        coord=tf.train.Coordinator()
        tf.train.start_queue_runners(sess,coord=coord)
        summaries_writer=tf.summary.FileWriter(logdir,sess.graph)
        current_batch=[]
        next_batch_data=sess.run(net.get_train_dict_ops())
        step=0
        step_accuracy=0.0
        comp_time_list=[]
        total_time_list=[]
        start_time=timer()
        while step<net.BATCH_NO:
            step_start_time=timer()
            logger.info("Started step {0}.".format(step))
            profile_step=do_profile and step%100==5
            if profile_step:
                run_options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE,output_partition_graphs=True)
                run_metadata=tf.RunMetadata()
            else:
                run_options=None
                run_metadata=None
            step_comp_start_time=timer()
            current_batch={net.x:next_batch_data[0],net.y:next_batch_data[1]}
            step_comp_time1=timer()
            [next_batch_data,_]=sess.run([net.get_train_dict_ops(),training_operation],feed_dict=current_batch,options=run_options,run_metadata=run_metadata)
            step_comp_time2=timer()
            
            if profile_step:
                tl=timeline.Timeline(run_metadata.step_stats,tf.get_default_graph())
                ctf=tl.generate_chrome_trace_format(True,True)
                summaries_writer.add_run_metadata(run_metadata,"step:{0}".format(step))
                with open(logdir+"/timeline_step{0}.json".format(step), "w") as f:
                    f.write(ctf)
                logger.info("Profiled as timeline_step{0}.json.".format(step))
            step_end_time=timer()
            logger.info("Times: pre:{0:.6f} comp:{1:.6f} post:{2:.6f} total:{3:.6f}.".format(step_comp_time1-step_start_time, step_comp_time2-step_comp_time1, step_end_time-step_comp_time2, step_end_time-step_start_time))
            comp_time_list.append(step_comp_time2-step_comp_time1)
            total_time_list.append(step_end_time-step_start_time)
            logger.info("Ended step {0}.".format(step))
            step+=1
        end_time=timer()
        net.saver.save(sess,logdir+"/model")
        coord.request_stop()
        coord.join()
        logger.info("Computation time: {0}.".format(end_time-start_time))
        logger.info("Comp stats: mean: {0} stdev: {1}".format(statistics.mean(comp_time_list), statistics.pstdev(comp_time_list)))
        logger.info("Total stats: mean: {0} stdev: {1}".format(statistics.mean(total_time_list), statistics.pstdev(total_time_list)))
    
if __name__ == "__main__":
    main()

