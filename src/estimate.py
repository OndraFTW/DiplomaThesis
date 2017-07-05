#projekt: Paralelní trénování hlubokých neuronových sítí
#Popis: Výpočet odhadů zrychlení
#Autor: Ondřej Šlampa, xslamp01@stud.fit.vutbr.cz

import argparse
import matplotlib.pyplot as plt

def async(server, grad, comm, d, n):
    if server=="central":
        train_best=max((n-1)*comm+(d/n)*(2*comm+grad), (d+1)*comm+grad)
        train_worst=(d/n)*(2*n*comm+grad)
        train=(train_best+train_worst)/2
    else:
        train=(d/n)*(2*comm*((n-1)/n)+grad)
    return (d*grad)/train

def sync_join(server, grad, comm, d, n):
    if server=="central":
        train_best=(n+1)*comm+grad
        train_worst=2*n*comm+grad
        train=(train_best+train_worst)/2
    else:
        train=2*comm*((n-1)/n)+grad
    return (d*grad)/(train*d/n)

def sync_split(server, grad, comm, d, n):
    if server=="central":
        train_best=(n+1)*comm+grad/n
        train_worst=2*n*comm+grad/n
        train=(train_best+train_worst)/2
    else:
        train=2*comm*((n-1)/n)+grad/n
    return (d*grad)/(train*d)

def parse_args():
    parser=argparse.ArgumentParser(description="Distributed neural network training estimator.")
    parser.add_argument("--t-grad",type=float,required=True,help="time to compute gradients")
    parser.add_argument("--t-comm",type=float,required=True,help="time to send/receive gradients/weights")
    parser.add_argument("--type",type=str,default="async",choices=["sync-join","sync-split","async"],help="type of training")
    parser.add_argument("--server",type=str,default="central",choices=["central","distributed"],help="type of server")
    parser.add_argument("--workers",type=int,required=True,help="number of worker nodes")
    parser.add_argument("--batches",type=int,default=128,help="number of batches")
    parser.add_argument("--output",type=str,default="single",choices=["single","plot","csv"],help="output type")
    return parser.parse_args()

def main():
    args=parse_args()

    if args.type=="async":
        f=async
    elif args.type=="sync-join":
        f=sync_join
    else:
        f=sync_split

    if args.output=="single":
        print(f(args.server, args.t_grad, args.t_comm, args.batches, args.workers))
    elif args.output=="csv":
        vals=[]
        for i in range(args.workers):
            vals.append(str(f(args.server, args.t_grad, args.t_comm, args.batches, i+1)))
        print(";".join(vals))
    else:
        vals=[]
        for i in range(args.workers):
            vals.append(str(f(args.server, args.t_grad, args.t_comm, args.batches, i+1)))
        indexes=[i+1 for i in range(args.workers)]
        plt.plot(indexes, vals)
        plt.ylabel("Speedup")
        plt.xlabel("Number of workers")
        plt.show()

if __name__=="__main__":
    main()

