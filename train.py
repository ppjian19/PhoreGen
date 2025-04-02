from run.run import Run
from argparse import ArgumentParser
import time
import os, sys

def parse_train_args():
    # General arguments
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    args = parser.parse_args()
    return args
    

def main():
    args = parse_train_args()
    train_run = Run()
    train_run.train(args)
    print(f"[{time.strftime('%Y/%m/%d-%H:%M:%S')}] Training Finished.")

def log():
    print(f"[{time.strftime('%Y/%m/%d-%H:%M:%S')}]")
    print(f"Current Working Dir: {os.getcwd()}")
    os.system("echo Current Hostname: $(hostname)")
    print(f'Current PID: {os.getpid()}')
    print(f"Current Command: {' '.join(sys.argv)}")
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        print(f"Current GPU: {os.environ['CUDA_VISIBLE_DEVICES']}")

if __name__ == '__main__':
    log()
    main()