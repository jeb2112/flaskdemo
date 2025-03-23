# script wraps the nnUNetv2_predict command to provide some arguments for
# local path on the back-end and stream stdout

import numpy as np
from sklearn.model_selection import train_test_split
import os
import argparse
import nibabel as nb
import shutil
import subprocess

def main(datadir,env,dataset,model):
    inputdir = os.path.join(datadir,'nnUNet_raw','flask','imagesTs')
    outputdir = os.path.join(datadir,'nnUNet_predictions','flask')

    try:
        shutil.rmtree(outputdir)
    except FileNotFoundError:
        pass
    os.makedirs(outputdir,exist_ok=True)
    # condapath = os.system('which conda')
    cmd = f"conda run -n {env} nnUNetv2_predict -i {inputdir} -o {outputdir} -d {dataset} -c {model}"

    process = subprocess.Popen(cmd,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    
    for line in iter(process.stdout.readline,""):
        print(line, end="")  # Print without adding extra newline    
    #     yield line
    process.stdout.close()
    process.wait()

    stderr = process.stderr.read()
    if stderr:
        print(f"Error: {stderr}")
    
    print(f"Process completed for model {model}.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", type=str, default="/media/jbishop/WD4/brainmets/sunnybrook/radnec2/") 
    parser.add_argument("--env", type=str, default="flask")
    parser.add_argument("--dataset", type=str, default="139") # ie whatever nnUNet dataset # has been used to identify the model
    parser.add_argument("--model", type=str,default='2d')
    args = parser.parse_args()
    main(args.datadir,args.env,args.dataset,args.model)
