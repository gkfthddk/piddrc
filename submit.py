#!python3
#/bin/python3 submit.py --epochs 100 --title drt4f_{cand}_{key}_70T_p{pool}_cct_tune4 --dform image --cand pi0_70T,gamma_70T --opt adamw --pool 2 --net cct --tune --hour 10 --keep
import argparse
import subprocess
import datetime,sys
import htcondor
import os
parser=argparse.ArgumentParser()
parser.add_argument("--name",type=str,default=None,help='save name model will be save in save/<argument>/')
parser.add_argument("--exe",type=str,default='run.py',help='pt for dataset')
parser.add_argument("--request_memory",type=str,default='48GB',help='HTCondor request_memory value, e.g. 48GB or 65536MB')

args, unknown = parser.parse_known_args()
if args.name is None:
    args.name = f"run_{datetime.datetime.now():%Y%m%d_%H%M%S}_{os.getpid()}"
jobname=args.name
transfer_inputs = [args.exe]
if os.path.isdir("pid"):
    transfer_inputs.append("pid")
if(os.path.isfile(f'condor/{jobname}.out')):
    os.remove(f'condor/{jobname}.out')
    os.remove(f'condor/{jobname}.error')
    os.remove(f'condor/{jobname}.log')
submit_desc={
    'JobBatchName': jobname,
    'universe':'vanilla',
    'getenv':'True',
    'should_transfer_files':'YES',
    'when_to_transfer_output':'ON_EXIT',
    'request_memory':args.request_memory,
    'request_cpus':f'4',
    'request_gpus':1,
    'request_disk':f'262144',
    'requirements':'(machine != "gpu01.sscc.uos")',
    'transfer_input_files':",".join(transfer_inputs),
    'executable':args.exe,
    'transfer_output_files':'save',
    'output':f'condor/{jobname}.out',
    'error':f'condor/{jobname}.error',
    'log':f'condor/{jobname}.log',
    'arguments':" ".join(["--name", jobname] + unknown)+" --no_progress_bar",
    #'priority':1,
}
submit_desc = {key: str(value) for key, value in submit_desc.items()}
print(submit_desc)
job=htcondor.Submit(submit_desc)
schedd = htcondor.Schedd()
submit_result = schedd.submit(job)

print(f'{jobname} {submit_result.cluster()} submitted')
