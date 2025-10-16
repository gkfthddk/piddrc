import os
import gc
import numpy as np
import random
import h5py
import tqdm
import multiprocessing
import time
import psutil
VERSION="version16"
basepid=os.getpid()
def print_memory_usage(tag="",pid=None):
    if(pid is None):
        process = psutil.Process(os.getpid())
    else:
        process = psutil.Process(pid)
    mem = process.memory_info().rss / (1024 ** 2)  # RSS: 실제 메모리 사용량 (bytes → MB)
    print(f"{tag}[PID {process.pid}] Memory usage: {mem:.2f} MB")    
    
    
def apply_jitter(batch_data,is_point=True): 
    """ Randomly jitter points. jittering is per point. 
    """ 
    sigma=batch_data.shape[-1]*np.mean(batch_data)/10
    clip=5*sigma
    #assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(*batch_data.shape), -1 * clip, clip)
    if(is_point):
        jittered_data[:,:,0]=np.round(jittered_data[:,:,0])
        jittered_data[:,:,1]=np.round(jittered_data[:,:,1])
        jittered_data[:,:,4]=np.round(jittered_data[:,:,4])
    jittered_data += batch_data
    return jittered_data


writekey=['C_amp', 'C_raw',
        'DRcalo3dHits.amplitude', 'DRcalo3dHits.amplitude_sum', 'DRcalo3dHits.cellID', 'DRcalo3dHits.position.x', 'DRcalo3dHits.position.y', 'DRcalo3dHits.position.z', 'DRcalo3dHits.time', 'DRcalo3dHits.time_end',  'DRcalo3dHits.type',
        'DRcalo2dHits.amplitude', 'DRcalo2dHits.cellID', 'DRcalo2dHits.position.x', 'DRcalo2dHits.position.y', 'DRcalo2dHits.position.z', 'DRcalo2dHits.type',
        'Reco3dHits_C.amplitude', 'Reco3dHits_C.position.x', 'Reco3dHits_C.position.y', 'Reco3dHits_C.position.z',
        'Reco3dHits_S.amplitude', 'Reco3dHits_S.position.x', 'Reco3dHits_S.position.y', 'Reco3dHits_S.position.z',
        'E_dep', 'E_gen', 'E_leak', 'GenParticles.PDG', 'GenParticles.momentum.phi', 'GenParticles.momentum.theta', 'seed', 'S_amp', 'S_raw','angle2',
        VERSION,
        ]
for pool in [4,8,14,28,56]:
    writekey.append(f"Reco3dHits{pool}_C.amplitude")
    writekey.append(f"Reco3dHits{pool}_C.position.x")
    writekey.append(f"Reco3dHits{pool}_C.position.y")
    writekey.append(f"Reco3dHits{pool}_C.position.z")
    writekey.append(f"Reco3dHits{pool}_S.amplitude")
    writekey.append(f"Reco3dHits{pool}_S.position.x")
    writekey.append(f"Reco3dHits{pool}_S.position.y")
    writekey.append(f"Reco3dHits{pool}_S.position.z")
    writekey.append(f"DRcalo3dHits{pool}.amplitude")
    writekey.append(f"DRcalo3dHits{pool}.amplitude_sum")
    writekey.append(f"DRcalo3dHits{pool}.cellID")
    writekey.append(f"DRcalo3dHits{pool}.position.x")
    writekey.append(f"DRcalo3dHits{pool}.position.y")
    writekey.append(f"DRcalo3dHits{pool}.position.z")
    writekey.append(f"DRcalo3dHits{pool}.time")
    writekey.append(f"DRcalo3dHits{pool}.time_end")
    writekey.append(f"DRcalo3dHits{pool}.type")
    writekey.append(f"DRcalo2dHits{pool}.amplitude")
    writekey.append(f"DRcalo2dHits{pool}.cellID")
    writekey.append(f"DRcalo2dHits{pool}.position.x")
    writekey.append(f"DRcalo2dHits{pool}.position.y")
    writekey.append(f"DRcalo2dHits{pool}.position.z")
    writekey.append(f"DRcalo2dHits{pool}.type")

def validated_indices(file_name):
    try:
        k=[]
        with h5py.File(file_name,'r') as f:
            S_amp=f['S_amp'][:]
            C_amp=f['C_amp'][:]
            for i in range(len(S_amp)):
                if(S_amp[i]==0 or S_amp[i]>2e+8):
                    continue
                if(C_amp[i]==0 or C_amp[i]>2e+8):
                    continue
                k.append(i)
        return k
    except:
        print("Error validated_indices",file_name)
        return []
            
def read_datasets(file_name, dataset_names, k):
    data_dict = {}
    
    with h5py.File(file_name, 'r') as f:
        for dataset_name in dataset_names:
            try:
                #data_dict[dataset_name] = f[dataset_name][k]
                data_dict[dataset_name] = np.take(f[dataset_name], k, axis=0)
            except Exception as e:
                print(f"Failed to read {dataset_name} from {file_name}: {e}")
                return file_name, None
    return file_name, data_dict

def read_datasets_wrapper(args):
    return read_datasets(*args)

def reader_worker(file_list, dataset_names, q):
    for i,file_name in enumerate(file_list):
        k = validated_indices(file_name)
        if not k:
            continue
        file_name, data_dict = read_datasets(file_name, dataset_names, k)
        q.put((file_name, data_dict))

def write_data(q,output_file,max_queue):
    with h5py.File(output_file,'a') as f:
        for j in tqdm.tqdm(range(max_queue)):
            #if(j%50==0 and f):
            #    f.close()
            #    f=h5py.File(output_file,'a')
            file_name,data_dict=q.get()
            if(file_name is None):
                break
            if(data_dict is None):
                continue
            if(True):
                for dataset_name, data_chunk in data_dict.items():
                    if(dataset_name in f):
                        dataset = f[dataset_name]
                        if(dataset.shape[0]+data_chunk.shape[0]>3e6):continue
                        dataset.resize(dataset.shape[0]+data_chunk.shape[0],axis=0)
                        dataset[-data_chunk.shape[0]:]=data_chunk
                    else:
                        maxshape=(3e6,)+data_chunk.shape[1:]
                        f.create_dataset(dataset_name,data=data_chunk,maxshape=maxshape,chunks=True,compression='lzf')
            del data_dict
            del data_chunk
            gc.collect()
    gc.collect()

def merge_files(file_list,output_file,dataset_names=None,entries=None):
    print_memory_usage('start merge')
    if(dataset_names is None):
        with h5py.File(file_list[0],'r') as f:
            dataset_names = list(f.keys())
        print(dataset_names)        
        
    manager = multiprocessing.Manager()
    q = manager.Queue(maxsize=20)
    
    num_reader=4
    chunk_size=len(file_list) // num_reader + 1
    reader_process = []
    for i in range(num_reader):
        sub_list = file_list[i*chunk_size:(i+1)*chunk_size]
        p=multiprocessing.Process(target=reader_worker, args=(sub_list,dataset_names,q))
        reader_process.append(p)
        p.start()

    writer_process = multiprocessing.Process(target=write_data,args=(q,output_file,len(file_list)))
    writer_process.start()
    
    
    for p in reader_process:
        p.join()
    print_memory_usage('merge before None,None')
    q.put((None,None))
    
    writer_process.join()
    print_memory_usage('merge end')
    
def check_reco(file_name):
    if(os.path.isfile(file_name)):
        try:
            hf=h5py.File(file_name,'r')
            keys=list(hf.keys())
            entries=len(hf['seed'])
            hf.close()
            if(VERSION in keys):
                return file_name,entries
        except:
            print("Error open",file_name)
    return None,None

def toh5(sample,max_num=1000):
    reco_path="/users/yulee/dream/tools/reco"
    out_path=f"h5s/{sample}.h5py"
    print("writing",out_path)
    hf=h5py.File(out_path,'w')
    hf.close()
    #max_num=max(100,len(os.listdir(f"{reco_path}/{sample}")))
    reco_list=os.listdir(f"{reco_path}/{sample}")
    if(max_num==0):
        max_num=len(reco_list)
    max_num=min(max_num,len(reco_list))
    file_list=[]          
    entries=0
    with multiprocessing.Pool(processes=8) as pool:
        results=[]
        for fi in range(max_num):
            recopath=f"{reco_path}/{sample}/{reco_list[fi]}"
            result = pool.apply_async(check_reco,args=(recopath,))
            results.append(result)
        for result in tqdm.tqdm(results,leave=False):
            file_name,entry=result.get()
            if(file_name):
                file_list.append(file_name)
            if(entry):
                entries+=entry
            
    print(out_path,len(file_list))
    if(len(file_list)==0):print("Error: file_list is empty")
    merge_files(file_list,output_file=out_path,dataset_names=writekey)
    #hf=h5py.File(out_path,'r')
    #for key in hf.keys():
    #    print(key,hf[key].shape)
    #hf.close()
    
#pids= ['kaon+','proton','neutron','kaon0L']
pids= ['pi+','pi0','gamma','e-']
#pids= ['gamma','e-']
#pids= ['mu']
#pids= ['e-','pi+','pi0','gamma','kaon+','proton','neutron','kaon0L']
for pid in pids:
  #for en in ["1"]:
  #for en in ["10","20","50","100"]:
  for en in ["1-100"]:
      for pool in [10]:
          if(pool==1):
              sample=f"{pid}_{en}GeV"
          else:
              sample=f"{pid}_{en}GeV_{pool}"
          toh5(sample,5000)
  #sample=f"{pid}_0T"
  #toh5(sample,2500)
  #sample=f"{pid}_line"
  #toh5(sample,2500)
gc.collect()
time.sleep(1)
print_memory_usage('end')
