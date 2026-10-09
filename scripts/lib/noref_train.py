import argparse
import json
import math
from pathlib import Path
import time
import noref_common as c


def epoch_batches(count, epochs, seed, batch_size=8):
    import torch
    g=torch.Generator().manual_seed(seed)
    return [(e+1, order[i:i+batch_size]) for e in range(epochs)
            for order in [torch.randperm(count,generator=g).tolist()] for i in range(0,count,batch_size)]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed',type=int,choices=c.SEEDS,required=True); p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--data',help='default: <runs-root>/planner/data/train.jsonl')
    p.add_argument('--out',help='default: <runs-root>/planner/sft/s<seed>'); c.add_path_args(p,model=True); a=c.configure(p.parse_args())
    a.data=a.data or str(c.TASK/'planner/data/train.jsonl'); a.out=a.out or str(c.TASK/'planner/sft'/f's{a.seed}')
    a.epochs=2
    c.runtime(gpu=a.gpu)
    import torch
    from lawn_mec.llm.policy import LoRAPolicy, lora_record
    from lawn_mec.llm.checkpoint import TrainingState
    from lawn_mec.llm.sft import restricted_loss, scaled_step, autocast_for
    from noref_input import traces_for
    from noref_sample import generate
    m=c.manifest('train'); data=c.rows(a.data); prep=json.loads(Path(a.data).with_name('complete.json').read_text())
    if not prep['complete'] or len(data)!=1747 or prep['train_sha256']!=c.sha(a.data):
        raise ValueError('all 1747 train instances required; build missing contexts and rerun preparation')
    by={r['instance_sha256']:r for r in data}
    if len(by)!=1747 or set(by)!={r['sha256'] for r in m['instances']}:
        raise ValueError('train instance coverage differs')
    data=[by[r['sha256']] for r in m['instances']]
    if any(r['variant']!='no_reference' or not r['single_label'] or c.sha(r['instance_path'])!=r['instance_sha256'] for r in data):
        raise ValueError('teacher provenance differs')
    out=c.output(Path(a.out)/'config.json').parent
    cfg=dict(seed=a.seed,epochs=a.epochs,lr=1e-4,batch_size=8,model=str(c.MODEL),lora=lora_record(),
        instances=1747,manifest_sha256=m['manifest_sha256'],data_sha256=c.sha(a.data),code_version=c.VERSION,
        variant='no_reference',precision='float16',loss='mean sequence NLL over non-forced canonical fields',
        extension=None,
        scripts={n:c.sha(c.LIB/n) for n in ('noref_train.py','noref_input.py','noref_common.py','noref_sample.py')})
    ch=c.pin(out/'config.json',cfg); reg=out/'registry.json'; c.pin(reg,{r['key']:r['path'] for r in m['instances']})
    with c.lease(out/'writer.lock'):
        if (out/'complete.json').exists():
            done=json.loads((out/'complete.json').read_text())
            if done['config_sha256']!=ch or done['adapter_sha256']!=c.sha(out/'adapter/adapter_model.safetensors'):
                raise ValueError('completed adapter differs')
            print('NOREF_TRAIN_DONE already complete',flush=True); return
        torch.set_num_threads(8); torch.manual_seed(a.seed)
        if not torch.cuda.is_available():
            raise ValueError('assigned GPU unavailable')
        policy=LoRAPolicy.load(c.MODEL,device='cuda',precision='float16')
        traces=traces_for(policy,data,reg); batches=epoch_batches(len(data),a.epochs,a.seed)
        optimizer=torch.optim.AdamW(list(policy.parameters()),lr=1e-4)
        state=TrainingState(policy.model,optimizer,config=cfg,seed=a.seed,directory=out/'resume',resume='auto')
        policy.training_state=state; tic=time.perf_counter()
        for step in range(state.step,len(batches)):
            epoch,ids=batches[step]; policy.model.train(); torch.cuda.reset_peak_memory_stats(); lap=time.perf_counter()
            def backward(scaler):
                total=0.
                for i in ids:
                    with autocast_for(policy.model):
                        loss=restricted_loss(policy,[traces[i]])/len(ids)
                    scaler.scale(loss).backward(); total+=float(loss.detach())
                return total
            loss,norm,retries=scaled_step(optimizer,state.scaler,policy.parameters(),backward)
            torch.cuda.synchronize()
            metric=dict(epoch=epoch,optimization_loss=loss,grad_norm=norm,amp_retries=retries,
                instances=len(ids),instance_indices=ids,wall_s=time.perf_counter()-lap,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
            state.commit(metric); c.write_json(out/'losses.json',state.history)
            print('NOREF_TRAIN_STEP',state.step,json.dumps(metric),flush=True)
        if state.step!=a.epochs*math.ceil(1747/8):
            raise AssertionError('wrong update count')
        if not (out/'adapter').exists():
            policy.save_adapter(out/'adapter')
        tune=c.manifest('tune'); treg=out/'tune.registry.json'; c.pin(treg,{r['key']:r['path'] for r in tune['instances']})
        monitor=generate(policy,tune['instances'],treg); metrics=c.stats(monitor)
        c.write_json(out/'metrics.json',dict(config=cfg,wall_s=time.perf_counter()-tic,tune=metrics,
            monitoring='full autoregressive field argmax; only after final update; no early stopping',history=state.history))
        c.write_json(out/'complete.json',dict(config_sha256=ch,seed=a.seed,epochs=a.epochs,steps=state.step,
            complete=True,adapter_sha256=c.sha(out/'adapter/adapter_model.safetensors')))
        print('NOREF_TRAIN_DONE',a.seed,'tune field accuracy',metrics['field_accuracy'],flush=True)


if __name__=='__main__':
    main()
