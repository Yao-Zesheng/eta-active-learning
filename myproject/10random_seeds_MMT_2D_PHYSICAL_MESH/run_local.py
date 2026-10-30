"""Background supervisor for the seed-101 physical-mesh USLW campaign."""
from pathlib import Path
import os,sys,json,subprocess,traceback,hashlib,shutil
ROOT=Path(__file__).resolve().parent;CASE=ROOT/'hard_s101';NAME='uslw_seed101_physical';OUT=CASE/'results'/NAME
OUT.mkdir(parents=True,exist_ok=True)
def status(phase,**kwargs):
 value=dict(phase=phase,supervisor_pid=os.getpid(),target_iteration=100,case=str(CASE),run_name=NAME,**kwargs)
 temp=ROOT/'status.tmp';temp.write_text(json.dumps(value,indent=2),encoding='utf8');temp.replace(ROOT/'status.json')
cmd=[sys.executable,'-u','run.py','--iterations','100','--threads','8','--device','cpu','--acq','uslw','--lhs-seed','101','--seed-base','1','--mesh','32','--adam','1800','--lbfgs','450','--name',NAME]
try:
 if (OUT/'config.json').exists():cmd+=['--resume']
 for filename in ['model.py','run.py','training.py','continuous_pdf.py','acquisition.py','problem.py']:
  target=OUT/(Path(filename).stem+'_used.py')
  if not target.exists():shutil.copy2(CASE/filename,target)
 if not (OUT/'reference_law_used.npz').exists():shutil.copy2(CASE/'data/reference_law.npz',OUT/'reference_law_used.npz')
 manifest={'command':cmd,'reference_sha256':hashlib.sha256((OUT/'reference_law_used.npz').read_bytes()).hexdigest(),'run_goal':'100 acquired labels plus 3 original LHS labels; first seed group only','model':'physical x/6 score features; 32 physical-uniform intervals per axis mapped to Gaussian probability coordinates; exact nonuniform triangle areas','pdf_metrics':'pdf_e_members_continuous: model integration; pdf_e_members and pdf_e_mean: historical grid KDE; ensemble mean has no hard-law guarantee'}
 (OUT/'launch_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
 status('starting',command=cmd)
 with (ROOT/'run.log').open('a',encoding='utf8') as log:
  proc=subprocess.Popen(cmd,cwd=CASE,stdout=log,stderr=subprocess.STDOUT)
  status('running',training_pid=proc.pid,command=cmd)
  rc=proc.wait()
 if rc:raise RuntimeError(f'Training process exit={rc}; see run.log')
 rows=json.loads((OUT/'history.json').read_text());assert len(rows)==101 and rows[-1]['iteration']==100 and rows[-1]['n']==103
 status('completed',last_completed_iteration=100,labels=103)
 subprocess.run([sys.executable,str(ROOT/'summarize_run.py')],cwd=ROOT,check=True)
except Exception:
 status('failed',error=traceback.format_exc());raise
