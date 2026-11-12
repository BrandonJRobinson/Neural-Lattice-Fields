from pathlib import Path
import ast,json,hashlib,collections,os
import numpy as np
R=Path(os.environ['NNLFT_LEGACY_ROOT']);B=Path(__file__).resolve().parent
ns={'np':np}
def functions(path,names):
 tree=ast.parse(path.read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names];assert len(nodes)==len(names)
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)
functions(R/'scripts/precision_upgrade/common.py',{'conservative_seed_combine'})
functions(R/'scripts/precision_upgrade/scalar_precision.py',{'point_rows'})
j=json.loads((R/'results_arch_campaign/architecture_campaign_summary.json').read_text());cfg=j['config'];L=cfg['L'];p=2*np.pi*np.arange(L)/L;G=1/(cfg['m2']+4*np.sin(p[:,None]/2)**2+4*np.sin(p[None,:]/2)**2);d2=2-np.cos(p[:,None])-np.cos(p[None,:]);cal=[]
for family in ['Gaussian','TwoScale']:
 c=j['level_A']['calibration'][family];params=c['opt_params'];sc=c['sigma_c']
 if family=='TwoScale':s1,s2,w=params;C=sc**2/2*(w*np.exp(-s1*s1*d2)+(1-w)*np.exp(-s2*s2*d2))
 else:C=sc**2/2*np.exp(-params[0]**2*d2)
 K=np.fft.fft2(C).real;err=float(np.sqrt(np.mean(((K-G)/G)**2)));norm=float(np.linalg.norm(K-G)/np.linalg.norm(G));assert abs(err-c['metrics']['l2_rel_full'])<1e-13
 C0=K/(sc*sc);alpha=float(np.sum(C0*G)/np.sum(C0*C0));assert abs(np.sqrt(alpha)-sc)<1e-13
 cal.append({'family':family,'params':params,'sigma_c':sc,'mode_relative_RMS':err,'relative_norm':norm,'amplitude_rescaling_matches':True})
records=[];inputs=[R/'results_arch_campaign/architecture_campaign_summary.json',R/'spectral_prior_architectures.py',R/'kernel_recalibration.py']
for name in ['2d_refined_crossings','3d_refined_wilson_fisher']:
 f=R/'results_precision_upgrade'/name/'summary.json';d=json.loads(f.read_text());rows=ns['point_rows'](d['per_seed']);old={(x['L'],x['m0sq']):x for x in d['points']}
 for row in rows:
  want=old[row['L'],row['m0sq']]
  for key in ['U4','U4_se','chi0','chi0_se','chi_abs','chi_abs_se','absM','effective_sample_size_sum']:
   assert np.isclose(row[key],want[key],rtol=1e-12,atol=1e-13),(name,key)
 fail=[x for x in d['per_seed'] if not x['diagnostic_pass']];assert len(fail)==d['classification_diagnostics']['failed_chain_diagnostics']
 pins=[]
 for x in d['runtime_metadata']['source_files']:
  path=R/x['path'];ok=path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest()==x['sha256'];pins.append({'path':x['path'],'match':ok})
 legacy={x['source_path']:x['source_sha256'] for x in d['per_seed'] if 'source_path' in x}
 for path,h in legacy.items():assert hashlib.sha256((R/path).read_bytes()).hexdigest()==h
 # Verify every archived crossing central value on its declared bracket, without new bootstrap draws.
 for c in d['crossings']:
  lo,hi=c['bracket_low'],c['bracket_high'];pair=[c['L_low'],c['L_high']]
  dif=[old[pair[1],m]['U4']-old[pair[0],m]['U4'] for m in [lo,hi]]
  value=lo-dif[0]*(hi-lo)/(dif[1]-dif[0]);assert abs(value-c['m_cross'])<1e-10
 records.append({'campaign':name,'seed_records':len(d['per_seed']),'points_reproduced':len(rows),'failed_diagnostic_records':len(fail),'crossing_central_values_reproduced':len(d['crossings']),'bootstrap_scope':'archived bracket-preserving seed-bootstrap errors recovered, not rerun','source_pins':pins,'legacy_import_hashes_verified':legacy,'archived_status':d['verdict']})
 inputs += [f,R/'results_precision_upgrade'/name/'binder_table.csv',R/'results_precision_upgrade'/name/'crossing_table.csv',R/'results_precision_upgrade'/name/'hmc_diagnostics.csv']
xi=R/'results_precision_upgrade/connected_xi/connected_xi_summary.json';d=json.loads(xi.read_text());bad=[x for x in d['rows'] if not np.isfinite(x['xi_connected_over_L'])];assert len(bad)==3
out={'N01':cal,'S01':records,'selected_xi_invalid_points':[{'dim':x['dim'],'L':x['L'],'m0sq':x['m0sq'],'policy':x['connected_policy']} for x in bad],'scope':'Record recovery and deterministic summary replay; archived failed diagnostics and PARTIAL campaign status retained. No new simulations, bootstrap draws or LaTeX rendering.'}
inputs += [xi,R/'scripts/precision_upgrade/scalar_precision.py',R/'scripts/precision_upgrade/common.py',R/'scripts/precision_upgrade/connected_xi.py']
out['input_hashes']={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
(B/'VERIFIED_LEGACY_RESULTS.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=1))
