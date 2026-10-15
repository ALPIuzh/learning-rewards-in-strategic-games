"""Render all publication figures to a separate candidate preview."""
import json
from dataclasses import replace
from pathlib import Path
from mairl_exp.config import REPOSITORY_ROOT
from mairl_exp.execution.artifacts import sha256_file
from reproducibility.figures.manifest import load_manifest
from reproducibility.figures.render import render_text

root=Path(__file__).resolve().parent
mapping={key:root/'exports/downstream'/name for key,name in [('summary','summary.json'),('policy_grids','policy_grids.npz')]}
mapping['environment']=REPOSITORY_ROOT/'reproducibility/data/lobeke/environment/provenance.json'
wrapper=root/'exports/downstream_figures.json'
wrapper.write_text(json.dumps({'schema_version':1,'inputs':{k:{'path':str(p.relative_to(REPOSITORY_ROOT)),'sha256':sha256_file(p)} for k,p in mapping.items()}},indent=2)+'\n')
replacements={'lobeke_source_convergence':root/'exports/source/provenance.json','lobeke_diagnostics':root/'exports/diagnostics/provenance.json','lobeke_behavioral_transfer':root/'exports/behavioral/provenance.json',**{k:wrapper for k in ('lobeke_transfer','lobeke_transfer_compact','lobeke_downstream')}}
tex=[r'\documentclass{article}',r'\input{reproducibility/tex/tex_base}',r'\begin{document}']
tex += [r'\noindent Final evaluation: 50 fresh learning seeds, 1,600 observations, MLE ridge $10^{-6}$.',r'Active IRL: warm-up 50 with $\lambda/\eta^2=0.03$; no warm-up with $\lambda/\eta^2=0.1$.',r'The source and transfer figures use the new evaluation. The environment, tuning, sparse-information, and lower-bound figures are unchanged.']
checks={}
for spec in load_manifest().figures:
 revised=replace(spec,data=replacements.get(spec.recipe,spec.data))
 body,_=render_text(revised)
 out=root/'figures'/spec.output
 out.parent.mkdir(parents=True,exist_ok=True);out.write_text(body)
 checks[spec.id]={'sha256':sha256_file(out),'updated':spec.recipe in replacements}
 if spec.recipe not in replacements:
  assert body==spec.output_path.read_text(),spec.id
 tex += [r'\clearpage',r'\noindent\textbf{'+spec.title+'}',r'\par\medskip',r'\begin{center}',r'\resizebox{\textwidth}{!}{\input{'+str(out.relative_to(REPOSITORY_ROOT))+'}}',r'\end{center}']
tex += [r'\end{document}']
(root/'preview.tex').write_text('\n'.join(tex)+'\n')
(root/'render-checks.json').write_text(json.dumps(checks,indent=2)+'\n')
print('Rendered all 10 figures; unchanged figures match main exactly.')
