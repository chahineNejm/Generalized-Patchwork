import fs from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import {FileBlob,SpreadsheetFile} from '@oai/artifact-tool';

const dir=fileURLToPath(new URL('.',import.meta.url)), file=dir+'GPW_experiment_plan.xlsx';
const wb=await SpreadsheetFile.importXlsx(await FileBlob.load(file));
const names=['Experiments','Ablations','Model settings','Paper settings','Datasets','Protocol'];
const old=new Map(names.map(n=>[n,wb.worksheets.getItem(n).getUsedRange().values]));
if(process.argv.includes('--dataset-paper-model')) {
 const exp=wb.worksheets.getItem('Experiments');
 const data=exp.getRange('A5:K25').values, links=exp.getRange('H6:H25').formulas;
 const preview=await wb.render({sheetName:'Experiments',range:'H5:K9',scale:1,format:'png'});
 await fs.writeFile(dir+'before_paper_model_mapping.png',new Uint8Array(await preview.arrayBuffer()));
 if(process.argv.includes('--preview')) process.exit(0);
 const references=[];
 const titles={pw:'Patchwork Kriging for Large-scale Gaussian Process Regression',vn:'Variational Nearest Neighbor Gaussian Process',sp:'ProSpar-GP: scalable Gaussian process modeling with massive non-stationary datasets',nk:'Nested Kriging predictions for datasets with large number of observations',sv:'Gaussian Processes for Big Data',sw:'Sparse within Sparse Gaussian Processes using Neighbor Information',gs:'gp2Scale: A Class of Compactly Supported Non-Stationary Kernels and Distributed Computing for Exact Gaussian Processes on 10 Million Data Points'};
 const urls={pw:'https://jmlr.org/papers/v19/17-042.html',vn:'https://proceedings.mlr.press/v162/wu22h/wu22h.pdf',sp:'https://arxiv.org/abs/2311.08752v1',nk:'https://arxiv.org/abs/1607.05432',sv:'https://arxiv.org/abs/1309.6835',sw:'https://proceedings.mlr.press/v139/tran21a.html',gs:'https://arxiv.org/abs/2512.06143v3'};
 function ref(dataset,paper,model,structure,training,design,limits,short) {references.push([dataset,titles[paper],model,structure,training,design,limits,urls[paper],short]);}
 for(const [dataset,p,b,kernel] of [
  ['TCO','{64,128,256,512}','{5,7}','SE and exponential'],
  ['TCO.L2','{256,512,1024}','{3,5}','SE and exponential'],
  ['ICETHICK','{64,128,256,512,1024}','{3,5,7}','SE and exponential'],
  ['PROTEIN','{64,128,256}','{2,3,4}','SE'],
  ['SARCOS','{128,256}','{3,5,7}','SE'],
  ['FLIGHT','1024','5','SE']]) {
  ref(dataset,'pw','Standard Patchwork',`P=${p}; stitches per boundary=${b}; ${kernel} kernel; recursive local-PC tree; uniform random boundary stitches.`,
   'Shared regional kernel hyperparameters estimated using all training observations. Optimizer budget, fitted values and initial kernel/noise values not recorded in checked inventory.',
   '90/10 random split. SARCOS uses first torque target.',
   'Paper configuration differs from our separate regional kernels, selected-target OISST baseline, and grid/Voronoi partitions. Our current kernel is RBF only.',
   `P=${p}, stitches/boundary=${b}, kernel=${kernel}; shared regional hyperparameters; PC tree; 90/10 split. Optimizer/initial/fitted values NR.`);
 }
 for(const dataset of ['PROTEIN','UKHousing','Precipitation']) {
  const small=dataset==='PROTEIN', budget=small?'{300,500}':'{100,300}', lr=small?'{0.005,0.01}':'{0.005,0.001}';
  const training=`ARD Matern-5/2; initial lengthscales/outputscale/likelihood-noise parameter=0.6931. Adam LR=${lr}, reported optimization budget=${budget} iterations. LR x0.1 at 75% and 90%.`;
  const design='64/16/20 split, 3 seeds, validation NLL selection. Batch sizes and winning configurations not recorded in checked inventory.';
  const limits='Our current kernels are RBF and use different initialization/training. Original split/snapshot equivalence not established. The paper noise parameter is not a reported noise SD.';
  ref(dataset,'vn','VNNGP','M=N_train, fixed observed locations; random ordering; mean-field q(U); K={32,256}.',training,design,limits,
   `M=N_train, K={32,256}, mean-field q; ARD Matern-5/2; init kernel/noise parameter=0.6931; LR=${lr}, budget=${budget}; decay at75/90%, 3 seeds. Batch NR.`);
  ref(dataset,'vn','SVGP','M=1024; full-covariance q; inducing points initialized by k-means.',training,design,limits,
   `M=1024, full q, k-means; ARD Matern-5/2; init kernel/noise parameter=0.6931; LR=${lr}, budget=${budget}; decay at75/90%, 3 seeds. Batch NR.`);
 }
 for(const dataset of ['Michalewicz','G-function','Satellite drag']) {
  const design=dataset==='Satellite drag'?'d=8; N_train={100k,175k,250k}; N_test=175k from 2M simulations; 10 repetitions/config.':'d={3,5,10} x N_train={50k,100k}; N_test=25k; LHS; observation-noise variance=1e-8; 10 repetitions/config.';
  ref(dataset,'sp','ProSpar','P=200 experts; m=25/expert; M=5000; local SE kernels; learn expert kernels, inducing locations, temperatures, exponent and nuggets.',
   'Adam; batch=1024. Learning rates, optimization budget and initial T_j/c/D_j not recorded in checked inventory.',design,
   'Supplied preprint v1. Our implementation is paper-derived, not verified author code. Do not fill missing initial values from library defaults.',
   'P=200,m=25,M=5000,local SE; Adam,batch=1024. LR/budget/T/c/nugget starts NR. '+design);
  ref(dataset,'sp','SVGP','M=1024; Matern-5/2 kernel.',
   'Batch=1024. Learning rates, optimization budget and initial kernel/noise values not recorded in checked inventory.',design,
   'Paper comparator, not our RBF preset. Observation-noise variance in the synthetic data design is not the fitted likelihood-noise value.',
   'M=1024,Matern-5/2,batch=1024. LR/budget/initial kernel-noise values NR. '+design);
 }
 ref('FLIGHT','sv','SVGP','ARD squared-exponential + bias + noise; M=1000 main; inducing sweep M={50,100,200,500,800,1000}.',
  'Batch=5000; variational LR=0.01; kernel LR=1e-5; kernel momentum=0.9. Optimization budget not explicit in checked prose.',
  '700,000 train /100,000 test. 10 main runs.',
  'No verified original 2013 split locally. Our RBF-only model does not include the paper bias kernel.',
  'M=1000,batch=5000,ARD SE+bias+noise; q LR=.01,kernel LR=1e-5,momentum=.9; 700k/100k split;10 runs. Budget NR.');
 ref('NK synthetic Matern example','nk','Nested Kriging','N=30; P=15; two consecutive sorted observations/expert; zero-mean Matern-5/2; variance=1; lengthscale=0.05; no observation noise.',
  'All kernel parameters held fixed; no fitting or optimizer.',
  '101 test-grid points; 50 independent function/design repetitions.',
  'Our LOO-fitting implementation is not used in this selected fixed-kernel example. Full GP/PoE/gPoE/BCM/rBCM omitted here because they are not among our implemented model classes.',
  'N=30,P=15,2 observations/expert; Matern-5/2,variance=1,lengthscale=.05,noise=0; no fitting;101 grid points,50 repetitions.');
 ref('PROTEIN','sw','SVGP','Small inducing comparison: M={16,32,64,128}. Large inducing comparison: SVGP M={128,512}. Matern-5/2; Gaussian likelihood.',
  'Adam LR=0.001; batch=64. Small-inducing comparison: 300,000 iterations; large-comparison training budget not recorded in checked inventory.',
  '10 folds. Large-comparison figure split: 41,157 train /4,573 test.',
  'SWSGP itself is not implemented locally; only its SVGP comparisons are included. Check each figure-specific configuration before replication.',
  'M={16,32,64,128} small /{128,512} large; Matern-5/2; Adam LR=.001,batch64;small budget300k iterations;10 folds. Large budget NR.');
 ref('California Housing','gs','SVGP','Paper: M=100, train/test=20,000/640. Inspected code: random observed inducing locations, trainable; full q; isotropic Matern-3/2 + outputscale; constant mean.',
  'Inspected code: 500 full-batch updates, Adam LR=0.1; likelihood noise not optimized. Initial kernel/noise values not recorded in checked inventory.',
  'Paper and pinned associated notebook details are distinguished, not merged into an assumed reproduction.',
  'Code inspected at b968d975bd80fe8e23dca3a2ad62e398eeb008d4. Our local SVGP family/preset differs.',
  'Paper M=100,20k/640 split. Code: isotropic Matern-3/2,constant mean,full q,500 full-batch Adam updates,LR=.1; noise not optimized.');
 ref('California Housing','gs','VNNGP','Paper reports M=20,000 and K=50; train/test=20,000/640.',
  'Matching VNNGP kernel, initialization and optimization settings NOT VERIFIED in associated code.',
  'Associated CA_nngp notebook actually uses independent local exact GPs, not VNNGP.',
  'Paper-reported VNNGP comparison is included because VNNGP is implemented locally, but cannot be claimed as a code-verified reproduction.',
  'Paper reports M=20,000,K=50,20k/640 split. Kernel/training details not code-verified; associated notebook is not VNNGP.');
 const implemented=new Set(['GPW','SVGP','ProSpar','VNNGP','Standard Patchwork','Nested Kriging']);
 if(references.some(r=>!implemented.has(r[2]))) throw new Error('Unimplemented model in reference mapping');
 const revised=[['Experiment ID','Experiment','Dataset','Models / candidates','Question','Status','Experiment hyperparameters','Full configuration','Paper(s) using this dataset','Our implemented models used in those papers','Reported hyperparameters by paper and model','Caveat / next decision']];
 for(const row of data.slice(1)) {
  const matching=references.filter(r=>r[0]===row[2]);
  const titlesUnique=[...new Set(matching.map(r=>r[1]))];
  const modelText=titlesUnique.map(title=>title+': '+[...new Set(matching.filter(r=>r[1]===title).map(r=>r[2]))].join(', ')).join('\n');
  const hyper=matching.map(r=>r[1]+' / '+r[2]+': '+r[8]+' Source: '+r[7]).join('\n');
  revised.push([...row.slice(0,8),titlesUnique.join('\n')||'No matching paper recorded in the checked reference set',modelText||'Not established for this dataset',hyper||'NOT RECORDED. No configuration inferred.',row[10]]);
 }
 exp.tables.items[0].delete();
 exp.getRange('A5:L25').clear({applyTo:'contents'});
 exp.getRange('A5:L25').values=revised;
 exp.getRange('H6:H25').formulas=links;
 exp.getRange('A3').values=[['Paper -> our implemented model(s) -> reported hyperparameters, separately for each dataset. Full experiment configurations remain linked. NR means not recorded, not a default.']];
 for(const [c,w] of [['I',64],['J',75],['K',110],['L',72]]) exp.getRange(`${c}1:${c}25`).format.columnWidth=w;
 exp.getRange('I5:L25').format.wrapText=true;
 exp.getRange('A5:L5').format={fill:'#215C49',font:{name:'Arial',size:11,bold:true,color:'#FFFFFF'},wrapText:true,rowHeight:46};
 for(let i=1;i<revised.length;i++) {
  const lines=Math.max(...revised[i].slice(8).map((v,j)=>String(v??'').split('\n').reduce((sum,t)=>sum+Math.ceil(t.length/[64,75,110,72][j]),0)));
  exp.getRange(`A${i+5}:L${i+5}`).format.rowHeight=Math.max(155,20*(lines+2));
 }
 exp.tables.add('A5:L25',true,'ExperimentsTable');
 const papers=wb.worksheets.getItem('Paper settings');
 papers.tables.items[0].delete();
 papers.getUsedRange().clear({applyTo:'contents'});
 papers.getRange('A2').values=[['Dataset, paper, implemented model and reported settings']];
 papers.getRange('A3').values=[['Only our six implemented model classes are included. Multiple tested values remain grids; absent settings are explicitly marked. No paper settings are adopted automatically.']];
 papers.getRange(`A5:H${references.length+5}`).values=[['Dataset','Paper title','Implemented model used','Structure / kernel','Training / initialization','Design / repetitions','Verification / missing values','Source'],...references.map(r=>r.slice(0,8))];
 papers.getRange(`A5:H${references.length+5}`).format={font:{name:'Arial',size:11,color:'#252B30'},wrapText:true,verticalAlignment:'top',rowHeight:110};
 papers.getRange('A5:H5').format={fill:'#215C49',font:{name:'Arial',size:11,bold:true,color:'#FFFFFF'},wrapText:true,rowHeight:42};
 [28,76,25,84,90,75,92,65].forEach((w,i)=>papers.getRangeByIndexes(0,i,references.length+5,1).format.columnWidth=w);
 papers.tables.add(`A5:H${references.length+5}`,true,'PapersettingsTable');
 for(const n of ['Ablations','Model settings','Datasets','Protocol']) if(JSON.stringify(old.get(n))!==JSON.stringify(wb.worksheets.getItem(n).getUsedRange().values)) throw new Error('Unrelated sheet changed: '+n);
 if(JSON.stringify(exp.getRange('A6:G25').values)!==JSON.stringify(data.slice(1).map(r=>r.slice(0,7)))) throw new Error('Experiment hyperparameters changed');
 wb.recalculate();
 for(const [sheet,range,name] of [['Experiments','I7:L8','paper_model_mapping'],['Paper settings','A5:H8','paper_model_details']]) {
  await fs.writeFile(dir+name+'.png',new Uint8Array(await (await wb.render({sheetName:sheet,range,scale:1,format:'png'})).arrayBuffer()));
 }
 console.log((await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?',options:{useRegex:true,maxResults:10}})).ndjson);
 await (await SpreadsheetFile.exportXlsx(wb)).save(file);
 console.log(JSON.stringify({paperDatasetModelRecords:references.length,models:[...new Set(references.map(r=>r[2]))],experimentRows:20,preserved:'All experiment configurations, ablations, local model settings and links'}));
 process.exit(0);
}
if(process.argv.includes('--paper-only')) {
 const sheet=wb.worksheets.getItem('Experiments');
 const before=sheet.getRange('A5:M25').values;
 const links=sheet.getRange('H6:H25').formulas;
 const preview=await wb.render({sheetName:'Experiments',range:'H5:M8',scale:1,format:'png'});
 await fs.writeFile(dir+'references_before_paper_only.png',new Uint8Array(await preview.arrayBuffer()));
 if(process.argv.includes('--preview')) process.exit(0);
 const alreadyPaperOnly=before[0][8]==='Reference models in the paper';
 const revised=before.map(row=>alreadyPaperOnly?row.slice(0,11):[...row.slice(0,8),...row.slice(10,13)]);
 revised[0][8]='Reference models in the paper';
 revised[0][9]='Reference hyperparameters reported in the paper';
 revised[0][10]='Caveat / next decision';
 const paperBase='https://proceedings.mlr.press/v162/wu22h/wu22h.pdf';
 const prosparPaper='https://arxiv.org/abs/2311.08752v1';
 const patchworkPaper='https://jmlr.org/papers/v19/17-042.html';
 const svgpCommon='SVGP M=1024, full-covariance q, k-means initialization, ARD Matern-5/2, Gaussian likelihood. Initial lengthscales/outputscale/noise parameter=0.6931. Adam; LR x0.1 at 75% and 90% of budget. Validation NLL selection. ';
 const prosparComparator='SVGP M=1024, Matern-5/2 kernel, batch size=1024. Learning rates, optimizer budget and initial kernel/noise values NOT REPORTED in the checked inventory. These are paper reference settings, not adopted local values. '+prosparPaper;
 const comparator={
  E03:['SVGP (baseline in VNNGP paper)',svgpCommon+'PROTEIN: reported budget {300,500} iterations, LR {0.005,0.01}; 4 optimizer pairs before seeds. 64/16/20 split, 3 seeds. Batch size not reported in this inventory. '+paperBase],
  E04:['SVGP (baseline in VNNGP paper)',svgpCommon+'UKHousing (>50k): reported budget {100,300} iterations, LR {0.005,0.001}; 4 optimizer pairs before seeds. Our reconstructed UK data are not the verified paper snapshot. Batch size not reported in this inventory. '+paperBase],
  E05:['rBCM (Patchwork-paper baseline; not implemented locally)','ICETHICK: expert counts {50,100,150,200,250,300}. Other kernel/optimizer values not recorded for this comparator in the checked inventory. Do not substitute Standard Patchwork hyperparameters. '+patchworkPaper],
  E06:['rBCM (Patchwork-paper baseline; not implemented locally)','TCO: expert counts {100,150,200,250,300,600}. Other comparator hyperparameters not recorded in the checked inventory. '+patchworkPaper],
  E06b:['rBCM (Patchwork-paper baseline; not implemented locally)','TCO.L2: expert counts {100,150,200,250,300,600}. Other comparator hyperparameters not recorded in the checked inventory. TCO and TCO.L2 are distinct sources. '+patchworkPaper],
  E07:['rBCM (Patchwork-paper baseline; not implemented locally)','SARCOS: expert counts {100,150,200,250,300}. Other comparator hyperparameters not recorded in the checked inventory. First torque is the Patchwork-paper target. '+patchworkPaper],
  E08:['rBCM (Patchwork-paper baseline; not implemented locally)','FLIGHT: 1024 experts. Other comparator kernel/optimizer settings not recorded in the checked inventory. Original SVGP paper settings remain separately in Paper settings. '+patchworkPaper],
  E09:['SVGP (baseline in ProSpar paper)',prosparComparator],
  E10:['SVGP (baseline in VNNGP paper)',svgpCommon+'Precipitation: >50k training group uses reported budget {100,300} iterations and LR {0.005,0.001}. Exact data source and batch size still need verification. '+paperBase],
  E11:['Full GP (reference in Nested Kriging paper; local diagnostic not configured)','All N=30 observations, zero-mean Matern-5/2, variance=1, lengthscale=0.05, noiseless; kernel held fixed, no hyperparameter fitting. 101 test-grid points, 50 independent function/design repetitions. https://arxiv.org/abs/1607.05432'],
  E12:['SVGP (baseline in ProSpar paper)',prosparComparator],
  E13:['SVGP (baseline in ProSpar paper)',prosparComparator],
 };
 const notes=new Map(wb.worksheets.getItem('Experiment configs').getUsedRange().values
  .filter(r=>r[2]==='(dataset protocol)').map(r=>[r[0],r[7]]));
 comparator.E02=['SVGP / VNNGP (baselines in gp2Scale paper)',
  'Paper: SVGP M=100; VNNGP M=20000,K=50; train/test=20000/640. Inspected SVGP code differs in implementation detail: isotropic Matern-3/2, constant mean, full q, random observed supports, 500 full-batch updates, Adam LR=0.1; noise not optimized. Claimed VNNGP code is actually independent local exact GPs, not verified VNNGP. Source inventory: https://github.com/chahineNejm/Generalized-Patchwork/blob/25d3ff5/target.txt'];
 for(const row of revised.slice(1)) {
  [row[8],row[9]]=comparator[row[0]]??['No paper comparator selected / recorded for this experiment','UNDECIDED. No paper reference hyperparameters inferred.'];
  if(!row[10]) row[10]=notes.get(row[0]);
 }
 sheet.tables.items[0].delete();
 sheet.getRange('A5:M25').clear({applyTo:'contents'});
 sheet.getRange('L5:M25').clear({applyTo:'all'});
 sheet.getRange('A5:K25').values=revised;
 sheet.getRange('H6:H25').formulas=links;
 sheet.getRange('A3').values=[['Reference columns describe the papers, not our local presets. Full experiment settings are linked separately; unchosen values remain UNDECIDED.']];
 sheet.getRange('I1:I25').format.columnWidth=44;
 sheet.getRange('J1:J25').format.columnWidth=100;
 sheet.getRange('K1:K25').format.columnWidth=70;
 sheet.getRange('A5:K5').format={fill:'#215C49',font:{name:'Arial',size:11,bold:true,color:'#FFFFFF'},wrapText:true,rowHeight:42};
 sheet.tables.add('A5:K25',true,'ExperimentsTable');
 for(const n of names.filter(n=>n!=='Experiments')) if(JSON.stringify(old.get(n))!==JSON.stringify(wb.worksheets.getItem(n).getUsedRange().values)) throw new Error('Unrelated sheet changed: '+n);
 if(JSON.stringify(sheet.getRange('A6:G25').values)!==JSON.stringify(before.slice(1).map(r=>r.slice(0,7)))) throw new Error('Experiment settings changed');
 wb.recalculate();
 await fs.writeFile(dir+'paper_references_only.png',new Uint8Array(await (await wb.render({sheetName:'Experiments',range:'H5:K9',scale:1,format:'png'})).arrayBuffer()));
 console.log((await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?',options:{useRegex:true,maxResults:10}})).ndjson);
 await (await SpreadsheetFile.exportXlsx(wb)).save(file);
 console.log('Reference columns now contain paper references only; experiment settings and configuration links preserved.');
 process.exit(0);
}
if(process.argv.includes('--inspect')) {
 console.log((await wb.inspect({kind:'table',range:'Experiments!A5:H11',tableMaxRows:7,tableMaxCols:8,maxChars:3000})).ndjson);
 await fs.writeFile(dir+'before_dataset_rows.png',new Uint8Array(await (await wb.render({sheetName:'Experiments',range:'A1:H11',scale:1,format:'png'})).arrayBuffer()));
 process.exit(0);
}
const catalog=old.get('Model settings').slice(5).filter(r=>r[0]);
const inventory=JSON.parse(await fs.readFile(dir+'settings_inventory.json','utf8'));
const paper=old.get('Paper settings').slice(5).filter(r=>r[0]);
const main=['GPW','SVGP','ProSpar','VNNGP'];
const std=['Standard Patchwork','Standard Patchwork (GPW regions)','Standard Patchwork (PC tree)'];
const all=[...main,...std];
const repo='https://github.com/chahineNejm/Generalized-Patchwork/blob/25d3ff5/';
const experiments=[
 {id:'E01',name:'Corrected model comparison',dataset:'OISST',models:all,scope:'OISST preset',status:'Existing preset; rerun needed',q:'Accuracy and calibration after corrections',refs:[],notes:'Three Standard variants use GPW initial selected targets. GPW trains on all Y. Historical variants remain separate.'},
 {id:'E02',name:'Small tabular comparison',dataset:'California Housing',models:all,status:'Hyperparameters UNDECIDED',q:'Small 8-feature benchmark',refs:[],notes:'Standard Patchwork currently 2D only. Do not drop features. Local presets do not exist for this dataset.'},
 {id:'E03',name:'Higher-dimensional comparison',dataset:'PROTEIN',models:all,status:'Hyperparameters UNDECIDED',q:'Generalization beyond spatial inputs',refs:[0,5],notes:'Current Standard geometry needs D>2 extension. CASP source/split equivalence to references is unverified.'},
 {id:'E04',name:'Spatial replication',dataset:'UKHousing',models:all,scope:'UKHousing preset',status:'Existing partial presets',q:'Replicate spatial regression findings',refs:[1],notes:'2018-flats reconstruction, not verified VNNGP snapshot. Three matched Standard variants need explicit UK setup; legacy adaptive/all-data settings are reference only.'},
 {id:'E05',name:'Ice thickness regression',dataset:'ICETHICK',models:all,status:'Hyperparameters UNDECIDED',q:'Local adaptation on a physical field',refs:[4],notes:'Verify imported data and schema. No dataset-specific model preset selected.'},
 {id:'E06',name:'Ozone regression',dataset:'TCO',models:all,status:'Hyperparameters UNDECIDED',q:'Spatial physical-field comparison',refs:[3],notes:'TCO is distinct from TCO.L2. Specify exact release and split.'},
 {id:'E06b',name:'Ozone regression',dataset:'TCO.L2',models:all,status:'Hyperparameters UNDECIDED',q:'Spatial physical-field comparison',refs:[3],notes:'Use the TCO.L2 reference grid, not the TCO grid.'},
 {id:'E07',name:'Robot dynamics regression',dataset:'SARCOS',models:all,status:'Hyperparameters UNDECIDED',q:'Higher-dimensional regression',refs:[5],notes:'Target remains to approve. Paper comparison uses first torque. Standard D>2 support pending.'},
 {id:'E08',name:'Flight delay regression',dataset:'FLIGHT',models:all,status:'Hyperparameters UNDECIDED',q:'Large-data regression',refs:[2,6],notes:'Register trusted pickle and checksum. Original 2013 SVGP split is not verified locally.'},
 {id:'E09',name:'Satellite drag regression',dataset:'Satellite drag',models:all,status:'Hyperparameters UNDECIDED',q:'Large anisotropic response',refs:[8],notes:'Approve simulation release, inputs and sample counts. Standard D>2 geometry pending.'},
 {id:'E10',name:'Precipitation regression',dataset:'Precipitation',models:all,status:'Hyperparameters UNDECIDED',q:'Spatial/time benchmark',refs:[],notes:'Exact dataset and likelihood undecided. Do not treat UKHousing/PROTEIN VNNGP settings as precipitation settings.'},
 {id:'E11',name:'Aggregation and uncertainty',dataset:'NK synthetic Matern example',models:['GPW','Nested Kriging','Full GP'],status:'Diagnostic protocol UNDECIDED',q:'Aggregation and variance versus exact GP',refs:[9],notes:'Reference design recorded, not an adopted GPW setup. Full-GP reference has no local implementation/preset yet.'},
 {id:'E12',name:'Controlled higher-dimensional regression',dataset:'Michalewicz',models:all,status:'Hyperparameters UNDECIDED',q:'Known-function nonstationarity',refs:[7],notes:'Dimensions, noise, sampling and budgets need approval. Standard D>2 extension pending.'},
 {id:'E13',name:'Controlled higher-dimensional regression',dataset:'G-function',models:all,status:'Hyperparameters UNDECIDED',q:'Known-function nonstationarity',refs:[7],notes:'Paper structural grids are not automatically adopted. Standard D>2 extension pending.'},
 {id:'E14',name:'Point and kernel recovery',dataset:'Orthogonal-anisotropy synthetic',models:['GPW'],status:'New function/layout UNDECIDED',q:'Visualize point, stitch and ARD adaptation',refs:[],notes:'Existing heat experiment is a different diagnostic. No new hyperparameters selected.'},
 {id:'E15',name:'Time and memory scaling',dataset:'Increasing synthetic sample sizes',models:main,status:'Scaling protocol UNDECIDED',q:'Accuracy versus resource cost as N grows',refs:[],notes:'N grid, repeats, hardware and compute-budget rules need approval.'},
 {id:'E16a',name:'No-stitches ablation',dataset:'UNDECIDED',models:['GPW'],delta:{stitches_per_edge:0},status:'Dataset/protocol UNDECIDED',q:'Contribution of stitching constraints',refs:[],notes:'Same supports, data and training settings as stitched reference. Runner support must be verified.'},
 {id:'E16b',name:'Fixed-stitches ablation',dataset:'UNDECIDED',models:['GPW'],delta:{train_stitches:false},status:'Dataset/protocol UNDECIDED',q:'Contribution of learning stitch positions',refs:[],notes:'Same initial stitch positions/count as learned reference. Freeze flag not exposed by UKConfig yet.'},
 {id:'E16c',name:'Learned-stitches control',dataset:'UNDECIDED',models:['GPW'],delta:{train_stitches:true},status:'Dataset/protocol UNDECIDED',q:'Reference for no/fixed stitch comparisons',refs:[],notes:'All non-ablated settings must match the selected reference experiment.'},
 {id:'E16d',name:'Rank-of-V ablation',dataset:'UNDECIDED',models:['GPW'],delta:{q_rank:'UNDECIDED'},status:'Rank values UNDECIDED',q:'Contribution of variational cross-patch rank',refs:[],notes:'Current OISST rank is 32. Tested ranks remain undecided as requested.'},
];
const rows=[], configRows=[], firstRow=new Map(), counts={};
function baseModel(model) {return model.startsWith('Standard Patchwork')?'Standard Patchwork':model;}
function localRows(e,model) {
 if(!e.scope) return [];
 return catalog.filter(r=>r[0]===model&&r[1]===e.scope);
}
function schemaRows(model) {
 const base=baseModel(model), preset=catalog.filter(r=>r[0]===model&&r[1]==='OISST preset');
 const api=catalog.filter(r=>r[0]===base&&['Constructor defaults','Prediction API'].includes(r[1]));
 const map=new Map();
 for(const r of [...preset,...api]) if(!map.has(r[2])) map.set(r[2],r);
 return [...map.values()];
}
function append(e,model,key,value,status,note,source,override=null) {
 if(!firstRow.has(e.id)) firstRow.set(e.id,configRows.length+6);
 configRows.push([e.id,e.dataset,model,key,value,status,override,note,source]);
}
function localReference(e) {
 if(e.dataset==='OISST') return ['GPW, SVGP, ProSpar, VNNGP; Standard checkerboard / GPW-center Voronoi / PC tree','GPW: M=512, P=8, m=64, r=32, graph K=2, 8 stitches/edge. SVGP M=512. ProSpar 8x64, T=1,c=1,nuggets=0.01,jitter=0. VNNGP M=N_train,K=32. Standard selected targets=512, grid4x2/Voronoi8/tree8, 5 stitches/boundary. All full settings: Model settings, Scope=OISST preset.'];
 if(e.dataset==='UKHousing') return ['GPW, SVGP, VNNGP; legacy all-data adaptive Standard Patchwork','GPW 8x128, M=1024,r=32,K_graph=2,8 stitches/edge. SVGP M=1024. VNNGP M=N_train,K=32,B=512. Legacy Standard <=512 targets/region,2 stitches/boundary,LR=0.01. Variational:100 epochs/20 warm-up; q LR0.002,other LR0.001. Full: Model settings, Scope=UKHousing preset. ProSpar required inputs missing.'];
 if(e.id.startsWith('E16')) return ['Available candidate: OISST GPW preset; NOT selected for this ablation','Candidate only: M=512,P=8,m=64,r=32,8 learned stitches/edge,100 epochs/20 warm-up. Dataset and complete ablation configuration remain UNDECIDED.'];
 if(e.dataset==='NK synthetic Matern example') return ['Nested Kriging class exists; no diagnostic run preset selected','Constructor defaults are NOT a run: Matern-5/2,ell=0.2,outputscale=1,noise=0,jitter=0. Reference paper ell=0.05 differs. See Model settings and Paper settings.'];
 return ['No dataset-specific local reference preset','UNDECIDED. Model constructors exist; constructor defaults are not a selected dataset configuration.'];
}
function literature(e) {
 if(e.dataset==='California Housing') return ['SVGP / claimed VNNGP in gp2Scale (reference only; not our preset)',
  'Paper: SVGP M=100, VNNGP M=20000,K=50; train/test=20000/640. Inspected SVGP code: isotropic Matern-3/2 + outputscale, constant mean, full q, random observed supports, 500 full-batch updates, Adam LR=0.1; likelihood noise not optimized. Claimed VNNGP notebook actually uses independent local exact GPs: not a verified VNNGP reproduction. Source inventory: '+repo+'target.txt'];
 if(!e.refs.length) return ['No dataset-specific reference configuration in the recorded inventory','NOT RECORDED. No borrowed values or inferred defaults.'];
 const selected=e.refs.map(i=>paper[i]);
 const value=selected.map(r=>r[0]+': '+r.slice(2,6).join(' ')+' Source: '+r[6]).join('\n');
 if(e.dataset==='TCO') return ['Standard Patchwork (paper)', 'P={64,128,256,512}; stitches/boundary={5,7}; 8 structural pairs. SE and exponential kernels. Shared regional hyperparameters, recursive PC tree, uniform random boundary stitches, 90/10 split. Fitted kernel/noise values and optimizer budget not reported. '+paper[3][6]];
 if(e.dataset==='TCO.L2') return ['Standard Patchwork (paper)', 'P={256,512,1024}; stitches/boundary={3,5}; 6 structural pairs. SE and exponential kernels. Shared regional hyperparameters, recursive PC tree, uniform random boundary stitches, 90/10 split. Fitted kernel/noise values and optimizer budget not reported. '+paper[3][6]];
 return [selected.map(r=>r[0]+' ('+r[1]+')').join('; '),value];
}
for(const e of experiments) {
 counts[e.id]=0;
 for(const model of e.models) {
  const local=localRows(e,model);
  const template=schemaRows(model);
  if(local.length) {
   for(const r of local) append(e,model,r[2],r[3],r[4],r[6],r[7],r[5]);
   const keys=new Set(local.map(r=>r[2]));
   for(const r of template) if(!keys.has(r[2])) append(e,model,r[2],r[3]==='REQUIRED'?'DATA INPUT':r[3],'API reference; see effective constructor.* values','Raw API default is not necessarily the runner argument.',r[7]);
   const cfg=inventory.presets.find(p=>p.dataset===e.dataset&&p.label===model)?.config;
   if(!cfg) throw new Error('Missing source preset '+e.id+':'+model);
   const base=baseModel(model), numerical={...inventory.numerical,weight_epsilon:cfg.weight_epsilon};
   const supplied={supports:'DATA: selected observed locations',train_x:'DATA: all training locations',
    x:'DATA: selected observed locations',y:'DATA: observed targets at selected rows',
    stitch_points:'DATA: initialized cross-patch midpoints',edges:'DATA: initial patch-neighbor graph',
    lengthscale:cfg.lengthscale,outputscale:cfg.outputscale,noise_std:cfg.noise_std,
    numerical,bounds:inventory.bounds,q_rank:cfg.q_rank,
    neighbors:cfg.vnngp_neighbors,batch_size:cfg.vnngp_batch_size,jitter:inventory.numerical.prior_jitter,
    partition:cfg.exact_partition==='gpw_voronoi'?'voronoi':cfg.exact_partition,
    maximum_points:cfg.exact_maximum_points,stitches_per_edge:cfg.exact_stitches_per_edge,
    grid_shape:cfg.exact_grid_shape,tree_patches:cfg.exact_tree_patches,
    centers:cfg.exact_partition==='gpw_voronoi'?'DATA: initial GPW cluster centers':null,seed:cfg.seed};
   if(base==='ProSpar') for(const k of ['temperatures','exponent','inducing_nuggets','prior_jitter']) supplied[k]=cfg.prospar===null?'UNDECIDED':cfg.prospar[k];
   for(const [key,entry] of Object.entries(inventory.constructors[base])) {
    const value=Object.hasOwn(supplied,key)?supplied[key]:entry.default;
    if(value!==null&&typeof value==='object'&&!Array.isArray(value)) {
     for(const [sub,v] of Object.entries(value)) append(e,model,'constructor.'+key+'.'+sub,typeof v==='object'?JSON.stringify(v):v,'Effective constructor argument','Resolved from runner and preset, not guessed from class defaults.',repo+'experiments/ukhousing_benchmark.py');
    } else append(e,model,'constructor.'+key,value===null?'None':typeof value==='object'?JSON.stringify(value):value,'Effective constructor argument','Resolved from runner and preset. Data arrays come from initialization, not free hyperparameters.',repo+'experiments/ukhousing_benchmark.py');
   }
  } else if(template.length) {
   for(const r of template) {
    const dataInput=['supports','stitch_points','edges','train_x','x','y','groups','predict.x','predict.state','predict.return_components','predict.return_weights','predict.observation_noise'].includes(r[2]);
    const inactive=r[4]==='Inactive config field'||r[4]==='Not used by this model';
    const value=inactive?'NOT APPLICABLE':dataInput?'DATA INPUT / CONTROL':'UNDECIDED';
    append(e,model,r[2],value,inactive?'Inactive for model':'No experiment value selected',inactive?r[6]:'Schema only. Do not copy the OISST preset or constructor default automatically.',r[7]);
   }
  } else append(e,model,'complete_configuration','UNDECIDED','No local reference implementation/preset','Full GP is a proposed small diagnostic reference, not a configured local run.',repo+'target.txt');
  for(const [key,val] of Object.entries(e.delta??{})) {
   const record=configRows.find(r=>r[0]===e.id&&r[2]===model&&r[3]===key);
   if(record) {record[4]=val;record[5]='Explicit ablation choice';record[7]=e.notes;}
   else append(e,model,key,val,'Explicit ablation choice',e.notes,repo+'src/pwgp/models/patchwork.py');
  }
  counts[e.id]++;
 }
 const [localModel,localHP]=localReference(e), [paperModel,paperHP]=literature(e);
 const protocol=e.dataset==='OISST'?{...inventory.data,test_fraction:0.2,input_scaling:'Fixed ROI to open unit square',target_scaling:'Training-only mean and population SD'}:e.dataset==='UKHousing'?{train_fraction:0.64,validation_fraction:0.16,test_fraction:0.2,input_scaling:'Fixed UK ROI to unit square',target_scaling:'log GBP, standardized from training targets',snapshot:'2018 flats reconstruction, not verified original VNNGP snapshot'}:{split:'UNDECIDED',input_scaling:'UNDECIDED',target_scaling:'UNDECIDED',repeated_seeds:'UNDECIDED'};
 for(const [key,v] of Object.entries(protocol)) append(e,'(dataset protocol)','data.'+key,v!==null&&typeof v==='object'?JSON.stringify(v):v,e.scope?'Existing data protocol':'Not selected',e.notes,repo+'experiments/oisst_benchmark.py');
 let summary=e.scope?localHP:e.models.map(m=>m+': UNDECIDED').join('; ');
 if(e.dataset==='UKHousing') summary='Existing GPW: M=1024,P=8,m=128,r=32,graph K=2,8 stitches/edge; SVGP M=1024; VNNGP M=N_train,K=32,B=512. Variational:100 epochs/20 warm-up, q LR0.002,hyper LR0.001. ProSpar required inputs and all three matched Standard configurations remain UNDECIDED. Full values linked next.';
 if(e.delta) summary=Object.entries(e.delta).map(([k,v])=>k+'='+v).join('; ')+'. All other experiment-specific values UNDECIDED; must match the selected learned-stitch control.';
 rows.push([e.id,e.name,e.dataset,e.models.join('; '),e.q,e.status,
  summary,
  'Full configuration '+e.id,localModel,localHP,paperModel,paperHP,e.notes]);
}
const s=wb.worksheets.getItem('Experiments');
s.tables.items[0].delete();
s.getUsedRange().clear({applyTo:'contents'});
s.getRange('A2').values=[['Experiments by dataset']];
s.getRange('A3').values=[['Experiment configs contains the full per-model settings. Local presets and paper reports are separate; unknown values are not filled.']];
const headers=['Experiment ID','Experiment','Dataset','Models / candidates','Question','Status','Experiment hyperparameters','Full configuration','Available local reference models','Local reference hyperparameters','Paper reference models','Paper reference hyperparameters','Caveat / next decision'];
s.getRange(`A5:M${rows.length+5}`).values=[headers,...rows];
s.getRange(`A5:M${rows.length+5}`).format={font:{name:'Arial',size:11,color:'#252B30'},wrapText:true,verticalAlignment:'top',rowHeight:155};
s.getRange('A5:M5').format={fill:'#215C49',font:{name:'Arial',size:11,bold:true,color:'#FFFFFF'},rowHeight:42};
const widths=[12,32,27,49,40,33,52,30,54,85,42,100,70];
widths.forEach((v,i)=>s.getRangeByIndexes(0,i,rows.length+5,1).format.columnWidth=v);
s.tables.add(`A5:M${rows.length+5}`,true,'ExperimentsTable');
for(let i=0;i<experiments.length;i++) s.getCell(i+5,7).formulas=[[`=HYPERLINK("#'Experiment configs'!A${firstRow.get(experiments[i].id)}","Full configuration ${experiments[i].id}")`]];
s.freezePanes.freezeRows(5);
const config=wb.worksheets.add('Experiment configs');
config.showGridLines=false;
config.getRange('A2').values=[['Full configurations by experiment and model']];
config.getRange('A2').format.font={name:'Arial',size:16,bold:true,color:'#215C49'};
config.getRange('A3').values=[['Filter Experiment ID and Model. UNDECIDED means not chosen; this sheet does not configure or run the package.']];
config.getRange(`A5:I${configRows.length+5}`).values=[['Experiment ID','Dataset','Model','Parameter / code key','Experiment value','Status / applicability','Your override','Decision / explanation','Source'],...configRows];
config.getRange(`A5:I${configRows.length+5}`).format={font:{name:'Arial',size:11,color:'#252B30'},wrapText:true,verticalAlignment:'top',rowHeight:52};
config.getRange('A5:I5').format={fill:'#215C49',font:{name:'Arial',size:11,bold:true,color:'#FFFFFF'},rowHeight:36};
[12,29,39,48,60,45,23,72,70].forEach((v,i)=>config.getRangeByIndexes(0,i,configRows.length+5,1).format.columnWidth=v);
config.getRange(`G6:G${configRows.length+5}`).format.fill='#FFF1CC';
config.tables.add(`A5:I${configRows.length+5}`,true,'ExperimentConfigsTable');
config.freezePanes.freezeRows(5);
for(const e of experiments) {
 if(counts[e.id]!==e.models.length) throw new Error('Missing model configuration '+e.id);
 for(const model of e.models) {
  const template=localRows(e,model).length?localRows(e,model):schemaRows(model);
  for(const r of template) if(!configRows.some(v=>v[0]===e.id&&v[2]===model&&v[3]===r[2])) throw new Error('Missing experiment parameter '+e.id+':'+model+':'+r[2]);
 }
}
for(const name of names.filter(n=>n!=='Experiments')) if(JSON.stringify(old.get(name))!==JSON.stringify(wb.worksheets.getItem(name).getUsedRange().values)) throw new Error('Unrelated sheet changed: '+name);
wb.recalculate();
for(const [name,range,suffix] of [['Experiments','A1:H9','dataset_plan'],['Experiments','H5:M8','reference_columns'],['Experiment configs','A1:I11','experiment_parameters']]) {
 const preview=await wb.render({sheetName:name,range,scale:1,format:'png'});
 await fs.writeFile(dir+suffix+'.png',new Uint8Array(await preview.arrayBuffer()));
}
console.log((await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?',options:{useRegex:true,maxResults:20}})).ndjson);
console.log((await wb.inspect({kind:'table',range:'Experiments!F5:M7',tableMaxRows:3,tableMaxCols:8,maxChars:2300})).ndjson);
await (await SpreadsheetFile.exportXlsx(wb)).save(file);
console.log(JSON.stringify({experiments:rows.length,configurationRows:configRows.length,modelsPerExperiment:counts,unchanged:'Ablations, Model settings, Paper settings, Datasets, Protocol'}));
