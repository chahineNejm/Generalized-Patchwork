import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Workbook, SpreadsheetFile } from '@oai/artifact-tool';

const out = path.dirname(fileURLToPath(import.meta.url));
const wb = Workbook.create();
const repo = 'https://github.com/chahineNejm/Generalized-Patchwork/blob/62ba647/';
const defaults = repo + 'experiments/default_hyperparameters.py';
const target = repo + 'target.txt';
const papers = {
  svgp: 'https://arxiv.org/abs/1309.6835',
  patchwork: 'https://jmlr.org/papers/v19/17-042.html',
  vnngp: 'https://proceedings.mlr.press/v162/wu22h/wu22h.pdf',
  prospar: 'https://arxiv.org/abs/2311.08752v1',
  nk: 'https://arxiv.org/abs/1607.05432',
};
const mainModels = 'GPW, SVGP, ProSpar, VNNGP';
const sheetSpecs = [];
function col(n) { let s = ''; for (; n; n = Math.floor((n - 1) / 26)) s = String.fromCharCode(65 + (n - 1) % 26) + s; return s; }
function sheet(name, title, note, headers, rows, widths) {
  const s = wb.worksheets.add(name), last = col(headers.length), end = rows.length + 5;
  s.showGridLines = false;
  s.getRange(`A1:${last}${end}`).format.font = { name: 'Arial', size: 11, color: '#252B30' };
  s.getRange('A2').values = [[title]];
  s.getRange('A2').format.font = { name: 'Arial', size: 16, bold: true, color: '#215C49' };
  s.getRange('A3').values = [[note]];
  s.getRange(`A3:${last}3`).format.rowHeight = 26;
  s.getRange(`A5:${last}${end}`).values = [headers, ...rows];
  s.getRange(`A5:${last}${end}`).format.wrapText = true;
  s.getRange(`A5:${last}${end}`).format.verticalAlignment = 'top';
  s.getRange(`A5:${last}5`).format = { fill: '#215C49', font: { name: 'Arial', size: 11, bold: true, color: '#FFFFFF' }, rowHeight: 32, verticalAlignment: 'center' };
  s.getRange(`A5:${last}5`).format.horizontalAlignment = 'center';
  widths.forEach((width, i) => { s.getRange(`${col(i+1)}1:${col(i+1)}${end}`).format.columnWidth = width; });
  for (let r = 6; r <= end; r++) {
    s.getRange(`A${r}:${last}${r}`).format.rowHeight = name === 'Model settings' ? 47 : 66;
    if (r % 2 === 0) s.getRange(`A${r}:${last}${r}`).format.fill = '#F1F5F3';
  }
  s.tables.add(`A5:${last}${end}`, true, name.replace(/ /g, '') + 'Table');
  s.freezePanes.freezeRows(5);
  sheetSpecs.push({ name, rows: rows.length, cols: headers.length });
  return s;
}

const experiments = [
 ['E01','OISST corrected comparison','OISST',mainModels + ', both Standard Patchwork modes','Measure accuracy and calibration after corrections','Existing setup','Ready for rerun','Historical numbers use different GPW/standard variants. Keep separate.'],
 ['E02','Small tabular comparison','California Housing',mainModels,'Fast non-spatial benchmark','Proposed next','Needs protocol','8-feature input. Do not silently reduce to latitude/longitude. Standard baseline needs D>2 implementation.'],
 ['E03','Higher-dimensional comparison','PROTEIN',mainModels,'Test performance beyond spatial inputs','Previously selected','Needs protocol','Use approved source and split. Historical VNNGP/CASP splits are not automatically identical.'],
 ['E04','Spatial replication','UKHousing',mainModels + ', Standard Patchwork','Test whether OISST findings repeat','Existing setup','Ready existing runner','Local 2018-flats reconstruction, not verified original VNNGP snapshot. UK baseline remains all-data/adaptive.'],
 ['E05','Ice thickness regression','ICETHICK',mainModels + ', Standard Patchwork if compatible','Local adaptation on a physical field','Previously selected','Needs source/protocol','Verify imported data and dimensions before selecting partition method.'],
 ['E06','Ozone regression','TCO / TCO.L2',mainModels + ', Standard Patchwork if compatible','Additional spatial benchmark','Previously selected','Needs source/protocol','TCO and TCO.L2 are distinct datasets. Specify which snapshot is used.'],
 ['E07','Robot dynamics regression','SARCOS',mainModels,'Higher-dimensional regression','Previously selected','Needs source/protocol','Choose target explicitly. Paper comparison uses first torque.'],
 ['E08','Flight delay regression','FLIGHT',mainModels,'Large-data accuracy and scaling','Previously selected','Needs data registration','User reported trusted pickle locally. Confirm path/schema/checksum. No claim of original 2013 split.'],
 ['E09','Satellite drag regression','Satellite drag',mainModels,'Large anisotropic physical response','Previously selected','Needs source/protocol','Select simulation release, inputs and sample counts before acquisition.'],
 ['E10','Precipitation regression','Precipitation',mainModels,'Spatial benchmark overlapping VNNGP','Discussed candidate','Needs source/protocol','Exact source, coordinates/time inputs and likelihood remain undecided.'],
 ['E11','Aggregation and uncertainty','NK synthetic Matern example','GPW, Nested Kriging, full GP; optional aggregation rules','Separate aggregation effects from local prediction','Previously selected','Needs diagnostic runner','Use the paper reference design from Paper settings. Additional rules are not all implemented.'],
 ['E12','Controlled nonstationarity','Michalewicz','GPW, SVGP, ProSpar, VNNGP','Higher-dimensional controlled test','Previously selected','Needs protocol','Dimensions, noise, sampling and budgets require approval.'],
 ['E13','Controlled nonstationarity','G-function','GPW, SVGP, ProSpar, VNNGP','Higher-dimensional controlled test','Previously selected','Needs protocol','Do not automatically adopt the ProSpar full paper sweep.'],
 ['E14','Point and kernel recovery','Orthogonal-anisotropy synthetic','GPW; references to decide','Visualize kernel and point movement','Previously selected, low priority','Existing heat diagnostic only','Periodic heat was difficult. Function and initial geometry for a new test remain undecided.'],
 ['E15','Time and memory scaling','Increasing synthetic sample sizes',mainModels,'Accuracy versus measured resource cost','Previously discussed, low priority','Needs protocol','Sample-size grid, hardware, repetitions and budget rules remain open.'],
];
const exp = sheet('Experiments', 'GPW experiment plan', 'As of 2026-11-12. Planned work, not completed results. Proposed choices require approval.',
 ['ID','Experiment','Dataset','Models','Question','Decision basis','Run status','Limits / next decision'], experiments, [9,29,25,39,40,28,27,66]);
exp.tabColor = '#215C49';
exp.getRange('G6:G20').dataValidation = { rule: { type: 'list', values: ['Ready for rerun','Ready existing runner','Needs protocol','Needs source/protocol','Needs data registration','Needs diagnostic runner','Existing heat diagnostic only','Running','Completed','Failed','Deferred'] } };

const ablations = [
 ['A01','Stitching','No stitches / fixed / learned','GPW','OISST or synthetic; decide','Same supports, kernel starts, split and training budget','Previously selected','No-stitch runner path must be verified. Fixed/learned counts and placement need specification.'],
 ['A02','Initial layout','Several deliberately mismatched layouts','GPW','Synthetic; decide','Same data and total supports','Previously selected','Save initial/final supports, stitches, weights and ARD history. Exact layouts undecided.'],
 ['A03','Local adaptation','Shared versus local kernel hyperparameters','GPW','Localized rapid variation synthetic','Same geometry and observations','Previously selected, low priority','Shared-kernel mode and parameter tying need approval/implementation.'],
 ['A04','Patch count','Vary patches at fixed total supports','GPW, ProSpar','OISST; proposed','Total inducing budget, split, optimizer policy','Proposed','Choose divisible budgets. Do not confound more patches with more points.'],
 ['A05','Inducing budget','Vary total supports','GPW, SVGP, ProSpar','California / OISST; decide','Split and tuning allowance','Proposed','For VNNGP, M=N is a different capacity convention. Report it explicitly.'],
 ['A06','Stitch density','Vary stitches per graph edge','GPW, Standard Patchwork','OISST; proposed','Graph/partition and training data','Proposed','GPW graph edges and Standard geometric boundaries are different. Match within method.'],
 ['A07','Variational precision','Vary low-rank correction rank','GPW','OISST; proposed','Patches, supports, split and optimizer','Proposed','Check whether rank zero is supported before including it. Record memory and calibration.'],
 ['A08','VNNGP neighbors','K=32 versus 256 (paper candidates)','VNNGP','PROTEIN / UKHousing; decide','Kernel, data and training protocol','Proposed from paper','Not adopted. K must be below training-point count. Include neighbor-build time.'],
 ['A09','VNNGP protocol','Current configuration versus paper-style configuration','VNNGP','PROTEIN; proposed','Same approved data split','Proposed','Kernel, initialization, optimizer/scheduler and bounds require explicit decisions.'],
 ['A10','Kernel-family control','Common RBF setup versus approved paper-kernel setup','Main models where supported','Selected real dataset; decide','Data and tuning allowance','Proposed','Do not attribute a kernel-family advantage solely to aggregation or stitching.'],
 ['A11','Observation information','Selected initial targets versus all training targets','Standard Patchwork','OISST','Fixed geometry where feasible','Proposed control','Current baseline is selected-target only. All-data fixed-region exact inference may be expensive.'],
 ['A12','Uncertainty decomposition','Inspect residual / variational / observation-noise terms','GPW, SVGP, VNNGP','Existing saved diagnostic or new approved run','Same trained model and predictions','Proposed diagnostic','Inspection only. No replacement of the probabilistic model or q-dependent weights.'],
];
sheet('Ablations','Ablation studies','Change one factor at a time. Candidate values are not approved run configurations.',
 ['ID','Factor','Variants / question','Models','Dataset','Hold fixed','Decision basis','Unresolved / visual outputs'], ablations,[9,24,44,29,31,44,29,64]);

const params = [];
function p(model, scope, key, value, role='Fixed setting', source=defaults) { params.push([model,scope,key,value,role,null,'',source]); }
for (const m of ['GPW','SVGP','ProSpar','VNNGP','Standard Patchwork']) {
 p(m,'OISST current','kernel','ARD RBF','Learned kernel family fixed');
 p(m,'OISST current','initial lengthscale',0.1,'Learned after initialization');
 p(m,'OISST current','initial output variance',1,'Learned after initialization');
 p(m,'OISST current','initial observation-noise SD',0.1,'Learned after initialization');
 p(m,'OISST current','epochs',100,'Training budget');
 p(m,'OISST current','warm-up epochs',m==='Standard Patchwork'?0:20,m==='Standard Patchwork'?'Not used':'Kernel/noise frozen initially');
 p(m,'OISST current','seed',31);
 p(m,'OISST current','snapshot interval (epochs)',10,'Display/save interval');
 if (m!=='Standard Patchwork') {
   p(m,'OISST current','q learning rate',0.002,'Optimization');
   p(m,'OISST current','kernel/noise learning rate',0.001,'Optimization');
   p(m,'OISST current','training batch size',512);
 }
}
for (const m of ['GPW','SVGP','ProSpar']) p(m,'OISST current','total inducing points',512,'Trainable locations');
for (const m of ['GPW','ProSpar']) { p(m,'OISST current','patches / experts',8); p(m,'OISST current','points per patch / expert',64); p(m,'OISST current','support/structure learning rate',0.001,'Optimization'); }
for (const [key,value,role] of [
 ['q precision rank',32,'Structured q(U)'],['graph neighbors',2,'Patch graph'],['stitches per graph edge',8,'Trainable stitch locations'],
 ['weight epsilon',0.01,'Explicit numerical regularization'],['weights','Residual covariance conditional on U and stitches','Independent of q(U)'],
 ['variational covariance','(D + V V^T)^-1','Learned q(U)'],['training observations','All training rows','Data policy'],
 ['initialization','Distinct observed locations sampled within initial clusters','Location policy']]) p('GPW','OISST current',key,value,role);
p('SVGP','OISST current','q covariance','Full Cholesky','Learned q(U)');
p('SVGP','OISST current','initialization','Distinct observed training locations','Location policy');
for (const [key,value] of [['temperature initial value',1],['exponent initial value',1],['inducing nugget initial value',0.01],['additional fixed prior jitter',0]]) p('ProSpar','OISST current',key,value,'Explicit approved initialization');
p('ProSpar','Implementation','variational covariance','Independent full covariance per expert','Learned q(U_j)');
p('ProSpar','Implementation','reference','Derived from v1 equations (6)-(13), not literal (14)/(16)','Not verified author code',papers.prospar);
for (const [key,value,role] of [
 ['total inducing points','N_train','All fixed observed locations'],['nearest neighbors K',32,'Fixed geometric neighbors'],['inducing batch size',512,'Stochastic KL/likelihood'],
 ['neighbor metric','Euclidean on normalized inputs','Not learned ARD distance'],['q covariance','Diagonal mean-field','Learned independent variances'],
 ['inducing locations','Fixed throughout training','M=N specialization']]) p('VNNGP','OISST current',key,value,role);
for (const [key,value,role] of [
 ['selected observations',512,'Exact GPW initial selected rows and targets'],['grid shape','4 x 2 (longitude x latitude)','Fixed checkerboard mode'],
 ['Voronoi regions',8,'Fixed initial GPW clustering centers'],['stitches per shared boundary',5,'Fixed geometric placement'],
 ['learning rate',0.01,'Full joint-likelihood Adam updates'],['objective','log p(Y, Delta_S=0)','Separate kernel parameters per region'],
 ['empty region policy','Raise error','No dropping or merging'],['exact_maximum_points','Not active for fixed grid / Voronoi','Legacy adaptive mode only']]) p('Standard Patchwork','OISST current',key,value,role);
for (const [key,value] of [['prior jitter',0.0001],['minimum variational Cholesky diagonal',0.001],['variance validity threshold',1e-12],['gradient clip norm',10],['weight decay',0],['lengthscale bounds','(0.02, 3.0)'],['output-variance bounds','(0.02, 30.0)'],['noise-SD bounds','(0.0001, 1.0)']]) p('Applicable models','Numerical controls',key,value,'Do not silently modify',repo+'src/pwgp/config.py');
p('Applicable models','OISST current','prediction batch size',256);
p('Applicable models','OISST current','runtime','CUDA / float64 / 2 CPU threads','Requested runtime');
for (const m of ['GPW','SVGP']) p(m,'UKHousing current','total inducing points',1024);
p('GPW','UKHousing current','patches / points per patch','8 / 128');
p('GPW and SVGP','UKHousing current','training batch size',1024);
p('Standard Patchwork','UKHousing current','training targets','All training observations','Legacy protocol differs from OISST');
p('Standard Patchwork','UKHousing current','partition','Adaptive rectangles, <=512 observations per region','No predefined region count');
p('Standard Patchwork','UKHousing current','stitches per boundary',2);
p('ProSpar','UKHousing runner','initialization and additional jitter','UNDECIDED','Required call-time values. No UK default preset.');
for (const m of ['GPW','SVGP']) p(m,'Heaton current','total inducing points',480);
p('GPW','Heaton current','patches / points per patch','8 / 60','4 / 120 also historically requested');
p('GPW and SVGP','Heaton current','training / prediction batch size',2048);
p('Nested Kriging','Implementation','groups','Explicit disjoint groups in arbitrary D','Exact local observations',repo+'src/pwgp/models/nested_kriging.py');
p('Nested Kriging','Implementation','kernel options','Shared ARD RBF or Matern-5/2','New experiment choice required',target);
p('Nested Kriging','Implementation','fitting','LOO MSE with Adam, not paper SPSA','Outputscale calibration requires noiseless/zero-jitter',target);
for (const m of ['GPW','SVGP','ProSpar','VNNGP','Standard Patchwork','Nested Kriging']) p(m,'New datasets','dataset-specific preset','UNDECIDED','Do not copy OISST defaults without approval');
const settings = sheet('Model settings','Model hyperparameters','Current values are reference presets, not prescriptions for new datasets. Your override cells are editable.',
 ['Model','Scope','Parameter','Current value / behavior','Meaning','Your override','Decision / note','Source'],params,[25,25,37,54,46,22,29,64]);
settings.getRange(`F6:G${params.length+5}`).format.fill='#FFF1CC';

const reported = [
 ['VNNGP','PROTEIN','M=N_train. K in {32,256}. ARD Matern-5/2. Mean-field q.','Init ell/output/noise parameter=0.6931. Adam LR {0.005,0.01}. Budget {300,500} reported iterations.','LR x0.1 at 75%/90%. 64/16/20 split. 3 seeds. Validation NLL selection.','8 K/LR/budget combinations before seeds. Winning values not reported.',papers.vnngp],
 ['VNNGP','UKHousing','M=N_train. K in {32,256}. ARD Matern-5/2.','For >50k: LR {0.005,0.001}, budget {100,300} reported iterations.','Same initialization/scheduler as above.','Our UK data reconstruction is not the paper snapshot.',papers.vnngp],
 ['SVGP','FLIGHT','M=1000 main. Batch=5000. ARD squared exponential + bias + noise.','q LR=0.01. Kernel LR=1e-5, momentum=0.9.','700k train /100k test. 10 main runs. M sweep {50,100,200,500,800,1000}.','Training budget not explicit in checked prose. Original split not verified locally.',papers.svgp],
 ['Standard Patchwork','TCO / TCO.L2','TCO: P {64,128,256,512}, stitches {5,7}. TCO.L2: P {256,512,1024}, stitches {3,5}.','Shared regional hyperparameters in real-data experiments. SE and exponential kernels.','Recursive local-PC partition, 90/10 random split.','8 and 6 structural combinations respectively. Not our current OISST geometry.',papers.patchwork],
 ['Standard Patchwork','ICETHICK','P {64,128,256,512,1024}. Stitches {3,5,7}.','SE and exponential kernels.','15 structural combinations.','Fitted kernel values and optimizer budget not reported in inventory.',papers.patchwork],
 ['Standard Patchwork','PROTEIN / SARCOS','PROTEIN: P {64,128,256}, stitches {2,3,4}. SARCOS: P {128,256}, stitches {3,5,7}.','Squared-exponential kernels.','9 and 6 structural combinations respectively. SARCOS first torque.','Our Standard Patchwork implementation currently 2D only.',papers.patchwork],
 ['Standard Patchwork','FLIGHT','P=1024, stitches=5, squared-exponential kernel.','Shared regional parameters in paper real-data runs.','All training observations, not inducing-only regression.','Not equivalent to our selected-target OISST baseline.',papers.patchwork],
 ['ProSpar','Michalewicz / G-function','P=200, m=25. Local SE kernels. Batch=1024.','d {3,5,10} x train N {50k,100k}; test=25k; LHS; noise variance=1e-8.','10 repetitions per combination. 6 data configurations per function.','LR, optimization budget and T/c/nugget starts not reported in inventory.',papers.prospar],
 ['ProSpar','Satellite drag','P=200, m=25. Inputs d=8.','Train N {100k,175k,250k}; test=175k from 2M simulations.','10 repetitions per configuration.','Supplied preprint v1, not verified final-publication replication.',papers.prospar],
 ['Nested Kriging','1D synthetic Matern','N=30, P=15, two consecutive sorted observations/expert.','Matern-5/2 variance=1, lengthscale=0.05, zero mean; held fixed.','101 test-grid points, 50 independent function/design repetitions.','Same experts for aggregation comparisons. Full GP uses all 30 observations.',papers.nk],
];
sheet('Paper settings','Paper-reported reference settings','Recorded literature inventory. These configurations have not been adopted as new run defaults.',
 ['Model','Dataset','Structure / kernel','Training / initialization','Design / repetitions','Caveat','Source'],reported,[25,28,60,68,65,62,65]);

const datasets = [
 ['OISST','2D spatial SST','Local data folder exists','Current benchmark split and loader','Existing notebook and runner','Current 64/16/20, seed 31. Day 2025-01-15.','Separate old/corrected model results.',defaults],
 ['California Housing','8 input features','Not in inspected local data folders','Acquire and approve snapshot/split','Needs dataset/runner integration','20,640 observations in standard dataset.','Full 8D benchmark not supported by current 2D Standard Patchwork.','https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset'],
 ['PROTEIN / CASP','9 input features','Not in inspected local data folders','Acquire and approve snapshot/split','Needs dataset/runner integration','45,730 observations in UCI CASP.','Do not claim VNNGP original split reproduction.','https://archive.ics.uci.edu/dataset/265/physicochemical+properties+of+protein+tertiary+structure'],
 ['UKHousing','2D location / flats prices','Local data folder exists','Verified 2018-flats reconstruction','Existing runner','Existing 64/16/20 split.','Not verified VNNGP snapshot. Legacy Standard baseline differs from OISST.',repo+'src/pwgp/data/ukhousing.py'],
 ['ICETHICK','Physical field; verify schema','Acquisition/import code exists; data presence unverified','Verified import workflow','Integration to comparison needs review','Source host has CAPTCHA in earlier workflow.','Verify exact imported file/checksum before running.',target],
 ['TCO / TCO.L2','Ozone; separate data releases','Preparation workflows recorded; files unverified','Choose dataset identity first','Integration to comparison needs review','Do not combine TCO and TCO.L2.','Verify dimensionality, units, rows and original split claims.',target],
 ['SARCOS','Robot dynamics, multidimensional','Data not verified locally','Source and torque target undecided','Needs dataset/runner integration','First torque used in recorded Patchwork reference.','Target selection requires approval.',target],
 ['FLIGHT','Delay regression','Trusted pickle reported by user; path not checked here','Trusted author-prepared pickle','Loader exists; comparison integration to review','Confirm schema/checksum and train/test rules.','Pickle is an explicitly trusted source, not arbitrary untrusted input.',target],
 ['Satellite drag','8D reference simulation','Source not verified locally','Simulation release and variables undecided','Needs dataset/runner integration','Do not claim downloaded/ready.','Sample counts and target transforms undecided.',target],
 ['Precipitation','Spatial/time inputs undecided','Source not verified locally','Exact benchmark undecided','Needs dataset/runner integration','Candidate discussed alongside UKHousing.','Identify release/likelihood before any benchmark claim.',papers.vnngp],
 ['NK synthetic','1D Matern function draws','Generate after protocol approval','Reference design in Paper settings','NK class implemented','No fitting in selected paper example.','GPW adaptation to reference design still to specify.',papers.nk],
 ['Michalewicz / G-function','Higher-dimensional synthetic','Generate after protocol approval','Function/sampling/noise definitions needed','Needs experiment definitions','Paper settings are references only.','Choose dimensions and budgets explicitly.',papers.prospar],
 ['Heaton / MODIS','2D satellite temperature','Local data folder exists','Existing official comparison workflow','Existing runner and notebook','Existing 480-support presets available.','Older experiment, separate from OISST.',defaults],
];
sheet('Datasets','Dataset readiness','Local availability is distinguished from an implemented loader or a literature candidate.',
 ['Dataset','Task / inputs','Availability','Source / split status','Implementation readiness','Known setup','Open decision / caution','Source'],datasets,[27,35,49,47,42,57,62,66]);

const protocol = [
 ['P01','Approval','Split and preprocessing','Current OISST/UK protocols already set. New datasets undecided.','Choose fractions, leakage controls, transforms and target units.','Do not introduce spatial holdout by default.'],
 ['P02','Approval','Repeated seeds','Current default seed=31 only.','Choose repeats, seeds and aggregation statistics.','Pair methods on identical split/seed per repeat.'],
 ['P03','Fairness','Data information','GPW uses all Y. OISST Standard uses selected initial targets.','Decide whether an all-data Standard control is also wanted.','Label reduced-data results explicitly.'],
 ['P04','Fairness','Capacity matching','OISST GPW/SVGP/ProSpar M=512; VNNGP M=N_train.','Choose inducing-budget comparisons separately from compute-budget comparisons.','M is not equivalent model capacity across all constructions.'],
 ['P05','Fairness','Optimization budget','100 epochs /20 warm-up current variational presets.','Choose matched time/update/convergence analysis.','An epoch is not equal work across models. Standard uses full hyperparameter updates.'],
 ['P06','Approval','Tuning allowance','No new tuning grid selected.','Choose validation metric, candidate count and selection rule.','Keep test set untouched during tuning. Record all attempted configurations.'],
 ['P07','Approval','Kernel family','Current comparisons use ARD RBF. Paper VNNGP real data uses Matern-5/2.','Choose common-kernel and/or paper-style tracks.','Do not silently change kernels under the same experiment label.'],
 ['P08','Evaluation','Accuracy and uncertainty','Existing metrics include RMSE, MAE, NLPD, CRPS, coverage and interval width.','Agree primary metric and units. Optional interval score and calibration curves.','Keep observation-noise and latent-variance reporting distinct.'],
 ['P09','Visuals','Mechanism evidence','Support/stitch and kernel histories available in selected runners.','Choose saved trajectories, residual maps, weights and variance decomposition.','History updates in place. Metric tables remain.'],
 ['P10','Resources','Runtime and memory','Existing runners log train/predict time and CUDA memory.','Choose hardware, warm-up, repeats and timing scope.','Include preparation and VNNGP graph-building separately.'],
 ['P11','Reproducibility','Source and artifacts','Latest-main Colab records resolved commit/config/environment.','Keep immutable batch IDs, checksums, predictions and full error records.','No silent retries, resume, skips or fallback.'],
 ['P12','Implementation','D>2 baseline capability','Standard Patchwork currently supports 2D geometry only.','Approve extension before claiming full-feature California/PROTEIN baseline.','Do not silently use a two-feature projection.'],
 ['P13','Interpretation','Negative controls','No-stitch/shared-kernel controls still require setup.','Specify what evidence would support or refute stitching benefit.','Avoid choosing only tasks where GPW wins.'],
 ['P14','Tracking','Results log','No new completed metrics entered in this workbook.','Record experiment ID, run path, model, seed, config and completion/failure.','Historical results from different variants are not pooled.'],
 ['P15','Model scope','Optional baselines','Nested Kriging implemented. CPoE/SWSGP not ready as comparison models.','Decide whether optional models justify extra implementation.','ProSpar is paper-derived, not verified author code.'],
];
sheet('Protocol','Before launching experiments','Additional items to settle. This sheet records decisions still needed; it does not select them.',
 ['ID','Category','Item','Current situation','Decision / action needed','Guardrail'],protocol,[9,24,32,67,68,66]);

wb.recalculate();
for (const spec of sheetSpecs) {
 const info = await wb.inspect({kind:'table',range:`'${spec.name}'!A5:${col(spec.cols)}8`,include:'values,formulas',tableMaxRows:4,tableMaxCols:4,maxChars:900});
 console.log(spec.name, spec.rows, info.ndjson);
 const image = await wb.render({sheetName:spec.name,range:`A1:${col(Math.min(spec.cols,5))}9`,scale:1,format:'png'});
 await fs.writeFile(path.join(out,spec.name.replace(/ /g,'_')+'.png'),new Uint8Array(await image.arrayBuffer()));
}
console.log((await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!',options:{useRegex:true,maxResults:25},summary:'Final error scan'})).ndjson);
const xlsx = await SpreadsheetFile.exportXlsx(wb);
await xlsx.save(path.join(out,'GPW_experiment_plan.xlsx'));
console.log(JSON.stringify({output:path.join(out,'GPW_experiment_plan.xlsx'),sheets:sheetSpecs}));
