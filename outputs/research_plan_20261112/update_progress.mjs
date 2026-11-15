import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import {fileURLToPath} from 'node:url';
import {FileBlob, SpreadsheetFile} from '@oai/artifact-tool';

const dir = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(dir, '../..');
const file = path.join(dir, 'GPW_experiment_plan.xlsx');
const source = path.join(root, 'results/archive_audit_20261113/audit.json');
const audit = JSON.parse(await fs.readFile(source, 'utf8'));
assert.equal(audit.checks_failed, 0);
assert.equal(audit.runs.length, 61);
const wb = await SpreadsheetFile.importXlsx(await FileBlob.load(file));
if (process.argv.includes('--polish')) {
  const exp = wb.worksheets.getItem('Experiments');
  for (let row = 6; row <= 25; row++) {
    const text = exp.getRange(`P${row}`).values[0][0];
    if (typeof text === 'string') exp.getRange(`P${row}`).values = [[text
      .replace(/SVGP: 31 \(M=(\d+),P=\d+\)/g, 'SVGP: seed31 (M=$1)')
      .replace(/Standard Patchwork \(PC tree\): 31 \(M=(\d+),P=\d+\)/g, 'Standard Patchwork (PC tree): seed31 ($1 selected observed targets)')]];
  }
  const config = wb.worksheets.getItem('Experiment configs');
  const values = config.getRange('A8212:I8907').values;
  for (let index = 0; index < values.length; index++) if (values[index][3] === 'config.model') {
    config.getRange(`E${8212+index}:F${8212+index}`).values = [['gpw','GPW scaling requested by user']];
  }
  wb.recalculate();
  for (const [sheetName,range,suffix] of [['Experiments','M5:Q8','experiment_status'],['Experiment configs','A8212:F8217','scaling_parameters']]) {
    const blob = await wb.render({sheetName,range,scale:1,format:'png'});
    await fs.writeFile(path.join(dir,`${suffix}_20261113.png`),new Uint8Array(await blob.arrayBuffer()));
  }
  await (await SpreadsheetFile.exportXlsx(wb)).save(file);
  const check = await SpreadsheetFile.importXlsx(await FileBlob.load(file));
  assert.equal(check.worksheets.getItem('Completed runs').getRange('A6:A66').values.length,61);
  assert.equal(check.worksheets.getItem('Experiments').getRange('N6').values[0][0],1);
  assert.equal(check.worksheets.getItem('Experiment configs').getRange('E8212').values[0][0],'gpw');
  assert.equal(check.worksheets.getItem('Experiments').getRange('C37').values[0][0],'FLIGHT');
  console.log('Saved workbook reopened; completed runs, seed counts, scaling rows and GPW model identity verified.');
  process.exit(0);
}
const preserved = ['Model settings', 'Paper settings', 'Datasets', 'Protocol'];
const before = new Map(preserved.map(name => [name, JSON.stringify(wb.worksheets.getItem(name).getUsedRange().values)]));
const exp = wb.worksheets.getItem('Experiments');
const abl = wb.worksheets.getItem('Ablations');
assert.equal(exp.getRange('M5').values[0][0], null);
const expRows = exp.getRange('A6:L25').values;
const originalExp = JSON.stringify(expRows);
const originalLinks = JSON.stringify(exp.getRange('H6:H25').formulas);
const originalAbl = JSON.stringify(abl.getRange('A6:H19').values);
const originalConfigs = JSON.stringify(wb.worksheets.getItem('Experiment configs').getUsedRange().values);
const displayNames = {oisst:'OISST', ukhousing:'UKHousing', modis:'MODIS', california:'California Housing',
  tco:'TCO', tco_l2:'TCO.L2', icethick:'ICETHICK', protein:'PROTEIN', sarcos:'SARCOS',
  michalewicz_d5_n50000:'Michalewicz', g_function_d5_n50000:'G-function', flight:'FLIGHT'};
const byDisplay = new Map(Object.entries(displayNames).map(([key, value]) => [value, key]));
const selected = ['California Housing', 'PROTEIN', 'TCO', 'OISST'];
const seeds = runs => [...new Set(runs.map(run => run.config.seed))].sort((a,b) => a-b);
const headerFormat = {fill:'#215C49', font:{name:'Arial',size:11,bold:true,color:'#FFFFFF'},
  wrapText:true, verticalAlignment:'center', horizontalAlignment:'center', rowHeight:42};
const progressHeaders = ['Execution status', 'Completed seeds (verified)', 'Completed seed IDs', 'Completed configurations / scope', 'Evidence / remaining work'];
const counts = [];
for (let index = 0; index < expRows.length; index++) {
  const row = expRows[index];
  const runs = audit.runs.filter(run => run.dataset === byDisplay.get(row[2]));
  if (runs.length) {
    const ids = seeds(runs);
    const scope = runs.map(run => `${run.name}: ${seeds([run]).join(',')} (M=${run.config.inducing_points},P=${run.config.patches})`).join('\n');
    counts.push(['Partially completed', ids.length, ids.join(', '), scope,
      `Audited archive: ${runs.length} runs. One seed per listed configuration, not ${runs.length} seeds. VNNGP and other unlisted configurations are not in this batch.`+
      (['Michalewicz','G-function'].includes(row[2]) ? ' Only D=5, training N=50000 is completed here; the full dimension/size grid is not.' : '')]);
  } else if (String(row[0]).startsWith('E16')) {
    counts.push(['In progress (reported)', 0, '', 'Planned: California Housing, PROTEIN, TCO, OISST; seeds31-33; ranks0/32; none/fixed/learned; 72 runs.',
      'Kaggle batch reported running. Screenshot confirms OISST seed31 rank0 stitches-none saved; no ablation archive received locally. Zero means zero locally verified seeds, not zero executed remotely. See Ablations.']);
  } else {
    counts.push([['E09','E10'].includes(row[0]) ? 'Deferred' : 'Not run (not verified)', 0, '', 'No completed result for this specified protocol in the audited batch.',
      row[0] === 'E14' ? 'Older heat diagnostics exist, but are not evidence that this still-undecided recovery experiment is completed.' :
      row[0] === 'E15' ? 'Prior exploratory scaling was discussed; no verified systematic scaling grid is counted. Dataset-specific GPW scaling studies are listed below.' :
      'Count refers to verified saved results, not implementation readiness.']);
  }
}
exp.getRange('M5:Q25').values = [progressHeaders, ...counts];
exp.getRange('M5:Q25').format.wrapText = true;
exp.getRange('M5:Q25').format.verticalAlignment = 'top';
exp.getRange('M5:Q5').format = headerFormat;
for (const [c,width] of [['M',29],['N',19],['O',19],['P',65],['Q',85]]) exp.getRange(`${c}1:${c}37`).format.columnWidth=width;
exp.getRange('N6:N37').setNumberFormat('0');
exp.getRange('A3').values = [['Execution columns M:Q updated 2026-11-13 from the audited 61-run archive. Historical plan/configuration columns are preserved; actual executed settings are in Completed runs and Run parameters.']];

const scaleDatasets = [...Object.keys(displayNames)];
const scaleRows = scaleDatasets.map((key,index) => {
  const name = displayNames[key];
  const reference = expRows.find(row => row[2] === name);
  return [`S${String(index+1).padStart(2,'0')}`, 'GPW scaling', name, 'GPW',
    'How do training/prediction time, peak memory, RMSE and NLPD change with data size and model capacity?',
    'Planned; protocol UNDECIDED',
    'UNDECIDED: scaling axis/axes (N, total inducing M, patches P, stitches); grids; fixed controls; seeds; hardware; training/stop policy. No sweep values approved.',
    'See Experiment configs: scaling fields UNDECIDED',
    reference ? reference[8] : 'No additional reference assigned',
    reference ? reference[9] : 'GPW is our method',
    reference ? reference[10] : 'No paper hyperparameters adopted for this scaling study',
    'Planning only. Do not treat the existing 4/8-patch comparisons as a completed scaling sweep. Preserve dataset-specific preprocessing; all scaling choices require approval.',
    key === 'flight' ? 'Not run; currently excluded' : 'Not run', 0, '',
    key.startsWith('michalewicz') || key.startsWith('g_function') ? 'Dataset anchor: D5/train50k; scaling grid UNDECIDED.' : 'No systematic GPW scaling grid completed.',
    'User requested dataset-specific GPW scaling studies on 2026-11-13. No training launched.'];
});
exp.getRange(`A26:Q${25+scaleRows.length}`).values = scaleRows;
exp.getRange(`A26:Q${25+scaleRows.length}`).format = {font:{name:'Arial',size:11,color:'#252B30'},wrapText:true,verticalAlignment:'top',rowHeight:180};
exp.tables.items[0].delete();
exp.tables.add(`A5:Q${25+scaleRows.length}`, true, 'ExperimentsTable');
exp.getRange(`M6:M${25+scaleRows.length}`).dataValidation = {rule:{type:'list',values:[
  'Completed','Partially completed','In progress (reported)','Not run','Not run (not verified)','Deferred','Not run; currently excluded']}};

const ablCounts = [];
for (const row of abl.getRange('A6:H19').values) {
  if (['A01a','A01b','A01c','A07'].includes(row[0])) {
    ablCounts.push(['In progress (reported)',0,'', '3 planned:31,32,33, per dataset/condition',
      'Current crossed study: California Housing, PROTEIN, TCO, OISST;1024 supports=8x128; ranks0/32;none/fixed/learned;100/20 schedule. Shared initial geometry per dataset/seed. No local ablation results yet. '+
      (row[0] === 'A01a' ? 'Screenshot confirms at least OISST seed31 rank0 none completed; other completed conditions/seeds are unknown until results arrive.' : 'Do not infer completion from the planned seed list or from older unmatched benchmark runs.')]);
  } else if (row[0] === 'A04') {
    ablCounts.push(['Partially completed',1,'31','GPW and ProSpar:4 versus8 patches at matched M in multiple audited datasets.',
      'Exploratory two-size comparisons only. See Completed runs for dataset/model/configuration coverage; not a systematic scaling sweep.']);
  } else {
    ablCounts.push(['Not run (not verified)',0,'','No verified matched ablation protocol.',
      row[0] === 'A03' ? 'Kernel study remains planned. Shared learned versus fixed common kernel parameters is UNDECIDED; not included in the current72-run study.' :
      'Historical diagnostics and a saved standard prediction are not counted as completion of this separate controlled ablation.']);
  }
}
abl.getRange('I5:M19').values = [progressHeaders.slice(0,3).concat(['Seed target / completed scope','Evidence / current approved setup']),...ablCounts];
abl.getRange('I5:M19').format = {font:{name:'Arial',size:11,color:'#252B30'},wrapText:true,verticalAlignment:'top'};
abl.getRange('I5:M5').format = headerFormat;
for (const [c,width] of [['I',29],['J',19],['K',19],['L',62],['M',95]]) abl.getRange(`${c}1:${c}19`).format.columnWidth=width;
abl.getRange('J6:J19').setNumberFormat('0');
abl.getRange('A3').values = [['Status updated 2026-11-13. Completed seeds means locally verified distinct seeds for the stated configuration. Current approved setup is in M; earlier planning notes in A:H are preserved.']];
abl.tables.items[0].delete();
abl.tables.add('A5:M19',true,'AblationsTable');

function tableSheet(name,title,note,headers,rows,widths) {
  const sheet = wb.worksheets.add(name);
  const last = String.fromCharCode(64+headers.length), end=rows.length+5;
  sheet.showGridLines=false;
  sheet.getRange('A2').values=[[title]];
  sheet.getRange('A2').format.font={name:'Arial',size:16,bold:true,color:'#215C49'};
  sheet.getRange('A3').values=[[note]];
  sheet.getRange(`A5:${last}${end}`).values=[headers,...rows];
  sheet.getRange(`A5:${last}${end}`).format={font:{name:'Arial',size:11,color:'#252B30'},wrapText:true,verticalAlignment:'top',rowHeight:48};
  sheet.getRange(`A5:${last}5`).format=headerFormat;
  widths.forEach((width,index)=>sheet.getRangeByIndexes(0,index,end,1).format.columnWidth=width);
  sheet.tables.add(`A5:${last}${end}`,true,name.replaceAll(' ','')+'Table');
  sheet.freezePanes.freezeRows(5);
  return sheet;
}
function flatten(value,prefix='') {
  return Object.entries(value).flatMap(([key,item])=>{
    const name=prefix ? `${prefix}.${key}` : key;
    if(item !== null && typeof item === 'object' && !Array.isArray(item)) return flatten(item,name);
    return [[name,item === null ? 'null (as saved)' : Array.isArray(item) ? JSON.stringify(item) : item]];
  });
}
const inventory = audit.runs.map((run,index)=>[index+1,displayNames[run.dataset],run.dataset,run.name,'Completed',
  run.config.seed,1,run.config.inducing_points,run.model === 'gpw' || run.model === 'prospar' ? run.config.patches : null,
  run.config.q_rank,run.training_summary.epochs_completed,run.metrics.rmse_original_target,run.metrics.nlpd_original_target,
  run.run,`audit.json: runs[${index}]; source commit ${audit.provenance.commit}`]);
const completed=tableSheet('Completed runs','Verified completed runs',
  '61 completed runs,11 datasets, seed31 only. One row is one configuration/run, not an additional seed. Source: '+source,
  ['Run ID','Dataset','Dataset key','Model / configuration','Status','Seed','Seeds for this configuration','Inducing M / selected-target budget','Patches (GPW/ProSpar)','Saved q_rank field','Actual epochs','RMSE (original target)','NLPD (original target)','Run path in audited archive','Evidence'],
  inventory,[10,25,32,35,18,10,18,23,21,18,16,23,23,36,74]);
completed.getRange('F6:K66').setNumberFormat('0');
completed.getRange('L6:M66').setNumberFormat('0.000000');
const parameters = audit.runs.flatMap((run,index)=>flatten({config:run.config,environment:run.environment,model_metadata:run.model_metadata})
  .map(([key,value])=>[index+1,displayNames[run.dataset],run.name,run.config.seed,key,value,'Saved effective value',`${source} : runs[${index}]`]));
tableSheet('Run parameters','Effective parameters for completed runs',
  'Saved configuration, numerical constraints, optimizer and environment. Inactive configuration fields remain explicitly saved, not claimed active. Original plan sheets are historical.',
  ['Run ID','Dataset','Model / configuration','Seed','Parameter / exact saved path','Effective value','Meaning','Source'],
  parameters,[10,25,35,10,55,65,28,80]);

const config=wb.worksheets.getItem('Experiment configs');
const configData=config.getRange('A1:I9000').values;
let last=9000;
while(last>0 && configData[last-1].every(value=>value===null || value==='')) last--;
const schema=flatten({config:audit.runs.find(run=>run.model==='gpw').config,
  numerical:audit.runs.find(run=>run.model==='gpw').environment.numerical,
  bounds:audit.runs.find(run=>run.model==='gpw').environment.bounds,
  optimizer:audit.runs.find(run=>run.model==='gpw').environment.optimizer});
const scaleParameters=scaleRows.flatMap(row=>[
  ...schema.map(([key])=>[row[0],row[2],'GPW',key,'UNDECIDED','Not approved for scaling study','',
    'Parameter inventory only; do not inherit an unapproved scaling configuration.','User request2026-11-13; current GPW schema']),
  ...['scaling_axes','sample_size_grid','inducing_budget_grid','patch_count_grid','stitch_count_grid','fixed_controls',
    'seeds','hardware','split_subset_policy','training_budget_policy','timing_protocol'].map(key=>[
    row[0],row[2],'GPW',`scaling.${key}`,'UNDECIDED','User decision required','','No numerical sweep chosen.','User request2026-11-13'])]);
config.getRange(`A${last+1}:I${last+scaleParameters.length}`).values=scaleParameters;
config.getRange(`A${last+1}:I${last+scaleParameters.length}`).format={font:{name:'Arial',size:11,color:'#252B30'},wrapText:true,verticalAlignment:'top',rowHeight:52};
config.tables.items[0].delete();
config.tables.add(`A5:I${last+scaleParameters.length}`,true,'ExperimentConfigsTable');
for(let index=0;index<scaleRows.length;index++) {
  const target=last+1+index*(schema.length+11);
  exp.getRange(`H${26+index}`).formulas=[[`=HYPERLINK("#'Experiment configs'!A${target}","Scaling configuration ${scaleRows[index][0]}")`]];
}

assert.equal(JSON.stringify(exp.getRange('A6:L25').values),originalExp);
assert.equal(JSON.stringify(exp.getRange('H6:H25').formulas),originalLinks);
assert.equal(JSON.stringify(abl.getRange('A6:H19').values),originalAbl);
const oldConfigs=JSON.parse(originalConfigs);
assert.deepEqual(config.getRange('A2:I'+last).values,oldConfigs);
for(const name of preserved) assert.equal(JSON.stringify(wb.worksheets.getItem(name).getUsedRange().values),before.get(name),`Unrelated sheet changed: ${name}`);
assert.equal(inventory.length,61);
assert.equal(new Set(inventory.map(row=>row[5])).size,1);
assert.equal(scaleRows.length,12);
assert.equal(new Set(scaleRows.map(row=>row[2])).size,12);
wb.recalculate();
console.log((await wb.inspect({kind:'table',range:'Experiments!M5:Q8',tableMaxRows:4,tableMaxCols:5,maxChars:2500})).ndjson);
console.log((await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!|#SPILL!',options:{useRegex:true,maxResults:30},summary:'Final formula error scan'})).ndjson);
for(const [name,range,suffix] of [['Experiments','M5:Q8','experiment_status'],['Experiments','A26:F28','scaling'],
  ['Ablations','I5:M9','ablation_status'],['Completed runs','A5:K10','completed'],['Run parameters','A5:G12','parameters'],
  ['Experiment configs',`A${last+1}:F${last+6}`,'scaling_parameters']]) {
  const blob=await wb.render({sheetName:name,range,scale:1,format:'png'});
  await fs.writeFile(path.join(dir,`${suffix}_20261113.png`),new Uint8Array(await blob.arrayBuffer()));
}
await (await SpreadsheetFile.exportXlsx(wb)).save(file);
console.log(JSON.stringify({file,completedRuns:inventory.length,scalingStudies:scaleRows.length,
  executedParameterRows:parameters.length,scalingParameterRows:scaleParameters.length,preservedSheets:preserved}));
