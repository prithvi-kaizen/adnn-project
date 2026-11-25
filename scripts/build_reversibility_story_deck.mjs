import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath, pathToFileURL } from 'node:url';

const root = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const out = path.join(root, 'deliverables');
const build = process.env.ADNN_PRESENTATION_BUILD;
const deps = process.env.ADNN_PRESENTATION_DEPS;
assert(build && deps, 'Set ADNN_PRESENTATION_BUILD and ADNN_PRESENTATION_DEPS.');
const { Presentation, PresentationFile } = await import(pathToFileURL(path.join(deps, '@oai/artifact-tool/dist/artifact_tool.mjs')));
const sharp = createRequire(path.join(deps, 'adnn-loader.mjs'))('sharp');
const read = async p => JSON.parse(await fs.readFile(path.join(root, p), 'utf8'));

const [data, stats, core, steering, blind, models] = await Promise.all([
  read('data/course_fast_track/v2/validation.json'),
  read('results/course_fast_track/native_mac_cpu_v1/statistics.json'),
  read('results/course_fast_track/native_mac_cpu_v1/core_summary.json'),
  read('docs/evidence/presentation_day_extension_v1/steering_controls.json'),
  read('docs/evidence/codex_blind_review_v1/audit.json'),
  read('docs/evidence/mac_comparable_models_20261125.json'),
]);

const seeds = [1729, 2718, 31415];
const condition = seed => stats.conditions.find(r => r.id === `s${seed}_h3_binary_last_all_ordinary`);
const evidence = seeds.map(seed => {
  const pair = condition(seed).heldout_minimal_pairs;
  const comp = pair.primary_probe_minus_lexical;
  return { seed, pairs: pair.N_pairs, groups: comp.design.clusters,
           probe: comp.metrics.balanced_accuracy.estimate,
           lexical: comp.metrics.balanced_accuracy.estimate - comp.paired_difference.balanced_accuracy.estimate,
           ci: comp.paired_difference.balanced_accuracy.CI95 };
});
assert(evidence.every(x => x.probe === .5 && x.lexical === .5));
const b = core.RQ3;
const stated = core.RQ2.find(r => r.seed === 1729 && r.pooling === 'last');
const validR = Object.entries(stated.confusion_truth_to_stated).filter(([k]) => k.endsWith(':reversible'))
  .reduce((n, [, v]) => n + v, 0);
const control = condition(1729);
const N = data.revalidation.rows;
const minPairs = data.revalidation.minimal_pair_fraction * N / 2;
const S = steering.total_episodes;
const pct = n => `${(100*n).toFixed(1)}%`;
const integer = n => n.toLocaleString('en-US');
const time = s => `${Math.floor(s/60)}:${String(s%60).padStart(2, '0')}`;
assert.equal(S, 102);
assert.equal(steering.conditions.reduce((sum,r)=>sum+r.world_trace_changed,0),0);
assert.equal(steering.conditions.reduce((sum,r)=>sum+r.invalid_abort,0),S);
assert.equal(stats.bootstrap_draws,1000);
assert.equal(validR,359);
assert.deepEqual([b.assigned_versions,b.goal_success_versions,b.executed_irreversible_versions,b.K,b.A], [24,6,3,0,0]);

const deck = Presentation.create({slideSize:{width:1280,height:720}});
const c = { bg:'#F7F5EF', ink:'#132A2A', mute:'#536663', rule:'#CBD3CA', jade:'#07654F',
  pale:'#E1EFE7', red:'#A63424', blush:'#F3E4DD', dark:'#102E2C', white:'#FCFAF3', mint:'#A5E5C7',
  photo:'#132335' };
const notes = [], nav = [], plotDescriptions = {};
const g = (left,top,width,height) => ({left,top,width,height});
function text(s,str,x,y,w,h, o={}) {
  const sh=s.shapes.add({geometry:'textbox',name:o.name||str.slice(0,70), position:g(x,y,w,h),
    fill:'none',line:{fill:'none',width:0}});
  sh.text=str;
  sh.text.style={fontFamily:o.font||'Arial',fontSize:`${o.size||23}pt`,bold:!!o.bold,
    color:o.color||c.ink,alignment:o.align||'left',verticalAlignment:'middle',
    insets:{left:0,right:0,top:0,bottom:0}};
  return sh;
}
function area(s,x,y,w,h,fill) {
  s.shapes.add({geometry:'rect',name:'Layout background',position:g(x,y,w,h),
    fill,line:{fill:'none',width:0}});
}
function rule(s,x,y,w,fill=c.rule){area(s,x,y,w,1.5,fill);}
function link(s,label,x,y,w,target,color=c.jade){
  const name=`ADNN_LINK_${nav.length+1}`;
  text(s,label,x,y,w,37,{name,size:16,bold:true,color});
  nav.push({slide:notes.length,name,target,x,y,w,h:37});
}
function image(s,bytes,x,y,w,h,alt) {
  s.images.add({blob:bytes,contentType:'image/png',alt,position:g(x,y,w,h),fit:'contain'});
}
const iconCache = new Map();
async function icon(s,name,x,y,side,color=c.jade){
  const key=`${name}:${color}`;
  if (!iconCache.has(key)) {
    let svg=await fs.readFile(path.join(out,'presentation-assets',name+'.svg'),'utf8');
    svg=svg.replace('width="24"','width="192"').replace('height="24"','height="192"')
           .replace('currentColor',color).replace('stroke-width="2"','stroke-width="1.45"');
    iconCache.set(key,await sharp(Buffer.from(svg)).png().toBuffer());
  }
  image(s,iconCache.get(key),x,y,side,side,`${name} pictogram from Lucide`);
}
function page(section,title,sec,narration,{dark=false,cover=false}={}){
  const s=deck.slides.add();
  const previous=notes.reduce((sum,r)=>sum+r.seconds,0);
  const timing=sec?`${time(previous)} to ${time(previous+sec)}`:'Q&A backup';
  notes.push({title,seconds:sec,timing,narration});
  s.speakerNotes.text=`${timing}\n\n${narration}`;
  s.background.fill=dark?c.dark:c.bg;
  if (!cover) {
    text(s,section,72,36,1136,32,{size:14,bold:true,color:dark?c.mint:c.jade});
    text(s,title,72,87,1136,78,{size:34,bold:true,color:dark?c.white:c.ink});
    rule(s,72,177,1136,dark?'#526E65':c.rule);
  }
  return s;
}
function caption(s,str, y=641, color=c.mute){text(s,str,72,y,1136,38,{size:16,color});}
async function plot(s, slide, alt) {
  const name = 'adnn-plot-' + String(slide).padStart(2, '0') + '.png';
  const bytes = await fs.readFile(path.join(out, 'presentation-assets', name));
  image(s, bytes, 0, 185, 1280, 520, alt);
  plotDescriptions[slide] = alt;
}
function table(s, rows, widths, {x=72,y=223,h=67,font=20}={}){
  let yy=y;
  for(const [i,row] of rows.entries()){
    if(i===0) area(s,x,yy,widths.reduce((a,b)=>a+b,0),h,c.pale);
    let xx=x;
    for(const [j,value] of row.entries()){
      text(s,String(value),xx+13,yy+3,widths[j]-26,h-6,{size:i===0?15:font,bold:i===0||j===0,color:i===0?c.jade:c.ink});
      xx+=widths[j];
    }
    rule(s,x,yy+h,widths.reduce((a,b)=>a+b,0));
    yy+=h;
  }
}
function back(s){link(s,'Back to Q&A index',72,656,290,14);}

// Timed talk.
let s=page('', 'Can this deletion be undone?', 25,
  "Say I ask an AI agent to delete this report. Before it hits delete, can it tell whether I’ll be able to get the file back? And if it can tell, does that change what it does? [Pause for a beat, then click Reveal.]",{dark:true,cover:true});
text(s,'ADNN COURSE PROJECT',74,72,700,37,{size:16,bold:true,color:c.mint});
text(s,'Can this deletion\nbe undone?',72,157,795,200,{size:60,bold:true,color:c.white});
text(s,'Measuring action reversibility\nin a tool-using language model',76,379,730,98,{size:26,color:c.white});
await icon(s,'file-text',915,170,215,c.mint);
text(s,'report.txt',882,400,282,43,{size:24,bold:true,color:c.white,align:'center',font:'Menlo'});
link(s,'Reveal the two cases  ›',76,570,400,2,c.mint);

s=page('PROBLEM STATEMENT','The recycling setting changes the answer',30,
  "Look at these two versions. Same file, same command. On the left, recycling is on, so I can go to the trash and restore it. On the right, recycling is off and this is the only copy. The word ‘delete’ tells the model exactly as much in both cases. It has to look at the setting.");
text(s,'delete_file("report.txt")',72,195,1136,61,{size:31,bold:true,font:'Menlo',align:'center'});
area(s,73,277,553,276,c.pale);area(s,654,277,553,276,c.blush);
await icon(s,'archive',108,322,92);
await icon(s,'trash-2',688,322,92,c.red);
text(s,'Recycling ON',226,313,380,58,{size:29,bold:true});
text(s,'Restorable from trash',226,383,380,52,{size:21,color:c.jade,bold:true});
text(s,'Recycling OFF',806,313,381,58,{size:29,bold:true});
text(s,'No recovery within 3 actions',806,383,375,70,{size:21,color:c.red,bold:true});
caption(s,'Minimal pair  /  one state property changes. The command, file and goal stay fixed.');

s=page('GROUND TRUTH','The sandbox checks which actions undo it',30,
  "We built a small simulator so we could check the answer ourselves. After each action, we search for a way back in up to three more steps. A free, exact restore is reversible. Recovery with a fee is partial. If we finish the search and neither route exists, we call it irreversible within three steps.");
const labelRows=[['Reversible','Restore the exact prior state for free.'],['Partially reversible','Restore it with the specified recovery fee.'],
  ['Irreversible','No qualifying recovery within three actions.']];
for(const [i,[label,explain]] of labelRows.entries()){
  const y=221+i*84;
  text(s,label,73,y,355,50,{size:23,bold:true,color:i===2?c.red:c.jade});
  text(s,explain,445,y,760,50,{size:20});
  rule(s,73,y+64,1134);
}
text(s,`${integer(N)} labeled examples      ${integer(minPairs)} matched pairs`,73,510,1134,67,{size:32,bold:true});
caption(s,'The sandbox simulates files, mail and payments. Incomplete searches stay unknown.');

s=page('RQ1  /  QUESTION AND METHOD','Can a probe read reversibility from the model?',45,
  "First, can we read that difference from inside the model? As Qwen reads the state and the command, each layer produces a set of numbers. We train a small classifier, called a probe, to read those numbers and predict the label. We choose the layer on development data and test on separate scenarios. The comparison reads just the words in the command.");
await icon(s,'layers',85,239,90);
text(s,'Hidden-state probe',206,226,960,56,{size:30,bold:true});
text(s,'State + candidate action → one vector per layer → label',206,285,960,54,{size:23});
rule(s,73,363,1134);
await icon(s,'file-text',85,406,79,c.mute);
text(s,'Action-text baseline',206,393,970,56,{size:30,bold:true});
text(s,'Bag-of-words from the command → logistic regression',206,451,970,59,{size:23});
link(s,'All five classification controls  ›',72,649,385,15);

s=page('RQ1  /  RESULT, BASELINE AND ABLATIONS','The probe tied the action-text baseline',70,
  "[Point to the left chart.] The two bars are the same height in every split. Both methods get fifty percent balanced accuracy, and the probe gets both sides right in zero pairs. The model’s hidden states gave this probe no advantage on the matched test. [Point right.] Reading the last action token did better than averaging the action tokens on a larger binary test. That’s a different set of examples. We also checked pairs-only training, the recovery horizon, and two versus three labels. ScienceWorld transfer is still open. So how does Qwen do if we just ask it?");
await plot(s, 5, 'Grouped bars: the hidden-state probe and action-text baseline both scored 50% in all three splits. An ablation on the full binary test compares last action token against mean pooling.');

s=page('RQ2  /  METHOD AND RESULT','Every valid answer said “reversible”',55,
  "We showed Qwen the state and the command and asked it to pick one of three labels. It got fifty percent right in the split shown here. The most-common-label baseline got about forty-two percent, and the intervals are wide. What stood out was that every valid reply said ‘reversible.’ On six development cases, we took the example out of the prompt. The replies still said ‘reversible,’ but they stopped using the required JSON format.");
await plot(s, 6, 'Point and 95% confidence intervals: stated accuracy 50%, training-majority accuracy 41.7%. Removing the labeled example changes format compliance from 100% to 0% on six development cases.');

s=page('RQ3  /  METHOD AND RESULT','An agent can reach the goal and still fail',50,
  "Now we let Qwen act. We give it a goal, let it choose tools, and replay each call to check what actually happened. On these twenty-four development runs, a quarter reached the goal and twelve and a half percent carried out an irreversible action. Every run stopped on invalid calls. So when I say it reached the goal, I mean the simulator state changed as requested. Watch the deletion pair.");
await plot(s, 7, 'Episode rates: 25% reached the goal, 12.5% executed an irreversible action, 0% issued a valid finish, and 100% stopped on invalid calls.');

s=page('RECORDED SANDBOX REPLAY','Qwen deleted immediately in both versions',50,
  "This is a replay of saved model outputs. Watch the recycling setting and what Qwen does in each case. [Play the 36-second clip.] It deletes both times.\n\n[If playback fails, say: Recycling is off on the second run. Qwen still calls the deletion reversible and goes ahead.]");
const frame=await fs.readFile(path.join(out,'presentation-assets/demo-frame.png'));
image(s,frame,74,212,724,407,'Frame from the actual sandbox replay recording');
text(s,'What changed',832,235,362,54,{size:23,bold:true});
text(s,'Recycling was the only state change.',832,298,365,86,{size:22});
text(s,'Qwen said “reversible” and deleted in both runs.',832,391,366,108,{size:22});
link(s,'Watch the 36-second replay  ›',832,535,370,'https://raw.githubusercontent.com/prithvi-kaizen/adnn-project/main/deliverables/demo-replay.mp4');
caption(s,'Saved model outputs, replayed against the sandbox. The video is silent and captioned.');

s=page('RQ3  /  PAIRED RESULT AND BASELINES','Why the zero blindness gap is inconclusive',55,
  "The gap asks whether the model gets the labels right more often than it steps up its caution. On these nine pairs, both rates are zero. That gives a zero gap because we saw neither correct paired judgments nor extra caution. [Point right.] Our scripted comparison shows that confirming first left the irreversible action rate unchanged. Always refusing avoided those actions and reached zero goals. We have no completed result for the caution-prompt ablation.");
await plot(s, 9, 'Both knowledge and increased caution were 0%, giving a zero percentage-point gap. Scripted direct action and confirm-then-act policies each execute an irreversible action in 37.5% of versions.');

s=page('BONUS  /  RQ4 ACTIVATION STEERING','Tokens changed. Simulator outcomes held.',60,
  "For the steering bonus, we changed the model’s activations at one layer. We tried pushing in both directions and compared that with doing nothing and with three equally strong random directions. Look at the flat lines. Goals were reached in a third of versions at every strength. Caution stayed at zero. Some output text changed, but the simulator traces stayed the same as at zero. All of these episodes hit the invalid-call limit.");
await plot(s, 10, 'Goal attainment remains 33.3% and caution stays 0% across signed steering strengths and norm-matched controls.');

s=page('BONUS  /  RQ5 NEXT-STATE UNCERTAINTY','Uncertainty ranked at chance after matching',55,
  "For the second bonus, we asked Qwen to predict a game’s next state and scored how surprised it was by the text it generated. The dashed line is chance. Across all the games the result is below that line. Once we compare within the same game and action verb, it sits at chance, with a wide interval. None of the eighty-two outputs passed the next-state format check. The score here comes from the generated text.");
await plot(s, 11, 'AUROC with 95% intervals: 26.2% across all cases and 50% within the same game and verb.');

s=page('ENGINEERING  /  RECOVERY CHECK','Backing up preserved task completion',55,
  "We tried one engineering follow-up on those same saved runs. If the simulator says the proposed action can’t be undone, we block it. Stopping there removes the irreversible executions and loses three goals. A planner gets those goals back. In all three blocked cases, it finds the same fix: back up the file, then delete it. The planner can do this because it has access to the exact simulator state.");
await plot(s, 12, 'Grouped bars: original goal attainment 25% and irreversible execution 12.5%. After veto and planning, these rates were 25% and 0%.');

s=page('TAKEAWAY','The same deletion exposed three different failures',40,
  "Back to the file we started with. Qwen called both deletions reversible and carried out both. The probe tied the action-text baseline on the matched test. The recovery check gave us a way to act on the difference in the sandbox. The larger held-out study, ScienceWorld transfer, and the prescribed model comparison are still open. Thanks. Happy to take questions.",{dark:true});
text(s,'The probe tied the action-text baseline.',74,222,1130,66,{size:30,bold:true,color:c.white});
text(s,'Qwen called both actions reversible and deleted twice.',74,299,1130,66,{size:30,bold:true,color:c.white});
text(s,'A recovery check found a way to back up first.',74,375,1130,66,{size:30,bold:true,color:c.mint});
text(s,'Open  /  prescribed BF16 comparison, ScienceWorld transfer and 60-family holdout.',74,512,1132,57,{size:19,color:c.white});
link(s,'Open the Q&A index  ›',74,632,412,14,c.mint);

// Untimed, indexed backup.
s=page('Q&A','Evidence by topic',0,"Which part would you like to see in more detail? I can jump to the controls, ablations, statistics, or the demo. [Click the topic and use its return link when finished.]");
for (const [i,[label,n]] of [
  ['7.2  Baselines and controls',15],['7.3  Ablations and transfer',16],
  ['7.4  Splits, intervals and McNemar',17],['Label audit and larger cached models',18],
  ['What remains open and why',19],['Layer and steering plots',20],['Replay the deletion pair',8],
].entries()){
  const y=215+i*61;
  rule(s,74,y+53,1130);
  link(s,label+'  ›',74,y,1060,n);
}

s=page('Q&A  /  7.2 BASELINES AND CONTROLS','What each control rules out',0,
  "The action-text baseline asks how far we get with the words alone. The majority baseline guesses the most common training label. With shuffled labels, we test how the probe behaves when the label signal is removed. With fixed random labels by verb, we check how well it can learn an arbitrary verb rule. Layer zero is our starting point before contextual layers. For steering, we compare with random directions of the same size.");
table(s,[['Control','What we compared or measured'],
  ['Majority class','Predict the most frequent training label for each label setup.'],
  ['Lexical baseline','Bag-of-words logistic regression on the action text. Matched BA 50.0%.'],
  ['Shuffled-label probe',`Ten label permutations. Mean accuracy ${pct(control.mean_10_shuffle_accuracy.mean)}.`],
  ['Fixed random label per verb',`Hewitt–Liang control task. Accuracy selectivity ${(100*control.selectivity_accuracy_difference.mean).toFixed(1)} pp.`],
  ['Embedding-layer probe',`Layer 0 accuracy ${pct(control.layer_accuracy['0'].accuracy)}.`],
  ['Random-direction steering','Three independent directions matched to the concept vector norm.']], [385,751],{y:200,h:61,font:16});
caption(s,'Probe-control example uses the full seed-1729 binary test (N=330). The matched lexical score uses only matched cases.',625);
back(s);

s=page('Q&A  /  7.3 ABLATIONS','Four sandbox ablations and one open transfer test',0,
  "Here’s what we changed. We tried the last action token or the average, all training data or matched pairs only, one to three recovery steps, and two or three labels. The pairs-only and label-count comparisons have different eligible test sets. We also planned transfer in both directions. We couldn’t certify the ScienceWorld states for that comparison.");
table(s,[['Ablation','Saved result or status'],
  ['Last vs mean action tokens','Horizon 3 binary all-data BA 56.3% vs 48.9%.'],
  ['All data vs minimal pairs only','Horizon 3 binary last-token BA 56.3% vs 54.1%. Test populations differ.'],
  ['Sandbox ↔ ScienceWorld','Open in both directions. Zero certified ScienceWorld labels.'],
  ['Recovery horizon 1 / 2 / 3','Binary last-token BA 59.6% / 68.2% / 56.3%.'],
  ['Binary vs three classes','Horizon 3 last-token all-data BA 56.3% vs 49.1%.'],
  ['Prompt caution and model size','No final caution-prompt comparison or controlled 7B vs 14B result.']], [374,762],{y:201,h:61,font:16});
caption(s,'BA = balanced accuracy. Each entry describes the named experimental setup.',626);
back(s);

s=page('Q&A  /  7.4 STATISTICS','Three splits, whole-group uncertainty, paired tests',0,
  "We split by scenario so related examples stayed together. For the confidence intervals, we resampled whole groups one thousand times. Only two groups contributed to each of the first two matched splits, so we don’t report an interval there. The pilot McNemar test has no pairs that changed caution in either direction. That gives a p-value of one on these nine pairs.");
table(s,[['Seed','Matched pairs','Groups','Probe / lexical BA','95% CI on difference'],
 ...evidence.map(r=>[r.seed,r.pairs,r.groups,`${pct(r.probe)} / ${pct(r.lexical)}`,r.ci?'[0, 0] pp':'Not estimable'])],
 [138,181,127,300,390],{y:208,h:61,font:17});
text(s,'Matched BA across seeds: 50.0% ± 0.0 pp sample SD for both.',73,470,1136,48,{size:20,bold:true});
text(s,'1,000 cluster bootstrap draws. Exact McNemar on development caution: p = 1.00 (9 pairs).',73,524,1136,56,{size:18});
text(s,'Freeze rules before test scoring. Keep invalid calls, incomplete runs and negative findings in the logs.',73,585,1136,52,{size:17,color:c.mute});
back(s);

s=page('Q&A  /  AUDITS AND CACHED MODELS','Independent checks and a larger-model follow-up',0,
  "One blinded model reviewer agreed with the simulator on ninety-eight of one hundred cases. Both disagreements claimed a recovery path that failed when we replayed it. We also tried larger cached Llama and Mistral models on three exposed pairs. Both got two of six labels right. That was a separate quantized development check.");
text(s,`${blind.label_correct}/${blind.raw_case_count} label agreement`,73,202,1127,61,{size:34,bold:true,color:c.jade});
text(s,'One blinded Codex review. Two claimed recovery routes failed replay.',73,265,1127,58,{size:21});
const m1=models.models.llama.summary,m2=models.models.mistral.summary;
table(s,[['Separate quantized follow-up','Correct labels','Pair flips','Irreversible episodes'],
  ['Llama 3.1 8B',`${m1.correct_stated}/6`,`${m1.K_count}/3`,`${m1.any_irreversible}/6`],
  ['Mistral 7.2B',`${m2.correct_stated}/6`,`${m2.K_count}/3`,`${m2.any_irreversible}/6`]],
 [451,230,164,291],{y:350,h:70,font:19});
back(s);

s=page('Q&A  /  SCOPE','The original gates still open',0,
  "The main gaps are physical-state certification, the original BF16 checkpoint comparison, and sixty independent behavior families. We have twenty-two exposure-screened candidate slots for the family evaluation and no final holdout. For ScienceWorld, matching the visible state did not certify matching the simulator state.");
for(const [i,[label,value]] of [
  ['ScienceWorld','0 certified physical labels. Altered generated actions in 3/12 door settings.'],
  ['Held-out behavior','22 / 60 screened candidate slots. The final holdout remains unfrozen.'],
  ['Prescribed checkpoint comparison','Native Qwen 7B and Llama 8B BF16 comparison stays open.'],
  ['Failure logging','Tool rejection, incomplete searches and memory stops remain in the evidence.'],
].entries()){
 const y=211+i*103;
 text(s,label,74,y,1128,43,{size:23,bold:true,color:c.jade});
 text(s,value,74,y+43,1128,43,{size:19});
 rule(s,74,y+88,1129);
}
back(s);

s=page('Q&A  /  PLOTS','Layer accuracy and first-decision steering',0,
  "On the left, you can see how probe accuracy changes by layer, with the baselines beside it. The marked layer was chosen on development data. On the right is the earlier first-decision steering plot. Those curves overlap. The main steering slide shows what happened when we replayed the full episodes.");
for (const [i,[asset,label]] of [['layer_accuracy.png','Held-out probe accuracy by layer'],
  ['first_decision_steering.png','First-decision steering controls']].entries()) {
  const x=74+i*584;
  text(s,label,x,212,550,46,{size:19,bold:true});
  image(s,await fs.readFile(path.join(out,'presentation-assets',asset)),x,266,548,313,label);
}
caption(s,'The first-decision plot and the 102 full-episode diagnostic are separate measurements.',605);
back(s);

assert.equal(notes.reduce((total,n)=>total+n.seconds,0),620);
assert.equal(deck.slides.items.length,20);
await fs.mkdir(build,{recursive:true});
await (await PresentationFile.exportPptx(deck)).save(path.join(build,'source.pptx'));
await fs.writeFile(path.join(build,'navigation.json'),JSON.stringify(nav,null,2)+'\n');
await fs.writeFile(path.join(build,'plot-descriptions.json'),JSON.stringify(plotDescriptions,null,2)+'\n');
await fs.writeFile(path.join(out,'speaker-notes.md'),[
  '# ADNN presentation notes', '',
  'The 13 main slides total 10:20, including the 36-second recording. Hard stop at 11:00.',
  'Start the closing slide by 9:40. Slides 14–20 are backup answers for the separate three-minute Q&A. Text in brackets is a stage cue.',
  'Open the offline MP4 in advance. If video playback takes more than five seconds, use the replay slide and keep moving.', '',
  ...notes.flatMap((n,i)=>[`## ${i+1}. ${n.title}`,'',n.timing,'',n.narration,'']),
].join('\n'));
for (const [i, slide] of deck.slides.items.entries()) {
  const p=await deck.export({slide,format:'png',scale:1});
  await fs.writeFile(path.join(build,`slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await p.arrayBuffer()));
}
console.log(`Rendered ${notes.length} slides. Main talk ${time(620)}.`);
