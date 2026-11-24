import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation, PresentationFile} from '@oai/artifact-tool';

const root=process.env.ADNN_RESEARCH_ROOT || process.cwd();
const build=process.env.ADNN_SLIDES_BUILD;
const skill=process.env.ADNN_PRESENTATIONS_SKILL;
const finalPath=process.env.ADNN_SLIDES_FINAL;
if(!build || !skill || !finalPath) throw new Error('Explicit build, skill and final paths required');
const {resolvePresentationFont,applyPresentationChartFont,finalizePresentation}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const font=resolvePresentationFont({fontFamily:'Arial'});
const read=async p=>JSON.parse(await fs.readFile(path.join(root,p),'utf8'));
const summary=await read('results/course_fast_track/native_mac_cpu_v1/core_summary.json');
const stats=await read('results/course_fast_track/native_mac_cpu_v1/statistics.json');
const validation=await read('data/course_fast_track/v2/validation.json');
const references=await read('artifacts/course_fast_track/native_mac_cpu_v1/references/report.json');
const diagnostic=await read('results/course_fast_track/native_first_decision_steering_v1/summary.json');
if(diagnostic.status!=='PASS_INDEPENDENT_FIRST_DECISION_REPLAY')throw new Error('Unverified diagnostic');
const deck=Presentation.create({slideSize:{width:1280,height:720}});
const ink='#17364B', rust='#AD4B29', green='#276B57';
const notes=[];
function text(slide,value,x,y,w,h,size=24,bold=false,color=ink){
 const s=slide.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 s.text=value;s.text.style={fontFamily:font,fontSize:`${size}pt`,bold,color};return s;
}
function slide(title,seconds,note){
 const s=deck.slides.add();s.background.fill='#FFFFFF';
 text(s,title,64,45,1152,105,32,true);
 s.speakerNotes.text=`Planned speaking time: ${seconds} seconds.\n${note}`;
 notes.push({slide:notes.length+1,title,seconds,notes:note});return s;
}
function body(s,lines,y=180,size=24){text(s,lines.join('\n\n'),74,y,1120,440,size);}
function table(s,values,widths,y=180,height=340){
 const t=s.tables.add({rows:values.length,columns:values[0].length,left:74,top:y,width:1132,height,values,columnWidths:widths});
 for(let r=0;r<values.length;r++)for(let c=0;c<values[0].length;c++){
  const cell=t.getCell(r,c);cell.fill=r===0?ink:'#FFFFFF';
  cell.text.style={fontFamily:font,fontSize:'20pt',color:r===0?'#FFFFFF':ink,bold:r===0};
 }
 t.borders.assign({style:'solid',fill:'#D9E1E6',width:1});return t;
}
function chart(s,type,config){
 const c=s.charts.add(type,{position:{left:74,top:175,width:1130,height:430},...config});
 applyPresentationChartFont(c,{fontFamily:font});
 for(const style of [c.legend.textStyle,c.xAxis.textStyle,c.yAxis.textStyle])style.fontSize=22;
 if(c.yAxis.title?.text)c.yAxis.title.textStyle.fontSize=22;
 return c;
}
let s=slide('Reversibility and tool use',45,'Introduce the reduced automated sandbox pilot. The report and raw records concern one native Qwen2.5-0.5B checkpoint on Mac CPU. Do not present this as the original two-family BF16 study. Source: paper/course_fast_track/report.pdf and core_summary.json.');
text(s,'A finite sandbox course project',74,220,1120,100,42,true);
text(s,'State-dependent labels, linear readouts and replayed agent traces',74,370,1100,150,26);
s=slide('The same deletion can have different recovery options',60,'Use the frozen delete-bin pair as the motivating example. State equality is the declared complete simulator state. All filenames and funds are fictional. A legal restore supplies a positive witness. A negative requires exhaustive legal search. Sources: sandbox_pool.jsonl and the data card.');
table(s,[['Property','Reversible version','Irreversible version'],['Candidate','Same delete_file request','Same delete_file request'],['Recycling','Enabled','Disabled'],['Recovery','restore_file from trash','No legal restore in h=3']], [240,446,446]);
s=slide('The runnable experiment',75,'The 1,800 rows include 1,620 unique pair members in 810 pairs. There are 17 conservative connected components, not 1,800 independent observations. Three split allocations overlap by 300 of 360 test rows. The model is native float32, not Ollama Q4. Source: freeze validation, execution config and feature manifest.');
table(s,[['Item','Actual condition'],['Checkpoint','Qwen2.5-0.5B-Instruct, pinned revision'],['Inference','Mac CPU, float32, four threads, greedy'],['Data','1,800 examples, 810 minimal pairs'],['Independence','17 connected components, three split seeds']], [300,832]);
s=slide('Recovery labels depend on the horizon',60,'Define reversible as exact free restoration, partial as the prespecified history-fee exception after excluding a free inverse, and irreversible as exhausted failure within the horizon. Unknown is excluded. These counts come directly from validation.json.');
chart(s,'bar',{categories:['h=1','h=2','h=3'],series:[['Reversible','reversible',ink],['Partial','partially_reversible',rust],['Irreversible','irreversible',green]].map(([name,key,fill])=>({name,values:[1,2,3].map(h=>validation.by_horizon[String(h)][key]),fill})),hasLegend:true,legend:{position:'bottom',overlay:false},barOptions:{direction:'column',grouping:'clustered'},yAxis:{title:{text:'Examples'}}});
s=slide('Representation, stated judgment and behavior',60,'Explain the distinction rather than assuming a chain of causation. Readout accuracy measures a separately trained linear classifier. Stated labels use a fixed strict JSON prompt. Actual actions pass the sandbox parser and state validator. Sources: COURSE_CORE_PROTOCOL.md and methods section.');
table(s,[['Measurement','Observed evidence'],['Representation','Action-token vectors and held-out linear predictions'],['Stated judgment','Raw label responses, including invalid output'],['Behavior','Executed tools, pre-state labels, goals and failures']], [330,802]);
s=slide('Aggregate scores need matched controls',90,'This editable chart shows selected last-token binary accuracy, action-only accuracy and majority accuracy across three overlapping splits. Full layer curves, controls and component intervals appear in the paper. The primary matched comparison follows on the next slide. Source: core_summary.json, h=3 binary last all ordinary.');
const p=[1729,2718,31415].map(seed=>summary.probe_conditions.find(r=>r.id===`s${seed}_h3_binary_last_all_ordinary`));
// Four decimal fractions preserve more precision than the displayed percentage
// and remain representable in the editable Excel chart snapshot.
chart(s,'bar',{categories:['1729','2718','31415'],series:[{name:'Probe',values:p.map(r=>Number(r.selected_metrics.accuracy.toFixed(4))),fill:ink},{name:'Action-only',values:p.map(r=>Number(r.lexical.accuracy.toFixed(4))),fill:rust},{name:'Majority',values:p.map(r=>Number(r.majority.accuracy.toFixed(4))),fill:green}],hasLegend:true,legend:{position:'bottom',overlay:false},barOptions:{direction:'column',grouping:'clustered'},yAxis:{min:0,max:1,numberFormatCode:'0%'}});
s=slide('The primary matched comparison is null',75,'Each pair retains the same action and changes one state property. All three last-token probe-minus-action-only BA differences are zero. Joint correctness is also zero. Two components fail the prespecified CI threshold in the first two splits. The last zero-width interval is conditional on three groups. Sources: statistics.json heldout_minimal_pairs, Table II.');
table(s,[['Seed','Pairs / groups','Delta BA','Joint correct'],...stats.conditions.filter(r=>r.condition.horizon===3 && r.condition.binary && r.condition.pooling==='last' && r.condition.subset==='all' && r.condition.control==='ordinary').map(r=>[String(r.condition.seed),`${r.heldout_minimal_pairs.N_pairs} / ${r.heldout_minimal_pairs.joint_pair_correctness.design.clusters}`,`${(100*r.heldout_minimal_pairs.primary_probe_minus_lexical.paired_difference.balanced_accuracy.estimate).toFixed(1)} points`,`${(100*r.heldout_minimal_pairs.joint_pair_correctness.mean).toFixed(1)}%`])],[190,280,330,332]);
s=slide('Valid JSON can conceal a constant-label answer',60,'For seed 1729, every valid response is reversible. One partial case is invalid. The fixed prompt has a reversible format example, so response anchoring and limited model capability remain possible explanations. Do not generalize to all prompting formats or larger models. Source: the stored confusion table and raw stated responses.');
const q=summary.RQ2.find(r=>r.seed===1729&&r.pooling==='last');
const labels=['reversible','partially_reversible','irreversible'];
table(s,[['Truth','Response R','Response P','Response I','Invalid'],...labels.map((a,i)=>[['R','P','I'][i],...labels.concat('invalid').map(b=>String(q.confusion_truth_to_stated[`${a}:${b}`]||0))])],[220,228,228,228,228]);
s=slide('A zero gap does not establish informed alignment',75,'The registered gap is mean(K minus A). Both K and A equal zero here. All 24 episodes terminate after two invalid calls, although six deletion versions achieve their goals before aborting. No version shows registered caution. Nine eligible pairs share three templates. Sources: replayed behavior records and core summary.');
body(s,[`K = ${summary.RQ3.K.toFixed(2)}   A = ${summary.RQ3.A.toFixed(2)}   Gap = ${summary.RQ3.Gap.toFixed(2)}`,`${summary.RQ3.goal_success_versions} of 24 goals hold at termination`,`${summary.RQ3.rejected_calls} rejected calls and 24 invalid-call aborts`,'Nine R/I pairs share three development templates'],180,26);
s=slide('Reference policies expose the tradeoff',60,'These are scripted, replayed policies, not model measurements. Confirmation increases the caution predicate but does not change how many versions execute an irreversible action. Always refusing avoids execution while achieving no goal. Sources: references/report.json, Table VI.');
const ref=new Map();for(const r of references.records){if(!ref.has(r.policy))ref.set(r.policy,[]);ref.get(r.policy).push(r.audit);}
table(s,[['Scripted policy','Goal / 24','Caution / 24','Any I / 24'],...[...ref].map(([name,rows])=>[name.replaceAll('_',' '),String(rows.filter(r=>r.goal_success).length),String(rows.filter(r=>r.caution).length),String(rows.filter(r=>r.executed_irreversible_action).length)])],[430,234,234,234]);
s=slide('Offline demo with an actual saved pair',120,'Run the saved-trace replay command. Contrast the recycling flag and oracle witnesses. Show the actual delete call, the two rejected prose calls and final goal/state. Explicitly say this replays real saved model output without regenerating a model response. The simulator never deletes host files. Source: demo_course_project.py and the two pilot-delete-0 episodes.');
body(s,['Compare the matched deletion versions','Replay the recorded model outputs and world states','Inspect the reached goal and invalid termination','Command: python scripts/demo_course_project.py --pair pilot-delete-0'],180,23);
s=slide('Scope and the next informative test',120,'Close with the negative matched result, low model/tool reliability and the limited independent groups. The study has one small native family, no certified physical data, no human annotations and fewer than 60 behavior families. Full steering and RQ5 remain separate. The next useful experiment needs reliable tool use and correct stated labels before interpreting a knowing/doing gap. Reserve the final minute for questions. Source: paper discussion and requirements matrix.');
body(s,['The matched readout comparison provides no primary advantage','Reliable tool use and paired stated knowledge remain prerequisites','The original BF16, physical and human requirements remain incomplete','Code, raw outputs, grouped statistics and replay records are available'],175,23);
s=slide('Backup: a bounded first-decision steering diagnostic',0,'Untimed backup slide, outside the 15-minute talk. This is one decision on six exposed versions in three templates, not the full RQ4 study. The raw block is development-selected; the direction uses training-only I-minus-R means. Four timing calls and six fresh zero checks are excluded from endpoint N. There are 96 fresh nonzero cells plus six shared saved baselines. Every endpoint is unchanged, with degenerate paired-difference intervals conditional on three templates. The paper provides the strength curves and wide absolute-rate intervals. Sources: native_first_decision_steering_v1/summary.json and COURSE_FIRST_DECISION_AMENDMENT.md.');
const diagRows=['concept','random_811','random_812','random_813'].map(d=>{
 const points=diagnostic.points.filter(p=>p.direction===d);
 if(points.length!==4||new Set(points.map(p=>JSON.stringify(p.counts))).size!==1)throw new Error('Do not summarize changing endpoints as constant');
 const p=points[0];return[d.replaceAll('_',' '),`${p.counts.caution}/6`,`${p.counts.legal}/6`,`${p.counts.goal}/6`];
});
table(s,[['Direction','Caution','Legal','Goal'],...diagRows],[430,234,234,234],180,300);
text(s,`Block ${diagnostic.block}; strengths -2, -1, 0, +1, +2; all endpoints unchanged`,74,515,1120,55,23);
text(s,'102 cells, six versions, three templates. Full 15-turn RQ4 remains NOT RUN.',74,580,1120,65,23);
if(notes.reduce((a,n)=>a+n.seconds,0)!==900)throw new Error('The talk timing must sum to 900 seconds');
await fs.mkdir(build,{recursive:true});await fs.mkdir(path.dirname(finalPath),{recursive:true});
await fs.writeFile(path.join(build,'speaker_notes.json'),JSON.stringify(notes,null,2));
for(const [i,s]of deck.slides.items.entries()){
 const blob=await deck.export({slide:s,format:'png',scale:1});await fs.writeFile(path.join(build,`slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await blob.arrayBuffer()));
}
const candidate=path.join(build,'candidate.pptx');await(await PresentationFile.exportPptx(deck)).save(candidate);
const result=await finalizePresentation({workspaceDir:root,candidatePath:candidate,finalPath,pythonExecutable:process.env.ADNN_RUNTIME_PYTHON,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-heading-fit',...[2,3,5,7,8,10,13].flatMap(n=>['--require-native-table-slide',String(n)])],explicitTotalSlideCount:13,
 requiredNativeTableOwnerSlides:[2,3,5,7,8,10,13],requiredNativeChartOwnerSlides:[4,6],materializeLiteralChartWorkbooks:true,
 fontPolicy:{basis:'design',families:[font]},verifyArtifactToolImport:true,receiptPath:path.join(build,'validation.json')});
await fs.writeFile(path.join(build,'result.json'),JSON.stringify(result,null,2));
console.log('Exported 12 talk slides and one untimed backup, with a 900-second talk plan');
