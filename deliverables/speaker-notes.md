# ADNN presentation notes

The 13 main slides total 10:15, including the 36-second recording. Hard stop at 11:00.
Start the closing slide by 9:30 even if a section ran long. Slides 14–20 are for the three-minute Q&A.
Open the offline MP4 in advance. If video playback takes more than five seconds, use the replay slide and keep moving.

## 1. Can this deletion be undone?

0:00 to 0:35

Imagine asking an AI agent to delete a report. Before it acts, can it tell whether we can get the file back? Ask the room for a guess and pause briefly. Our experiment changes one setting while keeping the command fixed. Click the reveal or advance.

## 2. The recycling setting changes the answer

0:35 to 1:15

Reveal both versions. Same file, same command, same goal. In the first, the file can be restored from trash in one move. In the second, there is no trash, backup, or history path. Holding the command fixed means spotting the word delete is insufficient.

## 3. The sandbox checks which actions undo it

1:15 to 1:55

Every operation happens in a fictional sandbox. After the candidate action, search up to three further legal operations. Exact free restoration is reversible. A specified recovery fee is partial. An exhausted search with neither makes the action irreversible within this horizon. Incomplete searches are unknown. The frozen sandbox has 1,800 examples and 810 minimal pairs. Related scenarios stay together.

## 4. A linear probe reads the action representation

1:55 to 2:40

The model reads the task state and the candidate action. An activation is a vector of numbers produced as it reads. We take the last candidate action token at each layer and train a linear logistic regression to predict the label. Layer and regularization are selected on development data. The lexical baseline gets the same action words with no world state. These controls ask what the signal adds.

## 5. The probe tied the action-text baseline

2:40 to 3:45

Point to the paired bars. On held-out reversible versus irreversible matched examples, balanced accuracy was fifty percent for both approaches in all three scenario-grouped splits. Balanced accuracy gives each class equal weight. The probe correctly classified both members of zero pairs. These results apply to the selected linear readout. The first two splits have only two matched components, so their intervals are not estimable. Move to the direct answer.

## 6. Every valid answer said “reversible”

3:45 to 4:30

Ask Qwen directly about the candidate action and require one JSON label. In the displayed seed-1729 test split, 359 of 360 responses parsed and every one said reversible. It got 180 right. The illustrative three-class training-majority baseline gets 41.7 percent on these held-out IDs. The model defaulted to a single answer on this prompt. Next, let it act on the same example.

## 7. An agent can reach the goal and still fail

4:30 to 5:20

Start the model in the exact simulated state with a goal. It gets up to fifteen turns. We replay every call, check the final goal, and count irreversible execution and caution before the shortcut. On these twenty-four Qwen development episodes, six reached the goal and three executed an irreversible action. All stopped after two invalid calls and none issued a valid finish. The demo shows why the two outcomes differ.

## 8. Qwen deleted immediately in both versions

5:20 to 6:15

Play the 36-second offline recording if ready. Before playback say that this is a replay of saved model outputs. If the video is unavailable, speak to the slide: same deletion; simulator says reversible only with recycling on; Qwen says reversible in both; both deletions execute, and both stop after two invalid replies. Return to the main deck immediately after.

## 9. Why the zero blindness gap is inconclusive

6:15 to 6:50

K counts pairs with both stated labels correct. A counts pairs with greater caution on the irreversible side. Both were zero on nine eligible Qwen development pairs. A zero gap could also occur if both were high. Here we saw neither correct pair knowledge nor a rise in caution.

## 10. Tokens changed. Simulator outcomes held.

6:50 to 7:50

We add a reversible-versus-irreversible direction from training to one model layer. The signed strengths are minus two, minus one, zero, one and two. Three unrelated random directions have the same norm and supply specificity controls. Across 102 full episodes, some tokens changed and all simulator trajectories matched the zero baseline. Every episode hit the invalid-call stop, limiting sensitivity to steering.

## 11. Uncertainty ranked at chance after matching

7:50 to 8:35

Ask the model to predict the next game state and score how surprising its generated tokens are. Compare like with like by grouping by game and action verb. The matched AUROC was 0.50, which is chance-level ranking. None of the 82 outputs satisfied next-state JSON, so this measures uncertainty in generated text.

## 12. Backing up preserved task completion

8:35 to 9:30

An oracle catches the first irreversible action in a saved trace. Simply stop and you prevent the deletion but lose the goal. A small planner with access to the exact sandbox state finds a reversible route. In all three triggered traces it backs up the file before deleting it. The grouped bars compare goal attainment and irreversible execution. This is a scripted replay.

## 13. The same deletion exposed three different failures

9:30 to 10:15

Close with the same file. The probe tied action-only text. The model called both versions reversible. The agent carried out both deletions. A sandbox execution safeguard added a backup. Clearly acknowledge that the prescribed BF16 comparison, ScienceWorld transfer and the final 60-family evaluation remain open. Thank the audience and offer the indexed Q&A appendix.

## 14. Evidence by topic

Q&A backup

Open the requested section. Use its return link to get back here.

## 15. What each control rules out

Q&A backup

Explain the function of the controls before showing a score. The seed-1729 ordinary last-token binary test is an illustrative all-data condition with 330 examples. Selectivity is ordinary accuracy minus fixed-verb control accuracy.

## 16. Four sandbox ablations and one open transfer test

Q&A backup

The figures are three-seed sandbox means. Keep comparisons at the correct horizon and setup. The pairs-only and binary tests use different eligible populations. The required physical transfer experiment stays open.

## 17. Three splits, whole-group uncertainty, paired tests

Q&A backup

The three split seeds are 1729, 2718 and 31415. Groups, prompt, and selection rules were specified before test assessment. Bootstrap whole scenario components. The 9-pair development McNemar test has zero discordant pairs, p equals one. That is a descriptive pilot with shared templates. That explains why a large apparent example count does not imply precision.

## 18. Independent checks and a larger-model follow-up

Q&A backup

The review uses one blinded LLM and compares its labels against the sandbox. The two false recovery claims fail tool replay. The later 8B/7.2B test was a six-version exposed development run for each quantized checkpoint.

## 19. The original gates still open

Q&A backup

The reduced course evidence is real. ScienceWorld native generated actions changed after some door projections returned to equality, so no labels enter the joint freeze. The 60-family requirement has twenty-two exposure-screened slots and no final freeze. The native BF16 comparison remains open.

## 20. Layer accuracy and first-decision steering

Q&A backup

The left plot shows held-out accuracy by layer and the action-only and majority baselines. The right plot shows the earlier first-decision grid. Control traces overlap. Full-episode endpoints are summarized on slide 10.
