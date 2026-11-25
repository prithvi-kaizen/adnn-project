# ADNN presentation notes

The 13 main slides total 10:20, including the 36-second recording. Hard stop at 11:00.
Start the closing slide by 9:40 even if a section ran long. Slides 14–20 are for the three-minute Q&A.
Open the offline MP4 in advance. If video playback takes more than five seconds, use the replay slide and keep moving.

## 1. Can this deletion be undone?

0:00 to 0:25

Imagine asking an AI agent to delete a report. Before it acts, can it tell whether we can retrieve the file? And if it can tell, does that change what it actually does? Take a quick guess, then click to reveal the two cases.

## 2. The recycling setting changes the answer

0:25 to 0:55

The requested deletion is identical. With recycling on, the file goes to trash and can be restored. With recycling off, the same command removes the only copy. So the problem is whether the model can use the state of the world to tell those cases apart.

## 3. The sandbox checks which actions undo it

0:55 to 1:25

We run all actions in a simulator. After the candidate action, we check for a legal route back within three further steps. Free and exact recovery means reversible. Recovery for a specified fee means partially reversible. Exhausting the search with neither makes it irreversible at that limit. The frozen sandbox has 1,800 examples, including 810 matched pairs.

## 4. Can a probe read reversibility from the model?

1:25 to 2:10

RQ1 asks whether we can read reversibility from the model’s hidden states. We give Qwen the state and candidate action, and train a small linear classifier on its internal numbers. We fit the classifier using the training scenarios and select the layer on development scenarios. We then test on separate, grouped scenarios. Think of the probe as a reader of the model, separate from the answer the model produces.

## 5. The probe tied the action-text baseline

2:10 to 3:20

On matched cases, the probe and the action-text baseline both score fifty percent in all three splits. The probe gets both members right in zero pairs. The baseline sees only the command, so this tie matters. We also tested a majority predictor, shuffled training labels, random labels fixed by verb, and the embedding layer. On the larger binary test, using the last action token gave a higher mean score than averaging the tokens. That result uses a different test population. We also varied pair-only training, the recovery horizon, and the number of classes. Two-way ScienceWorld transfer remains open. Now let’s ask the model directly.

## 6. Every valid answer said “reversible”

3:20 to 4:15

RQ2 asks whether Qwen can state the right label. We give it the state and action and require one of three labels in JSON. The left plot shows fifty percent accuracy against a 41.7 percent majority baseline, with wide intervals. All 359 valid replies in that split said reversible. In a separate six-case ablation, removing the labeled example dropped JSON compliance from one hundred percent to zero. The literal answer was still reversible in both formats. That test shows an effect on formatting. Next we let Qwen use tools.

## 7. An agent can reach the goal and still fail

4:15 to 5:05

RQ3 asks whether the agent is more careful when undoing is impossible. We give it a goal and let it choose tools on both versions. We replay each call for caution, irreversible execution, and goal completion. A quarter of the 24 development episodes reached the goal. Twelve and a half percent executed an irreversible action. Every run hit the invalid-call stop. The demo makes it easier to see how reaching the goal and finishing correctly are different.

## 8. Qwen deleted immediately in both versions

5:05 to 5:55

This recording replays saved model outputs. Watch what changes when recycling is off, and what Qwen does in response. [Play the 36-second clip.] It deletes in both cases.

## 9. Why the zero blindness gap is inconclusive

5:55 to 6:50

The blindness gap is K minus A. K is the share of pairs where both stated labels are correct. A is the share where the agent is more cautious on the irreversible side. Both were zero in nine exposed pairs, so this pilot gives a zero gap. The scripted baseline confirms that asking permission can leave the same irreversible actions in place. Always refusing avoids those actions but reaches no goals. The planned neutral versus caution prompt comparison has no completed agent result. Next we test a direct intervention inside the model.

## 10. Tokens changed. Simulator outcomes held.

6:50 to 7:50

RQ4 asks whether steering the model changes caution or goal attainment. We add a direction learned from training to a single layer. We compare the learned direction with zero steering and three random directions of the same size. We vary the intervention from negative through positive strengths. The curves overlap. Goal attainment stays at one third and caution stays at zero across conditions. Some tokens change, but no simulator trajectory changes from zero. Every episode hit the invalid-call limit, so this result describes the development setup we could test.

## 11. Uncertainty ranked at chance after matching

7:50 to 8:45

RQ5 asks whether prediction uncertainty can flag irreversible actions. We ask Qwen to predict the next game state and score how surprising its output tokens are. The dashed line is the chance baseline. Across all games, uncertainty ranked irreversible examples lower on average. When we compare the same game and action verb, AUROC rises to fifty percent, right at chance, with a wide interval. This sensitivity check matters because the game and action can influence the score. None of the 82 outputs was valid next-state JSON. We measured uncertainty in generated text.

## 12. Backing up preserved task completion

8:45 to 9:40

For the engineering follow-up, we replay the same 24 traces with a check that can detect an irreversible action before it runs. Blocking and stopping removes all three irreversible executions but sacrifices three goals. Blocking and planning also removes those executions and restores goal attainment to twenty-five percent. In each blocked trace, the planner backs up the file, then deletes. These are replay results from a planner with exact simulator state.

## 13. The same deletion exposed three different failures

9:40 to 10:20

Take the file example back with you. The probe did no better than the action words. Qwen said reversible in both cases and went ahead in both. A check at execution time found a safer route. This is a sandbox and development study. ScienceWorld transfer, the prescribed BF16 comparison, and the independent 60-family evaluation are still open. We can jump to the controls, ablations or statistics for questions.

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
