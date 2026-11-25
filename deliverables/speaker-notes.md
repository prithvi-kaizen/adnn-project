# ADNN presentation notes

The 13 main slides total 10:20, including the 36-second recording. Hard stop at 11:00.
Start the closing slide by 9:40. Slides 14–20 are backup answers for the separate three-minute Q&A. Text in brackets is a stage cue.
Open the offline MP4 in advance. If video playback takes more than five seconds, use the replay slide and keep moving.

## 1. Can this deletion be undone?

0:00 to 0:25

Say I ask an AI agent to delete this report. Before it hits delete, can it tell whether I’ll be able to get the file back? And if it can tell, does that change what it does? [Pause for a beat, then click Reveal.]

## 2. The recycling setting changes the answer

0:25 to 0:55

Look at these two versions. Same file, same command. On the left, recycling is on, so I can go to the trash and restore it. On the right, recycling is off and this is the only copy. The word ‘delete’ tells the model exactly as much in both cases. It has to look at the setting.

## 3. The sandbox checks which actions undo it

0:55 to 1:25

We built a small simulator so we could check the answer ourselves. After each action, we search for a way back in up to three more steps. A free, exact restore is reversible. Recovery with a fee is partial. If we finish the search and neither route exists, we call it irreversible within three steps.

## 4. Can a probe read reversibility from the model?

1:25 to 2:10

First, can we read that difference from inside the model? As Qwen reads the state and the command, each layer produces a set of numbers. We train a small classifier, called a probe, to read those numbers and predict the label. We choose the layer on development data and test on separate scenarios. The comparison reads just the words in the command.

## 5. The probe tied the action-text baseline

2:10 to 3:20

[Point to the left chart.] The two bars are the same height in every split. Both methods get fifty percent balanced accuracy, and the probe gets both sides right in zero pairs. The model’s hidden states gave this probe no advantage on the matched test. [Point right.] Reading the last action token did better than averaging the action tokens on a larger binary test. That’s a different set of examples. We also checked pairs-only training, the recovery horizon, and two versus three labels. ScienceWorld transfer is still open. So how does Qwen do if we just ask it?

## 6. Every valid answer said “reversible”

3:20 to 4:15

We showed Qwen the state and the command and asked it to pick one of three labels. It got fifty percent right in the split shown here. The most-common-label baseline got about forty-two percent, and the intervals are wide. What stood out was that every valid reply said ‘reversible.’ On six development cases, we took the example out of the prompt. The replies still said ‘reversible,’ but they stopped using the required JSON format.

## 7. An agent can reach the goal and still fail

4:15 to 5:05

Now we let Qwen act. We give it a goal, let it choose tools, and replay each call to check what actually happened. On these twenty-four development runs, a quarter reached the goal and twelve and a half percent carried out an irreversible action. Every run stopped on invalid calls. So when I say it reached the goal, I mean the simulator state changed as requested. Watch the deletion pair.

## 8. Qwen deleted immediately in both versions

5:05 to 5:55

This is a replay of saved model outputs. Watch the recycling setting and what Qwen does in each case. [Play the 36-second clip.] It deletes both times.

[If playback fails, say: Recycling is off on the second run. Qwen still calls the deletion reversible and goes ahead.]

## 9. Why the zero blindness gap is inconclusive

5:55 to 6:50

The gap asks whether the model gets the labels right more often than it steps up its caution. On these nine pairs, both rates are zero. That gives a zero gap because we saw neither correct paired judgments nor extra caution. [Point right.] Our scripted comparison shows that confirming first left the irreversible action rate unchanged. Always refusing avoided those actions and reached zero goals. We have no completed result for the caution-prompt ablation.

## 10. Tokens changed. Simulator outcomes held.

6:50 to 7:50

For the steering bonus, we changed the model’s activations at one layer. We tried pushing in both directions and compared that with doing nothing and with three equally strong random directions. Look at the flat lines. Goals were reached in a third of versions at every strength. Caution stayed at zero. Some output text changed, but the simulator traces stayed the same as at zero. All of these episodes hit the invalid-call limit.

## 11. Uncertainty ranked at chance after matching

7:50 to 8:45

For the second bonus, we asked Qwen to predict a game’s next state and scored how surprised it was by the text it generated. The dashed line is chance. Across all the games the result is below that line. Once we compare within the same game and action verb, it sits at chance, with a wide interval. None of the eighty-two outputs passed the next-state format check. The score here comes from the generated text.

## 12. Backing up preserved task completion

8:45 to 9:40

We tried one engineering follow-up on those same saved runs. If the simulator says the proposed action can’t be undone, we block it. Stopping there removes the irreversible executions and loses three goals. A planner gets those goals back. In all three blocked cases, it finds the same fix: back up the file, then delete it. The planner can do this because it has access to the exact simulator state.

## 13. The same deletion exposed three different failures

9:40 to 10:20

Back to the file we started with. Qwen called both deletions reversible and carried out both. The probe tied the action-text baseline on the matched test. The recovery check gave us a way to act on the difference in the sandbox. The larger held-out study, ScienceWorld transfer, and the prescribed model comparison are still open. Thanks. Happy to take questions.

## 14. Evidence by topic

Q&A backup

Which part would you like to see in more detail? I can jump to the controls, ablations, statistics, or the demo. [Click the topic and use its return link when finished.]

## 15. What each control rules out

Q&A backup

The action-text baseline asks how far we get with the words alone. The majority baseline guesses the most common training label. With shuffled labels, we test how the probe behaves when the label signal is removed. With fixed random labels by verb, we check how well it can learn an arbitrary verb rule. Layer zero is our starting point before contextual layers. For steering, we compare with random directions of the same size.

## 16. Four sandbox ablations and one open transfer test

Q&A backup

Here’s what we changed. We tried the last action token or the average, all training data or matched pairs only, one to three recovery steps, and two or three labels. The pairs-only and label-count comparisons have different eligible test sets. We also planned transfer in both directions. We couldn’t certify the ScienceWorld states for that comparison.

## 17. Three splits, whole-group uncertainty, paired tests

Q&A backup

We split by scenario so related examples stayed together. For the confidence intervals, we resampled whole groups one thousand times. Only two groups contributed to each of the first two matched splits, so we don’t report an interval there. The pilot McNemar test has no pairs that changed caution in either direction. That gives a p-value of one on these nine pairs.

## 18. Independent checks and a larger-model follow-up

Q&A backup

One blinded model reviewer agreed with the simulator on ninety-eight of one hundred cases. Both disagreements claimed a recovery path that failed when we replayed it. We also tried larger cached Llama and Mistral models on three exposed pairs. Both got two of six labels right. That was a separate quantized development check.

## 19. The original gates still open

Q&A backup

The main gaps are physical-state certification, the original BF16 checkpoint comparison, and sixty independent behavior families. We have twenty-two exposure-screened candidate slots for the family evaluation and no final holdout. For ScienceWorld, matching the visible state did not certify matching the simulator state.

## 20. Layer accuracy and first-decision steering

Q&A backup

On the left, you can see how probe accuracy changes by layer, with the baselines beside it. The marked layer was chosen on development data. On the right is the earlier first-decision steering plot. Those curves overlap. The main steering slide shows what happened when we replayed the full episodes.
