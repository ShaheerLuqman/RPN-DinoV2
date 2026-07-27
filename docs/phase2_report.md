# Phase 2 Test — Can the Computer Name Each Object Automatically?

## The question in plain terms

Our annotation pipeline has two jobs:

1. **Find** each object in a frame and draw a box around it.
2. **Name** each box (which of our 27 classes is it — a bin, a hand, a fixture…).

This test is only about **job 2 — naming**. To keep it fair, we gave the computer
*perfect boxes* (the human-drawn ones) so any mistake is purely a naming mistake,
not a box-drawing mistake.

**The idea being tested:** show the computer a small set of labeled examples, then
for every new box let it say *"this looks most like my examples of class X."* No
model training required — it just compares pictures by visual similarity.

**The short answer:** It works very well for clearly different objects (bins,
fixtures, parts — right ~90–99% of the time). It struggles with the look-alike
"a hand doing something" classes, which are hard to tell apart from a single
cropped picture.

---

## How the test was set up

- **Examples the computer learns from:** 419 frames spread evenly across the whole
  recording, chosen so **every class has at least 20 example frames** (the rare
  ones needed a boost — even spacing alone left some with only 4–5).
- **Frames it was tested on:** 1,000 different frames it had never seen (≈11,700
  objects).
- **What "similar" means:** each cropped object is turned into a list of numbers
  that captures its appearance (using a standard vision model, DINOv2). Two objects
  are "similar" if their numbers are close.

We tried **three ways** of comparing a new object to the examples. They differ only
in how much detail they keep about each class:

| Way of comparing | In plain words |
|---|---|
| **One-average** | Blend all examples of a class into a single "typical look," compare to that. |
| **Few-looks** | Allow a class to have a few distinct looks (e.g. seen from 2 camera angles) and compare to the closest one. |
| **All-examples** | Keep every example and let the 5 most-similar examples vote on the answer. |

---

## The headline result

| Way of comparing | Overall accuracy | Fair-average accuracy* |
|---|---:|---:|
| One-average | 73% | 71% |
| Few-looks | 82% | 71% |
| **All-examples (vote)** | **94%** | 70% |

\* **Overall accuracy** counts every object equally, so it is dominated by the few
very common classes. **Fair-average accuracy** counts every *class* equally, so one
rare class matters as much as one huge class.

**Why the two numbers differ so much matters more than the numbers themselves.**
Just three classes (`black_bin`, `hands`, `screw_bin`) make up about 70% of all
objects. The "All-examples" method gets those three almost perfectly, which pushes
overall accuracy to a shiny 94%. But when we give every class equal weight, the
score drops to 70% — because the rare classes are doing much worse.

> **Takeaway:** 94% does **not** mean "naming is basically solved." It means
> "naming is solved for the common, visually-distinct classes." The rare, look-alike
> classes are where the real work remains.

---

## What works and what doesn't

### ✅ Works well — reliable enough to auto-label today
Clearly different physical objects:
`black_bin` (99%), `parts_bin` (99%), `screw_bin` (96%), `hands` (98%),
`r_fixture1/2` (97–99%), `rotated_part_on_white_bin` (95%),
`right_part_on_white_bin` (93%).

These are distinct shapes that look the same every time — easy to recognize.

### ❌ Struggles — needs human review
The "hand doing a specific task" classes:
`hand_w_fixture` (3%), `r_hand_w_allen_wrench` (9%), `r_hand_w_part` (24%),
`r_hand_w_complete_part` (40%), and the `f1/f2 … hand_w_screwdriver` group (mixed,
often 40–70%).

**Why they're hard:** these classes are not really *different objects* — they are
the *same hand* doing subtly different things ("which corner of which fixture,
holding which tool"). From a single cropped picture, the computer just sees "a hand
with a screwdriver" and can't tell the fine variations apart. Giving it more
examples did **not** fix this — the limitation is that these distinctions barely
show up in a still crop at all. They likely need extra information (video/motion, or
where the hand is on the workstation), not a better comparison method.

---

## Accuracy for every class

Below is how often each class was named correctly, under each of the three
comparison methods. **"Best"** is the highest of the three — the accuracy we could
reach for that class if we picked the right method for it. Status:
✅ reliable (≥90%), 🟡 usable with review (70–89%), 🔴 weak (<70%).

| Class | One-average | Few-looks | All-examples | **Best** | Status |
|---|---:|---:|---:|---:|:--:|
| r_fixture2 | 99% | 100% | 100% | **100%** | ✅ |
| install_gaskit_pose | 100% | 100% | 90% | **100%** | ✅ |
| black_bin | 76% | 99% | 99% | **99%** | ✅ |
| parts_bin | 98% | 98% | 99% | **99%** | ✅ |
| hands | 49% | 53% | 98% | **98%** | ✅ |
| r_fixture1 | 91% | 94% | 97% | **97%** | ✅ |
| screw_bin | 87% | 83% | 96% | **96%** | ✅ |
| fixture_3 | 90% | 95% | 88% | **95%** | ✅ |
| rotated_part_on_white_bin | 70% | 93% | 95% | **95%** | ✅ |
| right_part_on_white_bin | 87% | 69% | 93% | **93%** | ✅ |
| empty_white_box | 91% | 93% | 87% | **93%** | ✅ |
| left_hand_w_screwdriver | 86% | 88% | 69% | **88%** | 🟡 |
| right_hand_w_screwdriver | 66% | 73% | 88% | **88%** | 🟡 |
| f1_FR_hand_w_screwdriver | 75% | 77% | 77% | **77%** | 🟡 |
| hand_w_fixture | 76% | 54% | 3% | **76%** | 🟡 |
| f1_FL_hand_w_screwdriver | 64% | 64% | 74% | **74%** | 🟡 |
| r_hand_w_complete_part | 70% | 35% | 40% | **70%** | 🟡 |
| f1_BL_hand_w_screwdriver | 65% | 67% | 65% | **67%** | 🔴 |
| f2_BL_hand_w_screwdriver | 61% | 66% | 58% | **66%** | 🔴 |
| f1_BR_hand_w_screwdriver | 54% | 61% | 65% | **65%** | 🔴 |
| f2_BR_hand_w_screwdriver | 65% | 46% | 54% | **65%** | 🔴 |
| f2_FL_hand_w_screwdriver | 55% | 60% | 45% | **60%** | 🔴 |
| r_hand_w_part | 57% | 57% | 24% | **57%** | 🔴 |
| r_hand_w_allen_wrench | 56% | 41% | 9% | **56%** | 🔴 |
| f2_FR_hand_w_screwdriver | 39% | 51% | 46% | **51%** | 🔴 |
| completed_part_on_white_bin | 14% | 42% | 48% | **48%** | 🔴 |

**Reading the table:**
- **11 classes are reliable** (green) — all of them are physical objects (bins,
  fixtures, parts) or plain hands. These can be auto-labeled with light checking.
- **The weak classes (red) are almost all the "hand-with-a-tool-at-a-position"
  group** — the same hand in slightly different situations, which a single crop
  can't distinguish.
- **No single method wins everywhere.** For common objects, *All-examples* is best;
  for several rare hand classes, *One-average* or *Few-looks* is far better (e.g.
  `hand_w_fixture` 76% vs 3%, `r_hand_w_allen_wrench` 56% vs 9%). That is why the
  "Best" column is higher than any single method's overall score — and why a future
  system should combine methods rather than pick just one.

---

## The practical payoff: a confidence dial

The "All-examples" method also reports **how sure** it is about each answer. If we
only auto-accept the confident ones and send the rest to a human:

| If we only keep answers the computer is sure about… | …we auto-label this share of boxes | …at this accuracy |
|---|---:|---:|
| keep everything | 100% | 94% |
| fairly sure | 95% | 97% |
| quite sure | 91% | 98% |
| very sure | **89%** | **98%** |

**In practice:** the computer can confidently name about **9 out of 10 boxes at ~98%
accuracy on its own**, and hand the uncertain 1 in 10 to a person to check. That is
already a large labor saving.

---

## How many frames should we annotate?

We guaranteed a minimum number of example frames per class and swept that floor from
20 all the way up to *every available frame*, measuring how naming accuracy responds.
(Because each frame contains many objects, common classes always keep their full,
large example set — only the rare classes are limited by how much we annotate.)

| Min frames per class | Frames to annotate | Overall accuracy | Fair-average accuracy |
|---:|---:|---:|---:|
| 20 | 419 | 94.0% | 69.5% |
| 30 | 559 | 94.9% | 75.5% |
| 40 | 699 | 94.9% | 75.8% |
| 50 | 838 | 95.5% | 78.7% |
| 75 | 1,184 | 96.1% | 82.4% |
| 100 | 1,522 | 96.2% | 82.2% |
| 150 | 2,172 | **97.0%** | **86.6%** |
| all | 6,748 | 97.8% | 89.1% |

**What this says:**
- **Overall accuracy barely moves** (94 → 98%). The common classes are already solved
  at 20 frames, so this number does not tell you how much to annotate.
- **Fair-average accuracy is the metric that responds** — it rises steadily with
  annotation because all the value goes into the rare, harder classes.
- **Returns diminish sharply.** The jump from 419 → 2,172 frames buys +17 fair-average
  points; annotating *every other frame* after that (another ~4,600 frames, to reach
  6,748) adds only **+2.5 more**. There is no reason to annotate 4,000–5,000 frames.
- **Sweet spot ≈ 800–1,200 frames** (min-50 to min-75): 96% overall / ~79–82%
  fair-average. Push to ~2,170 frames (min-150) only if lifting the rare classes to
  ~87% is worth roughly 1,000 extra frames.

### Per-class accuracy for every run

Below is the naming accuracy of **every class** under each annotation budget, using the
**All-examples (vote)** method — the same method the headline and confidence-dial
numbers are based on. Classes are ordered from strongest to weakest at the 50-frame
budget. The rare classes near the bottom have very few test examples (often only
15–50 boxes), so a few frames' difference swings their percentage — read those rows as
trends, not precise figures.

| Class | 20 | 30 | 40 | 50 |
|---|---:|---:|---:|---:|
| black_bin | 99% | 99% | 99% | 100% |
| install_gaskit_pose | 90% | 86% | 89% | 100% |
| r_fixture2 | 100% | 100% | 99% | 99% |
| parts_bin | 99% | 99% | 99% | 99% |
| screw_bin | 96% | 96% | 96% | 97% |
| hands | 98% | 97% | 98% | 97% |
| f1_FR_hand_w_screwdriver | 77% | 88% | 93% | 97% |
| r_fixture1 | 97% | 98% | 98% | 97% |
| rotated_part_on_white_bin | 95% | 93% | 96% | 95% |
| right_part_on_white_bin | 93% | 94% | 95% | 95% |
| empty_white_box | 87% | 94% | 90% | 94% |
| fixture_3 | 88% | 94% | 97% | 92% |
| right_hand_w_screwdriver | 88% | 91% | 90% | 92% |
| f1_BR_hand_w_screwdriver | 65% | 75% | 77% | 82% |
| f2_BL_hand_w_screwdriver | 58% | 64% | 70% | 80% |
| f1_FL_hand_w_screwdriver | 74% | 80% | 83% | 78% |
| f2_FL_hand_w_screwdriver | 45% | 64% | 78% | 73% |
| left_hand_w_screwdriver | 69% | 71% | 72% | 75% |
| f1_BL_hand_w_screwdriver | 65% | 76% | 78% | 72% |
| r_hand_w_complete_part | 40% | 50% | 28% | 67% |
| f2_FR_hand_w_screwdriver | 46% | 60% | 68% | 65% |
| f2_BR_hand_w_screwdriver | 54% | 70% | 46% | 60% |
| completed_part_on_white_bin | 48% | 53% | 46% | 51% |
| r_hand_w_allen_wrench | 9% | 22% | 45% | 37% |
| r_hand_w_part | 24% | 38% | 31% | 38% |
| hand_w_fixture | 3% | 10% | 12% | 16% |

*(Full per-method numbers for each run are in [phase2_results/](../phase2_results/):
`report_{centroid,knn,cluster}.txt` at the top level is the 20-frame run, and the
`min30/`, `min40/`, `min50/` subfolders hold the larger budgets.)*

The hard classes split into two groups — and only one is worth annotating more:

| Class | 20 | 30 | 40 | 50 | What it means |
|---|---:|---:|---:|---:|---|
| f2_FL_hand_w_screwdriver | 45% | 64% | 78% | 73% | **Was short on examples — more frames fix it** |
| f2_FR_hand_w_screwdriver | 46% | 60% | 68% | 65% | **Short on examples — big gains** |
| f1_BR_hand_w_screwdriver | 65% | 75% | 77% | 82% | **Short on examples — steady gains** |
| hand_w_fixture | 3% | 10% | 12% | 16% | **Genuinely hard — frames barely help** |
| completed_part_on_white_bin | 48% | 53% | 46% | 51% | **Genuinely hard — flat** |

1. **Under-supplied classes** (the "hand-with-screwdriver-at-a-position" group)
   improve a lot with more examples — they were simply starved at 20 frames.
2. **Genuinely hard classes** stay low no matter how many frames we add — for these
   the picture alone doesn't carry the answer, so more annotation is wasted effort.
   They need a different kind of information (motion/video, or where the hand is on
   the bench) or a stronger vision model.

### Recommendation
- **Annotate ~50–75 frames per class (≈840–1,180 frames total)** for the best
  value: 96% overall and ~79–82% fair-average, at a fraction of a full annotation
  pass.
- **Go to ~150 per class (≈2,170 frames) only if the rare classes matter** — that
  lifts fair-average to ~87%. Beyond that the curve is flat: labeling the remaining
  ~4,600 frames adds only ~2 points, so **4,000–5,000 frames is not worth it.**
- **Stop adding frames for the genuinely-hard classes** once they plateau — spend
  that effort elsewhere or solve them a different way.

---

## What if we skip balancing and just annotate N frames?

The budget numbers above use a per-class floor (top up rare classes until each is
covered). Here we test the simplest possible policy instead: **annotate N
evenly-spaced frames across the recording, with no per-class balancing at all**, and
see how per-class accuracy fills in as N grows. Same vote (All-examples) method, same
fixed 1,000-frame test set for every N.

| Class | 100 | 200 | 300 | 500 | 750 | 1000 | 1250 | 1500 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| black_bin | 99 | 99 | 99 | 99 | 100 | 100 | 100 | 100 |
| install_gaskit_pose | 0 | 56 | 100 | 100 | 100 | 100 | 100 | 100 |
| parts_bin | 98 | 98 | 98 | 99 | 99 | 99 | 100 | 99 |
| r_fixture2 | 99 | 99 | 99 | 99 | 99 | 100 | 100 | 99 |
| screw_bin | 90 | 93 | 95 | 97 | 97 | 98 | 98 | 99 |
| empty_white_box | 83 | 93 | 92 | 92 | 88 | 94 | 96 | 99 |
| r_fixture1 | 94 | 96 | 96 | 98 | 98 | 97 | 99 | 98 |
| hands | 96 | 95 | 97 | 97 | 97 | 98 | 97 | 97 |
| right_part_on_white_bin | 89 | 94 | 92 | 93 | 95 | 95 | 94 | 97 |
| rotated_part_on_white_bin | 94 | 95 | 95 | 94 | 95 | 97 | 97 | 96 |
| f1_FR_hand_w_screwdriver | 39 | 85 | 83 | 76 | 90 | 93 | 98 | 93 |
| right_hand_w_screwdriver | 86 | 88 | 86 | 91 | 90 | 93 | 94 | 92 |
| fixture_3 | 61 | 33 | 85 | 89 | 89 | 91 | 87 | 91 |
| f2_FL_hand_w_screwdriver | 22 | 50 | 45 | 50 | 72 | 78 | 72 | 88 |
| f1_BL_hand_w_screwdriver | 49 | 13 | 51 | 71 | 82 | 84 | 84 | 84 |
| f1_BR_hand_w_screwdriver | 86 | 77 | 34 | 86 | 84 | 84 | 82 | 84 |
| f1_FL_hand_w_screwdriver | 35 | 54 | 71 | 73 | 81 | 90 | 85 | 83 |
| f2_BR_hand_w_screwdriver | 56 | 29 | 56 | 66 | 71 | 76 | 71 | 83 |
| left_hand_w_screwdriver | 49 | 73 | 70 | 79 | 75 | 79 | 80 | 82 |
| f2_BL_hand_w_screwdriver | 40 | 70 | 72 | 65 | 81 | 86 | 74 | 81 |
| completed_part_on_white_bin | 29 | 18 | 35 | 69 | 68 | 77 | 81 | 79 |
| f2_FR_hand_w_screwdriver | 9 | 7 | 24 | 48 | 59 | 67 | 53 | 64 |
| r_hand_w_allen_wrench | 2 | 2 | 5 | 7 | 24 | 34 | 34 | 37 |
| r_hand_w_part | 4 | 13 | 9 | 17 | 39 | 30 | 26 | 26 |
| r_hand_w_complete_part | 11 | 33 | 22 | 17 | 28 | 33 | 39 | 22 |
| hand_w_fixture | 0 | 0 | 5 | 11 | 19 | 16 | 23 | 18 |
| **Overall** | **90.5** | **92.1** | **93.3** | **94.8** | **95.2** | **96.0** | **95.9** | **96.2** |
| **Fair-average** | **54.7** | **60.1** | **66.0** | **72.5** | **77.7** | **80.4** | **79.4** | **80.4** |

**What this shows:**
- **Overall accuracy is already ~90% at 100 frames** and only creeps to 96% — it is
  dominated by the common classes, which are essentially solved from the very first
  budget (`black_bin`, `hands`, `parts_bin`, `r_fixture2` all ~97–100% at 100 frames).
- **Fair-average is the metric that responds**, climbing steeply to **~1,000 frames
  (80%) and then flat** — 1,250 and 1,500 frames add nothing. **~1,000 frames is the
  practical ceiling** for plain even sampling.
- **Without a per-class floor, rare classes start out starved** — at 100 frames an even
  stride barely lands on them, so `install_gaskit_pose` and `hand_w_fixture` are 0%
  and `r_hand_w_allen_wrench`/`r_hand_w_part` are ~2–4%. This is precisely the gap the
  per-class floor was closing: the balanced min-50 run reached **78.7% fair-average at
  838 frames**, whereas plain sampling needs **~1,000 frames** to get there. Balancing
  buys the same fairness for roughly 150–200 fewer annotated frames.
- **The same three or four classes never recover** at any budget —
  `r_hand_w_allen_wrench` (37%), `r_hand_w_part` (26%), `r_hand_w_complete_part` (22%),
  `hand_w_fixture` (18%). These are the genuinely-hard look-alikes, not an
  annotation-volume problem.
- **Mid-tier hand poses are noisy across budgets** (e.g. `f1_BR` 86→77→34→86,
  `fixture_3` 61→33→85) because their test sets are small (30–60 boxes) and even
  sampling hits their few examples unevenly — read those rows as trends. The small dip
  at 1,250 is the same sampling noise, not a real regression.

**Takeaway:** plain even sampling is a fine default — it reaches ~96% overall / ~80%
fair-average by ~1,000 frames and then plateaus — but the per-class floor gets you the
same fairness with fewer frames, and neither approach can rescue the handful of
genuinely-hard classes.

---

## Does balancing the examples per class help?

Even with a per-class *frame* floor, the common classes still pile up far more
*examples* than the rare ones — one selected frame can add a `black_bin`, a
`screw_bin` and two `hands` crops, so after 838 frames `hands`/`black_bin`/`screw_bin`
have **thousands** of examples while a rare hand-pose class has ~50. The vote method
counts examples, so the crowd of common-class examples naturally wins.

So we ran the opposite experiment: **give every class exactly the same number of
examples** — cap each class to precisely *N* crops (evenly spread over time), even
when a selected frame contains other classes we don't need — and swept *N* to find the
best value. Same vote method throughout. The "frames to annotate" column is how many
frames you must label to obtain *N* examples of every class (driven by the rare ones).

| N per class | Frames to annotate | Overall accuracy | Fair-average accuracy |
|---:|---:|---:|---:|
| 20 | 419 | 74.9% | 77.6% |
| 50 | 838 | 78.6% | 83.4% |
| 75 | 1,184 | 81.0% | 87.1% |
| 100 | 1,522 | 83.8% | 88.7% |
| 150 | 2,172 | 84.4% | 89.7% |
| 200 | 2,791 | 85.9% | 91.3% |
| 300 | 4,011 | 87.7% | 92.6% |

**Reading the sweep:**

- **Fair-average climbs fast, then flattens.** The knee is around **N ≈ 75–100**:
  going 20 → 75 adds +9.5 points, 75 → 100 adds +1.6, and every step after that is
  ~1 point for a lot more annotation. **N ≈ 75–100 is the optimal balanced budget.**
- **Overall accuracy stays low (~81–84%)** — because forcing every class to just *N*
  examples also *starves the common classes* (`hands`, `black_bin`, `screw_bin`),
  which genuinely need their large, varied example sets. This is why exact-N is a poor
  setting for raw throughput even though it's great for the rare classes.
- **The rare classes hit a ceiling.** Several classes simply don't have *N* examples
  in the whole recording (`install_gaskit_pose` has 79, `r_hand_w_complete_part` 148,
  `r_hand_w_part` 214), so past N ≈ 150 you can't actually give them more — the extra
  frames only feed the mid- and high-frequency classes.

### Where balancing helps vs. hurts (vote method)

At the optimal N ≈ 100, balancing rescues the rare "hand-in-action" classes that the
common-class crowd had been swallowing — several dramatically (unbalanced = keep every
example; balanced = exactly 100/class):

| Class | Unbalanced | Balanced (100/class) |
|---|---:|---:|
| hand_w_fixture | 16% | **91%** |
| completed_part_on_white_bin | 51% | **97%** |
| left_hand_w_screwdriver | 75% | **96%** |
| f2_FL_hand_w_screwdriver | 73% | **95%** |
| r_hand_w_allen_wrench | 37% | **83%** |
| r_hand_w_complete_part | 67% | **78%** |

…but it hurts the very common classes, which lose their rich example sets:

| Class | Unbalanced | Balanced (100/class) |
|---|---:|---:|
| hands | 98% | **52%** |
| right_hand_w_screwdriver | 98% | **67%** |
| black_bin | 100% | **93%** |
| screw_bin | 99% | **96%** |

**Takeaway:** exact-N balancing is **not** the right default for auto-labeling — it
throws away the easy, free wins on the common classes and drags overall accuracy down.
But it is the **best setting for the rare classes**, and the optimal balanced budget is
**N ≈ 75–100 examples per class (~1,200–1,500 frames)**. The practical answer is a
**hybrid**: keep the full example set for the common classes, and additionally cap the
common classes only when scoring the rare group — plus a confidence gate — rather than
using one example budget for everyone.

*(Sweep uses the full example pool capped per class; run with `--ref_all
--samples_per_class N`. Frame counts from the reference-selection top-up logic.)*

---

## Does masking out the background help?

Every crop we compare is a **rectangle** — so it always contains some background
around the object (shelf, bench, the operator's sleeve). Does that background help
the computer (useful context) or hurt it (distraction)? We tested this by using
**SAM2** to trace the exact object outline inside each box and **blanking everything
outside it** (setting the background to black) before measuring appearance. The
object crops are otherwise identical; only the surroundings change. Reference and
test crops were masked the same way so the comparison stays fair.

We ran it on two datasets to see where masking pays off:

| Dataset | | Overall accuracy | Fair-average accuracy |
|---|---|---:|---:|
| **4079 End-Stop** (6 classes, tight boxes) | raw crop | 99.5% | 98.97% |
| | background masked | 99.2% | 98.95% |
| **gas_valve** (27 classes, hands & tools) | raw crop | 96.6% | 80.7% |
| | **background masked** | 96.2% | **83.0%** |

*(Naming-only test, same as the rest of this report: human-drawn boxes so any change
is purely a naming effect. gas_valve reference = 1,500 annotated frames; the raw-crop
row matches the 1,500-frame result in the sampling table above.)*

**What this shows — it depends entirely on the dataset:**

- **On 4079, masking does nothing** (a one-crop difference either way). The boxes are
  tight, the objects fill them, and naming is already near-perfect — there is no
  background problem to fix and no headroom to gain.
- **On gas_valve, masking helps the classes that matter most.** Overall accuracy dips
  a trivial 0.4 point, but **fair-average rises +2.3 points (80.7 → 83.0)**. The two
  numbers move in opposite directions on purpose: *overall* is dominated by the common,
  easy classes (bins, hands), which lose a hair; *fair-average* weights every class
  equally, so it reflects the **rare, look-alike "hand-with-a-tool" classes** — and
  those improve, because a cluttered, ever-changing background was leaking into their
  appearance and pulling the vote toward the wrong neighbor. Blanking it out lets the
  object itself carry the decision.

**Takeaway:** background masking is **worth enabling when classes are many, confusable,
and background-heavy** (like gas_valve) and you care about per-class accuracy — it lifts
the hard tail at negligible cost to the easy classes. It is **not worth it when boxes
are already tight and naming is near-ceiling** (like 4079): no gain, and it roughly
doubles compute (a SAM2 pass per frame). It is off by default and enabled per-dataset
with `phase2.mask: true` (writes to a separate `*_mask` run so baselines are untouched).

---

## Bottom line

1. **Automatic naming is ready** for the clearly-different objects (bins, parts,
   fixtures) — auto-label them and spot-check.
2. **It is not ready** for the look-alike "hand-in-action" classes. These aren't
   failing for lack of examples; a single cropped image simply doesn't contain
   enough to tell them apart.
3. **A confidence cutoff** lets us safely auto-label ~90% of boxes now and route the
   hard 10% to human review.

### Sensible next steps
- Show a picture-grid of the mix-ups so the pattern is visible at a glance.
- Score the "objects" and the "hand-actions" separately (the objects alone should
  look excellent).
- Try a newer/stronger vision model to see if the look-alike classes improve.
- Combine the strengths of the different comparison methods for the rare classes.

---

*Test details for reference: dataset `gas_valve_2view` (7,748 frames, 27 classes);
appearance model `facebook/dinov2-base`; boxes were the human-drawn ground truth so
this measures naming only. Full numbers and the script are in
[phase2_results/](../phase2_results/) and [phase2_eval.py](../scripts/phase2_eval.py).*
