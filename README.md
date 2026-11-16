# TakeMeter

> ### 👋 Start here
>
> **New to this repo? Read [RUNNING.md](RUNNING.md) first** — setup, the
> notebook, the baseline, and what to do when something breaks.
>
> Once `python test.py` passes:
>
> ```bash
> head -5 data/practice_labels.csv     # the shape your labels.csv needs
> ```
>
> Then open `takemeter.ipynb` **in this folder** — in VS Code, or with
> `jupyter notebook` if you prefer. Pick the kernel: the `.venv` inside this
> project. Run section 1, which reports the hardware you'll be training on.
> Everything else waits until you have data.
>
> Nothing to upload, nothing to connect, no accounts and no keys. The notebook
> runs on your machine and writes next to your code.
>
> **The rest of this file is your submission.** Fill it in as you go.

---

<!-- ─────────────────────────────────────────────────────────────────────────
     HOW TO USE THIS FILE

     Unit 5 asks for the first five sections. Unit 6 adds the five below them.

     Everything is pasted as TEXT. No screenshots, no images.

     ⚠️ The confusion matrix especially. The notebook prints one as a markdown
     table, ready to copy. A screenshot of a matrix earns nothing. Paste the
     table.
     ───────────────────────────────────────────────────────────────────────── -->

<!-- ═══════════════════════ UNIT 5 — THE BUILD ═══════════════════════ -->

## What This Does

<!-- Your community, and what your classifier sorts posts into. Three or four
     sentences. -->
     I built a text classifier for posts from the Discuss Cooking forum. It sorts food-related posts into food_review, food_story, or recommendation. A review evaluates a specific food or meal, a story recounts a food-related experience or memory, and a recommendation suggests what to try or where to eat.

device: cpu  (CPU)
torch:  2.14.1+cpu

---

## Label Taxonomy

<!-- Each label: a one-sentence definition and two real examples from your
     reading. Then your decision rule for the hardest boundary.

     The decision rule is worth a point on its own and it's the thing most
     people leave out. Every taxonomy has a hardest boundary. Name yours. -->

### `label_one` food_review

**Definition:** evaluates a specific food or meal, usually with reasons

**Example 1:** 
> One thing I've found here is that the food is very reasonably priced. Lunch, for many, is the main meal of the day. In fact, the word for lunch is la comida, which simply means "the meal." Our lunch yesterday was $36 per person for a three course meal, which included a bottle of wine and coffee. By the way, I am indulging in a few carbs on this trip. :smile:

**Example 2:**
> Pizza for dinner. I changed my strategy. I bought a basic cheese frozen pizza, and my own toppings, and I added some herbs and EVOO, and some parmigiano-reggiano. It was good, and about seven or eight bucks total?

### `label_two` food_story

**Definition:** recounts a meal or food-related memory without mainly reviewing or recommending it

**Example 1:**
> Wish I had taken a picture of our dinner last night. We each had a large toasted fresh Bolillo roll (think Mexican hot dog bun on steroids) containing two browned and beer simmered Johnson Brats, topped with brown mustard and kraut. Now that's a hot dog worth eating!! :yum: *burp*...scuze me. :wink: Bea, I sure wish we could get those white Snappy dogs..I know I'd like them!!

**Example 2:**
> I guess I have to add that my truest favorite is any wiener impaled on a willow twig that I cut myself along a creek, then cooked over a campfire. We had family weeny roasts like that often when I was a kid - do your hot dog, then use the same stick for a couple of marshmallows for s'mores. Great memories! :yum:

### `label_three` recommendation

**Definition:** tells others what to try or where to eat

**Example 1:**
> We had garlic baked butter chicken thighs, which I've prepared before and we really liked them. It's so easy and if you like garlic, this is a good 'un for you. Had some herbed and buttered noodles on the side and confetti cupcakes for dessert. Even enough left for our meal tomorrow. Good thing, too, 'cause Mother Nature is supposed to start a really bad COLD hissy fit tomorrow. I didn't have boneless skinless thighs this time, but that didn't seem to matter. Here's a link to the recipe if anyone is interested:

**Example 2:**
> Giordano's, Gino's East, and uno’s are all close to he cubs stadium off lakeshore drive. Lou malnati’s, I heard is very good but have never eaten there yet and not sure where it’s located near.

### The hardest boundary

**Which two labels:** food_review and recommendation

**The decision rule I used every time:**
<!-- e.g. "If the post names a specific checkable fact, it's `analysis`, even
     if the tone is heated." -->
If the post mainly evaluates a specific food or meal the writer had, I label it food_review. If its main purpose is to tell readers what to try or where to eat, I label it recommendation, even when it includes praise or reasons. If both fit, I choose based on the main takeaway for the reader. That makes the distinction about the post’s purpose, rather than whether it says something positive or negative.

---

## The Dataset

<!-- Where you collected from, how you labelled, your counts, and three hard
     cases. -->

**Where the posts came from:** The posts came from a cooking forum called Discuss Cooking https://www.discusscooking.com/ 

**How I labelled them:** <!-- Cold first? Pre-labelled with AI and corrected?
Say so plainly — the disclosure is required, not penalised. -->

I labeled the post by cold first doing about 60 of them, and then used the pre-labelled with AI to do the rest.

**Counts per label:**

| Label | Count | Share |
|---|---|---|
|Cold| 67 | 33% |
|Pre-labelled| 135 | 67% |
| **Total** |  | 100% |

**Three hard cases**

<!-- Any post that made you pause: what it was, which two labels it could have
     been, and what you chose. These are worth more than the easy 190. -->

**1.**
> *The post:*
> "We use them for different applications but we do make a lot of frangipane. There's a big difference, they're richer and creamier, softer more macadamia nut like. We use Cali almonds as well if it's in a recipe where it's not dominant but for plates where they are we only use Marcona. They aren't cheap, so be warned. I'm surprised your supplier, forget the name doesn't have them, it seems they would be right up their alley."
> *Could have been:*
> Food_review
> *I chose, because:* Recommendation, since they phrase their wording in a way that they want others to try these ingredients instead of the usual.

**2.**
> *The post:*
> "I also store nuts in the freezer if i am not using them fast enough."
> *Could have been:*
> recommendation
> *I chose, because:* Food_review since it gives more of a reasoning instead of why they store it in the freezer.

**3.**
> *The post:*
> "Cheese and onion sandwiches have been in my rotation from childhood. I don't shred the cheese or add mayo. I slice both the cheese, which is generally a cheddar and the onion, normally on toasted buttered sourdough. i'll add a fried egg if it's for breakfast."
> *Could have been:*
> recommendation
> *I chose, because:* Food_story since it starts with a memory of when they would eat the food and not have exact purpose of recommending it or critiquing it

---

## The Training Run

<!-- Your starting model, your settings, and anything you changed and why. -->

**Base model:** distilbert-base-uncased

**Settings:** <!-- epochs, learning rate, batch size, seed -->
 3 epochs, learning rate 2e-5, batch size 16, seed 42, maximum sequence length 128. I trained on CPU.

**Anything I changed from the defaults, and why:** I kept the default training settings

**Split sizes:** <!-- train / val / test, and per-label counts in the test
split. If a label had fewer than about 8 in test, say so — it explains a lot
of next unit's variance. -->
140 train / 31 validation / 31 test. The test split had 9 food_review, 9 food_story, and 13 recommendation posts.

---

## How I Used AI

<!-- Two specific moments — what you asked, what came back, what you changed.

     ⚠️ Plus disclosure of any pre-labelling. If you had a model pre-label a
     batch and then read and corrected every one, say that. It's an allowed
     workflow and disclosing it costs you nothing. Not disclosing it is the
     problem. -->

**Moment 1**

- *What I asked for:* I asked an AI assistant to stress-test my label definitions by writing eight posts near the category boundaries.
- *What came back:* It showed examples that could be confused between reviews, stories, and recommendations, and suggested a decision rule based on each post’s main purpose.
- *What I changed:* I removed hot_take and used three labels: food_review, food_story, and recommendation. I added the main-purpose rule to help distinguish reviews from recommendations.

**Moment 2**

- *What I asked for:* I gave the assistant my definitions and asked it to pre-label 135 forum posts.
- *What came back:* It suggested labels for the posts.
- *What I changed:* I reviewed every pre-labeled post and corrected labels that didn’t fit my definitions.

**Moment 3**

- *What I asked for:* I asked the assistant to look at my confusion matrix and failed predictions and tell me what pattern was recurring.
- *What came back:* It identified that many true food_review and food_story posts were being collapsed into recommendation, and that the model was favoring the recommendation class at the boundary where the post’s purpose was unclear.
- *What I changed:* I tightened the decision rule around the review/recommendation boundary and focused my relabeling on posts where the writer was evaluating a meal versus recommending it to others. This helped me diagnose the real issue as a boundary problem rather than a random training failure.

**Moment 4**

- *What I asked for:* I asked the assistant to compare the real labels to my model errors and tell me whether the mistakes looked like a single boundary issue or a broader dataset problem.
- *What came back:* It suggested that the recurring pattern was a label-boundary problem, especially around posts that mixed personal experience with a recommendation or an evaluation.
- *What I changed:* I focused my relabeling on posts where the writer’s main purpose was to evaluate a meal versus persuade others to try it, and I used the confusion-matrix pattern as evidence that the model was over-predicting recommendation. This helped me confirm that the issue was not random noise, but a systematic boundary confusion in my training data.

**Pre-labelling disclosure:**
I used an AI assistant to generate provisional labels for posts on lines 68–202, using my three label definitions. I reviewed every post and corrected any labels that do not fit before training.

<!-- ═══════════════════════ UNIT 6 — THE TEST ═══════════════════════

     Don't fill these in during unit 5.
     ═══════════════════════════════════════════════════════════════════ -->

---

## Baseline vs. Trained

<!-- Both models on the same posts. `python baseline.py --trained results.json`
     prints this table for you. -->

| Measure | Baseline | Trained | Difference |
|---|---|---|---|
| Overall accuracy | 0.290 | 0.516 | +0.226 |
| Macro F1 | 0.150 | 0.371 | +0.221 |
| F1 — `food_review` | 0.450 | 0.000 | -0.450 |
| F1 — `food_story` | 0.000 | 0.462 | +0.462 |
| F1 — `recommendation` | 0.000 | 0.650 | +0.650 |

**What I predicted before I looked:**
<!-- Milestone 1 asks you to write this BEFORE seeing the trained numbers. A
     prediction made afterwards isn't one. -->
     I expected the trained model to beat the baseline most on food_story and recommendation, because fine-tuning could learn those categories while the baseline appeared to recognize mainly food_review.

**What the gap actually means:**
<!-- If the baseline matched your trained model, your fine-tuning added
     nothing — and that is a real finding, not a failure. Say it plainly. -->
     Fne-tuning improved performance on the same 31 held-out posts: accuracy rose from 0.290 to 0.516, and macro F1 from 0.150 to 0.371. The trained model scored much better on food_story and recommendation, but its food_review F1 fell from 0.450 to 0.000. So fine-tuning helped overall, but the model still struggles to recognize reviews; this result is based on a small test set.



---

## Run Log — Before

<!-- Five criteria across three seeds. The notebook's section 6 prints the
     spread table; the Target and Verdict columns are yours. -->

| Criterion | Target | Seed 42 | Seed 7 | Seed 2024 | Verdict |
|---|---|---|---|---|---|
| 1. Overall accuracy | ≥ 0.60 | 0.516 | 0.419 | 0.419 | MISSED |
| 2. Food review F1 | ≥ 0.50 | 0.000 | 0.000 | 0.000 | MISSED |
| 3. Food story F1 | ≥ 0.40 | 0.4615 | 0.0000 | 0.0000 | MISSED |
| 4. Recommendation F1 | ≥ 0.50 | 0.6500 | 0.5909 | 0.5909 | MISSED |
| 5. Confidence gap | ≥ 0.10 | 0.60 | 0.80 | 0.30 | MET |

### Confusion matrix

<!-- ⚠️ TYPED AS A MARKDOWN TABLE. The notebook prints one ready to paste.
     An image of a matrix earns nothing. -->

| true \ predicted | food_review | food_story | recommendation |
|---|---|---|---|
| **food_review** | 0 | 1 | 8 |
| **food_story** | 0 | 3 | 6 |
| **recommendation** | 0 | 0 | 13 |

**My biggest off-diagonal number, and what it means:**
<!-- Not "the model made mistakes" — WHICH boundary it didn't learn, and which
     direction. "7 real analysis posts were called hot_take and only 3 went the
     other way" is a direction, not just an error rate. -->
     Eight real food_review posts were predicted as recommendation, while no recommendation posts were predicted as food_review. The model strongly favored recommendation at this boundary and failed to recognize reviews.



---

## Verdicts and Diagnoses

<!-- MET or MISSED against LAST UNIT's target. The target has to hold across
     all three seeds, not turn up sometimes. -->

| # | Criterion | Target | Verdict | How I decided |
|---|---|---|---|---|
| 1 | Overall accuracy | ≥ 0.60 | MISSED | The target does not hold across all three seeds: 0.516, 0.419, and 0.419 are all below 0.60 on at least one run. |
| 2 | Food review F1 | ≥ 0.50 | MISSED | Food review F1 is 0.000 in all three seeds, so the target never holds. |
| 3 | Food story F1 | ≥ 0.40 | MISSED | Seed 42 reaches 0.462, but Seed 7 and Seed 2024 both fall to 0.000; the target must hold across all three seeds. |
| 4 | Recommendation F1 | ≥ 0.50 | MET | Recommendation F1 is 0.650, 0.591, and 0.591, so it exceeds the target in every seed. |
| 5 | Confidence gap | ≥ 0.10 | MET | The confidence gap is 0.60, 0.80, and 0.30, so the model is more accurate in the most confident third than the least confident third in every seed. |

**Diagnoses**

<!-- For each miss: the cause, and how you know. The four common causes are:
     too few examples for a label, a boundary you applied inconsistently, a
     genuinely hard label pair, and a task the model can't reach from this
     much data.

     ⚠️ Use your agreement report as evidence. It is the only instrument you
     have that can tell a LABELLING problem from a MODEL problem, and this
     section is graded on whether you used it that way. -->

     The misses are driven by the same underlying problem: the model fails to learn the food_review boundary reliably. Across the three seeds, food_review F1 stays at 0.000, and the confusion matrix shows that eight true food_review posts are predicted as recommendation. That makes the main diagnosis a label-boundary problem rather than a random training failure. The agreement report reinforces this: several disagreements were cases where I treated a fact-based claim as reaction or hot_take, but the staff definitions are more consistent about treating evidence-heavy comparison posts as analysis. That mismatch explains why the model and my own labeling both over-prefer recommendation at the food_review boundary.


---

## Agreement Report

<!-- Your rate against the staff set, and every disagreement adjudicated.

     Remember you labelled these 30 under the STAFF taxonomy in
     data/staff_taxonomy.md, not your own — so every argument below is made
     from those definitions and those decision rules. -->

**Agreement rate:** __25_ / 30 = _83_%

<!-- Nobody grades this number. A 60% who argues every disagreement from the
     stated rules beats a 95% who wrote "staff was right" nine times. Several
     of the 30 were chosen because they're genuinely ambiguous — you should be
     winning some of these. -->

**Disagreements**

<!-- Three lines each: the post, both labels, and who you think is right and
     why — grounded in the staff definitions you were both applying.

     Then sort each into one of three piles:
       (a) the rule covered it and I applied it loosely → a consistency problem
       (b) the rule genuinely doesn't say               → a gap in the definitions
       (c) the rule is ambiguous here and my reading is defensible → argue it.
           This is a legitimate win.

     Pile (a) is the one that matters most for your diagnosis: if you applied a
     written rule two different ways on 30 posts, that is direct evidence about
     what you did across your own 200. -->

**1.**
> *The post:*
>  "I tracked sleep against session quality for eleven weeks. Under seven hours the next day's intervals were 6-8 seconds per mile slower, every time bar one. Small sample, one runner, but it's consistent enough that I've stopped scheduling hard days after late shifts." 
> *Staff said / I said:*
> analysis / reaction
> *My call, and why:*
> I think staff is right: the post is built around a checkable pattern across weeks and a specific time difference. The writer uses a fact to support a conclusion, even though it is framed in first person.
> *Which pile:*
     (c) the rule is ambiguous here and my reading is defensible

**2.**
> *The post:*
> "Everyone blames the hill at 18 miles. I split the field into people who ran the first half under 1:45 and over, and the slowdown after 18 was almost identical for both groups. If the hill were the problem the fast starters would fall apart harder."
> *Staff said / I said:*
> analysis / hot_take
> *My call, and why:*
> I think this is analysis because it tests a causal claim by comparing two groups and using a direct fact to evaluate the explanation. The comparison is the argument.
> *Which pile:*
     (a) the rule covered it and I applied it loosely

**3.**
> *The post:*
> "Checked the entry lists. 41% of this year's field have never run the distance before, against 26% five years ago. That's why the finish-line photos look different, not because standards dropped."
> *Staff said / I said:*
> analysis / reaction
> *My call, and why:*
> I think staff is right because the post relies on a direct percentage comparison across years to explain a trend. The numbers carry the argument, not just the feeling.
> *Which pile:*
     (a) the rule covered it and I applied it loosely

**4.**
> *The post:*
> "Ran the same loop in 4C and in 24C within a fortnight, same effort by heart rate. 90 seconds slower in the heat over six miles. People keep saying heat costs you 'a bit' and it's more than that."
> *Staff said / I said:*
> analysis / reaction
> *My call, and why:*
> I think this is analysis because it compares the same run under two conditions and gives a measured difference as evidence. The temperature effect is explicitly used as support.
> *Which pile:*
     (a) the rule covered it and I applied it loosely

**5.**
> *The post:*
> "Three of us ran the same session on the same day, different shoes, within 10 seconds of each other over 5 miles. I'm not saying shoes do nothing, but whatever they do was smaller than the noise between three people."
> *Staff said / I said:*
> analysis / hot_take
> *My call, and why:*
> I think this is analysis, although it is a borderline case: the writer gives a rough but concrete comparison and uses it to evaluate a causal claim. The fact is not decorative; it is the basis of the conclusion.
> *Which pile:*
     (c) the rule is ambiguous here and my reading is defensible

**What the pattern in my disagreements tells me:**
My disagreements cluster around the analysis/reaction boundary: I was weighting the writer's first-person framing and under-weighting whether the post used a checkable fact to support a claim. In several posts, the numbers or direct comparisons were the actual argument, even when the writing sounded personal; that is a consistency problem in how I applied the staff rules, not a gap in the rules themselves.

---

## The Improvement

**What I changed:**
I relabeled the food_review / recommendation boundary and applied the rule consistently across the dataset.
**Which diagnosis pointed at it:**
“The confusion matrix showed that true food_review posts were being predicted as recommendation, and my agreement disagreements clustered around that same boundary.”

### Run Log — After

| Criterion | Target | Seed 42 | Seed 7 | Seed 2024 | Verdict |
|---|---|---|---|---|---|
| 1. Overall accuracy | ≥ 0.60 | 0.387 | 0.484 | 0.419 | MISSED |
| 2. Food review F1 | ≥ 0.50 | 0.182 | 0.000 | 0.182 | MISSED |
| 3. Food story F1 | ≥ 0.40 | 0.000 | 0.533 | 0.000 | MISSED |
| 4. Recommendation F1 | ≥ 0.50 | 0.550 | 0.595 | 0.571 | MET |
| 5. Confidence gap | ≥ 0.10 | 0.70 | 0.60 | 0.40 | MET |

**Did it help, and how do I know:**

<!-- If it didn't, say so. Relabelling that didn't help is a genuinely
     interesting result and earns full credit. -->
     The improvement did not help. The target still fails for overall accuracy, food_review F1, and food_story F1, while recommendation remains above target. That is a valid result and a good diagnosis.

---

## What's Still Broken

<!-- For each criterion still missed: what you'd do, and why you stopped. -->

- Overall accuracy: I would add more boundary-heavy examples for food_review and food_story, especially posts where a meal is described in a personal or evaluative way but is not a direct recommendation. I stopped there because the relabel fix did not improve accuracy across seeds, so the issue is not just one mislabeled rule but a broader boundary problem in the training data.
- Food review F1: I would gather more clearly labeled review posts and re-check the review/recommendation rule to make sure the model sees more explicit “I ate this and here is my assessment” examples. I stopped because the model still predicts true food_review posts as recommendation in every seed, which shows the label boundary itself is still not learned well enough to fix with another small change.
- Food story F1: I would add more first-person memory posts and more examples that distinguish “telling a story about a meal” from “recommending a meal to try.” I stopped because the model still confuses story posts with recommendation, and the single improvement I made did not materially improve that pattern.

**The gap between what I meant my labels to capture and what the model
learned:**
<!-- Two sentences. Your confusion matrix is the evidence. -->
The model is still collapsing the review and story boundaries into recommendation. The confusion matrix shows eight true food_review posts predicted as recommendation and six true food_story posts predicted as recommendation, while no recommendation posts were predicted as food_review, which means the model learned the recommendation class too aggressively and still does not reliably separate evaluation from suggestion.


<!-- ═════════════════════════════════════════════════════════════════════

     SUBMISSION CHECKLIST — unit 5

       [ ] criteria.md has five numbered criteria, each naming a NUMBER
       [ ] Each has a reason underneath tied to your data or taxonomy
       [ ] labels.csv: at least 150 rows, text/label/note, ONE file not split
       [ ] No label above 70%
       [ ] All five unit 5 sections have real content
       [ ] Label Taxonomy includes the decision rule for your hardest boundary
       [ ] The Dataset includes three hard cases
       [ ] results.json and test_split.csv committed (the notebook does this)
       [ ] At least four new commits
       [ ] Repository URL submitted — WRITE IT DOWN

     SUBMISSION CHECKLIST — unit 6

       [ ] Baseline vs. Trained table, with your prediction written beforehand
       [ ] Run Log — Before, five criteria across three seeds
       [ ] Confusion matrix TYPED AS A MARKDOWN TABLE
       [ ] A verdict on every criterion
       [ ] A diagnosis for every miss, using the agreement report as evidence
       [ ] Agreement Report with every disagreement adjudicated
       [ ] One improvement, with Run Log — After
       [ ] What's Still Broken
       [ ] results_three_seeds_before.json, results_three_seeds_after.json,
           baseline_results.json and
           agreement_results.json committed
       [ ] At least four new commits
       [ ] The SAME repository URL as last unit

     Do not delete and recreate this repository.
     ═════════════════════════════════════════════════════════════════════ -->

---

📖 **How to run this project: [RUNNING.md](RUNNING.md)**

## The Capstone

## How I Used AI — Capstone

**Moment 1:** I asked an AI assistant why opening `/predict` in a browser returned an error. It explained that the browser sends GET while prediction requires POST; I tested the endpoint with a POST request and received a label and confidence.

**Moment 2:** I asked an AI assistant to help make repeated service tests reproducible and save their results. It suggested a Python harness. I ran it and inspected the reports; although the prediction and metrics results were recorded, `bad_inputs` was empty, so I did not count the error-handling criterion as demonstrated.

## The Shape: 
     - Shape A: I will package the classifier as a local API that starts with one documented command. It will return a predicted label and confidence, and record structured logs with timestamps, response times, request status, and predicted-label counts without saving submitted post text. I will include a text run record from a real session and document the Python version, pinned dependencies, and required model files and their sizes in `RUNNING.md`.

## Out of Scope:
I considered adding a web interface, user accounts, and automatic model retraining, but I’m not doing those as it would require more time than given. This extension will focus on deploying the API and measuring its operation and won’t change the classifier or store submitted post text.

## Local service run record

Started locally with `python serve.py` and sent eight prediction requests. All returned status 200. Mean logged latency was 142.23 ms; every prediction was `recommendation`. This is a small observed session, not a general performance guarantee.

text
{"timestamp": "2026-11-14T04:53:52Z", "status": 200, "latency_ms": 615.93, "label": "recommendation", "error": null}
{"timestamp": "2026-11-14T04:56:40Z", "status": 200, "latency_ms": 63.52, "label": "recommendation", "error": null}
{"timestamp": "2026-11-14T05:01:34Z", "status": 200, "latency_ms": 27.17, "label": "recommendation", "error": null}
{"timestamp": "2026-11-14T05:02:36Z", "status": 200, "latency_ms": 28.63, "label": "recommendation", "error": null}
{"timestamp": "2026-11-14T05:03:41Z", "status": 200, "latency_ms": 55.95, "label": "recommendation", "error": null}
{"timestamp": "2026-11-14T05:04:34Z", "status": 200, "latency_ms": 27.9, "label": "recommendation", "error": null}
{"timestamp": "2026-11-14T05:36:16Z", "status": 200, "latency_ms": 271.44, "label": "recommendation", "error": null}
{"timestamp": "2026-11-14T05:38:26Z", "status": 200, "latency_ms": 47.32, "label": "recommendation", "error": null}

## Criteria

Run Test - Before

| Criterion | Target | Run 1 | Run 2 | Run 3 | Verdict |
|---|---|---|---|---|---|
| 1. Prediction response | 10/10 requests return a valid prediction | 9 predictions, 1 `invalid_json` error | 10/10 predictions | 10/10 predictions | MISSED |
| 2. Logging and privacy | 10/10 requests have complete JSONL records; submitted text is not logged | 10 records; text absent | 10 records; text absent | 10 records; text absent | MET |
| 3. Metrics and latency | Metrics match all 10 requests and logged labels; average is within 0.1 ms of the log calculation | 10 requests, 1 error; 44.67 ms matches 44.667 ms | Log mean 25.24 ms; metrics output not provided | 10 requests; 27.84 ms matches 27.842 ms | MISSED |
| 4. Bad-input handling | All 3 invalid inputs return 400 and are logged; a valid request afterward returns 200 | The log shows the required error types and a later 200 | The test output shows malformed JSON instead of empty text | The test output shows malformed JSON instead of empty text | MISSED |
| 5. Local startup | `python serve.py` starts the service; `GET /metrics` returns 200 | Metrics response reported | Metrics response not recorded here | Metrics returned HTTP 200 | MISSED |

## Evidence excerpts

**Criterion 1 — Prediction responses**  
Source: `serve.py`, `Handler.do_POST`; recorded by `write_log` in `logs/run.jsonl`.  
Example successful response record:
```json
{"timestamp": "2026-11-16T00:31:36Z", "status": 200, "latency_ms": 28.17, "label": "recommendation", "error": null}
```

**Criterion 2 — Logging and privacy**  
Source: `serve.py`, `write_log`; `logs/run.jsonl`. Records contain result metadata, not submitted text.
```json
{"timestamp": "2026-11-16T00:31:36Z", "status": 200, "latency_ms": 26.18, "label": "food_story", "error": null}
```

**Criterion 3 — Metrics and latency**  
Source: `serve.py`, `Handler.do_GET` for `/metrics`; `write_log` for request records.
```json
{
"requests": 10,
"errors": 0,
"average_latency_ms": 27.84,
"predicted_labels": {
"recommendation": 9,
"food_review": 1
}
}
```
The matching ten log latencies average 27.842 ms, which rounds to 27.84 ms.

**Criterion 4 — Bad-input handling**  
Source: `serve.py`, `Handler.do_POST`; recorded by `write_log` in `logs/run.jsonl`.
```json
{"timestamp": "2026-11-16T00:04:15Z", "status": 400, "latency_ms": 0.13, "label": null, "error": "invalid_json"}
{"timestamp": "2026-11-16T00:08:34Z", "status": 400, "latency_ms": 0.19, "label": null, "error": "text_required"}
{"timestamp": "2026-11-16T00:09:12Z", "status": 400, "latency_ms": 0.11, "label": null, "error": "invalid_body_size"}
{"timestamp": "2026-11-16T00:17:08Z", "status": 200, "latency_ms": 36.52, "label": "recommendation", "error": null}
```

**Criterion 5 — Local startup**  
Source: `serve.py`, startup block and `Handler.do_GET` (`/metrics`).

```text
{"requests": 4, "errors": 4, "average_latency_ms": 0.15, "predicted_labels": {}}
HTTP_STATUS=200
```
The metrics endpoint returned HTTP 200. 4 POSTs had been recorded and all failed.

## Diagnoses for missed criteria

**Criterion 1 — Prediction response:** Run 1 missed the target at the PowerShell request-construction stage. One request reached `serve.py` as malformed JSON, so `Handler.do_POST` returned `400 invalid_json` before tokenization or model inference. This was a client payload error, not a classifier failure. Runs 2 and 3 passed, but the 10/10 target did not hold across all three runs.

**Criterion 3 — Metrics and latency:** This is an evidence gap, not a demonstrated metrics-calculation failure. Run 1’s 44.67 ms metric matches its log mean of 44.667 ms, and Run 3’s 27.84 ms metric matches its log mean of about 27.842 ms. Run 2’s `/metrics` output was not saved, so the target cannot be confirmed for all three runs. The metrics are in-memory and reset whenever the server restarts, which is why the response must be captured before stopping each run.

**Criterion 4 — Bad-input handling:** The failure was in the test procedure for Runs 2 and 3: the submitted test requests produced `invalid_json` instead of demonstrating the intended empty-text case (`text_required`). Therefore, those runs do not establish that all three distinct invalid inputs were handled as required. The Run 1 log does show `invalid_json`, `text_required`, and `invalid_body_size`, followed by a successful prediction. This is an incomplete-test-evidence miss, not proof that the handler failed those cases.

**Criterion 5 — Local startup:** No startup failure is shown. The saved output confirms `/metrics` returned HTTP 200 in one session, but equivalent evidence for all three runs is missing. In the excerpt showing `requests: 4, errors: 4`, the four POST requests errored; they were not four successful predictions. The metrics GET itself still returned 200. Thus the current MISS is due to incomplete evidence for the three-run target, not a demonstrated failure in the startup block or `Handler.do_GET`.

## Extension test results after the improvement

| Criterion | Target | Run 1 | Run 2 | Run 3 | Verdict |
|---|---|---|---|---|---|
| 1. Prediction response | 10/10 valid requests return HTTP 200 with a label and confidence | 10/10 | 10/10 | 10/10 | **MET** |
| 2. Logging and privacy | 10/10 requests have JSONL records; submitted text is not logged | 10 records; text absent | 10 records; text absent | 10 records; text absent | **MET** |
| 3. Metrics and latency | Metrics match 10 requests and logged labels; average is within 0.1 ms of log mean | 55.56 ms vs. 55.557 ms | 25.83 ms vs. 25.833 ms | 20.11 ms vs. 20.11 ms | **MET** |
| 4. Bad-input handling | All 3 invalid inputs return 400 and are logged; a valid request afterward returns 200 | Not shown in saved report | Not shown in saved report | All 3 returned 400; following valid request returned 200 | **MISSED** |
| 5. Local startup | Service starts and `GET /metrics` returns HTTP 200 | `/metrics`: 200 | `/metrics`: 200 | `/metrics`: 200 | **MET** |

### Run 3 evidence

Source: `logs/check_run_3.json`, produced by `run_service_checks.py`. The service
responses and request logs are produced by `serve.py` (`Handler.do_POST`,
`Handler.do_GET`, and `write_log`).

Prediction response:

```json
{"label": "recommendation", "confidence": 0.367935}
```

Example JSONL record:

```json
{"timestamp": "2026-11-16T02:26:41Z", "status": 200, "latency_ms": 34.5, "label": "recommendation", "error": null}
```

Metrics after ten predictions:

```json
{"requests": 10, "errors": 0, "average_latency_ms": 20.11, "predicted_labels": {"recommendation": 9, "food_story": 1}}
```
Startup metrics check:

```text
GET /metrics: HTTP 200
```
**Did it help, and how do I know:**

The Python harness helped make valid requests repeatable: the manual baseline had one `invalid_json` error in 10 requests, while each of the three harness runs recorded 10/10 successful predictions. It also saved per-run metrics evidence, making the results easier to compare.

This improvement changed how I tested the API; it did not change the server or improve the classifier. The saved reports show `bad_inputs: []`, so they do not prove that the three invalid-input tests passed in any of the after runs. I cannot claim the bad-input criterion improved. The trade-off is that the harness makes successful-request testing and evidence collection more consistent, but the error-handling criterion remains unverified.

## What It Costs

One test run means ten local prediction requests. The measured direct API/hosting cost was **$0.00 per run**: the service used the locally saved model and no paid API or hosted compute. That is $0.00 ÷ 10 = **$0.00 per prediction** in direct service fees; electricity, hardware depreciation, and my time were not measured.

For 100 people making one prediction each per day, the direct service-fee arithmetic is **100 people × 1 prediction/person × $0.00/prediction = $0.00/day**. This assumes the service runs on existing local hardware; it is not an estimate of electricity or the cost of operating a hosted service, and the local server was not tested for 100 users.

## What's Still Broken

- **Criterion 4 — bad-input handling: MISSED.** The after-run reports have empty `bad_inputs` arrays, so they don’t record the three expected 400 responses or prove recovery with a later valid request. The evidence gap is in the test/reporting stage; it does not establish that `serve.py` mishandled those requests. I stopped after collecting the three harness runs without obtaining complete failure-test records. I would fix or rerun the harness so each invalid request’s status and response are saved, then repeat all three trials.

## What I’d Do Next

First, I’d make the test harness save and verify the malformed-JSON, empty-text, and oversized-body responses in every run, because the current reports don’t establish Criterion 4. Then I’d test capacity and measure actual electricity use before making claims about serving 100 people per day.
