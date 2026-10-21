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
| Overall accuracy |  |  |  |
| Macro F1 |  |  |  |
| F1 — `label_one` |  |  |  |
| F1 — `label_two` |  |  |  |

**What I predicted before I looked:**
<!-- Milestone 1 asks you to write this BEFORE seeing the trained numbers. A
     prediction made afterwards isn't one. -->

**What the gap actually means:**
<!-- If the baseline matched your trained model, your fine-tuning added
     nothing — and that is a real finding, not a failure. Say it plainly. -->



---

## Run Log — Before

<!-- Five criteria across three seeds. The notebook's section 6 prints the
     spread table; the Target and Verdict columns are yours. -->

| Criterion | Target | Seed 42 | Seed 7 | Seed 2024 | Verdict |
|---|---|---|---|---|---|
| 1.  |  |  |  |  |  |
| 2.  |  |  |  |  |  |
| 3.  |  |  |  |  |  |
| 4.  |  |  |  |  |  |
| 5.  |  |  |  |  |  |

### Confusion matrix

<!-- ⚠️ TYPED AS A MARKDOWN TABLE. The notebook prints one ready to paste.
     An image of a matrix earns nothing. -->

| true \ predicted |  |  |  |
|---|---|---|---|
| **** |  |  |  |
| **** |  |  |  |
| **** |  |  |  |

**My biggest off-diagonal number, and what it means:**
<!-- Not "the model made mistakes" — WHICH boundary it didn't learn, and which
     direction. "7 real analysis posts were called hot_take and only 3 went the
     other way" is a direction, not just an error rate. -->



---

## Verdicts and Diagnoses

<!-- MET or MISSED against LAST UNIT's target. The target has to hold across
     all three seeds, not turn up sometimes. -->

| # | Criterion | Target | Verdict | How I decided |
|---|---|---|---|---|
| 1 |  |  |  |  |
| 2 |  |  |  |  |
| 3 |  |  |  |  |
| 4 |  |  |  |  |
| 5 |  |  |  |  |

**Diagnoses**

<!-- For each miss: the cause, and how you know. The four common causes are:
     too few examples for a label, a boundary you applied inconsistently, a
     genuinely hard label pair, and a task the model can't reach from this
     much data.

     ⚠️ Use your agreement report as evidence. It is the only instrument you
     have that can tell a LABELLING problem from a MODEL problem, and this
     section is graded on whether you used it that way. -->



---

## Agreement Report

<!-- Your rate against the staff set, and every disagreement adjudicated.

     Remember you labelled these 30 under the STAFF taxonomy in
     data/staff_taxonomy.md, not your own — so every argument below is made
     from those definitions and those decision rules. -->

**Agreement rate:** ___ / 30 = ___%

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
>
> *Staff said / I said:*
>
> *My call, and why:*
>
> *Which pile:*

**2.**
> *The post:*
>
> *Staff said / I said:*
>
> *My call, and why:*
>
> *Which pile:*

**What the pattern in my disagreements tells me:**



---

## The Improvement

**What I changed:**

**Which diagnosis pointed at it:**

### Run Log — After

| Criterion | Target | Seed 42 | Seed 7 | Seed 2024 | Verdict |
|---|---|---|---|---|---|
| 1.  |  |  |  |  |  |
| 2.  |  |  |  |  |  |
| 3.  |  |  |  |  |  |
| 4.  |  |  |  |  |  |
| 5.  |  |  |  |  |  |

**Did it help, and how do I know:**

<!-- If it didn't, say so. Relabelling that didn't help is a genuinely
     interesting result and earns full credit. -->



---

## What's Still Broken

<!-- For each criterion still missed: what you'd do, and why you stopped. -->



**The gap between what I meant my labels to capture and what the model
learned:**
<!-- Two sentences. Your confusion matrix is the evidence. -->



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
