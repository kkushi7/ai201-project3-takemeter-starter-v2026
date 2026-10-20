# Acceptance criteria — TakeMeter

Five criteria that say what "working" means for this classifier, written in
unit 5 **before** anything was trained.

**All five are yours this time.** None are given. You've had two projects of
practice.

An acceptance criterion names a number. *"The model is accurate"* is an
opinion. *"Every label has an F1 of at least 0.60 on the held-out set"* is a
criterion.

Under each, write a sentence or two on **why that number**. A reason that says
something about your data or your taxonomy earns credit — *"I picked 0.60 F1
for `reaction` because it's my smallest label and I only have about 50
examples of it"*. A reason that could be attached to any project does not.

> Missing your own targets next unit costs you nothing. Setting a target so
> easy you can't miss it does.

---

## Pick numbers you can defend

Not numbers that sound impressive. Three labels means a coin-flip guesser gets
about 33%, so a target of 0.40 is barely a target. Your number should sit
somewhere you'd honestly call useful.

**Cover at least three of these five areas.** They're here as prompts, not as a
form to fill in — a criterion that fits none of them is fine if it names a
number.

| Area | A question it could answer |
|---|---|
| Overall accuracy | How often does it need to be right to be worth using? |
| Per-label performance | Is one label allowed to be much worse than the others? |
| Balance | How lopsided can your label counts get before it's a problem? |
| Consistency | If someone else labelled the same posts, how often should you agree? |
| Confidence | Should a confident prediction be right more often than an unsure one? |

Two things worth knowing before you pick numbers, because both will affect
whether you hit them:

- **Your smallest label will have the jumpiest score.** If a label has 50
  examples, about 8 land in the test split. An F1 computed on 8 examples moves
  a lot between seeds. A target for that label should be looser than one for
  your biggest label, and saying so is a good reason.
- **Unit 6 tests across three seeds, and the target has to hold across all
  three.** A target of 0.65 against results of 0.71, 0.62, 0.68 is a **miss**.
  Pick with that in mind — it is stricter than it first sounds.

---

## 1. Overall accuracy

The model will achieve at least 0.60 accuracy on each of the three seeds.

**Why this target:** I chose 0.60 because a random guess across three labels would be right about 0.33 of the time. Requiring 0.60 means the classifier must do meaningfully better than guessing without assuming this small dataset will produce near-perfect results.
---

## 2. Food review performance

The `food_review` F1 score will be at least 0.50 on each of the three seeds.

**Why this target:** I chose 0.50 because some posts evaluate a meal while also telling a story or giving advice. This is a moderate floor that still requires the model to identify reviews usefully across all three seeds.

---

## 3. Food story performance

The `food_story` F1 score will be at least 0.40 on each of the three seeds.

**Why this target:** I chose 0.40 because stories can include opinions that sound like reviews, such as the dinner post that recounts a meal and calls it “a hot dog worth eating.” I set a lower target for this harder boundary, but still want performance above chance.
---

## 4. Recommendation performance

The `recommendation` F1 score will be at least 0.50 on each of the three seeds.

**Why this target:** I chose 0.50 because recommendations often include personal experience or evaluation, which can make them look like reviews. The target allows for that overlap while requiring the model to recognize advice consistently.

---

## 5. Confidence

On each of the three seeds, accuracy on the model’s most-confident third of predictions will be at least 0.10 higher than on its least-confident third.

**Why this target:** I chose a 0.10 accuracy gap because the test set should be about 30 posts, or roughly 10 per confidence group. A 10-point gap is about one more correct prediction in the most-confident third than in the least-confident third, making it a modest but meaningful difference



---

<!-- ─────────────────────────────────────────────────────────────────────────
     UNIT 6 — read this before you change anything above.

     If a criterion turns out to be BROKEN rather than merely unmet, you can
     revise it, and that earns credit. But never delete or edit the original
     line. Add the revision underneath, like this:

         ## 2. Every label performs acceptably

         The model performs well on all labels.

         **Why this target:** ...

         > **Revised in unit 6:** Every label has an F1 of at least 0.60 on
         > the held-out set.
         >
         > **Why revised:** "performs well" gave me nothing to check. I
         > couldn't produce a verdict from it at all.

     That's a revision because the criterion couldn't be MEASURED.

     Lowering a target because you missed it is not a revision, and it costs
     you the point:

         ✗ "Overall accuracy of at least 0.65" → "at least 0.55", because
            0.65 turned out to be optimistic for 200 examples.

     A number you missed stays where it is, gets diagnosed, and gets a fix
     attempted. That's where the points are.
     ───────────────────────────────────────────────────────────────────────── -->
