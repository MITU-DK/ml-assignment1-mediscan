# Report to the CEO - pneumonia screening model

From: Mitu Kumar (B24CS029), ML engineer
Every number here comes from a script in the repo; see `reports/tables.md` for where.

## Short version

- Your instructions 1 and 3 were right. **Instruction 2, "99% accuracy", is the wrong target.** At a camp where about
  1 child in 10 has pneumonia, a model can reach 99% accuracy while missing 1 sick child in 10, and a model that
  never says "pneumonia" at all already scores 90%. What we should ask for is: **catch at least 97% of sick children,
  and keep false alarms low enough that clinicians can review them, measured on X-rays from our own camp machine.**
- The new model is 1.8 MB, answers in about 30 ms on a laptop CPU, and runs offline. That meets instruction 3.
- On new data from the same hospital it catches 97.8% of pneumonia cases, with about 1 false alarm per 10 healthy children.
- **On X-rays from other hospitals it raised a false alarm on 37% of healthy children.** That is the same failure you saw
  in the pilot. The model is not ready for an unsupervised camp until it has been checked on our own machine's X-rays.

## 1. What the 98.4% really was

Rohit's number was probably real, but it measured the wrong thing:

- All the images come from one hospital in Guangzhou. In that data, healthy children's X-rays are saved as much larger
  files than sick children's (about 2.4 MP vs 0.9 MP). A model that looks only at **the file's size and shape, never at
  the lungs, is 84% accurate**. A model given only the edges of the image, with the lungs blanked out, is 89% accurate.
  So a high score on this data does not prove the model learned pneumonia.
- He tested on more images from the same hospital and the same saving process, so the shortcut kept working in the test.
- He trusted a "validation set" of 16 images. 16 out of 16 correct only tells you the true accuracy is somewhere between 79% and 100%.
- He did not leak duplicate images between train and test. I checked for that specifically, and it is not the problem.
- His model (ResNet50, about 98 MB) could never have gone out on the 25 MB channel.

A score measured on one hospital's own images says little about other hospitals, other machines or other patients.
I can't re-test Rohit's model, since there are no saved weights. But my own model shows the same pattern (section 4): excellent on
the source hospital, many false alarms on outside patients. That is the most likely story behind the pilot's 7 out of 10.

## 2. What the new model achieves, and at what setting

The model is a small network I designed (ArogyaNet, 0.4M parameters). It was tested once on 821 X-rays it had never seen,
from patients it had never seen:

| | result |
|---|---|
| sick children caught (sensitivity) | **97.8%** (581 of 594) |
| healthy children correctly cleared (specificity) | 90.3% (205 of 227) |
| ROC-AUC | 0.988 |
| file size / time per image on a laptop CPU | 1.8 MB / about 30 ms (95% of images under 60 ms) |

**Setting (threshold 0.448).** I did not use the default 0.5. A missed pneumonia was costed at 20 times a false alarm,
and the threshold was chosen to give the lowest expected cost at the camp's 1-in-10 prevalence. It was chosen on
validation data only, before the test.

**"Low confidence, refer to a clinician" band.** If the model's pneumonia probability is between 0.11 and 0.90, it does not
give an answer and asks for a clinician. That happened for 11.6% of test images. On the other 88%, it caught **99.4%**
of pneumonia (3 missed) with 95.3% specificity.

**Bad inputs.** A colour photo of a dog, a black image and a white image are all rejected. The same dog photo converted to
grey is also rejected, because it looks nothing like the training X-rays. None of them get a "pneumonia" answer.

## 3. Why "99% accuracy" is the wrong target: the camp arithmetic

Take 1,000 children at a camp, 100 with pneumonia (10%) and 900 without.

| model | sick caught | sick missed | false alarms | accuracy |
|---|---|---|---|---|
| says "NORMAL" to everyone | 0 | **100** | 0 | **90.0%** |
| a "99% accurate" model that misses 1 in 10 sick children (sens 90%, spec 100%) | 90 | **10** | 0 | **99.0%** |
| **our model at its chosen setting** (sens 97.8%, spec 90.3%) | 98 | **2** | 87 | **91.1%** |

Our model has the lowest accuracy of the three useful-looking options, but it misses the fewest children.
With the cost you implied (a missed child = 20 false alarms), the expected cost per child is 2.00 for "everyone normal",
0.20 for the "99% accurate" model, and **0.13 for ours**. Accuracy rewards the model for being right about the 900 healthy
children, which is easy, and hides the 100 sick ones, which is the whole point of the camp.

Also, of the 98 + 87 = 185 children our model flags, about 53% truly have pneumonia. So every flag needs a clinician's
look; the model narrows 1,000 children down to about 185 to examine. That is what a screening tool is for.

**What to ask for instead:**
1. Sensitivity of at least 97% (no more than 3 missed per 100 sick children), with specificity of at least 85%.
2. Measured on **200-300 labelled X-rays from our own camp machine**, not on the Kaggle data.
3. Keep instruction 3: under 200 ms and under 25 MB. Today: 30 ms and 1.8 MB.

## 4. The other-hospital test, and what it means

I tested once on a public database from other hospitals (the COVID-19 Radiography Database, Normal and Viral Pneumonia images).

- I found that **all 1,345 of its pneumonia images, and 1,341 of its normal ones, are copies of our own training data**,
  so those cannot measure anything new. I did not count them as an external result.
- On those copied images (re-saved by the database authors as small square PNGs) the model still got 96.5% of normals and
  96.7% of pneumonia right. So the code and the image format are fine.
- The remaining **8,851 normal X-rays really are from other hospitals. The model called pneumonia on 37% of them**
  (specificity 62.8%, against 90.3% on our own test). According to the database's documentation they come from the RSNA
  dataset, which is mostly **adult** chest X-rays. Our model has only ever seen children aged 1-5, and an adult chest looks different.
- I tried one fix, contrast equalisation (CLAHE), and **it made things worse** on those images (specificity 54.0%).

So the model I am delivering has the same weakness as Rohit's on new patients, only smaller.
Before deployment it needs to be fine-tuned or recalibrated on our own camp X-rays.

## 5. What it must not be used for

- Not a diagnosis. It is a screening aid; every "pneumonia" or "refer" result goes to a clinician.
- Not on a new X-ray machine or a new site until it has been checked on that site's labelled images. Expect many false alarms until then.
- Not for adults, lateral views, or other diseases (TB, COVID-19, fractures). It was trained on AP views of children aged 1-5 from one hospital.
- Not as the only reason to send a sick-looking child home. A "NORMAL" result must never override a clinician's concern.

## 6. What I recommend next

1. Collect 200-300 X-rays from our camp machine, with a radiologist's labels.
2. Measure sensitivity and specificity there; adjust the threshold or fine-tune on that data.
3. Run a supervised pilot in which a clinician reviews every flagged and every "low confidence" child.
4. Two technical changes I did not try, and would test on that camp data: saving the larger ResNet18 model at half precision
   (about 22 MB, inside the 25 MB limit, and slightly better on validation), and feeding images in without stretching them
   to a square, so the model can't lean on image shape.
