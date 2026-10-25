# Lab 6: Check Your Understanding — Mathematical Derivations & Theoretical Foundations
**Course:** Introduction to Applied AI | BS 7th Semester | Department of Electronics  
**Topic:** Computer Vision Pipeline Integrity, Group Splitting, Class Imbalance, Focal Loss, Bayesian Prevalence & Interpretability  

---

### Question 1: Why does a random split of chest X-rays overestimate performance?

#### Analytical Answer & Working:
In medical imaging datasets, multiple chest radiographs are frequently captured for a single patient over the course of treatment (e.g., initial baseline assessment, post-intubation check, supine vs upright orientations, and daily monitoring in an intensive care setting). 

These images possess profound, patient-invariant idiosyncratic signatures:
1. **Biological Morphology:** Individual skeletal geometry, thoracic diameter, rib cage spacing, cardiac silhouette size, diaphragm contour, and unique degenerative changes or scoliosis.
2. **Hardware & Acquisition Artifacts:** Sensor pixel-level fixed pattern noise from specific X-ray detector plates, focal distance differences, technician marker styles, and radiopaque orientation typography (`'R'`, `'L'`, `'PORTABLE'`).

When a **random split** is applied at the image level, multiple exposures from the same patient inadvertently populate both the training and test partitions (information leakage). Deep neural networks, being exceptional memorizers of surface-level statistical regularities, rapidly learn to recognize the patient's skeletal landmarks, body habitus, or hardware noise patterns and associate them with that patient's clinical outcome.

On the test set, the model does not identify pneumonia through alveolar consolidations or air bronchograms; rather, it simply performs patient identification. Consequently, published test accuracy and AUROC appear artificially stellar ($>98\%$), creating an illusion of clinical readiness. However, when the model is subsequently deployed in real-world clinical environments on entirely unseen patients, these patient-specific memorized cues are absent, causing the diagnostic performance to catastrophically degrade.

**Applied to this lab:** the official Kaggle test folder shares **170 patient groups (320/624 images)** with the training folder—exactly the leakage mechanism above. The train/validation split is group-disjoint (0 shared groups), and the official test score is therefore reported as an optimistic benchmark rather than an external validation.

---

### Question 2: Two classes have 900 and 100 training images. Compute both class weights with $N / (K \times n_c)$.

#### Given Data:
* Class 0 (Majority): $n_0 = 900$
* Class 1 (Minority): $n_1 = 100$
* Total training instances: $N = n_0 + n_1 = 900 + 100 = 1,000$
* Number of classes: $K = 2$

#### Formula:
$$w_c = \frac{N}{K \times n_c}$$

#### Step-by-Step Calculation:

1. **Weight for Class 0 (Majority):**
   $$w_0 = \frac{1,000}{2 \times 900} = \frac{1,000}{1,800} = \frac{5}{9} \approx \mathbf{0.5556}$$

2. **Weight for Class 1 (Minority):**
   $$w_1 = \frac{1,000}{2 \times 100} = \frac{1,000}{200} = \frac{10}{2} = \mathbf{5.0000}$$

#### Verification of Balance Condition:
$$\sum_{c=0}^{K-1} w_c \times n_c = (w_0 \times n_0) + (w_1 \times n_1)$$
$$\sum_{c=0}^{K-1} w_c \times n_c = \left(\frac{5}{9} \times 900\right) + (5.0 \times 100) = 500 + 500 = 1,000 = N \quad \checkmark$$

#### Interpretation:
The minority class is up-weighted by exactly a factor of **$9\times$** relative to the majority class:
$$\frac{w_1}{w_0} = \frac{5.0}{5/9} = 9.0$$
This forces a misclassification of a rare minority instance to incur 9 times more penalty in the cross-entropy objective than misclassifying a majority instance, preventing gradient descent from settling into a trivial majority-only solution.

---

### Question 3: With focal loss and $\gamma = 2$, by what factor is an example with $p_t = 0.8$ down-weighted?

#### Theoretical Background:
Standard binary cross-entropy loss is defined as:
$$\text{CE}(p_t) = -\log(p_t)$$
Where $p_t$ is the model's estimated probability for the ground-truth class.

Focal Loss (Lin et al., ICCV 2017) introduces a modulating dynamic factor $(1 - p_t)^\gamma$ to suppress the loss contributed by easy, well-classified examples:
$$\text{FL}(p_t) = -(1 - p_t)^\gamma \log(p_t) = \left[(1 - p_t)^\gamma\right] \times \text{CE}(p_t)$$

#### Given:
* Estimated ground-truth probability: $p_t = 0.8$
* Focusing parameter: $\gamma = 2$

#### Step-by-Step Calculation:
$$\text{Down-weighting Factor} = (1 - p_t)^\gamma$$
$$\text{Down-weighting Factor} = (1 - 0.8)^2 = (0.2)^2 = \mathbf{0.04}$$

#### Mathematical & Practical Conclusion:
The example is down-weighted by a factor of **$0.04$** (i.e., its loss contribution is reduced to **$4\%$** of its standard cross-entropy value, representing a **$25\times$** or **$96\%$** reduction in gradient magnitude). This ensures that large quantities of easily classified samples do not dominate parameter updates, focusing the model's learning capacity on hard, borderline instances where $p_t \le 0.5$.

---

### Question 4: Sensitivity 0.90, specificity 0.95, prevalence 2%. Compute PPV.

#### Given Data:
* **Sensitivity (True Positive Rate, $TPR$):** $\text{Sens} = 0.90$
* **Specificity (True Negative Rate, $TNR$):** $\text{Spec} = 0.95$
  * False Positive Rate ($FPR = 1 - \text{Spec}$): $1 - 0.95 = 0.05$
* **Prevalence ($P(D)$):** $\text{Prev} = 0.02$
  * Complement ($1 - \text{Prev}$): $1 - 0.02 = 0.98$

#### Formula (Bayes' Theorem / Diagnostic Epidemiology):
$$\text{PPV} = P(D \mid +) = \frac{P(+ \mid D) \times P(D)}{P(+)}$$
$$\text{PPV} = \frac{\text{Sensitivity} \times \text{Prevalence}}{[\text{Sensitivity} \times \text{Prevalence}] + [(1 - \text{Specificity}) \times (1 - \text{Prevalence})]}$$

#### Step-by-Step Arithmetic:
1. **Numerator (True Positives in population):**
   $$\text{Numerator} = 0.90 \times 0.02 = \mathbf{0.018}$$

2. **False Positives in population:**
   $$\text{FP} = (1 - 0.95) \times (1 - 0.02) = 0.05 \times 0.98 = \mathbf{0.049}$$

3. **Denominator (Total Positive Test Results):**
   $$\text{Denominator} = \text{Numerator} + \text{FP} = 0.018 + 0.049 = \mathbf{0.067}$$

4. **Positive Predictive Value (PPV):**
   $$\text{PPV} = \frac{0.018}{0.067} = \frac{18}{67} \approx \mathbf{0.2686567...} \implies \mathbf{26.87\%} \quad (\approx \mathbf{0.269})$$

#### Clinical Insight:
Despite the model exhibiting seemingly exceptional sensitivity ($90\%$) and specificity ($95\%$), because the background disease rate is low ($2\%$), the absolute number of healthy individuals falsely flagged ($0.049$) exceeds the number of true positive sick individuals detected ($0.018$) by nearly $3:1$. As a consequence, **$73.13\%$ of all positive alerts generated by the model are false alarms**, leaving only a $26.87\%$ probability that an alerted patient truly suffers from the disease.

---

### Question 5: When is macro-F1 a better selection metric than weighted F1?

#### Analytical Answer & Working:
To evaluate this distinction, consider the mathematical definitions of both metrics across $K$ classes:

1. **Macro-F1 (Unweighted Macro-Average):**
   $$\text{Macro-}F_1 = \frac{1}{K} \sum_{k=1}^K F_1^{(k)}$$
   Macro-F1 computes the harmonic mean of precision and recall independently for each individual class and takes their unweighted arithmetic mean. Every class contributes exactly $\frac{1}{K}$ to the final score, regardless of whether it has 10 samples or 10,000 samples.

2. **Weighted F1 (Support-Weighted Average):**
   $$\text{Weighted-}F_1 = \sum_{k=1}^K \left(\frac{n_k}{N}\right) F_1^{(k)}$$
   Weighted F1 weights each class's $F_1$ score proportionally by its sample size ($n_k / N$).

#### When Macro-F1 is Superior:
Macro-F1 is strictly superior whenever **the dataset exhibits class imbalance and detecting the rare minority class is critical to mission success or clinical safety**.

* **Failure Mode of Weighted F1:** In an imbalanced scenario (e.g., $95\%$ Class A and $5\%$ Class B), a naive model that predicts Class A for $100\%$ of cases achieves an $F_1$ of $0.974$ on Class A and $0.000$ on Class B. Its weighted F1 score is:
  $$\text{Weighted-}F_1 = (0.95 \times 0.974) + (0.05 \times 0.000) = \mathbf{0.925} \quad (92.5\%)$$
  Weighted F1 would falsely classify this completely defective model as high-performing.
* **Safeguard of Macro-F1:** Under the exact same scenario, Macro-F1 yields:
  $$\text{Macro-}F_1 = \frac{0.974 + 0.000}{2} = \mathbf{0.487} \quad (48.7\%)$$
  Macro-F1 severely penalizes the model for failing on the minority class. Therefore, whenever one cannot afford to sacrifice minority performance to maximize majority accuracy (such as rare pathology screening, fraud detection, or structural defect analysis), **Macro-F1 must be used for model checkpoint selection**.

---

### Question 6: Grad-CAM highlights a hospital label in the corner of X-rays. What do you do next?

#### Analytical Answer & Remediation Protocol:
When Grad-CAM isolates peak activation gradients on a peripheral radiopaque orientation tag (e.g., `'R'`, `'L'`, `'PORTABLE'`, or a clinic identifier stamp), it is definitive evidence of a **confounding shortcut**. The neural network is exploiting a spurious statistical correlation—such as severe pneumonia patients frequently being imaged with portable emergency scanners containing distinct physical markers, while ambulatory healthy patients are imaged on standard stationary hardware.

#### Actionable Multi-Stage Remediation Protocol:

1. **Phase 1: Diagnostic Ablation / Occlusion Experiment (Immediate Verification)**
   * Synthetically mask, blur, or inpaint the corner label region across the entire validation and test sets (setting that bounding region to neutral zero or average background intensity).
   * Re-run inference without retraining.
   * If model confidence or test sensitivity drops substantially, the shortcut hypothesis is conclusively proven.

2. **Phase 2: Upstream Anatomical Segmentation (Data Preprocessing)**
   * Integrate an automated thoracic segmentation model (such as a U-Net trained on the public Montgomery County or JSRT lung datasets) upstream in the pipeline.
   * Generate binary segmentation masks for the lung fields and crop strictly to the lung parenchyma, zeroing out all external clavicular markers, soft tissue, and collimator edges before feeding images to the classifier.

3. **Phase 3: Targeted Adversarial Augmentation (Training Pipeline)**
   * Apply aggressive corner cutouts (`RandomCutout` or `CoarseDropout`) during training to randomly blank out peripheral corners of images.
   * This forces the convolutional feature extractors to seek invariant pathological features within the pulmonary parenchyma rather than relying on static peripheral tokens.

4. **Phase 4: Multi-Site Curation & Re-Balancing**
   * If possible, collect additional normal radiographs from the intensive care portable units and pneumonia radiographs from outpatient stationary units to actively decouple scanner typography from pathological labels in the training distribution.
