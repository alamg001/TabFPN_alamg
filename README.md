# Neural networks versus TabPFN-3.5

Fashion-MNIST classification and computational efficiency

**Prior Labs**

**Author: Antonio Lam García**

Doctoral researcher, UNI, Lima – Peru

October 5, 2026

**Experimental status: executed.** MLP, three CNNs and Prior Labs TabPFN-3.5 with PCA32 were evaluated on the same 1,000 test images. TabPFN achieved 84.3% accuracy, the highest macro AUC (0.9858) and the lowest log-loss (0.4272); Deep CNN achieved the highest accuracy (84.8%).

![image](figures_en/samples.png)


<p align="left">
  <img src="./assets_pm10_italia/PM10_Italia_v1.png"
       alt="Infografía introductoria sobre PM10 en Italia"
       width="800">
</p>

Actual examples from all ten categories. Training, validation and test observations remain separate.

Source notebook: `Lab_1_CNN.ipynb`. Deliverables: executed notebook outputs, self-contained dashboard, evaluation data and reproducible LaTeX sources. English edition of the updated Spanish report.

## 1. Introduction and scientific rationale

Convolutional neural networks (CNNs) are a natural choice for image classification because they learn shared local filters and retain spatial structure. A multilayer perceptron (MLP) flattens pixels and learns global relationships. Both optimize their weights using examples from the target problem. TabPFN, a *Tabular Prior-data Fitted Network*, offers a different mechanism: a pretrained foundation model that conditions its predictions on labeled training observations supplied as context.

The TabPFN-3.5 technical report \[1\] describes a shared classification and regression model with Fourier encodings and column-wise empirical cumulative distribution function features. Its performance across tabular benchmarks motivates testing whether numerical image representations can benefit from this approach. This is an empirical hypothesis rather than a guarantee of superiority over CNNs on Fashion-MNIST.

### 1.1 Converting images into tabular features

Each observation $x_i\in[0,1]^{28\times28}$ is flattened into 784 columns. The main representation applies principal component analysis (PCA) to retain 32 components:

$$
z_i=(\operatorname{vec}(x_i)-\widehat\mu_{\rm train})\widehat V_{32},\qquad
 \widehat V_{32}^{\mathsf T}\widehat V_{32}=I_{32}.
$$

The mean and projection are fitted exclusively on training data. The reduction can decrease cost and redundancy, but may remove discriminative information. The evaluated object is therefore the complete PCA + TabPFN-3.5 pipeline.

The technical report uses frozen image embeddings followed by PCA in MulTaBench. Our direct pixel projection is simpler and does not reproduce that benchmark. A useful extension would compare frozen-extractor embeddings while including their cost for every method that uses them.

### 1.2 What fitting TabPFN means

Conceptually, its context-conditioned predictive distribution approximates an amortized inference procedure:

$$
p_\theta(y_*\mid x_*,D_{\rm train})\approx
 \int p(y_*\mid x_*,\phi)\,p(\phi\mid D_{\rm train})\,d\phi.
$$

This expression describes the motivation, not a certified exact posterior for Fashion-MNIST. Local fitting does not make foundation-model pretraining free. The report gives approximately 220 million parameters for TabPFN-3.5; the official checkpoint was successfully loaded. Its pretrained parameter count is distinct from the parameters optimized locally in this experiment.

### 1.3 Experimental question

Given the same observations and an explicit budget, which alternative offers higher predictive quality and lower local fitting and inference cost? Both dimensions are reported. Energy consumption, monetary cost and peak memory are not measured.

## 2. Experimental design and architectures

### 2.1 Data and leakage prevention

Official Fashion-MNIST contains 60,000 training images and 10,000 test images. This pilot selects 2,000 training and 500 validation images from the official training set, and 1,000 evaluation images exclusively from the official test set. Stratification uses seeds 42, 43 and 44 respectively. Each class contributes 200, 50 and 100 observations to these partitions.

Indices and SHA-256 hashes preserve execution traceability. Pixel division by 255 matches the original notebook’s `ToTensor()` transformation. There is no augmentation or synthetic-data generation. All four neural networks and the TabPFN input use the same selected observations.

### 2.2 Compared architectures

| Model         | Structure and purpose                                                                                                                         |
|:--------------|:----------------------------------------------------------------------------------------------------------------------------------------------|
| MLP           | 784–256–256–10 with ReLU; a flattened-pixel baseline.                                                                                         |
| Baseline CNN  | Conv(32)–pool–Conv(64)–pool–FC(128)–10; retains the source architecture.                                                                      |
| Deep CNN      | Conv(32)–Conv(32)–pool–Conv(64)–Conv(64)–pool–FC(128)–10; implements the source notebook’s proposed experiment.                               |
| BatchNorm CNN | Baseline CNN with BatchNorm between each convolution and ReLU.                                                                                |
| TabPFN-3.5    | PCA32 main representation, explicit `ModelVersion.V3_5`, one estimator and caching. The optional 784-pixel variant is disabled in this pilot. |

### 2.3 Budget and checkpoint selection

The neural networks use ten epochs, AdamW, learning rate $10^{-3}$, batch size 128 and the same initial seed. The checkpoint with the lowest validation loss is retained. Test observations are evaluated afterward and never select the epoch. He initialization is used with ReLU, and Xavier initialization for the output layer.

The executed TabPFN configuration uses one estimator for the CPU pilot, whereas the technical report describes eight default estimators for the base model. A stronger comparison should include that configuration and multiple seeds on GPU, without conflating the base model with Fast, Plus or Thinking variants.

### 2.4 Source-notebook audit

The source records CNN test accuracy of 0.9161, CNN validation accuracy of 0.9245 and MLP validation accuracy of 0.8905. Comparing the first value with the last mixes partitions. It also repeats loss plots under an accuracy heading and leaves the random split seed unspecified. The new notebook corrects those conditions and weights losses by observation count rather than averaging unequal batches with equal weight.

## 3. Executed results and uncertainty

**Highest descriptive pilot accuracy: Deep CNN, 0.8480.** The ranking includes all five executed alternatives, including PCA32 + TabPFN-3.5.

| Model            | Accuracy | Macro F1 | Macro AUC | Log-loss |
|:-----------------|---------:|---------:|----------:|---------:|
| MLP              |   0.8210 |   0.8207 |    0.9810 |   0.4984 |
| Baseline CNN     |   0.8410 |   0.8408 |    0.9839 |   0.4630 |
| Deep CNN         |   0.8480 |   0.8479 |    0.9842 |   0.4974 |
| BatchNorm CNN    |   0.8300 |   0.8288 |    0.9833 |   0.4706 |
| TabPFN-3.5 PCA32 |   0.8430 |   0.8426 |    0.9858 |   0.4272 |

Accuracy and F1 use $\widehat y_i=\arg\max_k p_{ik}$. The area under the receiver operating characteristic curve (ROC AUC) is computed from probabilities, one class versus the rest (OvR), then averaged with equal class weights. Log-loss evaluates the probability assigned to the correct label; lower values are better.

$$
\operatorname{Acc}=\frac1n\sum_i\mathbf1(\widehat y_i=y_i),\qquad
\operatorname{LL}=-\frac1n\sum_i\log p_{i,y_i},\qquad
\operatorname{AUC}_{\rm macro}=\frac1{10}\sum_{k=0}^9\operatorname{AUC}_{k,\rm OvR}.
$$

| Model            | Fit (s) | Test (s) | ms/image | Local weights |
|:-----------------|--------:|---------:|---------:|--------------:|
| MLP              |   0.479 |    0.011 |   0.0108 |       269,322 |
| Baseline CNN     |   5.840 |    0.084 |   0.0844 |       421,642 |
| Deep CNN         |  12.356 |    0.269 |   0.2686 |       467,818 |
| BatchNorm CNN    |   6.877 |    0.118 |   0.1184 |       421,834 |
| TabPFN-3.5 PCA32 |   5.595 |    2.032 |   2.0320 |             – |

Neural fitting includes training and validation for all ten epochs. Inference covers the common test set in batches of 128. These are single-run measurements on CPU with four threads, not single-request latency estimates. They are not compared with published GPU timings.

| Model            |      Accuracy CI | $\Delta$ vs. baseline CNN |      Difference CI |
|:-----------------|-----------------:|--------------------------:|-------------------:|
| MLP              | \[0.798, 0.840\] |                    -0.020 | \[-0.041, -0.001\] |
| Baseline CNN     | \[0.820, 0.862\] |                    +0.000 | \[+0.000, +0.000\] |
| Deep CNN         | \[0.828, 0.868\] |                    +0.007 | \[-0.009, +0.024\] |
| BatchNorm CNN    | \[0.807, 0.849\] |                    -0.011 | \[-0.028, +0.007\] |
| TabPFN-3.5 PCA32 | \[0.825, 0.865\] |                    +0.002 | \[-0.018, +0.022\] |

The intervals are 95% percentile estimates from 200 stratified bootstrap replicates. Differences are paired because predictions refer to the same test observations. These intervals condition on the fitted models and exclude training-seed variation; there is no multiple-comparison correction. The Deep CNN improvement of 0.7 percentage points has an interval containing zero and is not conclusive.

**TabPFN-3.5 result.** Accuracy: 0.843; macro F1: 0.8426; macro AUC: 0.9858; log-loss: 0.4272. Its accuracy difference from Baseline CNN is +0.2 percentage points, with a 95% CI of $[-1.8;\,2.2025]$ points. This does not establish higher accuracy. Its better AUC and log-loss are descriptive; no statistical inference was computed for those metrics.

## 4. ROC curves and class discrimination

![image](figures_en/roc_macro.png)

Macro ROC curves for the common test set. Legend AUC values use actual predicted probabilities. The PCA32 + TabPFN-3.5 curve is included.

![image](figures_en/roc_classes.png)

TabPFN-3.5 PCA32: actual one-versus-rest curves and AUC for each category.

The horizontal axis is false positive rate and the vertical axis is true positive rate. The macro trace averages interpolated per-class curves on a common grid. Its legend reports the average of the original per-class areas, not an area inferred from hard labels or a confusion matrix.

## 5. TabPFN-3.5 confusion matrix

![image](figures_en/TabPFN-3.5_PCA32_confusion.png)

Prior Labs TabPFN-3.5 with PCA32: 843 correct and 157 incorrect predictions on 1,000 images.

Each category has 100 test examples, so diagonal counts divided by 100 give class recall. Upper-body garment categories account for important confusions. Row-normalized matrices and all prediction probabilities are supplied for independent class-level checks.

## 6. Confusion matrices: baseline architectures

![image](figures_en/MLP_confusion.png)

MLP: actual test counts. Rows are true labels; columns are predictions.

![image](figures_en/CNN_base_confusion.png)

Baseline CNN: the same 1,000 test observations.

Diagonal cells count correct predictions; off-diagonal cells identify confusions. The notebook also exports row-normalized matrices to inspect class-wise recall. Normalization does not change the number of evaluated observations.

## 7. Confusion matrices: convolutional variants

![image](figures_en/CNN_profunda_confusion.png)

Deep CNN: two convolutions per block before pooling.

![image](figures_en/CNN_BatchNorm_confusion.png)

BatchNorm CNN: normalization between convolution and activation.

All five confusion matrices use real predictions on the same test partition. Tables and ROC curves are verified against those probabilities.

## 8. Learning dynamics and local efficiency

![image](figures_en/learning.png)

Training loss, validation loss and validation accuracy per epoch. Final evaluation uses the checkpoint with minimum validation loss.

Decreasing training loss with stagnant or increasing validation loss can indicate overfitting. Validation-based checkpoint selection avoids using the test set to respond to this pattern. This pilot does not optimize each architecture; it describes behavior under a shared budget.

### 8.1 An operational efficiency criterion

For a workload of $B$ prediction batches, approximate local cost is

$$
C_m(B)=t_{m,\rm prep+fit}+B\,t_{m,\rm prep+predict}.
$$

Neural training can be amortized over future predictions. TabPFN context processing and caching also have costs. For PCA + TabPFN, PCA fitting and transformation must be included. Model download and a preliminary Iris execution took place before the benchmark and are excluded from timings. TabPFN fitting on Fashion-MNIST includes loading during fit and PCA fitting; test inference includes PCA transformation. No repeated warm-up or timing-variance protocol was measured. The local-weight column reports neural-network parameters; TabPFN uses pretrained weights and is marked –.

A method dominates another if it offers at least equal quality and at least equal speed, with a strict improvement in one dimension. Higher quality at higher cost requires an operational threshold or budget rather than a universal efficiency claim.

Here, MLP has the lowest total local time, while Deep CNN has the highest descriptive accuracy. AUC and accuracy need not rank models identically.

### 8.2 Limits of interpretation

One seed, one subsample and one timing measurement do not support extrapolation to the full benchmark. Energy use and peak memory are not measured. TabPFN achieved 0.843 accuracy with 7.627 s total cost, compared with 5.924 s for Baseline CNN and 12.625 s for Deep CNN. Its inference took 2.032 s for the full test batch, longer than the neural networks. Baseline CNN has lower total cost, while TabPFN improves descriptive AUC and log-loss. There is no single winner across all metrics. Original full-test CNN accuracy belongs to a different execution and is retained only as background evidence.

## 9. Time advantage over the Deep CNN

**Favorable result for Prior Labs TabPFN-3.5.** In the CPU pilot, PCA32 + TabPFN required **54.7% less fitting time** and **39.6% less total time** than Deep CNN, for one fit and one evaluation on the same 1,000 test images.

![image](figures_en/comparacion_tiempos.png)

Measured time decomposition. Total time includes fitting and inference; PCA is included for TabPFN.

### 9.1 Magnitude of the observed advantage

TabPFN fitting took 5.595 s, compared with 12.356 s for Deep CNN: a reduction of 6.761 s. Including prediction, total times were 7.627 s and 12.625 s, saving 4.997 s. Percentages use Deep CNN time as the reference:

$$
R_{\rm fit}=100\left(1-\frac{5.595391}{12.356298}\right)=54.7\%,\qquad
 R_{\rm total}=100\left(1-\frac{7.627411}{12.624907}\right)=39.6\%.
$$

TabPFN accuracy was 84.3%, compared with 84.8% for Deep CNN: a descriptive difference of 0.5 percentage points. TabPFN achieved higher macro AUC and lower log-loss. The time advantage therefore accompanies similar descriptive predictive quality in this pilot, without establishing statistical equivalence between models.

### 9.2 Operational interpretation and scope

This supports a time advantage for the specific workload of *fitting once and evaluating once*, compared with Deep CNN trained for ten epochs. TabPFN uses a pretrained model; its pretraining and model download are excluded from the measured local budget. This comparison does not imply identical historical development or pretraining costs for all alternatives.

TabPFN inference took 2.032 s, compared with 0.269 s for Deep CNN, approximately 7.6 times longer. Its total time also exceeded Baseline CNN (5.924 s). For many predictions after fitting, the evaluated CNNs offer lower inference times. This is a single measurement with four threads, one TabPFN estimator and one seed; timing uncertainty and single-image latency were not estimated. The advantage must be reassessed when hardware, data size or configuration changes.

## 10. Dashboard and evaluation controls

![image](figures_en/dashboard.png)

Static overview of executed results. The accompanying HTML dashboard explores the same metrics.

### 10.1 Available controls

- Model selector with accuracy, macro AUC, macro F1, total time and milliseconds per image.

- Minimum-accuracy and maximum-time filters; a ranking of models meeting both values.

- Confusion matrix and macro ROC curve updated when the selected model changes.

- Explicit execution status for every alternative and the experiment configuration.

The HTML includes all numerical values and visualization code, with no external content-delivery dependency. Controls filter existing results rather than launch training. Experimental controls are configured in the notebook’s first code cell and require reexecution.

### 10.2 Reading the dashboard

First verify that a model was executed, then inspect global quality and relevant class confusions, and finally check its timing budget. A pending row is not zero accuracy. TabPFN-3.5 PCA32 is marked as executed and is included in metrics, ROC curves and confusion matrices. The optional 784-pixel variant remains disabled, with no results attributed to it.

## 11. Reproduction, access and reviewed documents

### 11.1 Reproducing the TabPFN-3.5 comparison

1.  Sign in to the Prior Labs account that issued the API key and accept the exact TabPFN-3.5 license under Licenses: <https://platform.priorlabs.ai>. Use the API key from the same account.

2.  Supply a Prior Labs key through `TABPFN_TOKEN`. A Hugging Face key belongs to `HF_TOKEN`; changing a token prefix does not change its issuer. Keep credentials outside source files and delivered outputs.

3.  Install the accompanying dependency versions and open `Lab_1_TabPFN35_Comparacion.ipynb`. The environment used Python 3.12, PyTorch 2.14.1 and `tabpfn==9.1.0`.

4.  Execute the cells in order. The model is pinned to `V3_5`; another version or hosted Plus variant is not silently substituted. Inspect execution status and exported predictions before interpreting metrics.

5.  Extend the pilot using GPU, multiple seeds, the full official test set and eight estimators, while retaining common partitions for every method.

The complete nine-code-cell sequence was rerun, including all neural networks and TabPFN-3.5, directly in Python with captured display outputs and without a Jupyter server. Weights were downloaded through the official authenticated package. No credentials or model weights are included in deliverables.

### 11.2 Lessons from the supplied tutorials

The TabPFGen documents describe scaling, feature generation, stochastic gradient Langevin dynamics (SGLD) refinement and TabPFN target prediction. That generator is distinct from the TabPFN-3.5 classifier. Its dependency declaration `tabpfn>=2.0.1` does not select version 3.5.

Their examples combine nearest-neighbor proximity and mean within-class distance into an energy, with updates of the form

$$
x^{(t+1)}=x^{(t)}-\eta\nabla E(x^{(t)})+
 \sigma\sqrt{2\eta}\,\varepsilon^{(t)},\qquad \varepsilon^{(t)}\sim N(0,I).
$$

These operations do not guarantee visually plausible synthetic images, privacy protection or exact final class balance after argmax labeling. The energy-defined distribution is not automatically the real image distribution. The seven chapters, duplicate copies, tests, dependencies and technical-report source archive were reviewed. Software tests concern shapes, balance and reproducibility rather than predictive superiority. Synthetic generation was therefore excluded from the main comparison; a future augmentation study must be an independent training-only ablation.

## 12. Exact sources and traceability

## References

1.  Jäger, B., et al. (2026). *TabPFN-3.5: Technical Report*. Version 2, September 22. <https://arxiv.org/abs/2609.17895v2>. Supplied source: `arXiv-2609.17895v2.tar.gz`.

2.  Prior Labs (2026). *TabPFN: official code and documentation*. Accessed October 5, 2026. <https://github.com/PriorLabs/TabPFN>.

3.  Xiao, H., Rasul, K., and Vollgraf, R. (2017). *Fashion-MNIST*. Official data: <https://github.com/zalandoresearch/fashion-mnist>.

4.  *Lab_1_CNN.ipynb*. User-supplied notebook, architectures and original execution results.

5.  Haan, S. *TabPFGen*. Supplied tutorials, tests and distribution configuration 0.1.4. <https://github.com/sebhaan/TabPFGen>. Methodological background rather than the TabPFN-3.5 API.

### 12.1 Reading inventory

All 18 supplied reference files below were reviewed. Suffixed copies were retained as separate sources; their main explanations are equivalent.

| Exact filename                                                 | Content               |
|:---------------------------------------------------------------|:----------------------|
| `arXiv-2609.17895v2.tar.gz`                                    | 3.5 report            |
| `01_tabpfgen_class_.md`                                        | Generator interface   |
| `02_tabpfn_integration_.md`                                    | Predictor integration |
| `03_classification_generation___generate_classification___.md` | Class generation      |
| `04_regression_generation___generate_regression___.md`         | Regression/quantiles  |
| `05_sgld_sampling____sgld_step___.md`                          | SGLD sampling         |
| `06_energy_function____compute_energy___.md`                   | Energy/distances      |
| `07_visualization___visuals_py___.md`                          | Diagnostics           |
| `test_tabpfgen.py`                                             | Software tests        |
| `01_tabpfgen_class_ (1).md`                                    | Chapter copy 1        |
| `02_tabpfn_integration_ (1).md`                                | Chapter copy 2        |
| `06_energy_function____compute_energy___ (1).md`               | Chapter copy 6        |
| `05_sgld_sampling____sgld_step___ (1).md`                      | Chapter copy 5        |
| `04_regression_generation___generate_regression___ (1).md`     | Chapter copy 4        |
| `06_energy_function____compute_energy___ (2).md`               | Chapter copy 6        |
| `07_visualization___visuals_py___ (1).md`                      | Chapter copy 7        |
| `setup.py`                                                     | Distribution 0.1.4    |
| `requirements.txt`                                             | Dependencies          |

**Conclusion.** This pilot includes real Prior Labs TabPFN-3.5 results. Deep CNN maximizes accuracy; TabPFN maximizes AUC and minimizes log-loss; MLP minimizes total time. TabPFN also reduces fitting time by 54.7% and total time by 39.6% relative to Deep CNN for the evaluated local budget. The accuracy advantage of TabPFN over Baseline CNN is inconclusive. Model selection must specify its quality metric and operating budget and be validated across seeds and sample sizes.

