# Research review

All five papers were read in full text (arXiv versions noted). Each idea below is tagged
**[orig]** demonstrated in the paper, **[adapted]** changed for this project, or **[new]** proposed
here. Adapted and new methods carry only the evidence reported in `docs/validation.md`.

## 1. Contrastive Mask Fidelity (Cao et al., arXiv 2608.09101v1)

1. **Problem.** Overlap metrics score agreement with a human mask, not faithfulness to the image. When labels are coarse, incomplete or shifted, a better mask scores worse (the "evaluation paradox").
2. **Method.** Reference-free score for one class mask: composite a *keep* view (mask region shown, rest Gaussian-blurred) and an *erase* view (mask blurred, rest shown); ask a frozen vision-language judge for the class probability on each.
3. **Architecture (GeoVeritas).** Stage I Seg-Probe makes candidate masks (SegEarth-OV3 on a frozen SAM 3 encoder, plus synonym pooling, two scales, flip TTA). Stage II CMF scores candidate and human mask with the same judge and arbitrates per class. Stage III retrains a segmenter on arbitrated labels to test usefulness.
4. **Equations.** P(c|r) = softmax_k(tau <phi(r), psi(c_k)>) over the dataset's class texts; x_keep = x*M + B(x)*(1-M); x_erase = x*(1-M) + B(x)*M; F(M) = P(c|x_keep) - P(c|x_erase) in [-1,1]; verdict dF = F(pred) - F(annotation) > 0; arbitration replaces the annotation only if dF > delta (default 0), conflicts fall back to the human label.
5. **Models and data.** RemoteCLIP ViT-L/14 judge at 224 px, blur sigma 24, tau 100. Ten RGB/IRRG benchmarks at 0.05-1 m GSD, 10,731 image-class pairs.
6. **Results.** Corruption controls: shift detected on 70-92% of pairs, erosion 52-86%, dilation 0-61%. The contrast cuts correlation with mask area from 0.61 to 0.08. Predictions win 56.6% of pairs overall, 62-85% for buildings/roads/cars, fewer for amorphous land cover. Agreement with a blinded three-expert majority: 81.0% on 279 pairs (88.4% at high margin, 70.5% on amorphous classes). Arbitrated labels improve cross-domain mIoU by about 1.2 over confidence-based replacement.
7. **Limits.** Inherits judge bias; all four judges tested are CLIP-family; measures class-relative semantics, not boundary geometry; weak on dilation; weakest exactly on amorphous land cover; label-conditioned audit; 300-pair expert study. The authors note Vaihingen's IRRG false colour is already outside the RGB judge's domain.
8. **Reusable.** The keep/erase contrast; symmetric treatment of competing masks; conservative arbitration with a margin; corruption controls as a validity test; the triage reading of |dF|.
9. **Needs adaptation.** The judge. RemoteCLIP has no defined behaviour on SWIR or red-edge bands, was validated at sub-metre GSD, and "burned area" is an amorphous class at 20 m, the regime where CMF is least reliable.
10. **Do not transfer.** Feeding SWIR false-colour composites to an RGB VLM and trusting the score; treating F as accuracy; replacing reference validation with it.

**Decision.** [adapted]/[new] `geoburn/qa/cmf.py` keeps the construction on reflectance bands and swaps the judge for a frozen physical index (BAI: red + NIR only), fitted once per scene without seeing any mask. It is used for ranking and arbitration only. Evidence so far: corruption controls on one real scene.

## 2. Data-centric benchmark for label-noise ranking (Nogueira et al., arXiv 2603.00604v2)

1. **Problem.** Find and rank noisy segmentation labels; noise in masks is continuous, not clean/noisy.
2. **Method.** Rank training images from least to most noisy; evaluate rankings against a known ordering.
3. **Benchmark design.** Clean SpaceNet8 building labels -> one of seven synthetic perturbations per patch (global or one-sided scaling, rotation, translation, deletion, vertex addition, false-positive insertion) -> ground-truth rank by IoU(clean, noisy).
4. **Algorithms.** *Augmented Ensemble Ranking*: RefineNet fine-tuned with strong augmentation, K=10 members, pixel majority vote, noise score = 1 - IoU(prediction, given label). *Regularized Variance Ranking*: ScaleMAE + UperNet, K=8 over 4 folds with increasing L2; S = IoU_max - (0.5 - IoU_max) * mean pixel variance across members.
5. **Data.** 5,000 training and 1,298 validation/test patches of 256 px, WorldView-3 RGB at 0.3-0.8 m.
6. **Results.** Kendall tau 0.61 (AER) and 0.57 (RVR) against 0.17 for CleanLab and 0.13 for an uncertainty-quantification baseline. Training on the best-ranked 50% beats training on all noisy data (U-Net F1 80.3 vs 78.3).
7. **Limits.** Synthetic noise only; binary buildings; RGB sub-metre; shrink/expand noise ranks worst because boundary shifts keep semantics; image-level scores, no pixel map.
8. **Reusable.** Ensemble disagreement and prediction-label agreement as reliability signals; ranking instead of a hard decision; the corruption taxonomy as a test generator.
9. **Needs adaptation.** No labels exist at inference time, so "label" becomes "the other method's mask"; the unit becomes a detected region, not a training patch; a pixel-level map is needed.
10. **Do not transfer.** The reported correlations (they describe buildings with synthetic noise); the supervised ensembles themselves.

**Decision.** [adapted] ensemble with bootstrap, feature bagging and radiometric jitter (`seg/models.py`); [adapted] RVR formula per region with method-agreement IoU (`qa/uncertainty.py::rank_regions`); [new] per-pixel uncertainty combining score entropy, method disagreement, member variance and a boundary band.

## 3. RS-Agent (Xu et al., arXiv 2406.07089v4)

1. **Problem.** MLLMs describe images but cannot run multi-step, tool-based remote-sensing workflows.
2. **Method.** LLM controller that first infers a task type, retrieves an expert workflow for it, then plans and calls tools.
3. **Architecture.** Central Controller (LangChain; perception, planning, retrieval, action), Toolkit (27 tools, optical and SAR, three-level hierarchy), Solution Space (34 solution documents + Solution Searcher), Knowledge Space (knowledge graph + Knowledge Searcher).
4. **Algorithms.** Task-Aware Retrieval: g_s = f_s(tp, D_s, k=|tp|), i.e. retrieve with the inferred task types as key (m3e-base embeddings, FAISS). DualRAG: LLM extracts keywords with weights 1-10; a global query retrieves N documents; each keyword retrieves n_i = floor(w_i / sum(w) * N), remainder by descending weight; union, N=30.
5. **Models and data.** Qwen2.5-32B by default; nine datasets, 18 tasks (DOTA, RSVQA-LR, AID, UCMerced, RSSDIVCS).
6. **Results.** Over 95% task-planning accuracy with strong LLMs. Ablation: single-tool accuracy 69.1% -> 93.5% and multi-tool 9.4% -> 90.6% only when task inference and solution retrieval are combined. Gains over MLLMs come from routing to specialist models.
7. **Limits (authors').** Task inference over-abstracts; malformed arguments; upstream errors propagate; no intermediate-step evaluation; small hand-curated databases. Also: RGB image in, text out; no georeferencing, spectral bands, data acquisition or uncertainty.
8. **Reusable.** The four-part architecture; task type as retrieval key; workflows as data; modality-aware tool hierarchy.
9. **Needs adaptation.** Tools must be geospatial processes with typed arguments and preconditions; workflows need declared fallbacks and quality gates, which the paper lists as future work.
10. **Do not transfer.** The specific tools (detectors, captioners); planning accuracy figures; dependence on a large LLM for routing a small task set.

**Decision.** [orig] architecture and Task-Aware Retrieval; [adapted] DualRAG on a TF-IDF index with a fixed domain weight table instead of LLM-extracted keywords and a graph store; [new] validated tool calls, post-conditions and workflow fallback branches.

## 4. ThinkGeo (Shabbir et al., arXiv 2505.23752v2)

1. **Problem.** No benchmark tests whether tool-using agents actually execute remote-sensing tasks correctly.
2. **Method.** 486 human-curated, tool-implicit tasks with 1,773 verified ReAct steps (436 optical, 50 SAR), 14 tools.
3. **Design.** ReAct loop over AgentLego tools grouped as perception, logic, operation; step-by-step and end-to-end modes.
4. **Metrics.** InstAcc, ToolAcc, ArgAcc, SummAcc per step; per-category tool accuracy; final answer judged by an LLM with curated questions.
5. **Data.** Chips from DOTA, xBD, FloodNet, iSAID, AID and others, 0.015-3.5 m optical, 1-15 m SAR.
6. **Results.** Best models: tool accuracy about 64-73%, argument accuracy 33-38%, final answers about 10-12% correct. Tool selection correlates most with answer accuracy. Error types: no action, invalid JSON, wrong argument values, premature final answers.
7. **Limits.** RGB chips, generic vision tools, no multispectral processing or data acquisition; LLM judge; evaluates the LLM, not a full geospatial system.
8. **Reusable.** Tool-implicit queries; separate tool and argument accuracy; trace-level evaluation; the error taxonomy.
9. **Needs adaptation.** Our tools and outputs differ, so the benchmark itself cannot be run; we need our own cases plus numeric scoring of the final map.
10. **Do not transfer.** Its scores as expectations for this system; LLM-as-judge for a product that can be measured against a reference raster.

**Decision.** [adapted] `agent/evaluate.py` scores task, tool-sequence, argument and status accuracy on 12 tool-implicit cases, and step success from stored traces. [new] call validation before execution targets the argument errors ThinkGeo found most common.

## 5. Agentic AI in Remote Sensing survey (Talemi et al., arXiv 2601.01891v1)

1. **Problem.** Organise the move from static models to agents in Earth observation.
2. **Method.** Taxonomy and review: single-agent copilots vs multi-agent systems; planning, RAG, memory, tool orchestration; benchmarks; limitations.
3. **Architectures covered.** Single controller (RS-Agent, RS-ChatGPT, TREE-GPT, Geode); planner-worker and role-specialised teams (ShapefileGPT, GeoLLM-Squad, STA-CoT with a verifier, MineAgent with judging agents).
4. **Key concepts.** Full-path vs end-to-end evaluation; trajectory metrics paired with remote-sensing metrics; error propagation into geophysical products.
5. **Coverage.** About 25 agent systems and the main benchmarks (ThinkGeo, GeoBenchX, Earth-Bench, GeoLLM-Engine, CORE).
6. **Findings.** Tool breadth and composability drive capability; tool accuracy tracks answer accuracy.
7. **Limits it names.** Weak geospatial grounding (mostly RGB, poor on spectral products and SAR); fragile tool orchestration (ignored preconditions, repeated failing calls); shallow memory (repeated downloads); fragmented evaluation; reliance on large proprietary LLMs. As a survey it offers no validated design.
8. **Reusable.** Single agent as the default; RAG over tool and domain documentation; logging intermediate results; pairing trajectory with product metrics; unsolvable queries as tests.
9. **Needs adaptation.** Memory here is a band cache and run traces, not a learned long-term store.
10. **Do not transfer.** Multi-agent orchestration without a measured benefit.

**Decision.** Single controller with specialist modules. A verifier role exists as a module (quality gates, CMF, reference validation), not as a second autonomous agent.

## How the five fit together

| Idea | Source | Used as | Status here |
|---|---|---|---|
| Controller + toolkit + solution space + knowledge space | RS-Agent | system skeleton | implemented |
| Task-Aware Retrieval | RS-Agent | workflow lookup by task type | implemented (rule-based task inference) |
| DualRAG weighted dual-path retrieval | RS-Agent | knowledge lookup | simplified (TF-IDF, fixed weights) |
| Tool-implicit queries, ToolAcc / ArgAcc, traces | ThinkGeo | agent evaluation | implemented on 12 own cases |
| Precondition checks, single agent, trajectory + product metrics | Survey | design rules | implemented |
| Ensemble + augmentation disagreement | Label-noise paper | uncertainty, second mask | adapted, weakly supervised |
| Regularized Variance Ranking | Label-noise paper | region review order | adapted |
| Keep/erase contrast with a frozen judge | CMF | mask arbitration without reference | adapted with a spectral judge |
| Corruption controls | CMF, label-noise paper | validity test of the fidelity score | run on one real scene |
| Class-wise arbitration with margin | CMF | choose between competing masks | implemented, advisory only |

RS-Agent supplies the structure, ThinkGeo and the survey say where such agents fail (tool choice,
arguments, missing checks), and the two data-quality papers supply ways to doubt a mask when no
ground truth is at hand. None of the five works with multispectral, georeferenced data; that part
comes from the wildfire remote-sensing literature in `docs/remote_sensing_methodology.md`.

## Wider landscape checked

- **Baselines:** NBR/dNBR (Key & Benson; USGS), RdNBR (Miller & Thode 2007), RBR (Parks et al. 2014), BAIS2 (Filipponi 2018), MIRBI, BAI.
- **Pretrained models:** IBM/NASA Prithvi-EO burn-scar fine-tunes (HLS, six bands, 30 m) are the closest published fit; SAM-family and SegEarth-OV models are RGB and open-vocabulary and have no documented skill on burned vegetation at 20 m; RemoteCLIP is RGB. None could be downloaded or run here, so none is claimed to work in this system.
- **Datasets:** HLS Burn Scars (US, MTBS-derived), CEMS-Wildfire (Europe, Sentinel-2 + Copernicus EMS masks; used here), Sen2Fire, MTBS, EFFIS, FireCCI51, MCD64A1.
- **Data access:** Earth Search STAC (public COGs) and the CDSE STAC endpoint with S3 credentials. The CDSE collection endpoint was confirmed from published metadata; neither endpoint was reachable from the build machine.
