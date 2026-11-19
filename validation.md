# Validation

## What has actually been measured

| Evidence | Data | Status |
|---|---|---|
| Unit and integration tests (19) | synthetic scenes, mocked catalogue | pass |
| End-to-end run with independent reference | real Sentinel-2, Copernicus EMS EMSR382 | one tile |
| Bi-temporal dNBR accuracy on real data | none | not measured |
| Live STAC acquisition | none | not run |
| Agent plan benchmark | 12 cases written for this project | 12/12 |

One scene supports no general accuracy claim. Everything below is a description of that scene.

## Real-data case: EMSR382, Gran Canaria, August 2019

- **Imagery.** `EMSR382_AOI01_01_S2L2A.tiff` from the CEMS-Wildfire dataset repository at commit `d635dea0`. Its request file records Sentinel-2 L2A, bbox -15.7638, 27.9657, -15.6497, 28.0908, least-cloud mosaic of 2019-08-17 to 2019-10-06, all 12 bands delivered at about 10 m in EPSG:4326. Re-gridded here to 20 m UTM 28N. The mosaic has a visible seam, so pixels come from more than one date.
- **Reference.** The dataset's DEL (delineation) and GRA (damage grade) rasters derived from Copernicus EMS rapid-mapping products, plus its CloudSen12 cloud mask.
- **No pre-fire image** is in the sample. The agent chose the mono-temporal branch and withheld severity. The identification of the activation with the mid-August 2019 Gran Canaria wildfire rests on the bbox and dates in the request file.
- **Run.** 9 tool calls, all succeeded; 11,718 ha analysed; 13.5% of the tile masked as cloud or shadow.

| Method (no reference labels used by any) | Precision | Recall | F1 | IoU | Boundary F1 (2 px) | Area error |
|---|---|---|---|---|---|---|
| 1 Spectral baseline | 0.746 | 0.614 | 0.674 | 0.508 | 0.229 | -17.7% |
| 2 Segmentation model | 0.764 | 0.560 | 0.647 | 0.478 | 0.236 | -26.6% |
| 3 Hybrid | 0.771 | 0.600 | 0.675 | 0.509 | 0.274 | -22.2% |

Predicted 3,691 ha against 4,745 ha of reference inside the analysed area.

**Reading it.**
- The hybrid is not meaningfully better than the baseline on overlap (F1 +0.001). It is slightly more precise and its outline follows the reference better (boundary F1 +0.045). The segmentation model alone is worse on recall. Because that model is trained on the baseline's seeds, a large gain was not expected.
- Recall depends on damage grade. Share of each CEMS grade mapped as burned: "destroyed" 99.5% (738 ha), "damaged" 68.4% (2,969 ha), "possibly damaged" 8.4% (1,098 ha). A post-fire-only char signal finds strong burn and misses light damage; this is the case for requiring a pre-fire image.
- 12.1% of the area outside the reference was mapped as burned. In the false-colour image much of it, west of the delineation, looks spectrally like burn scar. It may be dark volcanic terrain or burned land outside this activation's delineated area. That was not checked and should not be counted for or against the method.
- Boundary agreement is low for all methods: a generalised analyst polygon versus a 20 m pixel mask with unburned islands.
- Calibration: ECE of the fused score is 0.162, so it must not be read as a probability.
- Uncertainty is informative but coarse: error rate is 33.1% in the more-uncertain half of pixels against 13.8% in the other half (overall 23.4%).

### Multispectral CMF adaptation on this scene

- **Corruption controls** (clean reference mask ranked above its corrupted copy, 64 px chips with 10-90% burned): shift by 6 px 13/13, erosion by 3 px 12/13, dilation by 3 px 8/13. The ordering (shift easiest, dilation hardest) matches the paper's pattern, with far too few chips to estimate rates.
- **Reference vs final mask:** F = -0.008 vs +0.020 over 45 chips; the final mask scored higher on 78% of chips. This does not show the prediction is better. The judge is a char index and so favours masks limited to strong char, the same bias that makes the method miss "possibly damaged" land. It is the judge-bias limitation the CMF authors describe.
- **Baseline vs segmentation model:** F = 0.0274 vs 0.0277, a tie.
- **Conclusion:** the keep/erase contrast transfers mechanically and responds to gross mask errors. It is not validated as an arbiter for burned area. A RemoteCLIP judge on RGB composites was not tested.

## Agent evaluation

`python -m geoburn.cli eval-agent` scores 12 tool-implicit requests: task type, tool sequence,
key arguments and plan status (ok / needs input / not implemented). Result 12/12 on each. The
cases were written alongside the rule-based planner, so this is a regression test. It says
nothing about paraphrases, other languages, or an LLM planner. Executed-trace metrics: the real
run had step success 9/9 with no recoveries; fallback branches are exercised in tests with a mocked
catalogue.

## Protocol for a proper evaluation (not yet run)

1. **Events.** At least 30 fires across biomes, each with a pre/post pair and an independent perimeter.
2. **Splits by event and region,** never by neighbouring patches. For any trained model, hold out whole fires and whole regions; keep dates of test fires out of training years where possible.
3. **Three methods** on identical grids and masks: spectral baseline, segmentation model, hybrid. Add an independently trained burn-scar model as a fourth.
4. **Metrics.** Precision, recall, F1, IoU, boundary F1 at 1-2 px, area error; per-event and pooled, with bootstrap intervals over events. Stratify by reference severity and land cover.
5. **Uncertainty.** ECE and reliability tables, error rate by uncertainty quantile, share of errors inside the top 20% most uncertain area.
6. **Agent.** Tool-implicit queries written by someone other than the developer, including unsolvable ones (no imagery, wrong sensor, missing AOI); ToolAcc, ArgAcc, step success, recovery rate, completion rate.
7. **Thresholds** fixed before looking at test events.

## Reference products are not interchangeable

| Product | Grid / unit | Made from | Caveat for a 20 m map |
|---|---|---|---|
| MTBS (US) | 30 m | Landsat dNBR, analyst thresholds | fires above about 1000 acres (west) / 500 acres (east); perimeters include unburned islands |
| EFFIS (Europe) | polygons | mainly Sentinel-2 in recent years | minimum size about 5 ha; rapid mapping updated during the season |
| Copernicus EMS | polygons | analyst mapping, often VHR, at activation time | only requested AOIs; generalised outlines; dated to the activation |
| FireCCI51 / MCD64A1 | 250 m / 500 m | MODIS | far coarser; suited to regional totals, not perimeters |

Compare like with like: resample to the coarser grid, match dates, apply the reference product's
minimum mapping unit, and report disagreements as disagreements.
