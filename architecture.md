# Architecture

## Choice: one controller, specialist modules

The survey's taxonomy offers single-agent copilots and multi-agent systems. This project uses a
single controller. The work is a mostly linear pipeline with a few known branch points; ThinkGeo
and the survey attribute failures to wrong tools, bad arguments and skipped checks, and extra
agents multiply those hand-offs. The requested "agents" B to G are modules behind tools. A second
autonomous agent would be justified once there is a measured gain, for example an independent
verifier that re-plans; nothing here measures such a gain yet.

```
 request ──► Controller (geoburn/agent/controller.py)
              │ 1 perceive   parse AOI + dates (nothing guessed; asks if missing)
              │ 2 infer task rules (default) or LLM (optional)        ◄── Task inference
              │ 3 retrieve   workflow by task type                    ◄── Solution Space  knowledge/workflows.yaml
              │              knowledge chunks (dual-path)             ◄── Knowledge Space knowledge/*.md
              │ 4 plan       fill workflow arguments from the request
              │ 5 act        validate call → run tool → check post-condition
              │ 6 recover    declared fallback branch / retry transient / stop with reason
              │ 7 record     trace.json (tool, args, status, observation, seconds)
              ▼
 Toolkit (geoburn/agent/tools.py → geoburn/pipeline.py)
   data        search_scenes · select_image_pair · select_post_only · load_scenes · load_local_scene
   quality     quality_check (valid fraction, cloud, co-registration, date gap → mode)
   analysis    compute_burn_indices · vegetation_change · segment · fuse_and_delineate · summarize_severity
   assurance   assess_uncertainty · validate_against_reference
   output      export_results
   control     check_sar_availability · report_insufficient_evidence
              ▼
 Modules
   geoburn/data    Provider interface · STAC adapter (Earth Search, CDSE) · local GeoTIFF adapter · cache
   geoburn/rs      grid (UTM analysis grid, true pixel areas) · indices · quality · burn · sar
   geoburn/seg     PixelEnsembleSegmenter · Prithvi adapter slot · hybrid fusion
   geoburn/qa      uncertainty + region ranking · multispectral CMF · reference metrics
   geoburn/report  GeoTIFF / GeoJSON / CSV / report / Leaflet map
```

## Mapping to the requested components

| Requested | Where | Notes |
|---|---|---|
| A Central controller | `agent/controller.py` | workflow-grounded planning, not free-form ReAct |
| B Data acquisition | `data/` | provider-independent `Scene`; asset names live in YAML |
| C Remote sensing analysis | `rs/` | NBR, dNBR, RdNBR, RBR, NDVI, BAIS2, MIRBI; SAR log-ratio |
| D Segmentation | `seg/` | one runnable model; foundation models are a slot |
| E Quality and validation | `qa/` | uncertainty, ranking, CMF adaptation, reference metrics |
| F Knowledge and workflow retrieval | `agent/knowledge.py`, `knowledge/` | Task-Aware Retrieval + dual-path TF-IDF |
| G Mapping and reporting | `report/`, `web/template.html`, `server.py` | self-contained map per run |

## Control flow and branch points

1. Request without AOI or date → `needs_input`, nothing runs.
2. Task recognised but not implemented (active fire, flood, land cover, buildings) → `not_implemented` with the reason. Active fire is matched before burned area so "is it burning now" is never answered with a burn-scar map.
3. No pre-event scene → branch `post_only`: mono-temporal indices, extent only, severity withheld.
4. Image or pair unusable (cloud, no data) → branch `check_sar`: report Sentinel-1 availability, then stop with `insufficient_evidence`. SAR is not auto-processed into a burn map.
5. Segmenter cannot run (too few confident seeds, missing weights) → skipped; result is labelled "spectral baseline only".
6. Catalogue unreachable or empty → `insufficient_evidence` with the underlying cause.

Every call is validated first (unknown tool, unknown or missing argument, wrong type, unmet
precondition) and checked afterwards (expected state keys exist). Transient I/O errors get one retry.

## Data contracts

- `Scene`: reflectance bands as float32 on one analysis grid, a validity mask, transform, CRS, resolution and a provenance dict (item id, date, catalogue, native band resolutions, cloud fraction).
- Analysis grid: UTM zone of the AOI at 20 m. 10 m bands are averaged down; 20 m bands are never upsampled. Every export states 20 m.
- Run state: plain dict passed between tools; arrays stay in memory, summaries go to the trace.
- Scores: `p_spectral`, `p_seg`, `p_final` are monotone evidence scores in [0,1]. They are not calibrated probabilities; calibration is measured only when a reference exists.

## Extending

- New data source: add a provider block in YAML (STAC) or subclass `Provider`.
- New task: add a workflow to `knowledge/workflows.yaml` and any new tools to the registry. Flood mapping, for example, needs an RTC Sentinel-1 adapter plus a water-index tool; the controller does not change.
- New model: implement `predict(post, pre, p_spectral, valid)` returning `p_seg` and `var_seg`.
- LLM controller: `infer_task_llm` is the only LLM touch point today. Letting an LLM choose tools beyond the workflow should be gated by the existing call validation and evaluated with `agent/evaluate.py` first.

## Known architectural gaps

- No geocoder, so place names are not understood.
- No tiling: the AOI is held in memory (fine to roughly 50 x 50 km at 20 m).
- No multi-scene mosaicking across orbits or MGRS tiles; one pre and one post scene.
- Memory is a band cache plus run traces; there is no cross-run learning.
