# GeoBurn AI

An agent that maps wildfire burn scars from satellite imagery: it works out what analysis a
request needs, finds imagery, computes burn indices, segments, scores its own uncertainty,
checks itself against reference data when there is any, and writes a georeferenced result with
an interactive map. It refuses to produce a map when the evidence is not there.

Status in one paragraph: the full pipeline runs and is tested (19 tests). It has been run on one
real Sentinel-2 scene with an independent Copernicus EMS reference (results below). The STAC
download path is implemented and unit-tested with mocked catalogues, but it has **not** been run
against a live catalogue, because the machine this was written on could not reach one.
Sentinel-1 analysis works on arrays; there is no SAR download/terrain-correction adapter.
The only segmentation model that runs is a weakly supervised ensemble; the foundation-model
adapter (Prithvi burn scar) is an empty slot. See "What is and is not done".

## Install

    python -m venv .venv && . .venv/bin/activate
    pip install -r requirements.txt
    python -m pytest -q tests          # 19 tests, a few seconds
    python -m geoburn.cli doctor       # shows which catalogues and packages are reachable

## Run

Real data, no account needed (downloads about 47 MB from GitHub at a pinned commit):

    python scripts/demo_cems_emsr382.py
    python -m geoburn.cli serve --runs runs     # then open http://127.0.0.1:8000

Live bi-temporal analysis from a STAC catalogue (first real test of that path):

    python scripts/demo_stac_live.py
    python -m geoburn.cli run "Outline burned areas and burn severity after the wildfire of 2019-08-17" \
        --bbox -15.76 27.96 -15.65 28.09 --event-end 2019-08-25 --out runs/my_run

Your own GeoTIFFs (12-band Sentinel-2 order B01..B12 without B10, reflectance 0-1):

    python -m geoburn.cli run "find burned vegetation" --post post.tif --pre pre.tif --out runs/local

`python -m geoburn.cli plan "<request>"` prints the plan without running it.
The agent asks for an area of interest if the request has none; it has no geocoder and will not guess one.

## Configuration

`config/default.yaml` holds thresholds, resolution, minimum mapping unit and catalogue definitions.
Environment variables: `GEOBURN_CACHE` (cache directory), `GEOBURN_CONFIG`, `GEOBURN_LOG`,
`CDSE_S3_ACCESS_KEY` / `CDSE_S3_SECRET_KEY` (only for the `cdse` provider),
`ANTHROPIC_API_KEY` + `GEOBURN_LLM_MODEL` (optional LLM task inference; untested, off by default).
No credentials are stored in files.

## Outputs of a run

`burned_areas.geojson`, `layers.tif` (scores, indices, uncertainty), `classes.tif` (mask, severity),
`regions.csv`, `report.md`, `summary.json`, `trace.json` (every tool call, argument, outcome) and
`map/index.html` (layer toggles, opacity, before/after swipe, selectable polygons, provenance).
The map loads Leaflet and basemap tiles from the internet.

## Real-data result (EMSR382, Gran Canaria, August 2019)

Post-fire Sentinel-2 mosaic only, so the agent took the mono-temporal branch and did not report
severity. Compared at 20 m with the Copernicus EMS delineation, 11,718 ha analysed, 13.5% cloud:

| method | precision | recall | F1 | IoU | boundary F1 | area error |
|---|---|---|---|---|---|---|
| spectral baseline | 0.746 | 0.614 | 0.674 | 0.508 | 0.229 | -17.7% |
| segmentation model | 0.764 | 0.560 | 0.647 | 0.478 | 0.236 | -26.6% |
| hybrid (final) | 0.771 | 0.600 | 0.675 | 0.509 | 0.274 | -22.2% |

This is a modest result and one scene. `docs/validation.md` explains what it does and does not show.

## What is and is not done

| Component | State |
|---|---|
| Indices, masking, MMU, polygons, area, severity classes | done, tested |
| Bi-temporal dNBR workflow | done, tested on synthetic scenes only |
| Mono-temporal fallback | done, tested on real data |
| STAC search / pair selection / band loading / cache | done; logic tested with mocks; never run live |
| CDSE provider | configured from documentation; never run |
| Weakly supervised ensemble segmenter | done, tested |
| Foundation-model segmenter (Prithvi, SAM, SegEarth-OV) | not integrated; adapter slot only |
| Uncertainty map, region ranking | done; checked against one reference |
| Multispectral CMF adaptation | done; corruption controls on one scene (13 chips) |
| RemoteCLIP judge | not integrated |
| Agent: task inference, workflow retrieval, validation, fallbacks, trace | done, tested (rule-based inference) |
| LLM task inference | written, never executed |
| Sentinel-1 change evidence | array-level function, tested on synthetic arrays; no data adapter |
| Reference validation | done for rasters on any grid; MTBS / EFFIS loaders not written |
| Flood, land cover, building change, active fire | recognised and declined with a reason |

Imagery: contains modified Copernicus Sentinel data (2019). Reference: Copernicus Emergency
Management Service, activation EMSR382, as redistributed in the CEMS-Wildfire dataset.
