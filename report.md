# Burned-area analysis report

Generated 2026-11-19T02:40:07+00:00 by geoburn 0.1.0.

**Request:** Find burned vegetation in this region and estimate the affected area in hectares.

## Result

- Burned area detected: **3,691.1 ha** in 84 regions (minimum mapping unit applied; analysis grid 20 m, EPSG:32628).
- Area analysed: 11,718.4 ha. Not observed (cloud, shadow, no data): 4,061.7 ha. Burned area under cloud is not included.
- This is a model prediction compared against a reference product.
- Mode: monotemporal; delineation: hybrid (spectral x segmentation, hysteresis).
- Burned / unburned threshold: char composite z(-NBR, BAIS2, MIRBI) = 0.349 (scene-adaptive (Otsu); no pre-fire image).
- Burn severity: not reported. It needs a pre-fire image; a post-fire image alone only supports extent.

## Confidence and uncertainty

- Mean confidence inside the burned mask: 0.535; share of burned pixels with confidence < 0.5: 0.385.
- Confidence is a heuristic (score entropy, method disagreement, ensemble variance, boundary band). It is not an accuracy estimate.
- Mask fidelity (multispectral CMF adaptation, mean over 44 chips): F = 0.0208.
- CMF arbitration, spectral baseline vs segmentation model: F = 0.0274 vs 0.0277 (prefers segmentation model).
- The segmentation model was trained on seeds from the spectral baseline, so agreement between the two is not independent confirmation.

## Comparison with reference: Copernicus EMS EMSR382 delineation (via CEMS-Wildfire dataset)

| method | precision | recall | F1 | IoU | boundary F1 | area error % |
|---|---|---|---|---|---|---|
| spectral_baseline | 0.7463 | 0.6139 | 0.6737 | 0.5079 | 0.2286 | -17.7429 |
| segmentation_model | 0.764 | 0.5605 | 0.6467 | 0.4778 | 0.2363 | -26.6328 |
| final_hybrid | 0.771 | 0.5997 | 0.6746 | 0.509 | 0.2743 | -22.2131 |

- Calibration of the final score: ECE = 0.1618.
- Uncertainty vs error: {"overall_error_rate": 0.2342, "q50": {"error_rate_above": 0.331, "error_rate_below": 0.1375}, "q80": {"error_rate_above": 0.418, "error_rate_below": 0.1883}, "q95": {"error_rate_above": 0.4261, "error_rate_below": 0.2241}}
- Share of each reference class mapped as burned: {"0": {"area_ha": 6913.8, "predicted_burned_share": 0.121}, "2": {"area_ha": 1097.7, "predicted_burned_share": 0.084}, "3": {"area_ha": 2968.7, "predicted_burned_share": 0.684}, "4": {"area_ha": 738.2, "predicted_burned_share": 0.995}}
- CMF corruption controls on the reference mask: {"dilate": {"detection_rate": 0.615, "n_chips": 13}, "erode": {"detection_rate": 0.923, "n_chips": 13}, "shift": {"detection_rate": 1.0, "n_chips": 13}}

## Data provenance

```json
{
 "post": {
  "id": "/home/claude/geoburn/data/cems_sample/EMSR382_AOI01_01_S2L2A.tiff",
  "provider": "local",
  "analysis_res_m": 20,
  "bbox4326": [
   -15.763778600999956,
   27.96566434500005,
   -15.649711279999963,
   28.090828762000058
  ],
  "cloud_fraction": 0.13514414520687054,
  "source_crs": "EPSG:4326",
  "source_res_m": 9.99,
  "source_shape": [
   1380,
   1129
  ],
  "source": "CEMS-Wildfire-Dataset@d635dea0 (Sentinel-2 L2A via Sentinel Hub process API)",
  "time_range": "2019-08-17..2019-10-06 (leastCC mosaic)",
  "activation": "EMSR382 AOI01",
  "note": "all bands were delivered resampled to ~10 m; re-gridded here to 20 m UTM"
 },
 "pre": null
}
```

## Quality checks

```json
{
 "post": {
  "valid_fraction": 0.8549,
  "cloud_fraction": 0.13514414520687054,
  "usable": true
 }
}
```

## Limitations

- SWIR bands are 20 m; nothing here is a 10 m product.
- Thresholds are defaults or scene-adaptive; they are not calibrated to field data (CBI) for this ecosystem.
- Spectrally similar surfaces (dark bare soil, lava, shadow, harvested fields, water edges) can be mapped as burned; low-severity and under-canopy burns can be missed.
- Burned-area extent, burn severity and active fire are different products; this report covers the first, and the second only when a pre-fire image exists.
