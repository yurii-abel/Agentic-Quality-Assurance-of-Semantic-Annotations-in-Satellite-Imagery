# Remote sensing methodology

## Three different products

| Product | Question | Evidence | In this system |
|---|---|---|---|
| Active fire | Is it burning now, where is the front? | thermal anomalies (VIIRS, MODIS, SLSTR), SWIR saturation | recognised, declined |
| Burned area | What burned? | post-fire reflectance, pre/post change | implemented |
| Burn severity | How much did vegetation and soil change? | dNBR / RdNBR / RBR classes, ideally tied to field CBI | implemented when a pre-fire image exists |

The agent never substitutes one for another: a post-fire image alone yields extent, not severity.

## Bi-temporal workflow (preferred)

1. **Scenes.** Sentinel-2 L2A surface reflectance. The pair is the least cloudy pre-event and post-event scene closest to the event, preferring the same MGRS tile and relative orbit to limit view-angle and registration differences.
2. **Grid.** UTM zone of the AOI at 20 m. B8A, B11, B12 and red-edge bands are 20 m native and are read as they are; 10 m bands are block-averaged to 20 m. Results are 20 m products. Reflectance = (DN + offset) / 10000, with the -1000 offset for processing baseline 04.00 and later unless the catalogue has already applied it.
3. **Masking.** SCL classes 0, 1, 3, 8, 9, 10, 11 are excluded in each date; water (SCL 6 or a green/NIR test) is excluded from burn candidates. Masked area is reported as unobserved, never as unburned.
4. **Checks.** Jointly valid fraction, days between dates (warning above 90), and a phase-correlation estimate of pre/post shift on B8A (warning at one pixel or more).
5. **Indices.** NBR = (B8A - B12) / (B8A + B12); dNBR = NBR_pre - NBR_post; RdNBR = dNBR / sqrt(|NBR_pre|); RBR = dNBR / (NBR_pre + 1.001); dNDVI as a vegetation-loss condition. B8A is used rather than B8 so NIR and SWIR share a native grid.
6. **Burned / unburned.** Evidence score = logistic(dNBR around threshold t) x logistic(dNDVI around 0.05). t defaults to 0.10 and, when `adaptive` is on, is replaced by an Otsu threshold constrained to [0.08, 0.30] for the scene. The report states which was used.
7. **Severity.** dNBR classes at 0.10 / 0.27 / 0.44 / 0.66 (the Key & Benson break points that USGS publishes) inside the final mask. These are configurable defaults. They were derived for particular US ecosystems with Landsat; MTBS analysts choose thresholds per fire. Without field data they describe relative spectral change, not ecological severity. RdNBR and RBR are exported for users who need a relativized measure.

## Mono-temporal fallback

Used only when no pre-event scene exists. A char composite averages robust z-scores of -NBR,
BAIS2 and MIRBI; an Otsu threshold between the scene median and 99th percentile separates
candidates. This detects strong char. It confuses dark bare soil, lava, shadow and recently
ploughed fields with burn, and misses low-severity and under-canopy fire (measured on real data:
see validation). The run is labelled mono-temporal and severity is withheld.

## Segmentation and fusion

1. The spectral score gives an interpretable baseline mask.
2. A second candidate comes from `PixelEnsembleSegmenter`: six gradient-boosted classifiers trained on high-confidence seeds from the baseline (score above 0.9 or below 0.1), each with a bootstrap sample, a random 70% of features and 5% radiometric jitter. Features are all bands, NBR, NDVI, NBR2, 5x5 local mean and standard deviation, and pre/post differences when available. **It learns from the baseline's own confident pixels, so it is not independent evidence**; what it adds is multi-band and neighbourhood context for the ambiguous middle, and member disagreement.
3. Fusion is the geometric mean of the two scores with hysteresis (seed above 0.65, grow to 0.35), so a region needs support from both.
4. Cleaning: 3x3 opening and closing, removal of regions below the minimum mapping unit (1 ha default, about 25 pixels), filling of holes smaller than the MMU. Larger unburned islands are kept.
5. All candidate masks are kept and exported so disagreement can be inspected.

A general-purpose segmenter (SAM family, open-vocabulary models) is not assumed to understand
burned vegetation, and none is wired in. A model trained on burn scars (for example a Prithvi-EO
burn-scar fine-tune) would be the first truly independent candidate; its adapter is a stub.

## Area

Area is the sum of true pixel areas inside each region: constant on the UTM grid, geodesic per row
if a geographic grid is ever used. Polygons are vectorised on the analysis grid and reprojected to
longitude/latitude for GeoJSON; reported hectares come from the raster, not from the reprojected
polygon. Area below the MMU and area under cloud are excluded and both are stated.

## Sentinel-1

`rs/sar.py` computes VV and VH log-ratio change after boxcar multilooking in linear power and
robust z-scores, flagging strong VH decrease. Inputs must be terrain-corrected gamma0 from the same
relative orbit. Backscatter also changes with rain, soil moisture, snow, harvest and flooding, so
the output is "change consistent with canopy loss", never "burned". There is no adapter that
downloads, calibrates and terrain-corrects GRD scenes; the agent only reports whether Sentinel-1
acquisitions exist when optical data fails.

## Uncertainty

Per pixel, u = 1 - (1 - H(p_final)) (1 - D) (1 - V), then max(u, 0.5 x boundary band), where H is
binary entropy of the fused score, D is disagreement among the candidate masks, V is scaled member
variance, and the boundary band covers three pixels either side of the outline. Regions are ranked
for review with S = IoU - (0.5 - IoU) x mean variance, IoU being agreement between the two
candidate masks around the region. Both are heuristics for ranking. Confidence is not accuracy.

## Known failure modes

Topographic shadow and low sun; thin cloud and smoke missed by SCL; cloud shadow; agricultural
burning and harvest; phenological drying between dates; rapid regrowth in grass and savanna;
low-severity understory fire under intact canopy; dark volcanic or wet soil; water-level change;
mixed pixels along the perimeter at 20 m; multi-date mosaics with seams.
