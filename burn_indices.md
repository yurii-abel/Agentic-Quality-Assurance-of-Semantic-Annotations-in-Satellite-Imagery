# Burned-area indices and severity

## NBR and dNBR
NBR = (NIR - SWIR2) / (NIR + SWIR2). Healthy vegetation is bright in NIR and dark in SWIR2; char and exposed soil reverse this. dNBR = NBR_prefire - NBR_postfire; positive values indicate vegetation loss. For Sentinel-2 use B8A (865 nm, 20 m) with B12 (2190 nm, 20 m) so both bands share a native grid; B8 (10 m) is broader and must be aggregated to 20 m before use. Landsat 8/9: B5 and B7 at 30 m. Source: USGS Landsat Normalized Burn Ratio; Key and Benson (2006).

## dNBR severity thresholds
USGS publishes break points proposed by Key and Benson for interpretation: enhanced regrowth below -0.1, unburned -0.1 to 0.1, low 0.1 to 0.27, moderate-low 0.27 to 0.44, moderate-high 0.44 to 0.66, high above 0.66. These came from specific US ecosystems and Landsat data. They are starting points only: MTBS analysts set thresholds per fire, and thresholds shift with vegetation type, sensor, timing and atmosphere. Calibrate against field Composite Burn Index plots where possible.

## RdNBR and RBR
RdNBR = dNBR / sqrt(|NBR_prefire|) (Miller and Thode 2007) removes the bias of absolute dNBR toward dense pre-fire vegetation; unstable when pre-fire NBR is near zero. RBR = dNBR / (NBR_prefire + 1.001) (Parks et al. 2014) is a more stable relativization. Prefer RBR or RdNBR when comparing severity across sparse and dense vegetation.

## Mono-temporal indices
Without a pre-fire image only the post-fire state is available. BAIS2 (Filipponi 2018) uses red-edge, NIR and SWIR2; MIRBI (Trigg and Flasse 2001) uses SWIR1 and SWIR2; BAI (Chuvieco 2002) uses red and NIR distance to a charcoal convergence point. All are confused by dark soil, lava, shadow and water edges, and miss low-severity or under-canopy burns. Report extent only, never severity.

## Timing
Initial assessment images are taken soon after the fire; extended assessments use the next growing season. Pre and post images should match in phenology (anniversary dates if not immediate). A long gap lets regrowth or seasonal drying change NBR without fire.

## Active fire versus burned area versus severity
Active-fire detection finds thermal anomalies while the fire burns (VIIRS/MODIS FIRMS, SLSTR). Burned-area mapping outlines what burned. Burn severity grades the degree of ecological change. They answer different questions with different sensors and must not be substituted for each other.
