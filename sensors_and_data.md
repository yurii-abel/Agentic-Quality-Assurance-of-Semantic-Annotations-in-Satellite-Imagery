# Sensors, catalogues and reference products

## Sentinel-2 bands and resolution
10 m: B2 blue, B3 green, B4 red, B8 NIR. 20 m: B5, B6, B7 red edge, B8A narrow NIR, B11 SWIR1, B12 SWIR2, and the SCL scene classification. 60 m: B1, B9. Any index using B11 or B12 is a 20 m product. Resample 10 m bands by averaging to 20 m; never present an upsampled SWIR index as 10 m detail. L2A reflectance = (DN + offset) / 10000; processing baseline 04.00 and later (from 2022-01-25) carries BOA_ADD_OFFSET = -1000.

## Cloud and shadow masking
SCL classes 3 (cloud shadow), 8, 9 (cloud medium/high probability), 10 (thin cirrus), 11 (snow), 0 (no data), 1 (saturated) are excluded. SCL misses thin cloud and some shadow; residual shadow lowers NIR and SWIR and can look like burn. Smoke over an active fire also corrupts indices.

## STAC catalogues
Earth Search (earth-search.aws.element84.com/v1), collection sentinel-2-l2a: public cloud-optimised GeoTIFFs, no credentials. Copernicus Data Space Ecosystem (stac.dataspace.copernicus.eu/v1), collection sentinel-2-l2a: the official archive; asset download goes through S3 (eodata.dataspace.copernicus.eu) with user S3 credentials. Landsat Collection 2 is available through USGS and other STAC endpoints.

## Sentinel-1 SAR
C-band VV and VH. Fire in forest usually lowers VH (less volume scattering) but rain, soil moisture, snow, harvest and flooding also change backscatter. Compare only the same relative orbit and direction, calibrated to gamma0, terrain-corrected (RTC), speckle-filtered in linear power. Treat as supporting evidence when optical data is cloudy, not as proof of burning.

## Reference products
MTBS (US, Landsat 30 m, fires above about 1000 acres west / 500 acres east, analyst thresholds). EFFIS (Europe, mapped mainly from Sentinel-2, about 5 ha minimum in recent years). Copernicus EMS rapid mapping (analyst delineation and damage grading for activations, often from VHR imagery). FireCCI51 (global, 250 m MODIS) and MCD64A1 (500 m). These differ in resolution, minimum mapping unit, date and purpose; none is interchangeable ground truth for a 20 m map.

## Area measurement
Compute area on a projected grid (UTM zone of the AOI) or with geodesic pixel areas; never count pixels on a latitude/longitude grid as equal. Apply and report a minimum mapping unit. Area under cloud is unobserved, not unburned.
