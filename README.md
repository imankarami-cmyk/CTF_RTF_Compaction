================================================================================
SOIL COMPACTION ANALYSIS -- FULL PIPELINE
Iman Karami | Institut fur Landtechnik | Season 2025
================================================================================

PURPOSE
-------
This folder contains all Python scripts used to produce the analysis and
figures in the report:
  "Soil Compaction Analysis Using LiDAR Point Clouds and GPS Wheel-Track Data
   -- Comparing Controlled Traffic Farming and Random Traffic Farming"

Three fields were analysed:
  F1  Flaeche 1  CTF  ~1337 m2 sub-AOI
  F2  Flaeche 2  RTF  ~1326 m2 sub-AOI
  F3  Flaeche 3  CTF  ~1310 m2 sub-AOI

Primary comparison: F2 (RTF) vs F3 (CTF) -- geographically close fields.
F1 (CTF) is the second CTF reference field.

================================================================================
COMPLETE PIPELINE -- RUN SCRIPTS IN THIS ORDER
================================================================================

STEP 0a --> STEP 0b --> STEP 1 --> STEP 2 (optional) --> STEP 3 (optional)

   GPS raw data (.gpkg)
        |
        v
  [STEP 0a]  ctf_wheel_polygons_Klein_Gross.py     (Fendt 728 + Fendt 724)
  [STEP 0b]  ctf_wheel_polygons_third_tractor.py   (third tractor / F2)
        |
        v
  Wheel-track polygon shapefiles (.shp / .gpkg)
        |
        v
  [STEP 1]   ctf_rtf_analysis.py                   (main analysis + all report figures)
        |
        v
  [STEP 2]   ctf_lane_analysis.py                  (optional: GPS accuracy + TCI)
  [STEP 3]   rut_profile_analysis.py               (optional: rut cross-sections)


================================================================================
STEP 0a: ctf_wheel_polygons_Klein_Gross.py
STEP 0b: ctf_wheel_polygons_third_tractor.py
================================================================================
WHAT THESE DO
-------------
These two scripts convert raw GPS point data (exported from the tractor ISOBUS
system as .gpkg files) into wheel-track polygon shapefiles.

Each GPS point records where the tractor centre was at one moment in time.
The script converts each consecutive pair of GPS points into two polygons --
one for the left wheel and one for the right wheel -- using the known tractor
dimensions (tyre width and outer track width).

Each polygon represents the exact ground area covered by one wheel during the
movement between two GPS positions. The polygons are what every downstream
script uses as input.

WHICH SCRIPT FOR WHICH TRACTOR
-------------------------------
ctf_wheel_polygons_Klein_Gross.py  -->  Fendt 728 Vario (Gross) and Fendt 724 Vario (Klein)
                                        GPS data format A:
                                        Has GroundBasedMachineSpeed, CourseOverGround (radians)
                                        Run once with POLYGON_SOURCE = "gross" for Fendt 728
                                        Run again with POLYGON_SOURCE = "klein" for Fendt 724

ctf_wheel_polygons_third_tractor.py -->  Third tractor / F2 RTF tractor
                                         GPS data format B:
                                         Has Latitude, Longitude, DateTime only
                                         Course and speed are COMPUTED from position differences
                                         and smoothed with rolling median before polygon building

TRACTOR DIMENSIONS IN THE CONFIG BLOCK
---------------------------------------
Change these at the top of the script before running:

  Fendt 728 (Gross):    AUSSENBREITE = 2.70 m   REIFENBREITE = 0.70 m
  Fendt 724 (Klein):    AUSSENBREITE = 2.40 m   REIFENBREITE = 0.60 m
  Third tractor (F2):   AUSSENBREITE = 2.21 m   REIFENBREITE = 0.48 m

  AUSSENBREITE = outer width (Spur + Reifenbreite = centre-to-centre + one tyre)
  REIFENBREITE = rear tyre width (Hinterreifen)

  Note: the AUSSENBREITE in the config block for the third tractor script
  was originally 2.21 m but has been corrected to match actual measurements.

HOW THE POLYGON GEOMETRY WORKS
--------------------------------
For each GPS point the script computes the four corners of the tractor's
footprint by rotating the tractor's outer width around its centre position
using the CourseOverGround angle. The left wheel polygon uses corners p1-p2
(left outer and left inner) and the right wheel polygon uses p3-p4 (right
inner and right outer). Each polygon spans from the previous GPS position to
the current one, creating a continuous tiled coverage along the route.

OUTPUT FIELDS IN THE SHAPEFILE
--------------------------------
  PolySrc      Tractor identifier ("gross", "klein", "third", etc.)
  Wheel        Side ("left" or "right")
  RouteDate    Operation date in DD.MM.YYYY format
  RouteISO     Operation date in YYYY-MM-DD format (for sorting)
  Direction    Course over ground in degrees
  COG_rad      Course over ground in radians
  Speed_kmh    Ground speed in km/h
  PtIndex      Index of the GPS point in the source file
  + additional ISOBUS telemetry fields (EngSpeed, EngFuelRate, etc.)

HOW TO RUN
-----------
1. Edit the CONFIG block at the top of the script:
     INPUT_FOLDER    = path to folder containing raw .gpkg GPS files
     OUTPUT_FOLDER   = where to save the output .shp and .gpkg
     OUTPUT_BASENAME = filename prefix for output files
     POLYGON_SOURCE  = label for PolySrc field (e.g. "gross", "klein", "third")
     AUSSENBREITE    = outer width for this tractor [m]
     REIFENBREITE    = tyre width for this tractor [m]

2. Run:
     python ctf_wheel_polygons_Klein_Gross.py   (for Gross and Klein)
     python ctf_wheel_polygons_third_tractor.py (for third tractor / F2)

3. After both scripts finish, clip the output polygons to each field AOI in
   QGIS (or use geopandas intersection) to remove passes outside the field
   boundary before using them in STEP 1.

   The clipped files used in STEP 1 are:
     F1:  f1_traffic_clipped1.shp   (Gross + Klein passes combined)
     F2:  f2_traffic_clipped1.shp   (third tractor passes)
     F3:  f3_traffic_clipped222.shp (Gross + Klein passes combined)

INSTALL
--------
  pip install geopandas shapely tqdm pandas numpy fiona pyogrio


================================================================================
STEP 1: ctf_rtf_analysis.py  [MAIN ANALYSIS -- RUN THIS FIRST]
================================================================================
WHAT IT DOES
-------------
The primary analysis script. Takes the clipped polygon shapefiles from STEP 0
and the LiDAR point clouds as inputs. Produces all the maps, charts, and CSV
files used in the report.

Analysis for each field:
  - Pass-count raster: how many wheel passes each 10 cm soil pixel received
  - Inter-Row Reference DTM: local terrain model built from undisturbed soil
    between the wheel tracks (2 m smoothing window for all three fields)
  - Rut depth raster: minimum LiDAR elevation minus reference elevation
  - Rut depth class raster: very deep / deep / moderate / shallow
  - Residual histogram: confirms reference surface quality (GOOD if std < 3 cm)
  - Rut location proof map: deepest LiDAR points overlaid on wheel polygons
    (proves all detections are inside the wheel tracks, not random noise)
  - 4-panel diagnostic map: reference zones, reference DTM, depth map, depth class

F2 vs F3 focused comparison outputs:
  - Side-by-side pass maps at same colour scale
  - Side-by-side depth maps at same colour scale
  - Overlaid depth histograms + cumulative curves
  - Depth class stacked bar chart
  - 6-panel key metrics bar chart (F2 and F3 primary, F1 for context)

All-field comparison outputs:
  - Coverage bar charts
  - Pass-count stacked bar charts
  - Depth class stacked bar charts
  - Soil displacement bar charts (3 normalisations)
  - Summary CSV with all key statistics

OUTPUTS PER FIELD (saved to F1/analysis_sub/, F2/analysis_sub/, F3/analysis_sub/)
  *_pass_count.tif           Pass-count raster (uint16)
  *_pass_class.tif           Pass classified (0=none, 1=1-2, 2=3-5, 3=6+)
  *_pass_class_areas.csv     Area and % of field in each pass class
  *_interrow_mask.tif        Reference zone mask
  *_ref_dtm.tif              Smoothed reference DTM at 50 cm
  *_ref_fine.tif             Reference DTM resampled to 10 cm
  *_depth_full.tif           Full-field depth raster (negative = rut)
  *_depth_ruts.tif           Depth clipped to wheel footprint, ruts only
  *_depth_class.tif          Depth classified (1=very deep...4=shallow)
  *_depth_class_areas.csv    Area and % of rut area in each depth class
  *_rut_points.gpkg          Deepest LiDAR point locations (for QGIS)
  *_rut_proof_map.png        Rut location proof map
  *_diagnostic_4panel.png    4-panel diagnostic map
  *_residual_histogram.png   Reference quality histogram

F2/F3 comparison (saved to COMPARISON_SUB/F2_F3_comparison/)
  F2_F3_pass_comparison.png
  F2_F3_depth_comparison.png
  F2_F3_depth_distribution.png
  F2_F3_depth_class.png
  F2_F3_key_metrics.png

All-field comparison (saved to COMPARISON_SUB/)
  comparison_coverage.png
  comparison_pass_class_stacked.png
  comparison_depth_class_stacked.png
  comparison_rut_depth.png
  comparison_displacement.png
  summary_all_fields.csv
  pass_class_areas_all.csv
  depth_class_areas_all.csv

HOW TO RUN
-----------
1. Edit the FIELDS config block at the top:
     Each field needs: aoi (.shp), traffic_shps (list of .shp), las_dir, out_dir
2. Edit COMPARISON_OUT to your desired comparison folder
3. Run:  python ctf_rtf_analysis.py

KEY SETTINGS
-------------
  PIXEL_SIZE     = 0.10 m   fine depth raster resolution
  REF_PIXEL_SIZE = 0.50 m   reference DTM resolution
  TRACK_BUFFER_M = 0.20 m   buffer when dissolving polygons
  SMOOTH_WIN     = 4 px     2 m window -- same for ALL fields
                            Chosen by sensitivity analysis (see Table 1 in report):
                              F1: best at 2 m (std=1.34 cm) vs 5 m (1.78) vs 10 m (2.24)
                              F2: best at 2 m (std=1.23 cm) vs 5 m (2.03) vs 10 m (2.84)
                              F3: best at 2 m (std=1.41 cm) vs 5 m (1.85) vs 10 m (2.35)


================================================================================
STEP 2: ctf_lane_analysis.py  [OPTIONAL -- CTF GPS ACCURACY ANALYSIS]
================================================================================
WHAT IT DOES
-------------
Additional analysis for the CTF fields only (F1 and F3). NOT used in the
main report but useful for understanding GPS guidance performance.

Answers four questions:
  1. How wide is each tractor's confirmed wheel track across the season?
     (GPS navigation accuracy -- confirmed lane width vs nominal tyre width)
  2. How well do the two tractors follow the same permanent lanes?
     (Track Confinement Index: overlap area / union area)
  3. Does the heavier Fendt 728 cause deeper ruts than the Fendt 724?
     (Load-class attribution by tractor-wheel zone)
  4. Which field operation deviated most from the permanent lanes?
     (Off-route detection per operation date)

WHY NOT IN THE REPORT
----------------------
These four questions were addressed in an earlier version of the report.
The final report replaced sections 4.5-4.7 (GPS accuracy, tractor attribution,
operation off-route) with the F2/F3 focused comparison (Section 4.6 in final
report), because the reviewer asked for the F2/F3 paired comparison as the
primary result. The GPS accuracy and tractor attribution analysis remains
valid supplementary information but is not the focus of the report.

INPUTS
-------
  Requires the depth rasters from STEP 1 (*_depth_ruts.tif)
  Requires the traffic shapefiles from STEP 0

OUTPUTS (saved to F1/tractor_alignment/, F3/tractor_alignment/)
  *_pass_count_per_tractor.png    Pass maps for Fendt 728 and Fendt 724
  *_track_confinement_map.png     Overlap zones: shared / Gross-only / Klein-only
  *_depth_by_tractor.png          Rut depth in each tractor zone
  *_best_vs_worst_operation.png   GPS accuracy: best vs worst operation date
  *_alignment_summary.csv         Key metrics summary
  *_operation_offroute.csv        Off-route % per operation date

HOW TO RUN
-----------
1. Run STEP 1 first so the depth rasters exist
2. Edit CTF_FIELDS config in this script (F1 and F3 only)
3. Run:  python ctf_lane_analysis.py


================================================================================
STEP 3: rut_profile_analysis.py  [OPTIONAL -- RUT CROSS-SECTION PROFILES]
================================================================================
WHAT IT DOES
-------------
Extracts cross-section profiles perpendicular to each dissolved CTF/RTF lane
and measures rut depth and lateral offset along the route.

Key difference from STEP 1:
  STEP 1 uses individual wheel polygons (one pass of one tractor = one polygon)
  STEP 3 dissolves ALL polygons per field into cumulative lane geometries
  (all passes of the whole season combined) then samples cross-sections.
  This gives the true cumulative rut depth after a full season, which is
  what the LiDAR scan at end of season actually measures.

OUTPUTS per field (saved to F1/rut_profile/, etc.)
  *_dissolved_lanes.gpkg            Dissolved permanent lane geometries (QGIS)
  *_rut_centreline.gpkg             Rut centreline (connects deepest points)
  *_cross_section_stats.csv         Per-cross-section depth and offset
  *_lane_profiles.csv               Summary per lane
  *_fig1_cross_section_sample.png   Example cross-section profiles
  *_fig2_along_lane_depth.png       Depth along each lane
  *_fig3_rut_offset_boxplot.png     Lateral offset boxplot
  *_fig4_depth_map.png              Depth map with rut centreline

HOW TO RUN
-----------
1. Run STEP 1 first (needs depth rasters)
2. Edit FIELDS config to match your paths
3. Run:  python rut_profile_analysis.py


================================================================================
DATA REQUIREMENTS
================================================================================

INPUT DATA FOR STEP 0 (polygon building)
  Raw GPS .gpkg files exported from ISOBUS system
  One .gpkg per operation date, one folder per tractor
  Required columns (Format A -- Gross/Klein):
    CourseOverGround       (radians)
    GroundBasedMachineSpeed (m/s)
    GroundBasedMachineDirection
  Required columns (Format B -- third tractor):
    Latitude, Longitude, DateTime (or Date + Time separately)

INPUT DATA FOR STEP 1 (main analysis)
  Per field:
    AOI shapefile                -- boundary of sub-AOI analysis zone
    Traffic shapefile(s)         -- clipped wheel-track polygons from STEP 0
                                    Required columns: PolySrc, Wheel, RouteDate
    LAS/LAZ folder               -- LiDAR point clouds
                                    Ground returns classified as class 2

EXPECTED FOLDER STRUCTURE
  Analysing_full/
    F1/
      AOI_F1_sub.shp
      f1_traffic_clipped1.shp
      las/
        *.las or *.laz
      analysis_sub/         <- created by STEP 1
      tractor_alignment/    <- created by STEP 2
      rut_profile/          <- created by STEP 3
    F2/
      AOI_F2_sub.shp
      f2_traffic_clipped1.shp
      las/
    F3/
      AOI_F3_sub.shp
      f3_traffic_clipped222.shp
      las/
    COMPARISON_SUB/         <- created by STEP 1
      F2_F3_comparison/
    GPS-points_of_tractors/
      INPUT/
        gross/              <- .gpkg files for Fendt 728
        klein/              <- .gpkg files for Fendt 724
        3rd_tractor/        <- .gpkg files for F2 tractor
      OUTPUT/
        gross/
        klein/
        3rd_tractor/


================================================================================
PYTHON ENVIRONMENT
================================================================================

Python version: 3.13
Required packages (all scripts combined):

  laspy           LAS/LAZ point cloud reading             (STEP 1, 3)
  geopandas       Vector polygon operations               (all steps)
  shapely         Geometry operations                     (all steps)
  rasterio        Raster read/write (GeoTIFF)             (STEP 1, 2, 3)
  numpy           Array mathematics                       (all steps)
  pandas          CSV and tabular data                    (all steps)
  scipy           Spatial filtering                       (STEP 1, 2, 3)
  matplotlib      Plots and maps                          (STEP 1, 2, 3)
  tqdm            Progress bars                           (STEP 0, 1, 3)
  fiona           Shapefile I/O backend                   (STEP 0)
  pyogrio         Fast vector I/O backend                 (STEP 0)
  pyproj          Coordinate system transforms            (STEP 0b)

Install everything:
  pip install laspy geopandas shapely rasterio numpy pandas scipy matplotlib tqdm fiona pyogrio pyproj

Coordinate system: ETRS89 / UTM zone 33N (EPSG:32633)
All data are reprojected to this CRS inside the scripts.


================================================================================
KEY METHODOLOGICAL DECISIONS AND JUSTIFICATIONS
================================================================================

1. TRACTOR DIMENSIONS (used in STEP 0)
   Fendt 728 (Gross):  Hinterreifen=0.70 m, Spur=2.00 m, AUSSENBREITE=2.70 m
   Fendt 724 (Klein):  Hinterreifen=0.60 m, Spur=1.80 m, AUSSENBREITE=2.40 m
   Third tractor (F2): Hinterreifen=0.48 m, Spur=1.73 m, AUSSENBREITE=2.21 m
   AUSSENBREITE = Spur + Hinterreifen (centre-to-centre + one full tyre width)

2. SMOOTHING WINDOW = 2 m for ALL THREE FIELDS (used in STEP 1)
   Sensitivity analysis results:
     F1:  2m=1.34 cm  5m=1.78 cm  10m=2.24 cm  25m=3.10 cm  --> use 2 m
     F2:  2m=1.23 cm  5m=2.03 cm  10m=2.84 cm  20m=5.01 cm  --> use 2 m
     F3:  2m=1.41 cm  5m=1.85 cm  10m=2.35 cm  25m=3.22 cm  --> use 2 m
   The 2 m window was best for all three fields. Using the SAME window for
   all fields is essential -- different windows produce different depth values
   independent of actual compaction, making comparison invalid.

3. EQUAL-AREA SUB-AOIs (~1330 m2 each)
   Full field AOIs are different sizes (1.01, 0.49, 0.41 ha).
   Equal-area sub-AOIs make percentage-based comparisons directly valid.

4. F2/F3 AS PRIMARY COMPARISON
   F2 and F3 are geographically close, same landscape, similar field conditions.
   Any measured difference between them is attributable to traffic management
   (RTF vs CTF) rather than location or soil variability.
   F1 is included as a second CTF reference for context.

5. RUT PROOF MAPS (Method A, STEP 1)
   Validates that all deepest LiDAR returns fall inside wheel-track polygon
   boundaries before reporting depth values. Ensures measurements reflect real
   wheel-induced compaction rather than terrain noise.

6. DISSOLVED LANES vs INDIVIDUAL POLYGONS (STEP 3)
   Individual polygons = one pass = one day = underestimates cumulative depth
   Dissolved lanes = all passes + all tractors + whole season = true cumulative
   The LiDAR scan at end of season measures cumulative compaction so dissolved
   lanes are the correct analysis unit for seasonal rut depth.


================================================================================
OUTPUTS USED IN REPORT
================================================================================

Section                     Script      Output file
------------------------    --------    ----------------------------------------
4.1  Coverage               STEP 1      pass_class_areas_all.csv
                                        summary_all_fields.csv
4.2  Permanent CTF lanes    STEP 1      F1_CTF_pass_map.png
                                        F2_RTF_pass_map.png
                                        F3_CTF_pass_map.png
4.3  Reference quality      STEP 1      F1_CTF_residual_histogram.png
                                        F2_RTF_residual_histogram.png
                                        F3_CTF_residual_histogram.png
4.4  Rut validation         STEP 1      F1_CTF_rut_proof_map.png
                                        F2_RTF_rut_proof_map.png
                                        F3_CTF_rut_proof_map.png
4.5  Rut depth              STEP 1      F1_CTF_diagnostic_4panel.png
                                        F2_RTF_diagnostic_4panel.png
                                        F3_CTF_diagnostic_4panel.png
                                        summary_all_fields.csv
4.6  F2 vs F3 comparison    STEP 1      F2_F3_pass_comparison.png
                                        F2_F3_depth_comparison.png
                                        F2_F3_depth_distribution.png
                                        F2_F3_depth_class.png
                                        F2_F3_key_metrics.png
4.7  Depth class            STEP 1      depth_class_areas_all.csv
4.8  Displacement           STEP 1      summary_all_fields.csv
5.   All-field comparison   STEP 1      comparison_coverage.png
                                        comparison_pass_class_stacked.png
                                        comparison_depth_class_stacked.png
                                        comparison_displacement.png


================================================================================
SCRIPTS NOT USED IN THIS REPORT
================================================================================


ctf_lane_analysis.py
  GPS accuracy and tractor attribution analysis (STEP 2 above).
  Was used in an earlier draft of the report (Sections 4.5-4.7: GPS accuracy,
  Track Confinement Index, operation off-route detection). Replaced in the
  final report by the F2/F3 focused comparison at the reviewer's request.
  Results are still valid supplementary information.

traffic_analysis_Full.py
  Older version of ctf_rtf_analysis.py using:
    - Full field AOIs (not equal-area sub-AOIs)
    - SMOOTH_WIN=40 for F2 (before sensitivity analysis showed 2 m is better)
    - No F2/F3 focused comparison
    - No rut proof maps
  Kept for reproducibility of earlier results only.

================================================================================
END OF README
================================================================================