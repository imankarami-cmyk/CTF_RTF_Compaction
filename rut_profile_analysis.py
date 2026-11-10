3.13
"""
ctf_rtf_analysis.py
====================
Unified CTF vs RTF traffic and rut-depth analysis.
F1 (CTF), F2 (RTF), F3 (CTF).

DEPTH METHOD: same for ALL THREE FIELDS
----------------------------------------
Dissolved-footprint inter-row reference DTM:
  1. All wheel polygons dissolved + buffered into one compacted footprint.
  2. LiDAR points OUTSIDE that footprint = undisturbed inter-row soil.
  3. Reference DTM built from those inter-row points (mean Z, then smoothed).
     The smoother bridges across compacted strips by interpolating from
     the reference strips on each side.
  4. depth = min_Z_inside_footprint - interpolated_reference_Z
     (negative = below undisturbed soil = rut)

For F2 (RTF) the dissolved footprint covers more of the field (random traffic)
so the smoothing window is set wider (SMOOTH_WIN_RTF) to bridge larger gaps.
This is scientifically valid: same method, same question, wider neighbourhood.

ANALYSES PRODUCED
-----------------
Per field:
  - Coverage % (area under wheel / field area)
  - Pass-count raster + class map
  - Reference DTM (inter-row local terrain)
  - Depth raster + depth class raster
  - 4-panel diagnostic map
  - Residual histogram (reference quality check)
  - Per-field stats CSV

Cross-field comparison (one plot per metric, all three fields side by side):
  - % field under wheel
  - Depth class area breakdown (stacked bar)
  - Mean / p10 / deepest rut depth
  - Soil displacement proxy (m3/ha)

Run:
    python ctf_rtf_analysis.py
"""

import os
import glob
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import laspy
import rasterio
from rasterio.transform import from_origin
from rasterio.features import rasterize
from scipy.ndimage import uniform_filter, distance_transform_edt, zoom as nd_zoom
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
from tqdm import tqdm

warnings.filterwarnings("ignore")


# =========================================================
#  CONFIGURATION -- EDIT THIS BLOCK
# =========================================================

BASE = r"C:\Users\imankarami\Data\Analysing_full"

FIELDS = {

    "F2": {
        "method":       "RTF",
        "aoi":          os.path.join(BASE, "F2", "AOI_F21.shp"),
        "traffic_shps": [os.path.join(BASE, "F2", "f2_traffic_clipped1.shp")],
        "las_dir":      os.path.join(BASE, "F2", "las"),
        "out_dir":      os.path.join(BASE, "F2", "analysis30_7_2"),
    },

}

COMPARISON_OUT = os.path.join(BASE, "COMPARISON")
TARGET_EPSG    = 32633

# Raster resolutions
PIXEL_SIZE     = 0.10   # m  fine depth / pass raster
REF_PIXEL_SIZE = 0.50   # m  coarse reference DTM

# Track dissolve buffer -- merges adjacent polygons whose gap < 2*this
TRACK_BUFFER_M = 0.20   # m

# Smoothing window for reference DTM (in reference pixels at REF_PIXEL_SIZE)
#
# WHAT THE WINDOW DOES:
#   The reference DTM is built from LiDAR points in inter-row (untrafficked)
#   pixels. Inside the wheel footprint those pixels are NaN. The smoother
#   bridges across the NaN gaps by averaging from neighbouring inter-row pixels.
#   Window must be wide enough to bridge the widest compacted strip,
#   but narrow enough to follow the real terrain without blurring hills.
#
# HOW TO CHOOSE THE WINDOW (run f3_debug_diagnostic.py first):
#   The correct window = smallest residual std at inter-row points.
#   From the sensitivity analysis:
#     F3 CTF: win=4 px (2 m) gives residual std=1.47 cm  -> use 4
#     F1 CTF: run the same diagnostic; likely similar to F3
#     F2 RTF: wider footprint needs wider window; 40 px (20 m) is a starting point
#
# Rule of thumb:
#   window_px > (widest_compacted_strip_m / REF_PIXEL_SIZE)
#   For CTF with ~1.4 m strips: window > 1.4/0.5 = 2.8 px -> use 4
#   For RTF with wider gaps:    window > 5/0.5   = 10 px -> use 40
SMOOTH_WIN = {
    "F1": 4,    # px = 4 * 0.5 = 2 m  (CTF narrow strips, best from sensitivity)
    "F2": 2,   # px = 40 * 0.5 = 20 m (RTF wide coverage, wider window needed)
    "F3": 4,    # px = 4 * 0.5 = 2 m  (CTF, confirmed best from sensitivity analysis)
}

# Depth classes (m, negative = deeper than reference)
CLASS_BREAKS = [-999.0, -0.10, -0.05, -0.02, 0.0]
CLASS_LABELS = {
    1: "< -10 cm  (very deep)",
    2: "-10 to -5 cm  (deep)",
    3: "-5 to -2 cm  (moderate)",
    4: "-2 to 0 cm  (shallow)",
}
CLASS_COLORS = ["#7f0000", "#d7301f", "#fc8d59", "#fee8c8"]

# Pass-count classes
PASS_BREAKS = [0, 1, 3, 6]
PASS_LABELS = {0: "no traffic", 1: "1-2 passes",
               2: "3-5 passes",  3: "6+ passes"}
PASS_COLORS = ["#d9d9d9", "#fee08b", "#f46d43", "#a50026"]

# Field colours for comparison plots
FIELD_COLORS = {"F1": "#2166ac", "F2": "#d6604d", "F3": "#4dac26"}


# =========================================================
#  HELPERS
# =========================================================

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def find_las(folder):
    files = sorted(set(
        glob.glob(os.path.join(folder, "*.las")) +
        glob.glob(os.path.join(folder, "*.laz")) +
        glob.glob(os.path.join(folder, "**", "*.las"), recursive=True) +
        glob.glob(os.path.join(folder, "**", "*.laz"), recursive=True)
    ))
    if not files:
        raise FileNotFoundError(
            f"No LAS/LAZ files found under: {folder}\n"
            "Update las_dir in FIELDS config.")
    return files


def load_las(folder, aoi_union):
    files = find_las(folder)
    print(f"    {len(files)} LAS/LAZ file(s)")
    minx, miny, maxx, maxy = aoi_union.bounds
    xs, ys, zs, cs = [], [], [], []
    for fp in files:
        las = laspy.read(fp)
        x = np.asarray(las.x); y = np.asarray(las.y); z = np.asarray(las.z)
        c = (np.asarray(las.classification)
             if hasattr(las, "classification")
             else np.zeros(len(x), dtype=np.uint8))
        bb = (x>=minx)&(x<=maxx)&(y>=miny)&(y<=maxy)&np.isfinite(z)
        xs.append(x[bb]); ys.append(y[bb])
        zs.append(z[bb]); cs.append(c[bb])
    x=np.concatenate(xs); y=np.concatenate(ys)
    z=np.concatenate(zs); c=np.concatenate(cs)
    print(f"    {len(x):,} points loaded")
    return x, y, z, c


def load_traffic(paths, aoi_union, epsg):
    parts = [gpd.read_file(p).to_crs(epsg=epsg) for p in paths]
    gdf = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True),
                           crs=f"EPSG:{epsg}")
    gdf["geometry"] = gdf.buffer(0)
    gdf = gdf[gdf.geometry.notnull() & ~gdf.geometry.is_empty].copy()
    gdf = gdf[gdf.intersects(aoi_union)].copy().reset_index(drop=True)
    print(f"    {len(gdf):,} wheel polygons")
    return gdf


def build_grid(bounds, px):
    minx, miny, maxx, maxy = bounds
    w = int(np.ceil((maxx-minx)/px))
    h = int(np.ceil((maxy-miny)/px))
    return w, h, from_origin(minx, maxy, px, px)


def rast_mask(geom, shp, transform):
    return rasterize([(geom,1)], out_shape=shp,
                     transform=transform, fill=0, dtype="uint8").astype(bool)


def save_tif(arr, transform, epsg, path, dtype="float32", nodata=np.nan):
    ensure_dir(os.path.dirname(path))
    with rasterio.open(path, "w", driver="GTiff",
                       height=arr.shape[0], width=arr.shape[1],
                       count=1, dtype=dtype, crs=f"EPSG:{epsg}",
                       transform=transform, nodata=nodata) as dst:
        dst.write(arr.astype(dtype), 1)


def pts_to_mean_raster(x, y, z, bounds, px):
    w, h, t = build_grid(bounds, px)
    minx, _, _, maxy = bounds
    c = np.floor((x-minx)/px).astype(int)
    r = np.floor((maxy-y)/px).astype(int)
    ok = (c>=0)&(c<w)&(r>=0)&(r<h)&np.isfinite(z)
    flat = r[ok]*w + c[ok]; v = z[ok]
    S = np.zeros(h*w, dtype=np.float64)
    N = np.zeros(h*w, dtype=np.int32)
    np.add.at(S, flat, v); np.add.at(N, flat, 1)
    out = np.full(h*w, np.nan, dtype="float32")
    hit = N>0; out[hit] = (S[hit]/N[hit]).astype("float32")
    return out.reshape((h,w)), t


def pts_to_min_raster(x, y, z, bounds, px):
    w, h, t = build_grid(bounds, px)
    minx, _, _, maxy = bounds
    c = np.floor((x-minx)/px).astype(int)
    r = np.floor((maxy-y)/px).astype(int)
    ok = (c>=0)&(c<w)&(r>=0)&(r<h)&np.isfinite(z)
    flat = r[ok]*w + c[ok]; v = z[ok]
    order = np.lexsort((v, flat)); fs, vs = flat[order], v[order]
    uf, fi = np.unique(fs, return_index=True)
    out = np.full(h*w, np.nan, dtype="float32")
    out[uf] = vs[fi].astype("float32")
    return out.reshape((h,w)), t


def smooth_fill(arr, win):
    """Weighted uniform filter that correctly handles NaN borders."""
    valid  = np.isfinite(arr).astype("float32")
    filled = np.where(np.isfinite(arr), arr, 0.0).astype("float32")
    sv = uniform_filter(filled, size=win, mode="reflect")
    sw = uniform_filter(valid,  size=win, mode="reflect")
    with np.errstate(invalid="ignore", divide="ignore"):
        result = np.where(sw>0, sv/sw, np.nan)
    still = ~np.isfinite(result)
    if still.any():
        sv2 = uniform_filter(filled, size=win*3, mode="reflect")
        sw2 = uniform_filter(valid,  size=win*3, mode="reflect")
        with np.errstate(invalid="ignore", divide="ignore"):
            fb = np.where(sw2>0, sv2/sw2, np.nan)
        result = np.where(still, fb, result)
    return result.astype("float32")


def resample_to(src, dst_h, dst_w):
    """Bilinear zoom with NaN-safe nearest-neighbour pre-fill."""
    nan_mask = ~np.isfinite(src)
    if nan_mask.all():
        return np.full((dst_h, dst_w), np.nan, dtype="float32")
    if nan_mask.any():
        idx = distance_transform_edt(nan_mask,
                                     return_distances=False, return_indices=True)
        filled = src[idx[0], idx[1]].astype("float32")
    else:
        filled = src.astype("float32")
    zy = dst_h/src.shape[0]; zx = dst_w/src.shape[1]
    zoomed     = nd_zoom(filled,                     (zy,zx), order=1, mode="nearest")
    nan_zoomed = nd_zoom(nan_mask.astype("float32"), (zy,zx), order=1, mode="nearest")
    zoomed[nan_zoomed > 0.5] = np.nan
    out = np.full((dst_h, dst_w), np.nan, dtype="float32")
    ry = min(zoomed.shape[0], dst_h); rx = min(zoomed.shape[1], dst_w)
    out[:ry, :rx] = zoomed[:ry, :rx]
    return out


def classify_depth(d):
    out = np.zeros_like(d, dtype=np.uint8)
    for i,(lo,hi) in enumerate(zip(CLASS_BREAKS[:-1], CLASS_BREAKS[1:]), 1):
        out[(d>=lo)&(d<hi)&np.isfinite(d)] = i
    return out


def classify_pass(p):
    out = np.zeros_like(p, dtype=np.uint8)
    out[(p>=1)&(p<=2)] = 1
    out[(p>=3)&(p<=5)] = 2
    out[p>=6]          = 3
    return out


def legend_outside(ax, handles=None, labels=None, title=None, loc="upper left"):
    """Place legend to the right of the axes, outside the plot area."""
    kw = dict(bbox_to_anchor=(1.01, 1), loc=loc, borderaxespad=0, fontsize=8)
    if title:
        kw["title"] = title
    if handles is not None:
        ax.legend(handles=handles, labels=labels, **kw)
    else:
        ax.legend(**kw)


# =========================================================
#  PER-FIELD ANALYSIS
# =========================================================

def analyse_field(fid, cfg):
    method  = cfg["method"]
    out_dir = cfg["out_dir"]
    ensure_dir(out_dir)
    pfx = f"{fid}_{method}"
    smooth_win = SMOOTH_WIN.get(fid, (40 if method == "RTF" else 4))

    print(f"\n{'='*60}\n  {fid}  ({method})\n{'='*60}")

    # -- AOI --------------------------------------------------
    aoi = gpd.read_file(cfg["aoi"]).to_crs(epsg=TARGET_EPSG)
    aoi["geometry"] = aoi.buffer(0)
    aoi = aoi[aoi.geometry.notnull() & ~aoi.geometry.is_empty].copy()
    aoi_union   = aoi.union_all()
    aoi_area_m2 = aoi_union.area
    bounds      = aoi_union.bounds
    minx, miny, maxx, maxy = bounds

    # -- Traffic ----------------------------------------------
    print("  Traffic polygons ...")
    traffic = load_traffic(cfg["traffic_shps"], aoi_union, TARGET_EPSG)

    # =========================================================
    #  A. COVERAGE & PASS-COUNT
    # =========================================================
    # Dissolve for coverage (no buffer -- exact wheel area)
    print("  Coverage ...")
    tr_union_exact = traffic.geometry.union_all().intersection(aoi_union)
    trafficked_area = tr_union_exact.area
    pct_trafficked  = 100 * trafficked_area / aoi_area_m2

    # Pass-count raster
    dep_w, dep_h, dep_transform = build_grid(bounds, PIXEL_SIZE)
    pass_r = np.zeros((dep_h, dep_w), dtype=np.uint16)
    for geom in tqdm(traffic.geometry, desc="  Pass raster", leave=False):
        pass_r += rasterize([(geom,1)], out_shape=(dep_h,dep_w),
                            transform=dep_transform, fill=0, dtype="uint16")
    aoi_mask_dep = rast_mask(aoi_union, (dep_h,dep_w), dep_transform)
    pass_r[~aoi_mask_dep] = 0
    pass_class = classify_pass(pass_r)

    save_tif(pass_r,     dep_transform, TARGET_EPSG,
             os.path.join(out_dir, f"{pfx}_pass_count.tif"), dtype="uint16", nodata=0)
    save_tif(pass_class, dep_transform, TARGET_EPSG,
             os.path.join(out_dir, f"{pfx}_pass_class.tif"), dtype="uint8", nodata=0)

    # Pass-class areas -- count ONLY pixels inside the AOI polygon
    # (not the full bounding-box raster which is larger than the AOI)
    pa = PIXEL_SIZE**2
    aoi_px = int(aoi_mask_dep.sum())  # total AOI pixels = true field area denominator
    pass_class_rows = []
    for k, lab in PASS_LABELS.items():
        n = int(((pass_class == k) & aoi_mask_dep).sum())
        pass_class_rows.append({
            "fid": fid, "method": method,
            "pass_class": k, "pass_label": lab,
            "area_m2":  n * pa,
            "area_ha":  n * pa / 10000,
            "pct_field": 100 * n / aoi_px if aoi_px > 0 else np.nan,
        })
    pass_df = pd.DataFrame(pass_class_rows)

    # Sanity check -- all pass classes should sum to 100%
    total_pct = pass_df["pct_field"].sum()
    if abs(total_pct - 100) > 1.0:
        print(f"  WARNING: pass-class pct sum = {total_pct:.1f}% "
              f"(expected ~100%). Check AOI mask.")
    else:
        print(f"  Pass-class pct sum = {total_pct:.1f}%  [OK]")

    # =========================================================
    #  B. DISSOLVED FOOTPRINT FOR DEPTH REFERENCE
    # =========================================================
    print(f"  Dissolving footprint (buffer={TRACK_BUFFER_M} m) ...")
    dissolved = traffic.geometry.buffer(TRACK_BUFFER_M).union_all()\
                       .intersection(aoi_union)
    dissolved_area = dissolved.area
    pct_diss = 100 * dissolved_area / aoi_area_m2

    gpd.GeoDataFrame({"geometry": [dissolved]},
                     crs=f"EPSG:{TARGET_EPSG}").to_file(
        os.path.join(out_dir, f"{pfx}_dissolved_footprint.gpkg"),
        driver="GPKG")

    ref_w, ref_h, ref_transform = build_grid(bounds, REF_PIXEL_SIZE)
    diss_mask_ref = rast_mask(dissolved, (ref_h,ref_w), ref_transform)
    ir_mask_ref   = rast_mask(aoi_union, (ref_h,ref_w), ref_transform) & ~diss_mask_ref

    # Save inter-row mask (green=reference, red=compacted)
    save_tif(ir_mask_ref.astype("uint8"), ref_transform, TARGET_EPSG,
             os.path.join(out_dir, f"{pfx}_interrow_mask.tif"),
             dtype="uint8", nodata=255)

    # Strip width check
    dist_px = distance_transform_edt(ir_mask_ref) * REF_PIXEL_SIZE
    ir_dists = dist_px[ir_mask_ref]
    min_sw   = float(np.min(ir_dists)*2)   if len(ir_dists) else 0
    med_sw   = float(np.median(ir_dists)*2) if len(ir_dists) else 0
    sw_m     = smooth_win * REF_PIXEL_SIZE
    print(f"    Compacted footprint: {pct_diss:.1f}%  |  "
          f"Inter-row: {100-pct_diss:.1f}%")
    print(f"    Narrowest strip: {min_sw:.2f} m  |  Median: {med_sw:.2f} m  |  "
          f"Smooth window: {sw_m:.1f} m")

    # =========================================================
    #  C. LiDAR -- REFERENCE DTM + DEPTH RASTER
    # =========================================================
    print("  LiDAR ...")
    x, y, z, cls = load_las(cfg["las_dir"], aoi_union)
    gm = cls == 2
    xg, yg, zg = (x[gm],y[gm],z[gm]) if gm.sum()>=500 else (x,y,z)
    if gm.sum() >= 500:
        print(f"    class-2 ground: {gm.sum():,}")
    else:
        print(f"    all points (class-2 insufficient): {len(x):,}")

    # Separate inter-row points
    col_g = np.floor((xg-minx)/REF_PIXEL_SIZE).astype(int)
    row_g = np.floor((maxy-yg)/REF_PIXEL_SIZE).astype(int)
    in_b  = (col_g>=0)&(col_g<ref_w)&(row_g>=0)&(row_g<ref_h)
    is_ir = in_b & ~diss_mask_ref[
        np.clip(row_g,0,ref_h-1), np.clip(col_g,0,ref_w-1)]
    x_ir,y_ir,z_ir = xg[is_ir], yg[is_ir], zg[is_ir]
    n_ir  = len(x_ir)
    pct_ir = 100*n_ir/len(xg) if len(xg)>0 else 0
    print(f"    Inter-row LiDAR: {n_ir:,} ({pct_ir:.1f}%)")

    # Reference DTM
    print("  Building reference DTM ...")
    ref_raw, _ = pts_to_mean_raster(x_ir, y_ir, z_ir, bounds, REF_PIXEL_SIZE)
    aoi_mask_ref = rast_mask(aoi_union, (ref_h,ref_w), ref_transform)
    ref_raw[~aoi_mask_ref] = np.nan
    ref_smooth = smooth_fill(ref_raw, smooth_win)
    ref_smooth[~aoi_mask_ref] = np.nan

    save_tif(ref_raw,    ref_transform, TARGET_EPSG,
             os.path.join(out_dir, f"{pfx}_ref_dtm_raw.tif"))
    save_tif(ref_smooth, ref_transform, TARGET_EPSG,
             os.path.join(out_dir, f"{pfx}_ref_dtm_smooth.tif"))

    # Residuals at inter-row points (quality check)
    ir_c = np.clip(np.floor((x_ir-minx)/REF_PIXEL_SIZE).astype(int), 0, ref_w-1)
    ir_r = np.clip(np.floor((maxy-y_ir)/REF_PIXEL_SIZE).astype(int), 0, ref_h-1)
    res  = z_ir - ref_smooth[ir_r, ir_c]
    res  = res[np.isfinite(res)]
    res_mean = float(np.mean(res)); res_std = float(np.std(res))
    res_p5   = float(np.percentile(res, 5))
    res_p95  = float(np.percentile(res, 95))
    if abs(res_mean*100)<2.0 and res_std*100<4.0:
        assessment = "GOOD"
    elif abs(res_mean*100)<5.0 and res_std*100<8.0:
        assessment = "ACCEPTABLE"
    else:
        assessment = "POOR -- increase SMOOTH_WIN and re-run"
    print(f"    Residual mean={res_mean*100:+.2f} cm  "
          f"std={res_std*100:.2f} cm  -> {assessment}")

    # Min-Z surface + depth
    print("  Depth raster ...")
    min_surf, _ = pts_to_min_raster(xg, yg, zg, bounds, PIXEL_SIZE)
    min_surf[~aoi_mask_dep] = np.nan
    ref_fine = resample_to(ref_smooth, dep_h, dep_w)
    ref_fine[~aoi_mask_dep] = np.nan

    depth_full = min_surf - ref_fine
    depth_full[~aoi_mask_dep] = np.nan

    diss_mask_dep = rast_mask(dissolved, (dep_h,dep_w), dep_transform)
    depth_tracks  = depth_full.copy(); depth_tracks[~diss_mask_dep] = np.nan
    depth_ruts    = np.where(np.isfinite(depth_tracks)&(depth_tracks<0),
                             depth_tracks, np.nan).astype("float32")

    save_tif(ref_fine,    dep_transform, TARGET_EPSG,
             os.path.join(out_dir, f"{pfx}_ref_dtm_fine.tif"))
    save_tif(depth_full,  dep_transform, TARGET_EPSG,
             os.path.join(out_dir, f"{pfx}_depth_full.tif"))
    save_tif(depth_ruts,  dep_transform, TARGET_EPSG,
             os.path.join(out_dir, f"{pfx}_depth_ruts.tif"))

    depth_class_r = classify_depth(depth_ruts)
    save_tif(depth_class_r, dep_transform, TARGET_EPSG,
             os.path.join(out_dir, f"{pfx}_depth_class.tif"),
             dtype="uint8", nodata=0)

    # Depth class areas -- count only pixels inside AOI AND inside dissolved footprint
    # depth_ruts is already NaN outside footprint, so np.isfinite is the correct mask
    rut_vals    = depth_ruts[np.isfinite(depth_ruts)]
    rut_px_total = len(rut_vals)   # total rut pixels (negative depth inside footprint)
    depth_class_rows = []
    for k, lab in CLASS_LABELS.items():
        # depth_class_r is 0 outside footprint/outside rut, so no extra masking needed
        n = int((depth_class_r == k).sum())
        depth_class_rows.append({
            "fid": fid, "method": method,
            "depth_class": k, "depth_label": lab,
            "area_m2":  n * pa,
            "area_ha":  n * pa / 10000,
            "pct_of_rut_area": 100 * n / rut_px_total if rut_px_total > 0 else np.nan,
        })
    depth_class_df = pd.DataFrame(depth_class_rows)

    # Depth stats
    if len(rut_vals) > 0:
        d_mean   = float(np.mean(rut_vals))
        d_p10    = float(np.percentile(rut_vals, 10))
        d_median = float(np.median(rut_vals))
        d_min    = float(np.min(rut_vals))
        vol      = float(np.sum(-rut_vals) * pa)
        vol_ha   = vol / (aoi_area_m2 / 10000)
    else:
        d_mean = d_p10 = d_median = d_min = vol = vol_ha = np.nan

    print(f"    mean={d_mean*100:.2f} cm  p10={d_p10*100:.2f} cm  "
          f"deepest={d_min*100:.2f} cm  vol={vol_ha:.4f} m3/ha")

    # =========================================================
    #  D. 4-PANEL DIAGNOSTIC MAP (per field)
    # =========================================================
    print("  Maps ...")
    ext = [minx, maxx, miny, maxy]

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle(f"{fid} ({method}) -- Analysis diagnostics",
                 fontsize=14, fontweight="bold")

    # Panel 1: reference zone map
    ir_vis = ir_mask_ref.astype("float32"); ir_vis[~aoi_mask_ref] = np.nan
    im1 = axes[0,0].imshow(ir_vis, extent=ext, origin="upper",
                            cmap=ListedColormap(["#d73027","#4dac26"]),
                            vmin=0, vmax=1)
    aoi.boundary.plot(ax=axes[0,0], color="black", linewidth=1.2)
    axes[0,0].set_title(
        f"Reference zones\n"
        f"Compacted: {pct_diss:.1f}%  |  Inter-row: {100-pct_diss:.1f}%\n"
        f"Narrowest strip: {min_sw:.2f} m  |  Smooth window: {sw_m:.1f} m",
        fontsize=10)
    h_ref = [mpatches.Patch(color="#4dac26", label=f"Inter-row reference ({100-pct_diss:.1f}%)"),
             mpatches.Patch(color="#d73027", label=f"Compacted footprint ({pct_diss:.1f}%)")]
    legend_outside(axes[0,0], h_ref, title="Reference zones")
    axes[0,0].set_xlabel("Easting (m)"); axes[0,0].set_ylabel("Northing (m)")
    axes[0,0].set_aspect("equal")

    # Panel 2: reference DTM
    vn2 = np.nanpercentile(ref_fine, 2); vx2 = np.nanpercentile(ref_fine, 98)
    im2 = axes[0,1].imshow(ref_fine, extent=ext, origin="upper",
                            cmap="terrain", vmin=vn2, vmax=vx2)
    aoi.boundary.plot(ax=axes[0,1], color="black", linewidth=1.2)
    cb2 = plt.colorbar(im2, ax=axes[0,1], shrink=0.7, label="Z reference (m)")
    axes[0,1].set_title(
        f"Reference DTM (undisturbed terrain)\n"
        f"Residual: mean={res_mean*100:+.2f} cm  std={res_std*100:.2f} cm  "
        f"-> {assessment}",
        fontsize=10)
    axes[0,1].set_xlabel("Easting (m)"); axes[0,1].set_ylabel("Northing (m)")
    axes[0,1].set_aspect("equal")

    # Panel 3: depth raster (full field)
    vlim = max(0.05, float(np.nanpercentile(np.abs(depth_full[np.isfinite(depth_full)]), 98)))
    im3 = axes[1,0].imshow(depth_full, extent=ext, origin="upper",
                            cmap="RdBu_r", vmin=-vlim, vmax=vlim)
    aoi.boundary.plot(ax=axes[1,0], color="black", linewidth=1.2)
    cb3 = plt.colorbar(im3, ax=axes[1,0], shrink=0.7,
                        label="Depth (m)  [negative = rut]")
    axes[1,0].set_title("Depth map (min_Z - reference)\nBlue = depression / rut",
                         fontsize=10)
    axes[1,0].set_xlabel("Easting (m)"); axes[1,0].set_ylabel("Northing (m)")
    axes[1,0].set_aspect("equal")

    # Panel 4: depth class inside dissolved footprint
    cmap_cls = ListedColormap(["#eeeeee"] + CLASS_COLORS)
    norm_cls = BoundaryNorm([-0.5,0.5,1.5,2.5,3.5,4.5], cmap_cls.N)
    axes[1,1].imshow(depth_class_r, extent=ext, origin="upper",
                     cmap=cmap_cls, norm=norm_cls)
    aoi.boundary.plot(ax=axes[1,1], color="black", linewidth=1.2)
    h4 = [mpatches.Patch(color="#eeeeee", label="no rut / outside tracks")] + \
         [mpatches.Patch(color=CLASS_COLORS[i-1], label=CLASS_LABELS[i])
          for i in [1,2,3,4]]
    legend_outside(axes[1,1], h4, title="Depth class")
    axes[1,1].set_title(
        f"Rut depth class (inside dissolved footprint)\n"
        f"mean={d_mean*100:.1f} cm  p10={d_p10*100:.1f} cm  "
        f"deepest={d_min*100:.1f} cm",
        fontsize=10)
    axes[1,1].set_xlabel("Easting (m)"); axes[1,1].set_ylabel("Northing (m)")
    axes[1,1].set_aspect("equal")

    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(os.path.join(out_dir, f"{pfx}_diagnostic_4panel.png"),
                dpi=200, bbox_inches="tight")
    plt.close()

    # Pass-count map
    cmap_p = ListedColormap(PASS_COLORS)
    norm_p = BoundaryNorm([-0.5,0.5,1.5,2.5,3.5], cmap_p.N)
    fig, ax = plt.subplots(figsize=(10, 9))
    ax.imshow(pass_class, extent=ext, origin="upper", cmap=cmap_p, norm=norm_p)
    aoi.boundary.plot(ax=ax, color="black", linewidth=1.2)
    h_p = [mpatches.Patch(color=PASS_COLORS[i], label=PASS_LABELS[i])
           for i in range(4)]
    legend_outside(ax, h_p, title="Pass count")
    ax.set_title(f"{fid} ({method}) -- Wheel-pass intensity\n"
                 f"Area under wheel: {pct_trafficked:.1f}%", fontsize=11)
    ax.set_xlabel("Easting (m)"); ax.set_ylabel("Northing (m)")
    ax.set_aspect("equal")
    plt.tight_layout(rect=[0, 0, 0.82, 1])
    plt.savefig(os.path.join(out_dir, f"{pfx}_pass_map.png"),
                dpi=200, bbox_inches="tight")
    plt.close()

    # Residual histogram
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(np.clip(res,-0.10,0.10)*100, bins=200, color="#2ca02c",
            alpha=0.75, edgecolor="none")
    ax.axvline(0,              color="black",  lw=1.2, ls="--", label="zero")
    ax.axvline(res_mean*100,   color="red",    lw=1.5,
               label=f"mean = {res_mean*100:+.2f} cm")
    ax.axvline(res_p5*100,     color="orange", lw=1.0, ls=":",
               label=f"p5 = {res_p5*100:.1f} cm")
    ax.axvline(res_p95*100,    color="orange", lw=1.0, ls=":",
               label=f"p95 = {res_p95*100:.1f} cm")
    legend_outside(ax, title="")
    ax.set_xlabel("Inter-row residual (cm)  [clipped to +/-10 cm]")
    ax.set_ylabel("LiDAR point count")
    ax.set_title(f"{fid} ({method}) -- Inter-row residual histogram\n"
                 f"std = {res_std*100:.2f} cm  [{assessment}]")
    ax.grid(True, alpha=0.3)
    plt.tight_layout(rect=[0, 0, 0.82, 1])
    plt.savefig(os.path.join(out_dir, f"{pfx}_residual_histogram.png"),
                dpi=200, bbox_inches="tight")
    plt.close()

    # CSVs
    pass_df.to_csv(os.path.join(out_dir, f"{pfx}_pass_class_areas.csv"), index=False)
    depth_class_df.to_csv(os.path.join(out_dir, f"{pfx}_depth_class_areas.csv"),
                          index=False)

    summary = {
        "fid": fid, "method": method,
        "field_area_ha":      aoi_area_m2/10000,
        "trafficked_area_ha": trafficked_area/10000,
        "pct_trafficked":     pct_trafficked,
        "pct_NOT_trafficked": 100-pct_trafficked,
        "dissolved_pct":      pct_diss,
        "narrowest_strip_m":  min_sw,
        "smooth_win_px":      smooth_win,
        "smooth_win_m":       smooth_win * REF_PIXEL_SIZE,
        "ref_res_mean_cm":    res_mean*100,
        "ref_res_std_cm":     res_std*100,
        "ref_assessment":     assessment,
        "depth_mean_cm":      d_mean*100   if not np.isnan(d_mean)   else np.nan,
        "depth_p10_cm":       d_p10*100    if not np.isnan(d_p10)    else np.nan,
        "depth_median_cm":    d_median*100 if not np.isnan(d_median) else np.nan,
        "depth_min_cm":       d_min*100    if not np.isnan(d_min)    else np.nan,
        "vol_m3_total":       vol,           # total volume -- comparable if same field size
        "vol_m3_per_ha":      vol_ha,        # per ha -- normalised for field size comparison
        "vol_m3_per_trafficked_ha": (
            vol / (trafficked_area / 10000)  # per trafficked ha -- fairest comparison
            if trafficked_area > 0 else np.nan),
    }
    print(f"  Saved to: {out_dir}")
    return summary, pass_df, depth_class_df


# =========================================================
#  COMPARISON PLOTS
# =========================================================

def write_comparison(all_summaries, all_pass_dfs, all_depth_dfs):
    ensure_dir(COMPARISON_OUT)
    summary_df    = pd.DataFrame(all_summaries)
    pass_df_all   = pd.concat(all_pass_dfs,  ignore_index=True)
    depth_df_all  = pd.concat(all_depth_dfs, ignore_index=True)

    summary_df.to_csv(
        os.path.join(COMPARISON_OUT, "summary_all_fields.csv"), index=False)
    pass_df_all.to_csv(
        os.path.join(COMPARISON_OUT, "pass_class_areas_all.csv"), index=False)
    depth_df_all.to_csv(
        os.path.join(COMPARISON_OUT, "depth_class_areas_all.csv"), index=False)

    fids    = summary_df["fid"].tolist()
    methods = summary_df["method"].tolist()
    labels  = [f"{f}\n({m})" for f,m in zip(fids,methods)]
    colors  = [FIELD_COLORS[f] for f in fids]
    n       = len(fids)
    x_pos   = np.arange(n)

    def _bar(ax, values, title, ylabel, fmt="{:.1f}"):
        bars = ax.bar(x_pos, values, color=colors, width=0.5)
        for bar, v in zip(bars, values):
            if not np.isnan(v):
                ax.text(bar.get_x()+bar.get_width()/2,
                        bar.get_height()+abs(bar.get_height())*0.02,
                        fmt.format(v), ha="center", va="bottom", fontsize=9)
        ax.set_xticks(x_pos); ax.set_xticklabels(labels, fontsize=9)
        ax.set_title(title, fontsize=11); ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.3)

    # ---- Plot 1: % field under wheel + NOT under wheel ------
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle("Wheel coverage comparison -- all fields", fontsize=13,
                 fontweight="bold")
    _bar(axes[0], summary_df["pct_trafficked"].tolist(),
         "% of field under tractor wheel", "%")
    _bar(axes[1], summary_df["pct_NOT_trafficked"].tolist(),
         "% of field NOT under any wheel\n(protected soil -- higher is better)", "%")
    h_leg = [mpatches.Patch(color=FIELD_COLORS[f], label=f"{f} ({m})")
             for f,m in zip(fids,methods)]
    legend_outside(axes[1], h_leg, title="Field")
    plt.tight_layout(rect=[0,0,0.88,1])
    plt.savefig(os.path.join(COMPARISON_OUT, "comparison_coverage.png"),
                dpi=200, bbox_inches="tight")
    plt.close()

    # ---- Plot 2: depth metrics bar chart --------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle("Rut depth comparison -- all fields", fontsize=13,
                 fontweight="bold")
    _bar(axes[0], summary_df["depth_mean_cm"].tolist(),
         "Mean rut depth", "cm")
    _bar(axes[1], summary_df["depth_p10_cm"].tolist(),
         "p10 rut depth\n(10% of ruts deeper than this)", "cm")
    _bar(axes[2], summary_df["depth_min_cm"].tolist(),
         "Deepest point in field", "cm")
    legend_outside(axes[2], h_leg, title="Field")
    plt.tight_layout(rect=[0,0,0.88,1])
    plt.savefig(os.path.join(COMPARISON_OUT, "comparison_rut_depth.png"),
                dpi=200, bbox_inches="tight")
    plt.close()

    # ---- Plot 3: soil displacement (three metrics) ----------
    #
    # Three ways to express displacement -- all included because
    # the fields have very different sizes and traffic intensities:
    #
    #  vol_m3_total             -- raw total volume displaced
    #                              only fair if fields are similar size
    #  vol_m3_per_ha            -- per total field hectare
    #                              fair for comparing overall field impact
    #  vol_m3_per_trafficked_ha -- per trafficked hectare only
    #                              fairest: compares depth intensity under
    #                              the wheel regardless of how much of
    #                              the field was trafficked
    #
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Soil displacement proxy -- all fields\n"
                 "(three normalisations for fair comparison)",
                 fontsize=13, fontweight="bold")

    _bar(axes[0], summary_df["vol_m3_total"].tolist(),
         "Total displaced volume (m3)\n(only fair if fields are same size)",
         "m3", fmt="{:.2f}")

    _bar(axes[1], summary_df["vol_m3_per_ha"].tolist(),
         "Displacement per total field ha (m3/ha)\n"
         "Includes effect of how much field was trafficked",
         "m3/ha", fmt="{:.3f}")

    _bar(axes[2], summary_df["vol_m3_per_trafficked_ha"].tolist(),
         "Displacement per TRAFFICKED ha (m3/ha)\n"
         "Fairest comparison: depth intensity under the wheel only",
         "m3/ha", fmt="{:.3f}")

    legend_outside(axes[2], h_leg, title="Field")
    plt.tight_layout(rect=[0,0,0.88,1])
    plt.savefig(os.path.join(COMPARISON_OUT, "comparison_displacement.png"),
                dpi=200, bbox_inches="tight")
    plt.close()

    # ---- Plot 4: depth class -- TWO panels ----------------------
    # Panel A: absolute area (ha) -- affected by field size differences
    # Panel B: % of rut area -- size-normalised, best for cross-field comparison
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle("Rut depth class breakdown -- all fields", fontsize=13,
                 fontweight="bold")
    handles_dc = []
    for ai, (col, ylabel, title_note) in enumerate([
        ("area_ha",          "Area (ha)",
         "Absolute area by depth class\n(affected by field size -- use % panel for comparison)"),
        ("pct_of_rut_area",  "% of total rut area",
         "% of rut area by depth class\nSize-normalised -- best for cross-field comparison"),
    ]):
        ax = axes[ai]
        bottom = np.zeros(n)
        for k, lab in CLASS_LABELS.items():
            vals = []
            for fid in fids:
                row = depth_df_all[(depth_df_all["fid"]==fid) &
                                   (depth_df_all["depth_class"]==k)]
                vals.append(float(row[col].sum()) if len(row) else 0.0)
            bars = ax.bar(x_pos, vals, bottom=bottom,
                          color=CLASS_COLORS[k-1], width=0.5)
            if ai == 0:
                handles_dc.append(mpatches.Patch(color=CLASS_COLORS[k-1], label=lab))
            bottom += np.array(vals)
        ax.set_xticks(x_pos); ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylabel(ylabel); ax.set_xlabel("Field (method)")
        ax.set_title(title_note, fontsize=10)
        ax.grid(axis="y", alpha=0.3)

    legend_outside(axes[1], handles_dc, title="Depth class")
    plt.tight_layout(rect=[0,0,0.78,1])
    plt.savefig(os.path.join(COMPARISON_OUT, "comparison_depth_class_stacked.png"),
                dpi=200, bbox_inches="tight")
    plt.close()

    # ---- Plot 5: pass-class -- TWO panels ----------------------
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle("Wheel-pass count breakdown -- all fields", fontsize=13,
                 fontweight="bold")
    handles_pc = []
    for ai, (col, ylabel, title_note) in enumerate([
        ("area_ha",   "Area (ha)",
         "Absolute area by pass count\n(affected by field size)"),
        ("pct_field", "% of field area",
         "% of field area by pass count\nSize-normalised -- best for comparison"),
    ]):
        ax = axes[ai]
        bottom = np.zeros(n)
        for k, lab in PASS_LABELS.items():
            vals = []
            for fid in fids:
                row = pass_df_all[(pass_df_all["fid"]==fid) &
                                  (pass_df_all["pass_class"]==k)]
                vals.append(float(row[col].sum()) if len(row) else 0.0)
            ax.bar(x_pos, vals, bottom=bottom,
                   color=PASS_COLORS[k], width=0.5)
            if ai == 0:
                handles_pc.append(mpatches.Patch(color=PASS_COLORS[k], label=lab))
            bottom += np.array(vals)
        ax.set_xticks(x_pos); ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylabel(ylabel); ax.set_xlabel("Field (method)")
        ax.set_title(title_note, fontsize=10)
        ax.grid(axis="y", alpha=0.3)

    legend_outside(axes[1], handles_pc, title="Pass count")
    plt.tight_layout(rect=[0,0,0.82,1])
    plt.savefig(os.path.join(COMPARISON_OUT, "comparison_pass_class_stacked.png"),
                dpi=200, bbox_inches="tight")
    plt.close()

    # ---- Console summary ------------------------------------
    print(f"\n{'='*80}")
    print("  COMPARISON SUMMARY")
    print(f"{'='*80}")
    cols = ["fid", "method", "field_area_ha",
            "pct_trafficked", "pct_NOT_trafficked",
            "smooth_win_m", "ref_res_std_cm",
            "depth_mean_cm", "depth_p10_cm", "depth_min_cm",
            "vol_m3_total", "vol_m3_per_ha", "vol_m3_per_trafficked_ha"]
    print(summary_df[cols].to_string(index=False))
    print(f"{'='*80}")
    print()
    print("  VOLUME COMPARISON NOTE:")
    print("  vol_m3_total            -- raw total; only fair if fields are same size")
    print("  vol_m3_per_ha           -- per total hectare; shows overall field impact")
    print("  vol_m3_per_trafficked_ha -- per trafficked hectare; best for comparing")
    print("                             rut depth intensity independent of traffic %")
    print()
    print("  SMOOTHING WINDOW NOTE:")
    print("  smooth_win_m is the window used for the reference DTM.")
    print("  ref_res_std_cm is the reference quality (lower = better).")
    print("  Values < 3 cm are good. Values > 5 cm mean the window is too wide.")
    print(f"{'='*80}")
    print(f"Comparison outputs: {COMPARISON_OUT}")


# =========================================================
#  MAIN
# =========================================================

def main():
    all_summaries, all_pass_dfs, all_depth_dfs = [], [], []
    for fid, cfg in FIELDS.items():
        summary, pass_df, depth_df = analyse_field(fid, cfg)
        all_summaries.append(summary)
        all_pass_dfs.append(pass_df)
        all_depth_dfs.append(depth_df)
    write_comparison(all_summaries, all_pass_dfs, all_depth_dfs)
    print("\nOK  All done.")


if __name__ == "__main__":
    all_summaries = []
    all_pass_dfs = []
    all_depth_dfs = []

    # Loops over FIELDS (which now only contains F2)
    for fid, cfg in FIELDS.items():
        summary, pass_df, depth_df = analyse_field(fid, cfg)
        all_summaries.append(summary)
        all_pass_dfs.append(pass_df)
        all_depth_dfs.append(depth_df)