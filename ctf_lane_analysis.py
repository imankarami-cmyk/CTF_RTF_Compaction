"""
ctf_tractor_alignment.py
=========================
CTF tractor alignment analysis for the farmer report.
Produces clean, easy-to-read maps and charts suitable for a written report.

Tractors:
  "gross" in data  ->  Fendt 728 (Gross)
  "klein" in data  ->  Fendt 724 (Klein)

Four questions answered:
  1. How wide is each tractor's actual wheel track across the season?
     (GPS navigation accuracy)
  2. How well do the two tractors follow the same permanent lanes?
     (Track Confinement Index)
  3. Does the heavier Fendt 728 cause deeper ruts than the Fendt 724?
     (Load-class attribution)
  4. Which field operation deviated most from the permanent lanes?
     (Off-route detection -- actionable for the farmer)

Run:
    python ctf_tractor_alignment.py
"""

import os, glob, warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.transform import from_origin
from rasterio.features import rasterize
from shapely.ops import unary_union
from scipy.ndimage import distance_transform_edt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.gridspec import GridSpec
from tqdm import tqdm

warnings.filterwarnings("ignore")

# set a clean, professional style for all plots
plt.rcParams.update({
    "font.family":  "DejaVu Sans",
    "font.size":    11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# =========================================================
#  CONFIGURATION -- EDIT THIS BLOCK
# =========================================================

BASE = r"C:\Users\imankarami\Data\Analysing_full"

CTF_FIELDS = {
    "F1": {
        "aoi":         os.path.join(BASE, "F1", "AOI_F1.shp"),
        "traffic_shp": os.path.join(BASE, "F1", "f1_traffic_clipped1.shp"),
        "depth_tif":   os.path.join(BASE, "F1", "analysis23_7_v4", "F1_CTF_depth_ruts.tif"),
        "out_dir":     os.path.join(BASE, "F1", "tractor_alignment"),
    },
    "F3": {
        "aoi":         os.path.join(BASE, "F3", "AOI_F3.shp"),
        "traffic_shp": os.path.join(BASE, "F3", "f3_traffic_clipped222.shp"),
        "depth_tif":   os.path.join(BASE, "F3", "analysis23_7_v4", "F3_CTF_depth_ruts.tif"),
        "out_dir":     os.path.join(BASE, "F3", "tractor_alignment"),
    },
}

COMPARISON_OUT = os.path.join(BASE, "TRACTOR_ALIGNMENT_COMPARISON")
TARGET_EPSG    = 32633
PIXEL_SIZE     = 0.10   # m -- must match depth raster

# Tractor display names (what appears in all maps and charts)
TRACTOR_NAMES = {
    "gross": "Fendt 728 (Gross)",
    "klein": "Fendt 724 (Klein)",
}

# Corrected tyre and track dimensions
# Hinterreifen = rear tyre width, Spur = track gauge (centre to centre)
# Outer width (AUSSENBREITE) = Spur + Hinterreifen
NOMINAL = {
    "gross": {
        "tyre_width":      0.70,   # Hinterreifen
        "track_gauge":     2.00,   # Spur
        "outer_width":     2.70,   # Spur + Hinterreifen
    },
    "klein": {
        "tyre_width":      0.65,   # Hinterreifen
        "track_gauge":     1.90,   # Spur
        "outer_width":     2.55,   # Spur + Hinterreifen
    },
}

# Minimum pass count to identify a confirmed permanent CTF lane
MIN_PASSES = 3

# Shapefile column names
POLYSRC_COL = "PolySrc"
WHEEL_COL   = "Wheel"
DATE_COL    = "RouteDate"

# Colours -- consistent across all plots and maps
C = {
    "gross":        "#1a6faf",   # steel blue  -- Fendt 728
    "klein":        "#c0392b",   # brick red   -- Fendt 724
    "overlap":      "#6c3483",   # purple      -- both tractors
    "gross_only":   "#aed6f1",   # light blue  -- Fendt 728 only
    "klein_only":   "#f1948a",   # light red   -- Fendt 724 only
    "off_route":    "#e67e22",   # orange      -- off-route area
    "permanent":    "#27ae60",   # green       -- permanent lane
    "field":        "#ecf0f1",   # very light grey -- field background
}


# =========================================================
#  HELPERS
# =========================================================

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def tname(src): return TRACTOR_NAMES.get(src, src)

def load_tif(path):
    with rasterio.open(path) as src:
        return src.read(1).astype("float32"), src.transform, src.shape

def rast_mask(geom, shp, transform):
    return rasterize([(geom, 1)], out_shape=shp,
                     transform=transform, fill=0, dtype="uint8").astype(bool)

def save_tif(arr, transform, path, dtype="float32", nodata=np.nan):
    ensure_dir(os.path.dirname(path))
    with rasterio.open(path, "w", driver="GTiff",
                       height=arr.shape[0], width=arr.shape[1],
                       count=1, dtype=dtype, crs=f"EPSG:{TARGET_EPSG}",
                       transform=transform, nodata=nodata) as dst:
        dst.write(arr.astype(dtype), 1)

def load_traffic(shp_path, aoi_union):
    gdf = gpd.read_file(shp_path).to_crs(epsg=TARGET_EPSG)
    gdf["geometry"] = gdf.buffer(0)
    gdf = gdf[gdf.geometry.notnull() & ~gdf.geometry.is_empty].copy()
    gdf = gdf[gdf.intersects(aoi_union)].copy().reset_index(drop=True)
    col_lower = {c.lower(): c for c in gdf.columns}
    def _get(col, aliases, default):
        for cand in [col] + aliases:
            if cand.lower() in col_lower:
                return (gdf[col_lower[cand.lower()]]
                        .fillna(default).astype(str).str.lower().str.strip())
        return pd.Series([default]*len(gdf), index=gdf.index)
    gdf["src"]   = _get(POLYSRC_COL, ["polysrc","source","tractor"], "unknown")
    gdf["wheel"] = _get(WHEEL_COL,   ["wheel","side"],               "unknown")
    gdf["date"]  = _get(DATE_COL,    ["routedate","date","route_date"], "unknown")
    print(f"    Tractors found  : {sorted(gdf['src'].unique())}")
    print(f"    Unique dates    : {gdf['date'].nunique()}")
    return gdf

def build_pass_raster(geoms, shape_hw, transform):
    out = np.zeros(shape_hw, dtype=np.uint16)
    for geom in tqdm(geoms, leave=False, desc="      rasterising"):
        out += rasterize([(geom,1)], out_shape=shape_hw,
                         transform=transform, fill=0, dtype="uint16")
    return out

def confirmed_lane(pass_r, aoi_mask, min_passes):
    max_p  = int(pass_r[aoi_mask].max()) if aoi_mask.sum() > 0 else 0
    thresh = min_passes if max_p >= min_passes else max(2, int(max_p * 0.70))
    return (pass_r >= thresh) & aoi_mask, thresh, max_p

def lane_width(mask, px):
    dist = distance_transform_edt(mask) * px
    vals = dist[mask]
    return (float(np.mean(vals)*2), float(np.median(vals)*2)) if len(vals) else (np.nan, np.nan)

def depth_stats(depth_r, mask):
    if not mask.any(): return {}
    vals = depth_r[mask & np.isfinite(depth_r)]
    neg  = vals[vals < 0]
    if len(neg) == 0: return {"n": 0}
    return {
        "n":        int(len(neg)),
        "mean_cm":  float(np.mean(neg))*100,
        "p10_cm":   float(np.percentile(neg,10))*100,
        "min_cm":   float(np.min(neg))*100,
        "vol_m3":   float(np.sum(-neg)*PIXEL_SIZE**2),
    }

def add_scalebar(ax, length_m, x0_frac=0.05, y0_frac=0.05,
                 ext=None, color="black"):
    """Add a simple scale bar to a map axis."""
    if ext is None: return
    xmin, xmax, ymin, ymax = ext
    x0 = xmin + (xmax-xmin)*x0_frac
    y0 = ymin + (ymax-ymin)*y0_frac
    ax.plot([x0, x0+length_m], [y0, y0], color=color, lw=3, solid_capstyle="butt")
    ax.text(x0 + length_m/2, y0 + (ymax-ymin)*0.01,
            f"{length_m} m", ha="center", va="bottom", fontsize=8, color=color)

def legend_right(ax, handles, title=None):
    kw = dict(bbox_to_anchor=(1.02, 1), loc="upper left",
              borderaxespad=0, fontsize=9, frameon=True, framealpha=0.9)
    if title: kw["title"] = title
    ax.legend(handles=handles, **kw)


# =========================================================
#  PER-FIELD ANALYSIS
# =========================================================

def analyse_field(fid, cfg):
    out_dir = cfg["out_dir"]
    ensure_dir(out_dir)
    print(f"\n{'='*60}")
    print(f"  {fid}  --  CTF Tractor Alignment")
    print(f"{'='*60}")

    # -- Master grid from depth raster -----------------------
    depth_r, transform, (h, w) = load_tif(cfg["depth_tif"])
    print(f"  Grid: {w} x {h} px")

    # -- AOI -------------------------------------------------
    aoi = gpd.read_file(cfg["aoi"]).to_crs(epsg=TARGET_EPSG)
    aoi["geometry"] = aoi.buffer(0)
    aoi = aoi[aoi.geometry.notnull() & ~aoi.geometry.is_empty].copy()
    aoi_union   = aoi.union_all()
    aoi_area_m2 = aoi_union.area
    aoi_mask    = rast_mask(aoi_union, (h,w), transform)
    bnd = aoi_union.bounds
    ext = [bnd[0], bnd[2], bnd[1], bnd[3]]   # [xmin xmax ymin ymax]
    scale_m = max(10, int((bnd[2]-bnd[0])/8/10)*10)   # neat scale bar length

    # -- Traffic ---------------------------------------------
    print("  Loading traffic ...")
    traffic = load_traffic(cfg["traffic_shp"], aoi_union)
    tractors = [t for t in ["gross","klein"] if t in traffic["src"].unique()]

    R = {}   # results dict

    # =========================================================
    #  Q1: CONFIRMED LANES + TRACK WIDTH
    # =========================================================
    print("\n  Q1: Confirmed lanes per tractor ...")
    for src in tractors:
        grp  = traffic[traffic["src"] == src]
        name = tname(src)
        print(f"  {name}: {len(grp):,} polygons  "
              f"({grp['date'].nunique()} operation dates)")
        pass_r = build_pass_raster(grp.geometry, (h,w), transform)
        pass_r[~aoi_mask] = 0
        l_mask, thresh, max_p = confirmed_lane(pass_r, aoi_mask, MIN_PASSES)
        la = float(l_mask.sum()) * PIXEL_SIZE**2
        wm, wmed = lane_width(l_mask, PIXEL_SIZE)
        nom = NOMINAL[src]["tyre_width"]
        excess = wm - nom if not np.isnan(wm) else np.nan
        if np.isnan(excess):
            gps_note = "insufficient confirmed lane pixels"
        elif abs(excess) < 0.05:
            gps_note = "Excellent -- lane width matches tyre width"
        elif excess > 0:
            gps_note = (f"Lane {excess*100:.0f} cm wider than tyre "
                        f"-- GPS drift ~{excess*100/2:.0f} cm")
        else:
            gps_note = (f"Lane narrower than tyre -- confirmed core strip "
                        f"within {nom*100:.0f} cm tyre footprint")
        print(f"    Threshold: {thresh} passes  |  Max: {max_p}")
        print(f"    Lane area: {la:.0f} m2  |  Width: {wm:.3f} m "
              f"(nominal: {nom:.2f} m)")
        print(f"    GPS note: {gps_note}")
        save_tif(pass_r, transform,
                 os.path.join(out_dir, f"{fid}_{src}_pass_count.tif"),
                 dtype="uint16", nodata=0)
        save_tif(l_mask.astype("uint8"), transform,
                 os.path.join(out_dir, f"{fid}_{src}_confirmed_lane.tif"),
                 dtype="uint8", nodata=255)
        R[src] = {
            "pass_r": pass_r, "lane_mask": l_mask,
            "thresh": thresh, "max_p": max_p,
            "lane_area_m2": la, "w_mean": wm, "w_med": wmed,
            "gps_note": gps_note, "excess_m": excess,
            "n_polys": len(grp), "n_dates": grp["date"].nunique(),
        }

    # Combined permanent lane (gross OR klein)
    combined_lane = np.zeros((h,w), dtype=bool)
    for src in tractors:
        combined_lane |= R[src]["lane_mask"]

    # =========================================================
    #  Q2: TRACK CONFINEMENT INDEX
    # =========================================================
    print("\n  Q2: Track Confinement Index ...")
    if len(tractors) == 2:
        g_lane = R["gross"]["lane_mask"]
        k_lane = R["klein"]["lane_mask"]
        both   = g_lane & k_lane & aoi_mask
        g_only = g_lane & ~k_lane & aoi_mask
        k_only = k_lane & ~g_lane & aoi_mask
        either = (g_lane | k_lane) & aoi_mask
        tci    = float(both.sum()) / float(either.sum()) if either.sum() > 0 else np.nan
        ov_g   = float(both.sum()) / float((g_lane & aoi_mask).sum()) \
                 if (g_lane & aoi_mask).sum() > 0 else np.nan
        ov_k   = float(both.sum()) / float((k_lane & aoi_mask).sum()) \
                 if (k_lane & aoi_mask).sum() > 0 else np.nan
        dist_g_to_k = distance_transform_edt(~k_lane)[g_lane] * PIXEL_SIZE
        offset_m    = float(np.median(dist_g_to_k)) if len(dist_g_to_k) else np.nan
        if tci >= 0.80:
            tci_label = "Good alignment"
        elif tci >= 0.50:
            tci_label = "Moderate alignment"
        else:
            tci_label = "Poor alignment"
        print(f"    TCI = {tci:.3f}  ({tci*100:.1f}%)  -> {tci_label}")
        print(f"    {tname('gross')} lane overlapping with {tname('klein')}: "
              f"{ov_g*100:.1f}%")
        print(f"    {tname('klein')} lane overlapping with {tname('gross')}: "
              f"{ov_k*100:.1f}%")
        print(f"    Median lateral offset: {offset_m:.3f} m")
        combo = np.zeros((h,w), dtype=np.uint8)
        combo[g_only] = 1; combo[k_only] = 2; combo[both] = 3
        save_tif(combo, transform,
                 os.path.join(out_dir, f"{fid}_tractor_overlap_zones.tif"),
                 dtype="uint8", nodata=0)
        R["tci"] = {
            "tci": tci, "ov_g": ov_g, "ov_k": ov_k,
            "tci_label": tci_label,
            "overlap_m2": float(both.sum())*PIXEL_SIZE**2,
            "g_only_m2":  float(g_only.sum())*PIXEL_SIZE**2,
            "k_only_m2":  float(k_only.sum())*PIXEL_SIZE**2,
            "offset_m":   offset_m,
            "both": both, "g_only": g_only, "k_only": k_only,
            "combo": combo,
        }
    else:
        R["tci"] = {}

    # =========================================================
    #  Q3: DEPTH BY TRACTOR
    # =========================================================
    print("\n  Q3: Depth attribution ...")
    depth_by_zone = {}
    if "tci" in R and R["tci"]:
        for zone_key, mask, label in [
            ("gross_only", R["tci"]["g_only"], tname("gross")+" only"),
            ("klein_only", R["tci"]["k_only"], tname("klein")+" only"),
            ("overlap",    R["tci"]["both"],   "Both tractors (overlap)"),
        ]:
            st = depth_stats(depth_r, mask)
            depth_by_zone[zone_key] = st
            if st.get("n",0) > 0:
                print(f"    {label}: mean={st['mean_cm']:.2f} cm  "
                      f"p10={st['p10_cm']:.2f} cm  "
                      f"deepest={st['min_cm']:.2f} cm")

    # =========================================================
    #  Q4: OFF-ROUTE OPERATIONS
    # =========================================================
    print("\n  Q4: Off-route operations ...")
    dist_to_lane = distance_transform_edt(~combined_lane) * PIXEL_SIZE
    date_rows = []
    for (src, date), grp in traffic.groupby(["src","date"]):
        if date == "unknown": continue
        date_r = np.zeros((h,w), dtype=np.uint8)
        for geom in grp.geometry:
            date_r += rasterize([(geom,1)], out_shape=(h,w),
                                transform=transform, fill=0, dtype="uint8")
        d_mask  = (date_r > 0) & aoi_mask
        n_px    = int(d_mask.sum())
        if n_px == 0: continue
        on      = int((d_mask & combined_lane).sum())
        off     = n_px - on
        off_px  = d_mask & ~combined_lane
        m_off   = float(np.median(dist_to_lane[off_px])) if off_px.sum() > 0 else 0.0
        mx_off  = float(np.max(dist_to_lane[off_px]))    if off_px.sum() > 0 else 0.0
        st      = depth_stats(depth_r, d_mask & combined_lane)
        date_rows.append({
            "fid": fid, "src": src,
            "tractor_name": tname(src),
            "date": date,
            "n_polygons":     len(grp),
            "n_pixels":       n_px,
            "pct_on_lane":    100*on/n_px,
            "pct_off_lane":   100*off/n_px,
            "median_off_m":   m_off,
            "max_off_m":      mx_off,
            "depth_mean_cm":  st.get("mean_cm", np.nan),
            "depth_p10_cm":   st.get("p10_cm",  np.nan),
        })
    date_df = pd.DataFrame(date_rows).sort_values(
        ["src","pct_off_lane"], ascending=[True,False]).reset_index(drop=True)
    date_df.to_csv(
        os.path.join(out_dir, f"{fid}_operation_offroute.csv"), index=False)

    if len(date_df):
        print(f"  {'Tractor':<24} {'Date':<14} "
              f"{'On-lane':>9} {'Off-lane':>9} {'Med.offset':>11}")
        print(f"  {'-'*72}")
        for _, row in date_df.iterrows():
            flag = "  *** WORST" if row["pct_off_lane"] > 20 else ""
            print(f"  {row['tractor_name']:<24} {row['date']:<14} "
                  f"{row['pct_on_lane']:>8.1f}% {row['pct_off_lane']:>8.1f}% "
                  f"{row['median_off_m']:>10.3f} m{flag}")

    # =========================================================
    #  MAPS & PLOTS  (report quality)
    # =========================================================
    print("\n  Generating report maps ...")

    # ----------------------------------------------------------
    #  MAP 1: Pass-count per tractor side by side
    #  Simple, clean: shows where each tractor drove most
    # ----------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(f"Flaeche {fid} -- Where each tractor drove\n"
                 f"(darker colour = more passes over same spot)",
                 fontsize=13, fontweight="bold", y=1.01)
    for ax, src in zip(axes, tractors):
        pr = R[src]["pass_r"].astype("float32")
        pr[~aoi_mask] = np.nan
        vmax = float(np.nanpercentile(pr, 99))
        # Field background
        bg = np.where(aoi_mask, 1.0, np.nan)
        ax.imshow(bg, extent=ext, origin="upper",
                  cmap=ListedColormap([C["field"]]), vmin=0, vmax=1)
        img = ax.imshow(pr, extent=ext, origin="upper",
                        cmap="Blues" if src=="gross" else "Reds",
                        vmin=0, vmax=max(vmax,1), alpha=0.9)
        aoi.boundary.plot(ax=ax, color="#555555", linewidth=1.2)
        cb = plt.colorbar(img, ax=ax, shrink=0.6, pad=0.02)
        cb.set_label("Number of passes", fontsize=9)
        nom = NOMINAL[src]
        ax.set_title(
            f"{tname(src)}\n"
            f"{R[src]['n_polys']:,} wheel polygons  |  "
            f"{R[src]['n_dates']} field operations\n"
            f"Tyre width: {nom['tyre_width']*100:.0f} cm  |  "
            f"Track gauge: {nom['track_gauge']:.2f} m",
            fontsize=10, pad=8)
        ax.set_xlabel("Easting (m)"); ax.set_ylabel("Northing (m)")
        ax.set_aspect("equal")
        add_scalebar(ax, scale_m, ext=ext, color="#333333")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{fid}_map1_pass_count.png"),
                dpi=220, bbox_inches="tight")
    plt.close()

    # ----------------------------------------------------------
    #  MAP 2: Permanent lane identification
    #  Shows what the code considers the confirmed permanent lane
    # ----------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(f"Flaeche {fid} -- Confirmed permanent CTF lanes\n"
                 f"(green = pixels driven on >= {R[tractors[0]]['thresh']} times "
                 f"= permanent lane)",
                 fontsize=13, fontweight="bold", y=1.01)
    for ax, src in zip(axes, tractors):
        pr = R[src]["pass_r"].astype("float32"); pr[~aoi_mask] = np.nan
        vmax = float(np.nanpercentile(pr, 99))
        bg = np.where(aoi_mask, 1.0, np.nan)
        ax.imshow(bg, extent=ext, origin="upper",
                  cmap=ListedColormap([C["field"]]), vmin=0, vmax=1)
        ax.imshow(pr, extent=ext, origin="upper",
                  cmap="Blues" if src=="gross" else "Reds",
                  vmin=0, vmax=max(vmax,1), alpha=0.5)
        # Confirmed lane overlay
        lane_vis = np.where(R[src]["lane_mask"] & aoi_mask, 1.0, np.nan)
        ax.imshow(lane_vis, extent=ext, origin="upper",
                  cmap=ListedColormap([C["permanent"]]),
                  vmin=0, vmax=1, alpha=0.85)
        aoi.boundary.plot(ax=ax, color="#555555", linewidth=1.2)
        nom = NOMINAL[src]["tyre_width"]
        wm  = R[src]["w_mean"]
        excess_cm = (wm - nom)*100 if not np.isnan(wm) else np.nan
        ax.set_title(
            f"{tname(src)}\n"
            f"Confirmed lane area: {R[src]['lane_area_m2']:.0f} m2\n"
            f"Measured lane width: {wm:.2f} m  "
            f"(nominal tyre: {nom:.2f} m)\n"
            f"{R[src]['gps_note']}",
            fontsize=9, pad=8)
        ax.set_xlabel("Easting (m)"); ax.set_ylabel("Northing (m)")
        ax.set_aspect("equal")
        add_scalebar(ax, scale_m, ext=ext)
    h_leg = [mpatches.Patch(color=C["permanent"], label="Confirmed permanent lane"),
             mpatches.Patch(color="#aec7e8", label="Other passes (Fendt 728)"),
             mpatches.Patch(color="#f4a582", label="Other passes (Fendt 724)")]
    fig.legend(handles=h_leg, loc="lower center", ncol=3,
               fontsize=9, frameon=True, bbox_to_anchor=(0.5, -0.04))
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{fid}_map2_permanent_lanes.png"),
                dpi=220, bbox_inches="tight")
    plt.close()

    # ----------------------------------------------------------
    #  MAP 3: Track confinement -- where do the two tractors agree?
    #  The key question for the farmer: are both tractors using
    #  the same strips?
    # ----------------------------------------------------------
    if R["tci"]:
        tci_r = R["tci"]
        combo_vis = tci_r["combo"].astype("float32")
        combo_vis[~aoi_mask] = np.nan

        cmap_ov  = ListedColormap([C["field"], C["gross_only"],
                                   C["klein_only"], C["overlap"]])
        norm_ov  = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap_ov.N)

        fig, ax = plt.subplots(figsize=(10, 9))
        ax.imshow(combo_vis, extent=ext, origin="upper",
                  cmap=cmap_ov, norm=norm_ov)
        aoi.boundary.plot(ax=ax, color="#333333", linewidth=1.5)
        add_scalebar(ax, scale_m, ext=ext)

        h_ov = [
            mpatches.Patch(color=C["overlap"],
                           label=f"SHARED lane -- both tractors\n"
                                 f"   {tci_r['overlap_m2']:.0f} m2"),
            mpatches.Patch(color=C["gross_only"],
                           label=f"{tname('gross')} only\n"
                                 f"   {tci_r['g_only_m2']:.0f} m2"),
            mpatches.Patch(color=C["klein_only"],
                           label=f"{tname('klein')} only\n"
                                 f"   {tci_r['k_only_m2']:.0f} m2"),
            mpatches.Patch(color=C["field"],
                           label="Not a confirmed permanent lane"),
        ]
        legend_right(ax, h_ov, title="Lane zone")
        ax.set_title(
            f"Flaeche {fid} -- Do both tractors follow the same lanes?\n"
            f"\n"
            f"Track Confinement Index: {tci_r['tci']*100:.1f}%  "
            f"({tci_r['tci_label']})\n"
            f"{tci_r['tci']*100:.1f}% of all confirmed traffic is in shared lanes\n"
            f"Median lateral offset between tractors: "
            f"{tci_r['offset_m']:.2f} m",
            fontsize=10, pad=10)
        ax.set_xlabel("Easting (m)"); ax.set_ylabel("Northing (m)")
        ax.set_aspect("equal")
        plt.tight_layout(rect=[0, 0, 0.78, 1])
        plt.savefig(os.path.join(out_dir, f"{fid}_map3_track_confinement.png"),
                    dpi=220, bbox_inches="tight")
        plt.close()

    # ----------------------------------------------------------
    #  MAP 4: Rut depth by tractor zone
    # ----------------------------------------------------------
    if R["tci"] and depth_by_zone:
        tci_r = R["tci"]
        fig, axes = plt.subplots(1, 3, figsize=(20, 7))
        fig.suptitle(
            f"Flaeche {fid} -- Rut depth: which tractor caused more soil compaction?",
            fontsize=12, fontweight="bold", y=1.02)
        vlim = 0.12
        zone_specs = [
            ("gross_only", tci_r["g_only"],
             f"{tname('gross')} only\n(no Fendt 724 passes here)"),
            ("klein_only", tci_r["k_only"],
             f"{tname('klein')} only\n(no Fendt 728 passes here)"),
            ("overlap",    tci_r["both"],
             "Both tractors\n(overlap zone)"),
        ]
        for ax, (zk, mask, label) in zip(axes, zone_specs):
            bg = np.where(aoi_mask, 1.0, np.nan)
            ax.imshow(bg, extent=ext, origin="upper",
                      cmap=ListedColormap([C["field"]]), vmin=0, vmax=1)
            d_vis = depth_r.copy(); d_vis[~mask] = np.nan
            if np.isfinite(d_vis).any():
                img = ax.imshow(d_vis, extent=ext, origin="upper",
                                cmap="RdYlBu", vmin=-vlim, vmax=0)
                plt.colorbar(img, ax=ax, shrink=0.6, pad=0.02,
                             label="Rut depth (m)")
            aoi.boundary.plot(ax=ax, color="#333333", linewidth=1.2)
            st = depth_by_zone.get(zk, {})
            mean_cm = st.get("mean_cm", np.nan)
            p10_cm  = st.get("p10_cm",  np.nan)
            ax.set_title(
                f"{label}\n"
                f"Average depth: {mean_cm:.1f} cm\n"
                f"Deep 10th percentile: {p10_cm:.1f} cm",
                fontsize=10, pad=8)
            ax.set_xlabel("Easting (m)"); ax.set_ylabel("Northing (m)")
            ax.set_aspect("equal")
            add_scalebar(ax, scale_m, ext=ext)
        fig.text(0.5, -0.03,
                 "Blue = shallow rut.   Red = deep rut.   "
                 "White/grey = outside the confirmed lane zone.",
                 ha="center", fontsize=9, style="italic")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{fid}_map4_depth_by_tractor.png"),
                    dpi=220, bbox_inches="tight")
        plt.close()

    # ----------------------------------------------------------
    #  MAP 5: Worst off-route operation
    #  One map per tractor showing where it deviated from the lane
    # ----------------------------------------------------------
    if len(date_df) > 0:
        for src in tractors:
            sub = date_df[date_df["src"]==src]
            if len(sub) == 0: continue
            worst = sub.iloc[0]
            best  = sub.iloc[-1]
            w_date = worst["date"]
            grp_w  = traffic[(traffic["src"]==src)&(traffic["date"]==w_date)]
            wr_r   = np.zeros((h,w), dtype=np.uint8)
            for geom in grp_w.geometry:
                wr_r += rasterize([(geom,1)], out_shape=(h,w),
                                  transform=transform, fill=0, dtype="uint8")

            fig, axes = plt.subplots(1, 2, figsize=(16, 8))
            fig.suptitle(
                f"Flaeche {fid}  --  {tname(src)}\n"
                f"GPS navigation accuracy: best vs worst operation",
                fontsize=13, fontweight="bold", y=1.02)

            for ax, (d, grp_date, pct_off, label, col_oper) in zip(axes, [
                (w_date, grp_w, worst["pct_off_lane"], "WORST operation", C["off_route"]),
                (best["date"],
                 traffic[(traffic["src"]==src)&(traffic["date"]==best["date"])],
                 best["pct_off_lane"], "BEST operation", "#2980b9"),
            ]):
                op_r = np.zeros((h,w), dtype=np.uint8)
                for geom in grp_date.geometry:
                    op_r += rasterize([(geom,1)], out_shape=(h,w),
                                      transform=transform, fill=0, dtype="uint8")

                # Field background
                bg = np.where(aoi_mask, 1.0, np.nan)
                ax.imshow(bg, extent=ext, origin="upper",
                          cmap=ListedColormap([C["field"]]), vmin=0, vmax=1)
                # Permanent lane (green)
                lane_v = np.where(combined_lane & aoi_mask, 1.0, np.nan)
                ax.imshow(lane_v, extent=ext, origin="upper",
                          cmap=ListedColormap([C["permanent"]]),
                          vmin=0, vmax=1, alpha=0.6)
                # This operation
                op_v = np.where((op_r>0)&aoi_mask, 1.0, np.nan)
                ax.imshow(op_v, extent=ext, origin="upper",
                          cmap=ListedColormap([col_oper]),
                          vmin=0, vmax=1, alpha=0.75)
                aoi.boundary.plot(ax=ax, color="#333333", linewidth=1.2)
                add_scalebar(ax, scale_m, ext=ext)
                row = worst if label=="WORST operation" else best
                ax.set_title(
                    f"{label}:  {d}\n"
                    f"On permanent lane: {row['pct_on_lane']:.1f}%\n"
                    f"Off permanent lane: {row['pct_off_lane']:.1f}%\n"
                    f"Median deviation from lane: {row['median_off_m']:.2f} m",
                    fontsize=10, pad=8)
                ax.set_xlabel("Easting (m)"); ax.set_ylabel("Northing (m)")
                ax.set_aspect("equal")

            h_leg5 = [
                mpatches.Patch(color=C["permanent"],   label="Permanent CTF lane"),
                mpatches.Patch(color=C["off_route"],   label="Worst operation area"),
                mpatches.Patch(color="#2980b9",        label="Best operation area"),
            ]
            fig.legend(handles=h_leg5, loc="lower center", ncol=3,
                       fontsize=9, frameon=True, bbox_to_anchor=(0.5, -0.04))
            plt.tight_layout()
            plt.savefig(
                os.path.join(out_dir,
                             f"{fid}_map5_{src}_best_vs_worst_operation.png"),
                dpi=220, bbox_inches="tight")
            plt.close()

    # ----------------------------------------------------------
    #  CHART 1: GPS accuracy -- lane width comparison
    # ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    src_list = tractors
    x = np.arange(len(src_list))
    bar_colors = [C["gross"], C["klein"]]
    measured   = [R[s]["w_mean"] for s in src_list]
    nominal    = [NOMINAL[s]["tyre_width"] for s in src_list]
    names      = [tname(s) for s in src_list]

    bars = ax.bar(x, measured, 0.4, color=bar_colors, alpha=0.85,
                  label="Measured confirmed lane width", zorder=3)
    ax.scatter(x, nominal, color="black", zorder=5, marker="D", s=80,
               label="Nominal tyre width")
    for bar, m, n in zip(bars, measured, nominal):
        ax.text(bar.get_x()+bar.get_width()/2,
                bar.get_height()+0.005,
                f"{m:.2f} m", ha="center", va="bottom", fontsize=10)
        ax.text(bar.get_x()+bar.get_width()/2,
                n+0.008, f"Nominal:\n{n:.2f} m",
                ha="center", va="bottom", fontsize=8, color="#555555")

    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=11)
    ax.set_ylabel("Lane width (m)")
    ax.set_ylim(0, max(nominal)*1.6)
    ax.set_title(f"Flaeche {fid} -- GPS navigation accuracy\n"
                 f"How wide is each tractor's confirmed wheel track?",
                 fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.text(0.02, 0.97,
            "The measured lane width is the width of the strip\n"
            "that was consistently hit across all operations.\n"
            "A width close to the nominal tyre width = good GPS.",
            transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.7))
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{fid}_chart1_gps_accuracy.png"),
                dpi=220, bbox_inches="tight")
    plt.close()

    # ----------------------------------------------------------
    #  CHART 2: Off-route bar chart -- all operations
    # ----------------------------------------------------------
    if len(date_df) > 0:
        fig, ax = plt.subplots(figsize=(12, 6))
        date_plot = date_df.copy()
        date_plot["label"] = (date_plot["tractor_name"] + "\n" +
                               date_plot["date"])
        x2 = np.arange(len(date_plot))
        bar_cols2 = [C["gross"] if s=="gross" else C["klein"]
                     for s in date_plot["src"]]
        bars2 = ax.bar(x2, date_plot["pct_off_lane"], color=bar_cols2,
                       alpha=0.85, zorder=3)
        ax.axhline(10, color="orange", lw=1.5, ls="--",
                   label="10% threshold (caution)")
        ax.axhline(20, color="red",    lw=1.5, ls="--",
                   label="20% threshold (action needed)")
        for bar, val in zip(bars2, date_plot["pct_off_lane"]):
            if val > 5:
                ax.text(bar.get_x()+bar.get_width()/2,
                        bar.get_height()+0.3,
                        f"{val:.1f}%", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x2)
        ax.set_xticklabels(date_plot["label"], rotation=30, ha="right",
                           fontsize=8)
        ax.set_ylabel("% of operation area outside permanent lane")
        ax.set_title(
            f"Flaeche {fid} -- How well did each operation stay on the permanent lane?\n"
            f"(lower bar = better GPS accuracy on that day)",
            fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        h_t = [mpatches.Patch(color=C["gross"], label=tname("gross")),
               mpatches.Patch(color=C["klein"], label=tname("klein"))]
        ax.legend(handles=h_t + [
            mpatches.Patch(color="orange", label="10% caution threshold"),
            mpatches.Patch(color="red",    label="20% action threshold"),
        ], fontsize=9, loc="upper right")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{fid}_chart2_offroute_per_operation.png"),
                    dpi=220, bbox_inches="tight")
        plt.close()

    # ----------------------------------------------------------
    #  CHART 3: Depth by tractor zone (bar chart)
    # ----------------------------------------------------------
    if depth_by_zone:
        zones = ["gross_only", "klein_only", "overlap"]
        zlabels = [f"{tname('gross')}\nonly",
                   f"{tname('klein')}\nonly",
                   "Both\ntractors"]
        zcols   = [C["gross_only"], C["klein_only"], C["overlap"]]
        means   = [depth_by_zone.get(z,{}).get("mean_cm",np.nan) for z in zones]
        p10s    = [depth_by_zone.get(z,{}).get("p10_cm", np.nan) for z in zones]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(
            f"Flaeche {fid} -- Soil compaction by tractor\n"
            f"(measured as rut depth at confirmed permanent lane locations)",
            fontsize=11, fontweight="bold")
        for ax, vals, ylabel, title in [
            (axes[0], means, "Mean rut depth (cm)", "Average rut depth"),
            (axes[1], p10s,  "p10 depth (cm)",
             "Deep ruts (10% of area deeper than this)"),
        ]:
            bars = ax.bar(np.arange(3), vals, color=zcols, alpha=0.85,
                          width=0.5, zorder=3)
            ax.bar_label(bars, fmt="%.1f cm", padding=4, fontsize=10)
            ax.set_xticks(np.arange(3)); ax.set_xticklabels(zlabels, fontsize=10)
            ax.set_ylabel(ylabel)
            ax.set_title(title, fontsize=10)
            ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{fid}_chart3_depth_by_tractor.png"),
                    dpi=220, bbox_inches="tight")
        plt.close()

    # -- Summary CSV -----------------------------------------
    tci_r = R.get("tci", {})
    summary = {"fid": fid, "field_area_m2": aoi_area_m2}
    for src in tractors:
        r = R[src]; nom = NOMINAL[src]
        summary.update({
            f"{src}_name":         tname(src),
            f"{src}_n_polys":      r["n_polys"],
            f"{src}_n_dates":      r["n_dates"],
            f"{src}_max_pass":     r["max_p"],
            f"{src}_thresh":       r["thresh"],
            f"{src}_lane_m2":      r["lane_area_m2"],
            f"{src}_lane_width_m": r["w_mean"],
            f"{src}_tyre_nom_m":   nom["tyre_width"],
            f"{src}_width_dev_m":  r["excess_m"],
        })
    summary.update({
        "tci":               tci_r.get("tci",     np.nan),
        "tci_label":         tci_r.get("tci_label",""),
        "overlap_m2":        tci_r.get("overlap_m2", np.nan),
        "gross_only_m2":     tci_r.get("g_only_m2",  np.nan),
        "klein_only_m2":     tci_r.get("k_only_m2",  np.nan),
        "median_offset_m":   tci_r.get("offset_m",   np.nan),
        "depth_gross_cm":    depth_by_zone.get("gross_only",{}).get("mean_cm",np.nan),
        "depth_klein_cm":    depth_by_zone.get("klein_only",{}).get("mean_cm",np.nan),
        "depth_overlap_cm":  depth_by_zone.get("overlap",{}).get("mean_cm",np.nan),
    })
    pd.DataFrame([summary]).to_csv(
        os.path.join(out_dir, f"{fid}_alignment_summary.csv"), index=False)

    print(f"\n  All outputs saved to: {out_dir}")
    return summary, date_df


# =========================================================
#  COMPARISON F1 vs F3
# =========================================================

def write_comparison(all_summaries, all_date_dfs):
    ensure_dir(COMPARISON_OUT)
    df = pd.DataFrame(all_summaries)
    df.to_csv(os.path.join(COMPARISON_OUT,"alignment_comparison.csv"), index=False)

    if all_date_dfs:
        date_all = pd.concat(all_date_dfs, ignore_index=True)
        date_all.to_csv(
            os.path.join(COMPARISON_OUT,"operation_offroute_all.csv"), index=False)

    fids = df["fid"].tolist()
    x    = np.arange(len(fids))
    FCOL = {"F1":"#2166ac", "F3":"#4dac26"}
    cols = [FCOL.get(f,"#999") for f in fids]

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("CTF Tractor Alignment -- Comparison F1 vs F3",
                 fontsize=14, fontweight="bold")

    def _bar(ax, vals, title, ylabel, fmt="{:.2f}", ymax=None, ref=None):
        bars = ax.bar(x, vals, color=cols, width=0.5, zorder=3)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x()+bar.get_width()/2,
                        bar.get_height()+0.01*max(abs(v)+0.01,0.01),
                        fmt.format(v), ha="center", va="bottom", fontsize=10)
        if ref is not None:
            ax.axhline(ref, color="black", lw=1.5, ls="--",
                       label=f"Nominal: {ref}")
            ax.legend(fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels(fids, fontsize=11)
        ax.set_title(title, fontsize=10); ax.set_ylabel(ylabel)
        if ymax: ax.set_ylim(0, ymax)
        ax.grid(axis="y", alpha=0.3)

    def col(name):
        return df[name].tolist() if name in df.columns else [np.nan]*len(fids)

    _bar(axes[0,0], col("gross_lane_width_m"),
         f"Fendt 728 (Gross)\nConfirmed lane width", "m",
         ref=NOMINAL["gross"]["tyre_width"])
    _bar(axes[0,1], col("klein_lane_width_m"),
         f"Fendt 724 (Klein)\nConfirmed lane width", "m",
         ref=NOMINAL["klein"]["tyre_width"])
    _bar(axes[0,2], col("tci"),
         "Track Confinement Index\n(1.0 = perfect CTF alignment)", "",
         fmt="{:.3f}", ymax=1.1)
    _bar(axes[1,0], col("depth_gross_cm"),
         f"Fendt 728 (Gross) only\nMean rut depth", "cm")
    _bar(axes[1,1], col("depth_klein_cm"),
         f"Fendt 724 (Klein) only\nMean rut depth", "cm")
    _bar(axes[1,2], col("depth_overlap_cm"),
         "Overlap zone\nMean rut depth (both tractors)", "cm")

    plt.tight_layout()
    plt.savefig(os.path.join(COMPARISON_OUT,"comparison_F1_F3.png"),
                dpi=220, bbox_inches="tight")
    plt.close()

    print(f"\n{'='*65}")
    print("  COMPARISON SUMMARY")
    print(f"{'='*65}")
    show = [c for c in ["fid","tci","tci_label","median_offset_m",
                         "gross_lane_width_m","klein_lane_width_m",
                         "depth_gross_cm","depth_klein_cm",
                         "depth_overlap_cm"] if c in df.columns]
    print(df[show].to_string(index=False))
    print(f"{'='*65}")
    print(f"Outputs: {COMPARISON_OUT}")


# =========================================================
#  MAIN
# =========================================================

def main():
    all_summaries, all_date_dfs = [], []
    for fid, cfg in CTF_FIELDS.items():
        summary, date_df = analyse_field(fid, cfg)
        all_summaries.append(summary)
        if len(date_df): all_date_dfs.append(date_df)
    write_comparison(all_summaries, all_date_dfs)
    print("\nOK  All done.")


if __name__ == "__main__":
    main()