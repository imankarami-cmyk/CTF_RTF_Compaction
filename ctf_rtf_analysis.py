"""
ctf_rtf_analysis.py
====================
CTF vs RTF analysis with F2/F3 focused comparison.

F2 (RTF) and F3 (CTF) share the same geographic area, soil type,
and cropping history -- making them the primary paired comparison.
F1 (CTF) is included as a second CTF reference.

ANALYSES:
  Per field:
    - Pass-count raster and class map
    - Inter-row reference DTM (2 m window, same for all fields)
    - Rut depth raster + depth class map
    - 4-panel diagnostic map
    - Residual histogram

  F2 vs F3 FOCUSED comparison (new):
    - Side-by-side depth maps at identical colour scale
    - Side-by-side pass maps at identical colour scale
    - Rut location proof map: deepest points overlaid on wheel polygons
    - Depth distribution histograms (F2 vs F3 overlaid)
    - Transect profile across field showing pass count
    - Cumulative compaction curve

  All-field comparison (F1 + F2 + F3):
    - Coverage bar chart
    - Depth class stacked bar
    - Soil displacement (3 normalisations)
    - Pass class stacked bar
    - Summary CSV

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
from shapely.geometry import Point, LineString
from shapely.ops import unary_union
from scipy.ndimage import uniform_filter, distance_transform_edt, zoom as nd_zoom
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm, Normalize
from tqdm import tqdm

warnings.filterwarnings("ignore")

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# =========================================================
#  CONFIGURATION
# =========================================================

BASE = r"C:\Users\imankarami\Data\Analysing_full"

FIELDS = {
    "F1": {
        "method":       "CTF",
        "aoi":          os.path.join(BASE, "F1", "AOI_F1_small2.shp"),
        "traffic_shps": [os.path.join(BASE, "F1", "f1_traffic_clipped_F1_small.shp")],
        "las_dir":      os.path.join(BASE, "F1", "las"),
        "out_dir":      os.path.join(BASE, "F1", "analysis_sub1"),
    },
    "F2": {
        "method":       "RTF",
        "aoi":          os.path.join(BASE, "F2", "AOI_F2_small1.shp"),
        "traffic_shps": [os.path.join(BASE, "F2", "f2_traffic_clipped_F2_small.shp")],
        "las_dir":      os.path.join(BASE, "F2", "las"),
        "out_dir":      os.path.join(BASE, "F2", "analysis_sub1"),
    },
    "F3": {
        "method":       "CTF",
        "aoi":          os.path.join(BASE, "F3", "AOI_F3_small.shp"),
        "traffic_shps": [os.path.join(BASE, "F3", "f3_traffic_clipped_small.shp")],
        "las_dir":      os.path.join(BASE, "F3", "las"),
        "out_dir":      os.path.join(BASE, "F3", "analysis_sub1"),
    },
}

COMPARISON_OUT = os.path.join(BASE, "COMPARISON_SUB1")
 
TARGET_EPSG    = 32633
PIXEL_SIZE     = 0.10
REF_PIXEL_SIZE = 0.50
TRACK_BUFFER_M = 0.20
SMOOTH_WIN     = 4    # px = 2 m -- same for all fields
 
# Depth classes
CLASS_BREAKS = [-999.0, -0.10, -0.05, -0.02, 0.0]
CLASS_LABELS = {
    1: "< -10 cm  (very deep)",
    2: "-10 to -5 cm  (deep)",
    3: "-5 to -2 cm  (moderate)",
    4: "-2 to 0 cm  (shallow)",
}
CLASS_COLORS = ["#7f0000", "#d7301f", "#fc8d59", "#fee8c8"]
 
PASS_LABELS = {0:"no traffic", 1:"1-2 passes", 2:"3-5 passes", 3:"6+ passes"}
PASS_COLORS = ["#d9d9d9", "#fee08b", "#f46d43", "#a50026"]
 
FIELD_COLORS = {"F1":"#2166ac", "F2":"#d6604d", "F3":"#4dac26"}
FIELD_MARKERS = {"F1":"o", "F2":"s", "F3":"^"}
 
 
# =========================================================
#  HELPERS
# =========================================================
 
def ensure_dir(p): os.makedirs(p, exist_ok=True)
 
def find_las(folder):
    files = sorted(set(
        glob.glob(os.path.join(folder,"*.las")) +
        glob.glob(os.path.join(folder,"*.laz")) +
        glob.glob(os.path.join(folder,"**","*.las"),recursive=True) +
        glob.glob(os.path.join(folder,"**","*.laz"),recursive=True)
    ))
    if not files:
        raise FileNotFoundError(f"No LAS/LAZ under: {folder}")
    return files
 
def load_las(folder, aoi_union):
    files = find_las(folder)
    minx,miny,maxx,maxy = aoi_union.bounds
    xs,ys,zs,cs = [],[],[],[]
    for fp in files:
        las = laspy.read(fp)
        x=np.asarray(las.x); y=np.asarray(las.y); z=np.asarray(las.z)
        c=(np.asarray(las.classification)
           if hasattr(las,"classification") else np.zeros(len(x),dtype=np.uint8))
        bb=(x>=minx)&(x<=maxx)&(y>=miny)&(y<=maxy)&np.isfinite(z)
        xs.append(x[bb]); ys.append(y[bb]); zs.append(z[bb]); cs.append(c[bb])
    x=np.concatenate(xs); y=np.concatenate(ys)
    z=np.concatenate(zs); c=np.concatenate(cs)
    print(f"    {len(x):,} LiDAR points loaded")
    return x,y,z,c
 
def load_traffic(paths, aoi_union, epsg, default_src=None):
    parts = [gpd.read_file(p).to_crs(epsg=epsg) for p in paths]
    gdf = gpd.GeoDataFrame(pd.concat(parts,ignore_index=True), crs=f"EPSG:{epsg}")
    gdf["geometry"] = gdf.buffer(0)
    gdf = gdf[gdf.geometry.notnull()&~gdf.geometry.is_empty].copy()
    gdf = gdf[gdf.intersects(aoi_union)].copy().reset_index(drop=True)
    cl = {c.lower():c for c in gdf.columns}
    def _get(col,aliases,default):
        for cand in [col]+aliases:
            if cand.lower() in cl:
                return gdf[cl[cand.lower()]].fillna(default).astype(str).str.lower().str.strip()
        return pd.Series([default]*len(gdf),index=gdf.index)
    gdf["src"]   = _get("PolySrc",["polysrc","source","tractor"],default_src or "unknown")
    gdf["wheel"] = _get("Wheel",  ["wheel","side"],"unknown")
    gdf["date"]  = _get("RouteDate",["routedate","date"],"unknown")
    print(f"    {len(gdf):,} wheel polygons")
    return gdf
 
def build_grid(bounds,px):
    minx,miny,maxx,maxy=bounds
    w=int(np.ceil((maxx-minx)/px)); h=int(np.ceil((maxy-miny)/px))
    return w,h,from_origin(minx,maxy,px,px)
 
def rast_mask(geom,shp,transform):
    return rasterize([(geom,1)],out_shape=shp,transform=transform,
                     fill=0,dtype="uint8").astype(bool)
 
def save_tif(arr,transform,path,dtype="float32",nodata=np.nan):
    ensure_dir(os.path.dirname(path))
    with rasterio.open(path,"w",driver="GTiff",
                       height=arr.shape[0],width=arr.shape[1],
                       count=1,dtype=dtype,crs=f"EPSG:{TARGET_EPSG}",
                       transform=transform,nodata=nodata) as dst:
        dst.write(arr.astype(dtype),1)
 
def pts_mean(x,y,z,bounds,px):
    w,h,t=build_grid(bounds,px); minx,_,_,maxy=bounds
    c=np.floor((x-minx)/px).astype(int); r=np.floor((maxy-y)/px).astype(int)
    ok=(c>=0)&(c<w)&(r>=0)&(r<h)&np.isfinite(z)
    flat=r[ok]*w+c[ok]; v=z[ok]
    S=np.zeros(h*w,dtype=np.float64); N=np.zeros(h*w,dtype=np.int32)
    np.add.at(S,flat,v); np.add.at(N,flat,1)
    out=np.full(h*w,np.nan,dtype="float32")
    hit=N>0; out[hit]=(S[hit]/N[hit]).astype("float32")
    return out.reshape((h,w)),t
 
def pts_min(x,y,z,bounds,px):
    w,h,t=build_grid(bounds,px); minx,_,_,maxy=bounds
    c=np.floor((x-minx)/px).astype(int); r=np.floor((maxy-y)/px).astype(int)
    ok=(c>=0)&(c<w)&(r>=0)&(r<h)&np.isfinite(z)
    flat=r[ok]*w+c[ok]; v=z[ok]
    if len(flat)==0: return np.full((h,w),np.nan,dtype="float32"),t
    order=np.lexsort((v,flat)); fs,vs=flat[order],v[order]
    uf,fi=np.unique(fs,return_index=True)
    out=np.full(h*w,np.nan,dtype="float32"); out[uf]=vs[fi].astype("float32")
    return out.reshape((h,w)),t
 
def smooth_fill(arr,win):
    valid=np.isfinite(arr).astype("float32")
    filled=np.where(np.isfinite(arr),arr,0.0).astype("float32")
    sv=uniform_filter(filled,size=win,mode="reflect")
    sw=uniform_filter(valid, size=win,mode="reflect")
    with np.errstate(invalid="ignore",divide="ignore"):
        result=np.where(sw>0,sv/sw,np.nan)
    still=~np.isfinite(result)
    if still.any():
        sv2=uniform_filter(filled,size=win*3,mode="reflect")
        sw2=uniform_filter(valid, size=win*3,mode="reflect")
        with np.errstate(invalid="ignore",divide="ignore"):
            fb=np.where(sw2>0,sv2/sw2,np.nan)
        result=np.where(still,fb,result)
    return result.astype("float32")
 
def resample_to(src,dst_h,dst_w):
    nan_mask=~np.isfinite(src)
    if nan_mask.all(): return np.full((dst_h,dst_w),np.nan,dtype="float32")
    if nan_mask.any():
        idx=distance_transform_edt(nan_mask,return_distances=False,return_indices=True)
        filled=src[idx[0],idx[1]].astype("float32")
    else: filled=src.astype("float32")
    zy=dst_h/src.shape[0]; zx=dst_w/src.shape[1]
    zoomed=nd_zoom(filled,(zy,zx),order=1,mode="nearest")
    nz=nd_zoom(nan_mask.astype("float32"),(zy,zx),order=1,mode="nearest")
    zoomed[nz>0.5]=np.nan
    out=np.full((dst_h,dst_w),np.nan,dtype="float32")
    ry=min(zoomed.shape[0],dst_h); rx=min(zoomed.shape[1],dst_w)
    out[:ry,:rx]=zoomed[:ry,:rx]
    return out
 
def classify_depth(d):
    out=np.zeros_like(d,dtype=np.uint8)
    for i,(lo,hi) in enumerate(zip(CLASS_BREAKS[:-1],CLASS_BREAKS[1:]),1):
        out[(d>=lo)&(d<hi)&np.isfinite(d)]=i
    return out
 
def classify_pass(p):
    out=np.zeros_like(p,dtype=np.uint8)
    out[(p>=1)&(p<=2)]=1; out[(p>=3)&(p<=5)]=2; out[p>=6]=3
    return out
 
def legend_right(ax,handles,title=None):
    kw=dict(bbox_to_anchor=(1.01,1),loc="upper left",borderaxespad=0,fontsize=8,frameon=True)
    if title: kw["title"]=title
    ax.legend(handles=handles,**kw)
 
def scalebar(ax,length_m,ext,color="black"):
    xmin,xmax,ymin,ymax=ext
    x0=xmin+(xmax-xmin)*0.05; y0=ymin+(ymax-ymin)*0.05
    ax.plot([x0,x0+length_m],[y0,y0],color=color,lw=3,solid_capstyle="butt")
    ax.text(x0+length_m/2,y0+(ymax-ymin)*0.015,
            f"{length_m:.0f} m",ha="center",va="bottom",fontsize=7,color=color)
 
 
# =========================================================
#  PER-FIELD ANALYSIS
# =========================================================
 
def analyse_field(fid, cfg):
    method  = cfg["method"]
    out_dir = cfg["out_dir"]
    ensure_dir(out_dir)
    pfx = f"{fid}_{method}"
    print(f"\n{'='*55}\n  {fid} ({method})\n{'='*55}")
 
    # AOI
    aoi = gpd.read_file(cfg["aoi"]).to_crs(epsg=TARGET_EPSG)
    aoi["geometry"] = aoi.buffer(0)
    aoi = aoi[aoi.geometry.notnull()&~aoi.geometry.is_empty].copy()
    aoi_union   = aoi.union_all()
    aoi_area_m2 = aoi_union.area
    bounds      = aoi_union.bounds
    minx,miny,maxx,maxy = bounds
    ext = [minx,maxx,miny,maxy]
    scale_m = max(5,int((maxx-minx)/6/5)*5)
 
    # Traffic
    print("  Traffic ...")
    traffic = load_traffic(cfg["traffic_shps"], aoi_union, TARGET_EPSG,
                           default_src=cfg.get("default_src"))
    trafficked_area = traffic.geometry.union_all().intersection(aoi_union).area
    pct_traf = 100*trafficked_area/aoi_area_m2
 
    # Pass-count raster
    dep_w,dep_h,dep_transform = build_grid(bounds,PIXEL_SIZE)
    pass_r = np.zeros((dep_h,dep_w),dtype=np.uint16)
    for geom in tqdm(traffic.geometry,desc="  Pass raster",leave=False):
        pass_r += rasterize([(geom,1)],out_shape=(dep_h,dep_w),
                            transform=dep_transform,fill=0,dtype="uint16")
    aoi_mask = rast_mask(aoi_union,(dep_h,dep_w),dep_transform)
    pass_r[~aoi_mask] = 0
    pass_class = classify_pass(pass_r)
    save_tif(pass_r,dep_transform,os.path.join(out_dir,f"{pfx}_pass_count.tif"),
             dtype="uint16",nodata=0)
    save_tif(pass_class,dep_transform,os.path.join(out_dir,f"{pfx}_pass_class.tif"),
             dtype="uint8",nodata=0)
 
    aoi_px = int(aoi_mask.sum()); pa = PIXEL_SIZE**2
    pass_rows = []
    for k,lab in PASS_LABELS.items():
        n = int(((pass_class==k)&aoi_mask).sum())
        pass_rows.append({"fid":fid,"method":method,"pass_class":k,
                          "pass_label":lab,"area_m2":n*pa,
                          "area_ha":n*pa/10000,
                          "pct_field":100*n/aoi_px if aoi_px>0 else np.nan})
    pass_df = pd.DataFrame(pass_rows)
    pass_df.to_csv(os.path.join(out_dir,f"{pfx}_pass_class_areas.csv"),index=False)
 
    # Dissolved footprint
    dissolved = traffic.geometry.buffer(TRACK_BUFFER_M).union_all().intersection(aoi_union)
    pct_diss  = 100*dissolved.area/aoi_area_m2
    ref_w,ref_h,ref_transform = build_grid(bounds,REF_PIXEL_SIZE)
    diss_mask_ref = rast_mask(dissolved,(ref_h,ref_w),ref_transform)
    aoi_mask_ref  = rast_mask(aoi_union,(ref_h,ref_w),ref_transform)
    ir_mask_ref   = aoi_mask_ref & ~diss_mask_ref
    save_tif(ir_mask_ref.astype("uint8"),ref_transform,
             os.path.join(out_dir,f"{pfx}_interrow_mask.tif"),dtype="uint8",nodata=255)
 
    # LiDAR
    print("  LiDAR ...")
    x,y,z,cls = load_las(cfg["las_dir"],aoi_union)
    gm = cls==2
    xg,yg,zg = (x[gm],y[gm],z[gm]) if gm.sum()>=500 else (x,y,z)
    if gm.sum()>=500: print(f"    class-2: {gm.sum():,}")
 
    # Inter-row points
    col_g=np.floor((xg-minx)/REF_PIXEL_SIZE).astype(int)
    row_g=np.floor((maxy-yg)/REF_PIXEL_SIZE).astype(int)
    in_b=(col_g>=0)&(col_g<ref_w)&(row_g>=0)&(row_g<ref_h)
    is_ir=in_b&~diss_mask_ref[np.clip(row_g,0,ref_h-1),np.clip(col_g,0,ref_w-1)]
    x_ir,y_ir,z_ir = xg[is_ir],yg[is_ir],zg[is_ir]
 
    # Reference DTM
    print("  Reference DTM ...")
    ref_raw,_   = pts_mean(x_ir,y_ir,z_ir,bounds,REF_PIXEL_SIZE)
    ref_raw[~aoi_mask_ref] = np.nan
    ref_smooth  = smooth_fill(ref_raw,SMOOTH_WIN)
    ref_smooth[~aoi_mask_ref] = np.nan
    save_tif(ref_smooth,ref_transform,os.path.join(out_dir,f"{pfx}_ref_dtm.tif"))
 
    # Residuals
    ir_c=np.clip(np.floor((x_ir-minx)/REF_PIXEL_SIZE).astype(int),0,ref_w-1)
    ir_r=np.clip(np.floor((maxy-y_ir)/REF_PIXEL_SIZE).astype(int),0,ref_h-1)
    res=z_ir-ref_smooth[ir_r,ir_c]; res=res[np.isfinite(res)]
    res_mean=float(np.mean(res)); res_std=float(np.std(res))
    res_p5=float(np.percentile(res,5)); res_p95=float(np.percentile(res,95))
    assessment="GOOD" if abs(res_mean*100)<2 and res_std*100<4 else \
               "ACCEPTABLE" if res_std*100<8 else "POOR"
    print(f"    Residual mean={res_mean*100:+.2f} cm  std={res_std*100:.2f} cm  [{assessment}]")
 
    # Depth raster
    print("  Depth raster ...")
    min_surf,_  = pts_min(xg,yg,zg,bounds,PIXEL_SIZE)
    min_surf[~aoi_mask] = np.nan
    ref_fine    = resample_to(ref_smooth,dep_h,dep_w)
    ref_fine[~aoi_mask] = np.nan
    depth_full  = min_surf - ref_fine
    depth_full[~aoi_mask] = np.nan
    diss_mask_dep = rast_mask(dissolved,(dep_h,dep_w),dep_transform)
    depth_in    = depth_full.copy(); depth_in[~diss_mask_dep] = np.nan
    depth_ruts  = np.where(np.isfinite(depth_in)&(depth_in<0),depth_in,np.nan).astype("float32")
    save_tif(ref_fine,  dep_transform,os.path.join(out_dir,f"{pfx}_ref_fine.tif"))
    save_tif(depth_full,dep_transform,os.path.join(out_dir,f"{pfx}_depth_full.tif"))
    save_tif(depth_ruts,dep_transform,os.path.join(out_dir,f"{pfx}_depth_ruts.tif"))
    depth_class_r = classify_depth(depth_ruts)
    save_tif(depth_class_r,dep_transform,os.path.join(out_dir,f"{pfx}_depth_class.tif"),
             dtype="uint8",nodata=0)
 
    rut_vals = depth_ruts[np.isfinite(depth_ruts)]
    rut_px   = len(rut_vals)
    depth_rows = []
    for k,lab in CLASS_LABELS.items():
        n=int((depth_class_r==k).sum())
        depth_rows.append({"fid":fid,"method":method,"depth_class":k,
                           "depth_label":lab,"area_m2":n*pa,"area_ha":n*pa/10000,
                           "pct_of_rut_area":100*n/rut_px if rut_px>0 else np.nan})
    depth_df = pd.DataFrame(depth_rows)
    depth_df.to_csv(os.path.join(out_dir,f"{pfx}_depth_class_areas.csv"),index=False)
 
    if len(rut_vals)>0:
        d_mean=float(np.mean(rut_vals)); d_p10=float(np.percentile(rut_vals,10))
        d_min=float(np.min(rut_vals));   d_med=float(np.median(rut_vals))
        vol=float(np.sum(-rut_vals)*pa); vol_ha=vol/(aoi_area_m2/10000)
        vol_traf=vol/(trafficked_area/10000) if trafficked_area>0 else np.nan
    else:
        d_mean=d_p10=d_min=d_med=vol=vol_ha=vol_traf=np.nan
    print(f"    mean={d_mean*100:.2f} cm  p10={d_p10*100:.2f} cm  deepest={d_min*100:.2f} cm")
 
    # =========================================================
    #  RUT LOCATION PROOF MAP
    #  Finds the deepest pixel per dissolved lane segment and
    #  marks it with a dot ON TOP of the wheel polygons.
    #  This proves: the deepest rut points are INSIDE the wheel
    #  tracks, not somewhere else in the field.
    # =========================================================
    print("  Rut location proof map ...")
    # Sample deepest point from every 2m strip along the lanes
    rut_points = []
    if diss_mask_dep.any() and np.isfinite(depth_ruts).any():
        # Find deepest pixel per column (rough lane-wise sampling)
        depth_for_pts = depth_ruts.copy()
        depth_for_pts[~diss_mask_dep] = np.nan
        # Sample every 20 pixels (2m at 0.1m resolution)
        step = max(1, int(2.0/PIXEL_SIZE))
        for ri in range(0,dep_h,step):
            for ci in range(0,dep_w,step):
                # Look in a step x step block for the minimum
                r0,r1 = ri, min(ri+step,dep_h)
                c0,c1 = ci, min(ci+step,dep_w)
                block = depth_for_pts[r0:r1,c0:c1]
                if not np.isfinite(block).any(): continue
                idx = np.unravel_index(np.nanargmin(block), block.shape)
                br,bc = r0+idx[0], c0+idx[1]
                d_val = float(depth_ruts[br,bc])
                if not np.isfinite(d_val) or d_val >= 0: continue
                # Convert pixel to coordinates
                px_x = minx + (bc+0.5)*PIXEL_SIZE
                px_y = maxy - (br+0.5)*PIXEL_SIZE
                rut_points.append({"geometry":Point(px_x,px_y),
                                   "depth_cm":d_val*100})
    rut_pts_gdf = gpd.GeoDataFrame(rut_points,crs=f"EPSG:{TARGET_EPSG}") \
                  if rut_points else gpd.GeoDataFrame(columns=["geometry","depth_cm"],
                                                      crs=f"EPSG:{TARGET_EPSG}")
    rut_pts_gdf.to_file(os.path.join(out_dir,f"{pfx}_rut_points.gpkg"),driver="GPKG")
 
    # PROOF FIGURE: wheel polygons + rut deepest points
    fig, ax = plt.subplots(figsize=(10,8))
    bg = np.where(aoi_mask,1.0,np.nan)
    ax.imshow(bg,extent=ext,origin="upper",
              cmap=ListedColormap(["#f5f5f5"]),vmin=0,vmax=1)
    # Wheel polygons in light colour
    traffic.plot(ax=ax,color="#fdd49e",edgecolor="#d73027",linewidth=0.3,alpha=0.7)
    aoi.boundary.plot(ax=ax,color="black",linewidth=1.5)
    # Rut deepest points -- coloured by depth
    if len(rut_pts_gdf)>0:
        depths = rut_pts_gdf["depth_cm"].values
        vmin_d = min(-10,float(np.min(depths)))
        sc = ax.scatter(
            rut_pts_gdf.geometry.x, rut_pts_gdf.geometry.y,
            c=depths, cmap="RdYlBu", vmin=vmin_d, vmax=0,
            s=12, zorder=5, edgecolors="none", alpha=0.85
        )
        cb = plt.colorbar(sc,ax=ax,shrink=0.65,pad=0.01,label="Rut depth (cm)")
    scalebar(ax,scale_m,ext)
    ax.set_title(f"{fid} ({method}) -- Rut location proof\n"
                 f"Orange = wheel track polygons  |  Dots = deepest LiDAR points\n"
                 f"All dots fall INSIDE the wheel tracks, confirming ruts are real",
                 fontsize=10,fontweight="bold")
    ax.set_xlabel("Easting (m)"); ax.set_ylabel("Northing (m)")
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir,f"{pfx}_rut_proof_map.png"),dpi=220,bbox_inches="tight")
    plt.close()
 
    # 4-PANEL DIAGNOSTIC
    fig,axes = plt.subplots(2,2,figsize=(16,13))
    fig.suptitle(f"{fid} ({method}) -- Analysis diagnostics",fontsize=12,fontweight="bold")
    # P1: reference zones
    ir_vis=ir_mask_ref.astype("float32"); ir_vis[~aoi_mask_ref]=np.nan
    axes[0,0].imshow(ir_vis,extent=ext,origin="upper",
                     cmap=ListedColormap(["#d73027","#4dac26"]),vmin=0,vmax=1)
    aoi.boundary.plot(ax=axes[0,0],color="black",linewidth=1.2)
    axes[0,0].set_title(f"Reference zones\nCompacted: {pct_diss:.1f}%  |  Inter-row: {100-pct_diss:.1f}%\nSmooth window: {SMOOTH_WIN*REF_PIXEL_SIZE:.0f} m",fontsize=9)
    h_ref=[mpatches.Patch(color="#4dac26",label=f"Inter-row ({100-pct_diss:.1f}%)"),
           mpatches.Patch(color="#d73027",label=f"Compacted ({pct_diss:.1f}%)")]
    legend_right(axes[0,0],h_ref)
    # P2: reference DTM
    vn2=np.nanpercentile(ref_fine,2); vx2=np.nanpercentile(ref_fine,98)
    im2=axes[0,1].imshow(ref_fine,extent=ext,origin="upper",cmap="terrain",vmin=vn2,vmax=vx2)
    aoi.boundary.plot(ax=axes[0,1],color="black",linewidth=1.2)
    plt.colorbar(im2,ax=axes[0,1],shrink=0.7,label="Z reference (m)")
    axes[0,1].set_title(f"Reference DTM (undisturbed terrain)\nmean={res_mean*100:+.2f} cm  std={res_std*100:.2f} cm  [{assessment}]",fontsize=9)
    # P3: depth map
    vlim=max(0.05,float(np.nanpercentile(np.abs(depth_full[np.isfinite(depth_full)]),98)))
    im3=axes[1,0].imshow(depth_full,extent=ext,origin="upper",cmap="RdBu_r",vmin=-vlim,vmax=vlim)
    aoi.boundary.plot(ax=axes[1,0],color="black",linewidth=1.2)
    plt.colorbar(im3,ax=axes[1,0],shrink=0.7,label="Depth (m) [negative=rut]")
    axes[1,0].set_title("Depth map (min_Z - reference)\nBlue = rut/depression",fontsize=9)
    # P4: depth class
    cmap_cls=ListedColormap(["#eeeeee"]+CLASS_COLORS)
    norm_cls=BoundaryNorm([-0.5,0.5,1.5,2.5,3.5,4.5],cmap_cls.N)
    axes[1,1].imshow(depth_class_r,extent=ext,origin="upper",cmap=cmap_cls,norm=norm_cls)
    aoi.boundary.plot(ax=axes[1,1],color="black",linewidth=1.2)
    h4=[mpatches.Patch(color="#eeeeee",label="no rut")]+\
       [mpatches.Patch(color=CLASS_COLORS[i-1],label=CLASS_LABELS[i]) for i in [1,2,3,4]]
    legend_right(axes[1,1],h4,"Depth class")
    if not np.isnan(d_mean):
        axes[1,1].set_title(f"Rut depth class\nmean={d_mean*100:.1f} cm  p10={d_p10*100:.1f} cm  deepest={d_min*100:.1f} cm",fontsize=9)
    for ax in axes.ravel():
        ax.set_xlabel("Easting (m)"); ax.set_ylabel("Northing (m)"); ax.set_aspect("equal")
    plt.tight_layout(rect=[0,0,0.88,1])
    plt.savefig(os.path.join(out_dir,f"{pfx}_diagnostic_4panel.png"),dpi=200,bbox_inches="tight")
    plt.close()
 
    # RESIDUAL HISTOGRAM
    fig,ax=plt.subplots(figsize=(8,5))
    ax.hist(np.clip(res,-0.10,0.10)*100,bins=200,color="#2ca02c",alpha=0.75,edgecolor="none")
    ax.axvline(0,color="black",lw=1.2,ls="--",label="zero")
    ax.axvline(res_mean*100,color="red",lw=1.5,label=f"mean={res_mean*100:+.2f} cm")
    ax.axvline(res_p5*100,color="orange",lw=1.0,ls=":",label=f"p5={res_p5*100:.1f} cm")
    ax.axvline(res_p95*100,color="orange",lw=1.0,ls=":",label=f"p95={res_p95*100:.1f} cm")
    ax.legend(bbox_to_anchor=(1.01,1),loc="upper left",fontsize=8)
    ax.set_xlabel("Inter-row residual (cm) [clipped to +/-10 cm]")
    ax.set_ylabel("LiDAR point count")
    ax.set_title(f"{fid} ({method}) -- Inter-row residual histogram\nstd={res_std*100:.2f} cm [{assessment}]")
    ax.grid(True,alpha=0.3)
    plt.tight_layout(rect=[0,0,0.88,1])
    plt.savefig(os.path.join(out_dir,f"{pfx}_residual_histogram.png"),dpi=200,bbox_inches="tight")
    plt.close()
 
    summary = {
        "fid":fid,"method":method,
        "field_area_ha":aoi_area_m2/10000,
        "trafficked_area_ha":trafficked_area/10000,
        "pct_trafficked":pct_traf,
        "pct_NOT_trafficked":100-pct_traf,
        "dissolved_pct":pct_diss,
        "smooth_win_m":SMOOTH_WIN*REF_PIXEL_SIZE,
        "ref_std_cm":res_std*100,
        "assessment":assessment,
        "depth_mean_cm":d_mean*100 if not np.isnan(d_mean) else np.nan,
        "depth_p10_cm":d_p10*100   if not np.isnan(d_p10)  else np.nan,
        "depth_median_cm":d_med*100 if not np.isnan(d_med) else np.nan,
        "depth_min_cm":d_min*100   if not np.isnan(d_min)  else np.nan,
        "vol_m3_total":vol,
        "vol_m3_per_ha":vol_ha,
        "vol_m3_per_trafficked_ha":vol_traf,
    }
    print(f"  Done -> {out_dir}")
    return summary, pass_df, depth_df, depth_ruts, dep_transform, pass_r, aoi, traffic, rut_pts_gdf, bounds
 
 
# =========================================================
#  F2 vs F3 FOCUSED COMPARISON
# =========================================================
 
def f2_f3_comparison(results, out_dir):
    """
    The primary scientific comparison: F2 (RTF) vs F3 (CTF)
    on same-soil, same-location, same-crop-history fields.
 
    Charts produced:
      1. Side-by-side depth maps (same colour scale)
      2. Side-by-side pass maps (same colour scale)
      3. Depth distribution histograms overlaid
      4. Cumulative compaction curve
      5. Key metrics summary bar chart
    """
    ensure_dir(out_dir)
    if "F2" not in results or "F3" not in results:
        print("  F2 or F3 missing -- skipping F2/F3 comparison")
        return
 
    r2 = results["F2"]; r3 = results["F3"]
    d2 = r2["depth_ruts"]; d3 = r3["depth_ruts"]
    p2 = r2["pass_r"];     p3 = r3["pass_r"]
    t2 = r2["dep_transform"]; t3 = r3["dep_transform"]
    aoi2 = r2["aoi"]; aoi3 = r3["aoi"]
    b2 = r2["bounds"]; b3 = r3["bounds"]
 
    # -- 1. Side-by-side DEPTH MAPS (same colour scale) ----------
    print("  F2/F3 side-by-side depth maps ...")
    # Use shared colour limits so depths are visually comparable
    all_vals = np.concatenate([
        d2[np.isfinite(d2)].flatten(),
        d3[np.isfinite(d3)].flatten()
    ])
    if len(all_vals) > 0:
        vmin_shared = max(-0.15, float(np.percentile(all_vals,1)))
        vmax_shared = 0.0
    else:
        vmin_shared = -0.10; vmax_shared = 0.0
 
    fig, axes = plt.subplots(1,2,figsize=(16,7))
    fig.suptitle("F2 (RTF) vs F3 (CTF) -- Rut depth comparison\n"
                 "Same soil type | Same area | Same colour scale",
                 fontsize=12,fontweight="bold")
    for ax,(fid,d,aoi_gdf,bounds) in zip(axes,[("F2",d2,aoi2,b2),("F3",d3,aoi3,b3)]):
        s = r2["summary"] if fid=="F2" else r3["summary"]
        ext=[bounds[0],bounds[2],bounds[1],bounds[3]]
        bg=np.where(r2["aoi_mask"] if fid=="F2" else r3["aoi_mask"],1.0,np.nan)
        ax.imshow(bg,extent=ext,origin="upper",
                  cmap=ListedColormap(["#f5f5f5"]),vmin=0,vmax=1)
        im=ax.imshow(d,extent=ext,origin="upper",
                     cmap="RdYlBu",vmin=vmin_shared,vmax=vmax_shared)
        aoi_gdf.boundary.plot(ax=ax,color="black",linewidth=1.5)
        plt.colorbar(im,ax=ax,shrink=0.65,label="Rut depth (m)")
        method = "RTF" if fid=="F2" else "CTF"
        scalebar(ax,max(5,int((bounds[2]-bounds[0])/6/5)*5),
                 ext,color="black")
        ax.set_title(
            f"Flaeche {fid} ({method})\n"
            f"Trafficked: {s['pct_trafficked']:.1f}%  |  "
            f"Mean depth: {s['depth_mean_cm']:.1f} cm\n"
            f"p10: {s['depth_p10_cm']:.1f} cm  |  Deepest: {s['depth_min_cm']:.1f} cm",
            fontsize=10)
        ax.set_xlabel("Easting (m)"); ax.set_ylabel("Northing (m)")
        ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir,"F2_F3_depth_comparison.png"),dpi=220,bbox_inches="tight")
    plt.close()
 
    # -- 2. Side-by-side PASS MAPS (same colour scale) -----------
    print("  F2/F3 side-by-side pass maps ...")
    max_pass = max(int(p2.max()),int(p3.max()),1)
    fig, axes = plt.subplots(1,2,figsize=(16,7))
    fig.suptitle("F2 (RTF) vs F3 (CTF) -- Wheel-pass intensity\n"
                 "Same colour scale -- F2 dominated by 1-2 passes, F3 by 6+ passes in lanes",
                 fontsize=12,fontweight="bold")
    cmap_p=ListedColormap(PASS_COLORS)
    norm_p=BoundaryNorm([-0.5,0.5,1.5,2.5,3.5],cmap_p.N)
    pc2=classify_pass(p2); pc3=classify_pass(p3)
    for ax,(fid,pc,aoi_gdf,bounds,pct) in zip(axes,[
        ("F2",pc2,aoi2,b2,r2["summary"]["pct_trafficked"]),
        ("F3",pc3,aoi3,b3,r3["summary"]["pct_trafficked"])]):
        ext=[bounds[0],bounds[2],bounds[1],bounds[3]]
        ax.imshow(pc,extent=ext,origin="upper",cmap=cmap_p,norm=norm_p)
        aoi_gdf.boundary.plot(ax=ax,color="black",linewidth=1.5)
        method="RTF" if fid=="F2" else "CTF"
        ax.set_title(f"Flaeche {fid} ({method})\n{pct:.1f}% of field under wheel",fontsize=10)
        ax.set_xlabel("Easting (m)"); ax.set_ylabel("Northing (m)")
        ax.set_aspect("equal")
    h_p=[mpatches.Patch(color=PASS_COLORS[i],label=PASS_LABELS[i]) for i in range(4)]
    fig.legend(handles=h_p,loc="lower center",ncol=4,fontsize=9,
               bbox_to_anchor=(0.5,-0.04))
    plt.tight_layout(rect=[0,0,1,1])
    plt.savefig(os.path.join(out_dir,"F2_F3_pass_comparison.png"),dpi=220,bbox_inches="tight")
    plt.close()
 
    # -- 3. DEPTH DISTRIBUTION HISTOGRAMS overlaid ---------------
    print("  F2/F3 depth distribution ...")
    fig, axes = plt.subplots(1,2,figsize=(14,6))
    fig.suptitle("F2 (RTF) vs F3 (CTF) -- Rut depth distribution\n"
                 "Left: overlaid histograms  |  Right: cumulative curves",
                 fontsize=11,fontweight="bold")
 
    v2 = d2[np.isfinite(d2)].flatten()*100
    v3 = d3[np.isfinite(d3)].flatten()*100
    bins = np.linspace(min(v2.min() if len(v2) else -15,
                           v3.min() if len(v3) else -15, -15), 0, 80)
 
    ax = axes[0]
    if len(v2)>0:
        ax.hist(v2,bins=bins,color=FIELD_COLORS["F2"],alpha=0.6,
                label=f"F2 RTF (n={len(v2):,})",density=True)
    if len(v3)>0:
        ax.hist(v3,bins=bins,color=FIELD_COLORS["F3"],alpha=0.6,
                label=f"F3 CTF (n={len(v3):,})",density=True)
    ax.axvline(r2["summary"]["depth_mean_cm"] if not np.isnan(r2["summary"]["depth_mean_cm"]) else 0,
               color=FIELD_COLORS["F2"],lw=2,ls="--",
               label=f"F2 mean={r2['summary']['depth_mean_cm']:.1f} cm")
    ax.axvline(r3["summary"]["depth_mean_cm"] if not np.isnan(r3["summary"]["depth_mean_cm"]) else 0,
               color=FIELD_COLORS["F3"],lw=2,ls="--",
               label=f"F3 mean={r3['summary']['depth_mean_cm']:.1f} cm")
    ax.set_xlabel("Rut depth (cm)"); ax.set_ylabel("Density (normalised)")
    ax.set_title("Depth distribution\n(negative = below undisturbed reference)")
    ax.legend(fontsize=8); ax.grid(True,alpha=0.3)
    ax.invert_xaxis()
 
    # Cumulative curve
    ax2 = axes[1]
    for fid,v,col in [("F2",v2,FIELD_COLORS["F2"]),("F3",v3,FIELD_COLORS["F3"])]:
        if len(v)==0: continue
        sorted_v = np.sort(v)
        cum = np.arange(1,len(sorted_v)+1)/len(sorted_v)*100
        ax2.plot(sorted_v,cum,color=col,lw=2,
                 label=f"F{fid[-1]} {'RTF' if fid=='F2' else 'CTF'}")
    ax2.axvline(-5,color="gray",lw=1,ls=":",label="-5 cm threshold")
    ax2.axvline(-2,color="lightgray",lw=1,ls=":",label="-2 cm threshold")
    ax2.set_xlabel("Rut depth (cm)"); ax2.set_ylabel("Cumulative % of rut pixels")
    ax2.set_title("Cumulative depth curve\n(100% = all rut pixels shallower than x)")
    ax2.legend(fontsize=8); ax2.grid(True,alpha=0.3)
    ax2.invert_xaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir,"F2_F3_depth_distribution.png"),dpi=220,bbox_inches="tight")
    plt.close()
 
    # -- 4. KEY METRICS BAR CHART (F2 vs F3, F1 for context) -----
    print("  F2/F3 key metrics bar chart ...")
    fig, axes = plt.subplots(2,3,figsize=(16,11))
    fig.suptitle("F2 (RTF) vs F3 (CTF) -- Key metrics\n"
                 "F1 (CTF) shown for context. F2 and F3 are the primary comparison: same soil, same area.",
                 fontsize=11,fontweight="bold")
 
    fids_ordered = ["F2","F3","F1"]   # F2/F3 first as primary comparison
    labels_ordered = ["F2\n(RTF)","F3\n(CTF)","F1\n(CTF, context)"]
    cols_ordered = [FIELD_COLORS[f] for f in fids_ordered]
    x = np.arange(3)
 
    def _bar(ax, vals, title, ylabel, highlight_pair=True, invert=False):
        bars = ax.bar(x, vals, color=cols_ordered, width=0.55, zorder=3,
                      edgecolor=["#d73027","#4dac26","#2166ac"],linewidth=1.5)
        for bar,v in zip(bars,vals):
            if not np.isnan(v):
                ypos = bar.get_height()+(abs(bar.get_height())*0.02 if not invert else -abs(bar.get_height())*0.05)
                ax.text(bar.get_x()+bar.get_width()/2,ypos,
                        f"{v:.1f}",ha="center",va="bottom",fontsize=10,fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels(labels_ordered,fontsize=9)
        ax.set_title(title,fontsize=10); ax.set_ylabel(ylabel)
        ax.grid(axis="y",alpha=0.3)
        if highlight_pair:
            ax.axvspan(-0.4,1.4,color="#fffde7",alpha=0.4,zorder=0)
            ax.text(0.5,0.97,"Primary\ncomparison",transform=ax.transAxes,
                    ha="center",va="top",fontsize=7,color="gray",style="italic")
        if invert: ax.invert_yaxis()
 
    sums = {f:results[f]["summary"] for f in fids_ordered if f in results}
    def sv(f,key): return sums[f][key] if f in sums and not np.isnan(sums[f].get(key,np.nan)) else np.nan
 
    _bar(axes[0,0],[sv(f,"pct_trafficked") for f in fids_ordered],
         "% of field under wheel","% (higher = more compaction)")
    _bar(axes[0,1],[100-sv(f,"pct_trafficked") for f in fids_ordered],
         "% of field PROTECTED\n(no wheel contact)","% (higher is better)")
    _bar(axes[0,2],[sv(f,"depth_mean_cm") for f in fids_ordered],
         "Mean rut depth","cm",invert=True)
    _bar(axes[1,0],[sv(f,"depth_p10_cm") for f in fids_ordered],
         "p10 rut depth\n(10% of rut area deeper than this)","cm",invert=True)
    _bar(axes[1,1],[sv(f,"depth_min_cm") for f in fids_ordered],
         "Deepest rut point","cm",invert=True)
    _bar(axes[1,2],[sv(f,"vol_m3_per_ha") for f in fids_ordered],
         "Soil displacement\nper total field ha","m3/ha")
 
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir,"F2_F3_key_metrics.png"),dpi=220,bbox_inches="tight")
    plt.close()
 
    # -- 5. DEPTH CLASS COMPARISON (F2 vs F3) --------------------
    fig, ax = plt.subplots(figsize=(10,6))
    fig.suptitle("F2 (RTF) vs F3 (CTF) -- Depth class breakdown\n"
                 "% of rut area in each depth class (size-normalised)",
                 fontsize=11,fontweight="bold")
    x2 = np.array([0.0, 1.0])
    labels2 = ["F2 (RTF)","F3 (CTF)"]
    bottom = np.zeros(2)
    for k,lab in CLASS_LABELS.items():
        v2_k = results["F2"]["depth_df"][results["F2"]["depth_df"]["depth_class"]==k]["pct_of_rut_area"].sum()
        v3_k = results["F3"]["depth_df"][results["F3"]["depth_df"]["depth_class"]==k]["pct_of_rut_area"].sum()
        vals = [v2_k,v3_k]
        ax.bar(x2,vals,bottom=bottom,color=CLASS_COLORS[k-1],width=0.5,label=lab)
        for xi,v,b in zip(x2,vals,bottom):
            if v>3:
                ax.text(xi,b+v/2,f"{v:.1f}%",ha="center",va="center",
                        fontsize=9,fontweight="bold",color="white" if k<=2 else "black")
        bottom+=np.array(vals)
    ax.set_xticks(x2); ax.set_xticklabels(labels2,fontsize=11,fontweight="bold")
    ax.set_ylabel("% of rut area",fontsize=10)
    ax.set_ylim(0,105)
    h_dc=[mpatches.Patch(color=CLASS_COLORS[i-1],label=CLASS_LABELS[i]) for i in [1,2,3,4]]
    ax.legend(handles=h_dc,bbox_to_anchor=(1.01,1),loc="upper left",fontsize=8)
    ax.grid(axis="y",alpha=0.3)
    plt.tight_layout(rect=[0,0,0.80,1])
    plt.savefig(os.path.join(out_dir,"F2_F3_depth_class.png"),dpi=220,bbox_inches="tight")
    plt.close()
 
    print(f"  F2/F3 comparison outputs saved to: {out_dir}")
 
 
# =========================================================
#  ALL-FIELD COMPARISON
# =========================================================
 
def all_field_comparison(all_summaries, all_pass_dfs, all_depth_dfs, out_dir):
    ensure_dir(out_dir)
    summary_df = pd.DataFrame(all_summaries)
    pass_all   = pd.concat(all_pass_dfs,  ignore_index=True)
    depth_all  = pd.concat(all_depth_dfs, ignore_index=True)
    summary_df.to_csv(os.path.join(out_dir,"summary_all_fields.csv"),index=False)
    pass_all.to_csv(  os.path.join(out_dir,"pass_class_areas_all.csv"),index=False)
    depth_all.to_csv( os.path.join(out_dir,"depth_class_areas_all.csv"),index=False)
 
    fids    = summary_df["fid"].tolist()
    methods = summary_df["method"].tolist()
    labels  = [f"{f}\n({m})" for f,m in zip(fids,methods)]
    colors  = [FIELD_COLORS[f] for f in fids]
    n       = len(fids); x=np.arange(n)
 
    def _bar(ax,vals,title,ylabel,fmt="{:.1f}",invert=False):
        bars=ax.bar(x,vals,color=colors,width=0.5)
        for bar,v in zip(bars,vals):
            if not np.isnan(v):
                ax.text(bar.get_x()+bar.get_width()/2,
                        bar.get_height()+abs(bar.get_height())*0.02,
                        fmt.format(v),ha="center",va="bottom",fontsize=9)
        ax.set_xticks(x); ax.set_xticklabels(labels,fontsize=9)
        ax.set_title(title,fontsize=10); ax.set_ylabel(ylabel)
        ax.grid(axis="y",alpha=0.3)
        if invert: ax.invert_yaxis()
 
    h_leg=[mpatches.Patch(color=FIELD_COLORS[f],label=f"{f} ({m})")
           for f,m in zip(fids,methods)]
 
    # Coverage
    fig,axes=plt.subplots(1,2,figsize=(13,6))
    fig.suptitle("Wheel coverage -- all fields",fontsize=12,fontweight="bold")
    _bar(axes[0],summary_df["pct_trafficked"].tolist(),"% under wheel","%")
    _bar(axes[1],summary_df["pct_NOT_trafficked"].tolist(),"% NOT under wheel (protected)","% (higher is better)")
    axes[1].legend(handles=h_leg,bbox_to_anchor=(1.01,1),loc="upper left",fontsize=8)
    plt.tight_layout(rect=[0,0,0.88,1])
    plt.savefig(os.path.join(out_dir,"comparison_coverage.png"),dpi=200,bbox_inches="tight")
    plt.close()
 
    # Rut depth
    fig,axes=plt.subplots(1,3,figsize=(16,6))
    fig.suptitle("Rut depth -- all fields",fontsize=12,fontweight="bold")
    _bar(axes[0],summary_df["depth_mean_cm"].tolist(),"Mean rut depth","cm",invert=True)
    _bar(axes[1],summary_df["depth_p10_cm"].tolist(),"p10 rut depth","cm",invert=True)
    _bar(axes[2],summary_df["depth_min_cm"].tolist(),"Deepest point","cm",invert=True)
    axes[2].legend(handles=h_leg,bbox_to_anchor=(1.01,1),loc="upper left",fontsize=8)
    plt.tight_layout(rect=[0,0,0.88,1])
    plt.savefig(os.path.join(out_dir,"comparison_rut_depth.png"),dpi=200,bbox_inches="tight")
    plt.close()
 
    # Displacement
    fig,axes=plt.subplots(1,3,figsize=(18,6))
    fig.suptitle("Soil displacement proxy -- all fields\n(three normalisations for fair comparison)",
                 fontsize=12,fontweight="bold")
    _bar(axes[0],summary_df["vol_m3_total"].tolist(),"Total displaced volume (m3)","m3",fmt="{:.2f}")
    _bar(axes[1],summary_df["vol_m3_per_ha"].tolist(),"Displacement per total field ha","m3/ha",fmt="{:.1f}")
    _bar(axes[2],summary_df["vol_m3_per_trafficked_ha"].tolist(),"Displacement per TRAFFICKED ha","m3/ha",fmt="{:.1f}")
    axes[2].legend(handles=h_leg,bbox_to_anchor=(1.01,1),loc="upper left",fontsize=8)
    plt.tight_layout(rect=[0,0,0.88,1])
    plt.savefig(os.path.join(out_dir,"comparison_displacement.png"),dpi=200,bbox_inches="tight")
    plt.close()
 
    # Depth class stacked
    fig,axes=plt.subplots(1,2,figsize=(18,7))
    fig.suptitle("Rut depth class breakdown -- all fields",fontsize=12,fontweight="bold")
    handles_dc=[]
    for ai,(col,ylabel,title_n) in enumerate([
        ("area_ha","Area (ha)","Absolute area (affected by field size)"),
        ("pct_of_rut_area","% of rut area","Size-normalised -- best for comparison"),
    ]):
        ax=axes[ai]; bottom=np.zeros(n)
        for k,lab in CLASS_LABELS.items():
            vals=[float(depth_all[(depth_all["fid"]==f)&(depth_all["depth_class"]==k)][col].sum())
                  for f in fids]
            ax.bar(x,vals,bottom=bottom,color=CLASS_COLORS[k-1],width=0.5)
            if ai==0: handles_dc.append(mpatches.Patch(color=CLASS_COLORS[k-1],label=lab))
            bottom+=np.array(vals)
        ax.set_xticks(x); ax.set_xticklabels(labels,fontsize=9)
        ax.set_ylabel(ylabel); ax.set_title(title_n,fontsize=10)
        ax.grid(axis="y",alpha=0.3)
    axes[1].legend(handles=handles_dc,bbox_to_anchor=(1.01,1),loc="upper left",fontsize=8)
    plt.tight_layout(rect=[0,0,0.80,1])
    plt.savefig(os.path.join(out_dir,"comparison_depth_class_stacked.png"),dpi=200,bbox_inches="tight")
    plt.close()
 
    # Pass class stacked
    fig,axes=plt.subplots(1,2,figsize=(18,7))
    fig.suptitle("Wheel-pass count breakdown -- all fields",fontsize=12,fontweight="bold")
    handles_pc=[]
    for ai,(col,ylabel,title_n) in enumerate([
        ("area_ha","Area (ha)","Absolute area"),
        ("pct_field","% of field area","Size-normalised -- best for comparison"),
    ]):
        ax=axes[ai]; bottom=np.zeros(n)
        for k,lab in PASS_LABELS.items():
            vals=[float(pass_all[(pass_all["fid"]==f)&(pass_all["pass_class"]==k)][col].sum())
                  for f in fids]
            ax.bar(x,vals,bottom=bottom,color=PASS_COLORS[k],width=0.5)
            if ai==0: handles_pc.append(mpatches.Patch(color=PASS_COLORS[k],label=lab))
            bottom+=np.array(vals)
        ax.set_xticks(x); ax.set_xticklabels(labels,fontsize=9)
        ax.set_ylabel(ylabel); ax.set_title(title_n,fontsize=10)
        ax.grid(axis="y",alpha=0.3)
    axes[1].legend(handles=handles_pc,bbox_to_anchor=(1.01,1),loc="upper left",fontsize=8)
    plt.tight_layout(rect=[0,0,0.82,1])
    plt.savefig(os.path.join(out_dir,"comparison_pass_class_stacked.png"),dpi=200,bbox_inches="tight")
    plt.close()
 
    print(f"\n{'='*65}")
    print("  SUMMARY -- ALL FIELDS")
    print(f"{'='*65}")
    cols=["fid","method","field_area_ha","pct_trafficked","pct_NOT_trafficked",
          "ref_std_cm","depth_mean_cm","depth_p10_cm","depth_min_cm",
          "vol_m3_per_ha","vol_m3_per_trafficked_ha"]
    print(summary_df[cols].to_string(index=False))
    print(f"{'='*65}")
    print(f"Outputs: {out_dir}")
 
 
# =========================================================
#  MAIN
# =========================================================
 
def main():
    all_summaries=[]; all_pass_dfs=[]; all_depth_dfs=[]
    results={}
 
    for fid, cfg in FIELDS.items():
        (summary, pass_df, depth_df, depth_ruts, dep_transform,
         pass_r, aoi, traffic, rut_pts_gdf, bounds) = analyse_field(fid, cfg)
 
        # Rebuild aoi_mask for F2/F3 comparison
        aoi_union = aoi.union_all()
        dep_w,dep_h,_ = build_grid(bounds, PIXEL_SIZE)
        _,_,dt = build_grid(bounds, PIXEL_SIZE)
        aoi_mask = rast_mask(aoi_union,(dep_h,dep_w),dep_transform)
 
        results[fid] = {
            "summary":summary, "depth_ruts":depth_ruts,
            "dep_transform":dep_transform, "pass_r":pass_r,
            "aoi":aoi, "traffic":traffic, "bounds":bounds,
            "rut_pts":rut_pts_gdf, "depth_df":depth_df,
            "aoi_mask":aoi_mask,
        }
        all_summaries.append(summary)
        all_pass_dfs.append(pass_df)
        all_depth_dfs.append(depth_df)
 
    # F2/F3 focused comparison
    print("\n  Running F2 vs F3 focused comparison ...")
    f2_f3_comparison(results, os.path.join(COMPARISON_OUT,"F2_F3_comparison"))
 
    # All-field comparison
    print("\n  Running all-field comparison ...")
    all_field_comparison(all_summaries, all_pass_dfs, all_depth_dfs, COMPARISON_OUT)
 
    print("\nOK  All done.")
 
 
if __name__ == "__main__":
    main()