#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#  flightplot.py
#  ----------------
#  Generate a bundled flight summary figure from airdatauav exported CSV logs
#  for drone-related scenarios and engagement reports.
#
#  Author:  Alec Belsher (@brathdadair)
#
#  Created: 03-15-2021
#  Updated: 06-27-2025
#
#  License: MIT
#  Copyright © 2025 spooksec.  All rights reserved.
#
#  Usage:
#      python flightplot.py <flight.csv> [manual_events.json] [margin%]
#
#  Python >= 3.9,  Dependencies: pandas, numpy, matplotlib, python-dateutil
# --------------------------------------------------------------------

from __future__ import annotations
import sys, json, pathlib
from datetime import datetime, timedelta, timezone
from dateutil.parser import parse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import textwrap
from matplotlib.collections import LineCollection
import matplotlib.patheffects as pe


# column names
MSL_COL = "altitude_above_seaLevel(feet)"
AGL_COL = [
    "height_above_ground_at_drone_location(feet)",
    "altitudeAGL(feet)",
    "relativeAltitude(feet)"
]

# required columns (airdata defaults) - pick first match
REQ_COLS = [
    "latitude", "longitude", "speed(mph)",
    MSL_COL, "time(millisecond)", "datetime(utc)"
]

# extra columns (drone specific) - placeholder is example
EXTRA_COLS = {
    "photo": "Photo taken",
    "video": "Recording started",
    "c1":    "C1 activated",
    "c2":    "C2 activated"
}

# formatting helpers
fmt_hms_utc = lambda ms, ref_utc: (
        ref_utc + timedelta(milliseconds=ms)
).astimezone(timezone.utc).strftime("%H:%M:%S")

outline = [pe.withStroke(linewidth=2, foreground="black")]
numsty  = dict(color="white", fontsize=11, fontweight="bold",
               ha="center", va="center", path_effects=outline)

# csv helpers
def choose_alt(df: pd.DataFrame) -> str:
    for col in AGL_COL:
        if col in df.columns and pd.to_numeric(df[col], errors="coerce").notna().mean() > .9:
            return col
    return MSL_COL

def safe_dt(sample, ts_fallback) -> datetime:
    """Robust UTC parser for the AirData 'datetime(utc)' field."""
    if isinstance(sample, str):
        sample = " ".join(sample.split())        # collapse any weird spacing
        try:
            dt = parse(sample)
            # force UTC if tzinfo missing
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)
            return dt
        except (ValueError, OverflowError):
            pass
    # last-ditch fallback: file modification time
    return datetime.fromtimestamp(ts_fallback, tz=timezone.utc)

def load_csv(path: str):
    df = pd.read_csv(path)
    missing = [c for c in REQ_COLS if c not in df.columns]
    if missing:
        sys.exit(f"CSV missing: {', '.join(missing)}")

    alt_col = choose_alt(df)
    df[alt_col] = pd.to_numeric(df[alt_col], errors="coerce")

    # ── NEW: look for first real timestamp, not always iloc[0]
    dt_iter = next(
        (v for v in df["datetime(utc)"] if isinstance(v, str) and v.strip()),
        None
    )
    start_utc = safe_dt(
        dt_iter,
        pathlib.Path(path).stat().st_mtime
    )
    return df.dropna(subset=["latitude", "longitude", alt_col]), alt_col, start_utc

# event helpers
def detect_core(df, alt_col):
    events=[]; txt=[c for c in df.columns if df[c].dtype==object]
    for kw,label in EXTRA_COLS.items():
        hit=pd.Series(False,index=df.index)
        for col in txt: hit |= df[col].str.contains(kw,case=False,na=False)
        if hit.any():
            i=int(hit.idxmax())
            events.append(dict(idx=i,time_ms=int(df.at[i,"time(millisecond)"]),
                               lat=df.at[i,"latitude"],lon=df.at[i,"longitude"],
                               alt=df.at[i,alt_col],label=label))
    return events

def nearest_row(df,t):
    idx=np.searchsorted(df["time(millisecond)"].values,t)
    if idx==0: return 0
    if idx>=len(df): return len(df)-1
    a,b=df["time(millisecond)"].iloc[idx-1],df["time(millisecond)"].iloc[idx]
    return idx if abs(b-t)<abs(a-t) else idx-1

def load_manual(df,path,alt_col):
    if not path: return []
    with open(path) as f: data=json.load(f)
    if not (isinstance(data,list) and data and "time_ms" in data[0]): return []
    out=[]
    for m in data:
        t=int(m["time_ms"]); i=nearest_row(df,t)
        out.append(dict(idx=i,time_ms=t,lat=df.at[i,"latitude"],
                        lon=df.at[i,"longitude"],alt=df.at[i,alt_col],
                        label=m["event"]))
    return out

def lcoll(x,y,c):
    seg=np.concatenate([np.column_stack([x,y])[:-1,None,:],
                        np.column_stack([x,y])[1:,None,:]],axis=1)
    return LineCollection(seg, array=c, cmap="viridis",
                          linewidth=2.5, alpha=0.9, zorder=3)

# main plot
def plot_bundle(df,alt_col,start_utc,core_ev,manual_ev,margin_pct=15.0):

    events=core_ev+manual_ev
    events.sort(key=lambda d:d["time_ms"])
    for n,e in enumerate(events,1): e["num"]=n

    lon,lat,spd = df["longitude"],df["latitude"],df["speed(mph)"]
    df["tmin"]   = df["time(millisecond)"]/60000
    t,alt        = df["tmin"],df[alt_col]

    fig = plt.figure(figsize=(14,8), constrained_layout=True)
    gs  = fig.add_gridspec(2,3,width_ratios=[20,17,1],
                           height_ratios=[1,1], wspace=0.04, hspace=0.18)
    ax_evt = fig.add_subplot(gs[0,0]); ax_evt.axis("off")
    ax_map = fig.add_subplot(gs[0,1])
    ax_alt = fig.add_subplot(gs[1,0:2])
    cax    = fig.add_subplot(gs[:,2])

    # flight track
    ax_map.add_collection(lcoll(lon,lat,spd))
    ax_map.scatter(lon.iloc[0],lat.iloc[0],marker="^",c="limegreen",
                   s=90,edgecolors="black",linewidths=0.6,
                   path_effects=outline,zorder=4)
    ax_map.scatter(lon.iloc[-1],lat.iloc[-1],marker="s",c="red",
                   s=90,edgecolors="black",linewidths=0.6,
                   path_effects=outline,zorder=4)
    for e in events:
        ax_map.text(e["lon"],e["lat"],str(e["num"]),**numsty,zorder=6)
    m=max(lon.max()-lon.min(),lat.max()-lat.min())*margin_pct/100
    ax_map.set_xlim(lon.min()-m,lon.max()+m); ax_map.set_ylim(lat.min()-m,lat.max()+m)
    ax_map.set_aspect("equal"); ax_map.set_xticks([]); ax_map.set_yticks([])
    for s in ax_map.spines.values(): s.set_visible(False)
    ax_map.set_title("Flight Track", fontweight="bold", pad=8)

    # notable event plot
    header=["","Time (UTC)","Event","Latitude","Longitude"]; col=[0.03,0.12,0.32,0.68,0.90]
    ax_evt.spines["right"].set_visible(True)
    ax_evt.spines["right"].set_color("grey"); ax_evt.spines["right"].set_linewidth(1.2)
    ax_evt.set_title("Notable Events", fontweight="bold", pad=8)
    ax_evt.annotate('', xy=(0,1.02), xycoords='axes fraction',
                    xytext=(1,1.02), textcoords='axes fraction',
                    arrowprops=dict(arrowstyle='-',color='black',lw=1))
    for x,h in zip(col,header):
        ax_evt.text(x,0.83,h,transform=ax_evt.transAxes,
                    ha="left",va="center",fontsize=10,fontweight="bold")

    rows=[("▲",0,"Take-off",lat.iloc[0],lon.iloc[0])] + \
         [(str(e["num"]),e["time_ms"],e["label"],e["lat"],e["lon"]) for e in events] + \
         [("■",int(df["time(millisecond)"].iloc[-1]),"Landing",
           lat.iloc[-1],lon.iloc[-1])]
    y=0.70
    for sym,ms,lab,la,lo in rows:
        colc={"▲":"limegreen","■":"red"}.get(sym,"white")
        ax_evt.text(col[0],y,sym,transform=ax_evt.transAxes,
                    ha="left",va="center",fontsize=10,color=colc,
                    path_effects=outline)
        ax_evt.text(col[1],y,fmt_hms_utc(ms,start_utc),
                    transform=ax_evt.transAxes,ha="left",va="center",fontsize=10)
        lab_wr="\n".join(textwrap.wrap(lab,26))
        ax_evt.text(col[2],y,lab_wr,transform=ax_evt.transAxes,
                    ha="left",va="center",fontsize=10)
        ax_evt.text(col[3],y,f"{la:.5f}",transform=ax_evt.transAxes,
                    ha="left",va="center",fontsize=10)
        ax_evt.text(col[4],y,f"{lo:.5f}",transform=ax_evt.transAxes,
                    ha="left",va="center",fontsize=10)
        y-=0.11+0.05*lab_wr.count("\n")

    # altitude vs time
    ax_alt.add_collection(lcoll(t,alt,spd))
    ax_alt.set_ylim(alt.min()*0.98,alt.max()*1.02)
    ylabel="Altitude (AGL ft)" if alt_col in AGL_COL else "Altitude (MSL ft)"
    ax_alt.set_ylabel(ylabel); ax_alt.set_xlabel("Time (minutes)")
    ax_alt.set_title("Altitude vs Time",
                     fontweight="bold", pad=8)
    ax_alt.grid(True, linestyle="--", alpha=0.4)
    step=0.5 if t.max()<=5 else 1.0
    ticks=np.arange(0,t.max()+1e-9,step)
    ax_alt.set_xticks(ticks)
    ax_alt.set_xticklabels(
        [fmt_hms_utc(0,start_utc)] +
        [f"{tk:.1f}" for tk in ticks[1:-1]] +
        [fmt_hms_utc(int(df["time(millisecond)"].iloc[-1]), start_utc)]
    )
    for e in events:
        ax_alt.text(e["time_ms"]/60000,e["alt"],str(e["num"]),**numsty,zorder=6)

    mx_idx=alt.idxmax()
    ax_alt.annotate(f"{alt[mx_idx]:.0f} ft",(t[mx_idx],alt[mx_idx]),
                    xytext=(0,6), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, fontweight="bold",
                    path_effects=[pe.withStroke(linewidth=2,foreground="white")])

    sm=plt.cm.ScalarMappable(norm=lcoll(lon,lat,spd).norm,cmap="viridis")
    sm.set_array([])
    plt.colorbar(sm,cax=cax,label="Ground Speed (mph)")
    plt.show()

# cli
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: flight_table.py <flight.csv> [manual.json] [margin%]")
        sys.exit(1)

    csv_file   = sys.argv[1]
    margin_pct = 15.0
    manual_json = None

    # identify manual-events JSON (first .json arg)
    for a in sys.argv[2:]:
        if a.lower().endswith(".json"):
            manual_json = a
            break
    # numeric arg for margin (optional)
    for a in sys.argv[2:]:
        if a.replace('.', '', 1).isdigit():
            margin_pct = float(a)
            break

    df, alt_col, start_utc = load_csv(csv_file)
    core_events   = detect_core(df, alt_col)
    manual_events = load_manual(df, manual_json, alt_col)
    plot_bundle(df, alt_col, start_utc,
                core_events, manual_events,
                margin_pct=margin_pct)