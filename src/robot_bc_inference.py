#! /usr/bin/python3

"""Interactive task-space viewer for the robot behavior-cloning policy.

The CSV contains joint values and end-effector poses, but no URDF, DH parameters,
or link geometry. Consequently, this viewer renders the measured gripper pose,
object, target, trajectory, and controller actions in 3D rather than inventing an
articulated robot model. The recorded episode supplies environment transitions;
ONNX predicts an action from each complete rolling state-history window.

Run:
    python src/robot_bc_inference.py

Then open http://127.0.0.1:7860. Use --host 0.0.0.0 to expose it on your LAN.
The browser loads Plotly.js from its CDN; no Gradio or Plotly Python package is
required. Keep robot_bc.onnx.data beside robot_bc.onnx.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import webbrowser
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import onnxruntime as ort
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "robot_bc_grasp.csv"
MODEL_PATH = PROJECT_ROOT / "model" / "robot_bc.onnx"

STATE_COLUMNS = [
    "q0", "q1", "q2", "q3", "q4", "q5", "q6",
    "dq0", "dq1", "dq2", "dq3", "dq4", "dq5", "dq6",
    "ee_x", "ee_y", "ee_z", "ee_roll", "ee_pitch", "ee_yaw",
    "object_x", "object_y", "object_z",
    "target_x", "target_y", "target_z", "gripper_width",
]
ACTION_COLUMNS = [
    "action_dx", "action_dy", "action_dz",
    "action_droll", "action_dpitch", "action_dyaw", "action_gripper",
]


class EpisodeInference:
    """Load model/data once and cache fully inferred episodes for replay."""

    def __init__(self, model_path: Path, data_path: Path, cache_size: int = 5):
        if not model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset not found: {data_path}")

        self.session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        self.input_info = self.session.get_inputs()[0]
        self.output_info = self.session.get_outputs()[0]
        self.sequence_length = self._fixed_dimension(self.input_info.shape, 1)
        self.feature_count = self._fixed_dimension(self.input_info.shape, 2)
        if self.feature_count != len(STATE_COLUMNS):
            raise ValueError(
                f"Model expects {self.feature_count} features, but the dataset "
                f"schema defines {len(STATE_COLUMNS)}"
            )

        frame = pd.read_csv(data_path)
        required = {"episode", "t", "phase", *STATE_COLUMNS, *ACTION_COLUMNS}
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"Dataset is missing columns: {', '.join(missing)}")

        self.episodes = {
            self._json_scalar(episode_id): group.sort_values("t").reset_index(drop=True)
            for episode_id, group in frame.groupby("episode", sort=True)
        }
        self.episode_ids = list(self.episodes)
        self.cache_size = cache_size
        self.cache: OrderedDict[object, dict] = OrderedDict()
        self.cache_lock = threading.Lock()

    @staticmethod
    def _fixed_dimension(shape: list, index: int) -> int:
        value = shape[index]
        if not isinstance(value, int):
            raise ValueError(f"Expected a fixed ONNX input shape, received {shape}")
        return value

    @staticmethod
    def _json_scalar(value):
        return value.item() if isinstance(value, np.generic) else value

    def config(self) -> dict:
        return {
            "episodes": [
                {"id": episode_id, "frames": len(self.episodes[episode_id])}
                for episode_id in self.episode_ids
            ],
            "sequence_length": self.sequence_length,
            "state_features": self.feature_count,
            "actions": ACTION_COLUMNS,
            "model": MODEL_PATH.name,
            "input": {
                "name": self.input_info.name,
                "shape": self.input_info.shape,
                "type": self.input_info.type,
            },
            "output": {
                "name": self.output_info.name,
                "shape": self.output_info.shape,
                "type": self.output_info.type,
            },
            "geometry_note": (
                "Task-space view: no URDF or link geometry exists in this project."
            ),
        }

    def infer_episode(self, episode_id) -> dict:
        with self.cache_lock:
            cached = self.cache.get(episode_id)
            if cached is not None:
                self.cache.move_to_end(episode_id)
                return cached

        frame = self.episodes.get(episode_id)
        if frame is None:
            raise KeyError(f"Unknown episode: {episode_id}")

        states = frame[STATE_COLUMNS].to_numpy(dtype=np.float32)
        actual = frame[ACTION_COLUMNS].to_numpy(dtype=np.float32)
        predicted = np.full(actual.shape, np.nan, dtype=np.float32)
        latency_ms = np.full(len(frame), np.nan, dtype=np.float32)

        for end in range(self.sequence_length - 1, len(states)):
            history = np.ascontiguousarray(
                states[end - self.sequence_length + 1 : end + 1][None, ...]
            )
            started = time.perf_counter()
            predicted[end] = self.session.run(
                [self.output_info.name], {self.input_info.name: history}
            )[0][0]
            latency_ms[end] = (time.perf_counter() - started) * 1000.0

        valid = np.isfinite(predicted[:, 0])
        errors = predicted[valid] - actual[valid]
        overall_rmse = float(np.sqrt(np.mean(errors**2))) if valid.any() else None
        translation_rmse = (
            float(np.sqrt(np.mean(errors[:, :3] ** 2))) if valid.any() else None
        )
        rotation_rmse = (
            float(np.sqrt(np.mean(errors[:, 3:6] ** 2))) if valid.any() else None
        )
        gripper_rmse = (
            float(np.sqrt(np.mean(errors[:, 6] ** 2))) if valid.any() else None
        )

        payload = {
            "episode": episode_id,
            "frame_count": len(frame),
            "history_start": self.sequence_length - 1,
            "t": frame["t"].tolist(),
            "phase": frame["phase"].astype(str).tolist(),
            "q": states[:, :7].tolist(),
            "ee": states[:, 14:17].tolist(),
            "rpy": states[:, 17:20].tolist(),
            "object": states[:, 20:23].tolist(),
            "target": states[:, 23:26].tolist(),
            "gripper": states[:, 26].tolist(),
            "actual": actual.tolist(),
            "predicted": [
                row.tolist() if np.isfinite(row).all() else None for row in predicted
            ],
            "latency_ms": [
                float(value) if np.isfinite(value) else None for value in latency_ms
            ],
            "metrics": {
                "overall_rmse": overall_rmse,
                "translation_rmse": translation_rmse,
                "rotation_rmse": rotation_rmse,
                "gripper_rmse": gripper_rmse,
                "mean_latency_ms": (
                    float(np.nanmean(latency_ms)) if valid.any() else None
                ),
            },
        }

        with self.cache_lock:
            self.cache[episode_id] = payload
            self.cache.move_to_end(episode_id)
            while len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        return payload


HTML = r'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Robot BC · Task-Space Controller</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    :root {
      --bg: #081018; --panel: #101c27; --panel2: #142330; --line: #243746;
      --text: #e7f0f5; --muted: #8fa6b5; --cyan: #3ddbd9; --orange: #ff9f43;
      --green: #58d68d; --red: #ff6b6b; --purple: #b794f4;
    }
    * { box-sizing: border-box; }
    body { margin: 0; background: radial-gradient(circle at 25% 0%, #142838 0, var(--bg) 42%);
      color: var(--text); font: 14px/1.4 Inter, ui-sans-serif, system-ui, sans-serif; }
    header { display:flex; justify-content:space-between; align-items:end; padding:18px 22px 12px; }
    h1 { margin:0; font-size:22px; letter-spacing:-.02em; } .eyebrow { color:var(--cyan); font-size:11px;
      letter-spacing:.15em; text-transform:uppercase; font-weight:700; }
    .status { color:var(--muted); display:flex; gap:8px; align-items:center; }
    .dot { width:8px; height:8px; border-radius:50%; background:var(--green); box-shadow:0 0 10px var(--green); }
    .toolbar { margin:0 22px 12px; padding:12px; background:rgba(16,28,39,.92); border:1px solid var(--line);
      border-radius:12px; display:flex; flex-wrap:wrap; gap:10px; align-items:center; }
    button, select { background:var(--panel2); color:var(--text); border:1px solid #365064; border-radius:7px;
      padding:8px 10px; cursor:pointer; } button:hover, select:hover { border-color:var(--cyan); }
    button.primary { background:var(--cyan); color:#052225; border-color:var(--cyan); font-weight:750; }
    input[type=range] { accent-color:var(--cyan); min-width:220px; flex:1; }
    label { color:var(--muted); } .frame-label { min-width:110px; font-variant-numeric:tabular-nums; }
    main { padding:0 22px 22px; display:grid; grid-template-columns:minmax(0, 2fr) minmax(300px, .85fr); gap:12px; }
    .panel { background:rgba(16,28,39,.94); border:1px solid var(--line); border-radius:12px; overflow:hidden; }
    .panel-title { padding:10px 14px 0; color:var(--muted); font-size:11px; text-transform:uppercase;
      letter-spacing:.1em; font-weight:700; }
    #scene { height:610px; } #actions { height:290px; } #joints { height:260px; }
    .side { display:grid; gap:12px; align-content:start; }
    .cards { display:grid; grid-template-columns:repeat(2,1fr); gap:1px; background:var(--line); }
    .card { background:var(--panel); padding:13px; min-height:76px; }
    .card .name { color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.08em; }
    .card .value { font-size:19px; margin-top:5px; font-variant-numeric:tabular-nums; }
    .phase { color:var(--cyan)!important; }
    .legend { padding:12px 14px; display:flex; flex-wrap:wrap; gap:12px; color:var(--muted); }
    .swatch { display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:5px; }
    .note { padding:12px 14px; color:var(--muted); border-top:1px solid var(--line); font-size:12px; }
    #loading { position:fixed; inset:0; background:rgba(4,10,15,.78); display:none; place-items:center; z-index:5; }
    #loading.show { display:grid; } .loader { padding:20px 25px; background:var(--panel); border:1px solid var(--line);
      border-radius:12px; color:var(--cyan); }
    @media(max-width:950px){ main{grid-template-columns:1fr} #scene{height:520px} }
  </style>
</head>
<body>
  <div id="loading"><div class="loader">Running ONNX over the episode…</div></div>
  <header>
    <div><div class="eyebrow">Behavior cloning observatory</div><h1>Task-Space Controller Replay</h1></div>
    <div class="status"><span class="dot"></span><span id="modelStatus">Loading model metadata</span></div>
  </header>
  <section class="toolbar">
    <label>Episode <select id="episode"></select></label>
    <button id="back">−1</button><button id="play" class="primary">▶ Play</button><button id="forward">+1</button>
    <button id="reset">↺ Reset</button>
    <input id="timeline" type="range" min="0" max="999" value="0">
    <span id="frameLabel" class="frame-label">frame 0 / 0</span>
    <label>Speed <select id="speed"><option value="0.25">0.25×</option><option value="0.5">0.5×</option>
      <option value="1" selected>1×</option><option value="2">2×</option><option value="4">4×</option></select></label>
    <label>Stride <select id="stride"><option value="1">1 frame</option><option value="2">2 frames</option>
      <option value="5" selected>5 frames</option><option value="10">10 frames</option></select></label>
    <label>End <select id="endMode"><option value="loop">Loop episode</option><option value="next">Next episode</option>
      <option value="stop">Stop</option></select></label>
  </section>
  <main>
    <section class="panel">
      <div class="panel-title">3D workspace · drag to orbit · scroll to zoom</div><div id="scene"></div>
      <div class="legend">
        <span><i class="swatch" style="background:#3ddbd9"></i>gripper / path</span>
        <span><i class="swatch" style="background:#ff9f43"></i>object</span>
        <span><i class="swatch" style="background:#58d68d"></i>target</span>
        <span><i class="swatch" style="background:#b794f4"></i>predicted Δxyz</span>
        <span><i class="swatch" style="background:#fff"></i>recorded Δxyz</span>
      </div>
      <div class="note">The cyan seven-link chain is a schematic anchored to the measured gripper—not true forward kinematics. The CSV has no URDF, joint axes, or link dimensions. Motion and phase timing are recorded data; purple is ONNX controller output.</div>
    </section>
    <div class="side">
      <section class="panel cards">
        <div class="card"><div class="name">Phase</div><div id="phase" class="value phase">—</div></div>
        <div class="card"><div class="name">Dataset time</div><div id="time" class="value">—</div></div>
        <div class="card"><div class="name">History</div><div id="history" class="value">—</div></div>
        <div class="card"><div class="name">Gripper width</div><div id="gripper" class="value">—</div></div>
        <div class="card"><div class="name">Frame action RMSE</div><div id="frameRmse" class="value">—</div></div>
        <div class="card"><div class="name">ONNX latency</div><div id="latency" class="value">—</div></div>
        <div class="card"><div class="name">Episode action RMSE</div><div id="episodeRmse" class="value">—</div></div>
        <div class="card"><div class="name">Episode inference avg</div><div id="meanLatency" class="value">—</div></div>
      </section>
      <section class="panel"><div class="panel-title">Action channels · predicted solid / recorded dotted</div><div id="actions"></div></section>
      <section class="panel"><div class="panel-title">Joint configuration</div><div id="joints"></div></section>
    </div>
  </main>
<script>
const colors={cyan:'#3ddbd9',orange:'#ff9f43',green:'#58d68d',purple:'#b794f4',white:'#dbe8ef',muted:'#6f8999'};
const plotBg='rgba(0,0,0,0)', grid='#263b49', font={color:'#91a9b8',family:'Inter,system-ui,sans-serif'};
let config=null, episode=null, frame=0, playing=false, timer=null, camera=null, interacting=false;
const $=id=>document.getElementById(id);
const fmt=(x,d=4)=>x==null||!Number.isFinite(x)?'—':Number(x).toFixed(d);

async function getJson(path){const r=await fetch(path); if(!r.ok) throw new Error(await r.text()); return r.json();}
function rot(r,p,y){const cr=Math.cos(r),sr=Math.sin(r),cp=Math.cos(p),sp=Math.sin(p),cy=Math.cos(y),sy=Math.sin(y);
  return [[cy*cp,cy*sp*sr-sy*cr,cy*sp*cr+sy*sr],[sy*cp,sy*sp*sr+cy*cr,sy*sp*cr-cy*sr],[-sp,cp*sr,cp*cr]];}
function axisLine(origin,vec,len,name,color,width=5){return {type:'scatter3d',mode:'lines',name,x:[origin[0],origin[0]+vec[0]*len],
  y:[origin[1],origin[1]+vec[1]*len],z:[origin[2],origin[2]+vec[2]*len],line:{color,width},hoverinfo:'name',showlegend:false};}
function actionLine(origin,a,name,color,dash){if(!a)return axisLine(origin,[0,0,0],0,name,color);
  const scale=4; return {type:'scatter3d',mode:'lines+markers',name,x:[origin[0],origin[0]+a[0]*scale],
    y:[origin[1],origin[1]+a[1]*scale],z:[origin[2],origin[2]+a[2]*scale],line:{color,width:6,dash},
    marker:{size:[0,4],color},hovertemplate:name+'<br>Δx=%{customdata[0]:.4f}<br>Δy=%{customdata[1]:.4f}<br>Δz=%{customdata[2]:.4f}<extra></extra>',
    customdata:[a.slice(0,3),a.slice(0,3)],showlegend:false};}
function schematicArm(ee,q){
  const base=[0,0,0], reach=ee.map((v,j)=>v-base[j]);
  const mag=Math.hypot(...reach)||1, unit=reach.map(v=>v/mag);
  let side1=[-unit[1],unit[0],0], sideMag=Math.hypot(...side1);
  if(sideMag<1e-5){side1=[1,0,0];sideMag=1;} side1=side1.map(v=>v/sideMag);
  const side2=[unit[1]*side1[2]-unit[2]*side1[1],unit[2]*side1[0]-unit[0]*side1[2],unit[0]*side1[1]-unit[1]*side1[0]];
  const pts=[], joints=7; let angle=0;
  for(let j=0;j<=joints;j++){const s=j/joints;if(j>0)angle+=q[j-1];const envelope=Math.sin(Math.PI*s);
    const b1=.11*envelope*Math.sin(angle),b2=.07*envelope*Math.cos(angle*.7+q[0]);
    pts.push(base.map((v,d)=>v+reach[d]*s+side1[d]*b1+side2[d]*b2));}
  return {type:'scatter3d',mode:'lines+markers',name:'Schematic 7-link arm',x:pts.map(v=>v[0]),y:pts.map(v=>v[1]),z:pts.map(v=>v[2]),
    line:{color:'rgba(61,219,217,.78)',width:11},marker:{color:'#d9ffff',size:5,line:{color:colors.cyan,width:2}},
    hovertemplate:'schematic joint %{pointNumber}<extra></extra>',showlegend:false};}
function sceneTraces(i){const ee=episode.ee[i],obj=episode.object[i],target=episode.target[i],R=rot(...episode.rpy[i]);
  const jaw=R.map(row=>row[1]), half=episode.gripper[i]/2;
  return [
    {type:'scatter3d',mode:'lines',name:'Full recorded path',x:episode.ee.map(v=>v[0]),y:episode.ee.map(v=>v[1]),z:episode.ee.map(v=>v[2]),line:{color:'rgba(61,219,217,.13)',width:3},hoverinfo:'skip',showlegend:false},
    {type:'scatter3d',mode:'lines',name:'Traversed path',x:episode.ee.slice(0,i+1).map(v=>v[0]),y:episode.ee.slice(0,i+1).map(v=>v[1]),z:episode.ee.slice(0,i+1).map(v=>v[2]),line:{color:colors.cyan,width:7},hoverinfo:'skip',showlegend:false},
    {type:'scatter3d',mode:'markers',name:'Gripper',x:[ee[0]],y:[ee[1]],z:[ee[2]],marker:{size:7,color:colors.cyan,symbol:'diamond'},hovertemplate:'gripper<br>x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>',showlegend:false},
    {type:'scatter3d',mode:'markers',name:'Object',x:[obj[0]],y:[obj[1]],z:[obj[2]],marker:{size:9,color:colors.orange},hovertemplate:'object<extra></extra>',showlegend:false},
    {type:'scatter3d',mode:'markers',name:'Target',x:[target[0]],y:[target[1]],z:[target[2]],marker:{size:11,color:colors.green,symbol:'square-open',line:{width:4}},hovertemplate:'target<extra></extra>',showlegend:false},
    schematicArm(ee,episode.q[i]),
    axisLine(ee,R.map(row=>row[0]),.045,'gripper x','#ff6b6b',4),axisLine(ee,R.map(row=>row[1]),.045,'gripper y','#58d68d',4),axisLine(ee,R.map(row=>row[2]),.045,'gripper z','#5dade2',4),
    {type:'scatter3d',mode:'lines',name:'Gripper jaws',x:[ee[0]-jaw[0]*half,ee[0]+jaw[0]*half],y:[ee[1]-jaw[1]*half,ee[1]+jaw[1]*half],z:[ee[2]-jaw[2]*half,ee[2]+jaw[2]*half],line:{color:colors.cyan,width:12},hoverinfo:'skip',showlegend:false},
    actionLine(ee,episode.predicted[i],'Predicted Δxyz',colors.purple,'solid'),actionLine(ee,episode.actual[i],'Recorded Δxyz',colors.white,'dot')
  ];}
function sceneLayout(){const all=[[0,0,0],...episode.ee,...episode.object,...episode.target], xs=all.map(v=>v[0]),ys=all.map(v=>v[1]),zs=all.map(v=>v[2]);
  const range=(a,p=.08)=>[Math.min(...a)-p,Math.max(...a)+p]; return {paper_bgcolor:plotBg,plot_bgcolor:plotBg,font,margin:{l:0,r:0,t:0,b:0},
    uirevision:'keep-camera',scene:{bgcolor:plotBg,aspectmode:'cube',camera:camera||{eye:{x:1.45,y:1.45,z:.95}},xaxis:{title:'X (m)',range:range(xs),gridcolor:grid,zerolinecolor:grid},
    yaxis:{title:'Y (m)',range:range(ys),gridcolor:grid,zerolinecolor:grid},zaxis:{title:'Z (m)',range:[Math.min(0,...zs)-.02,Math.max(...zs)+.1],gridcolor:grid,zerolinecolor:grid}},showlegend:false};}
function actionPlot(i){const x=episode.t, pred=episode.predicted, actual=episode.actual, traces=[];
  const specs=[[0,'Δx',colors.cyan],[1,'Δy',colors.orange],[2,'Δz',colors.green],[6,'grip',colors.purple]];
  for(const [j,name,color] of specs){traces.push({x,y:pred.map(v=>v?v[j]:null),name:name+' pred',type:'scattergl',mode:'lines',line:{color,width:2}});
    traces.push({x,y:actual.map(v=>v[j]),name:name+' actual',type:'scattergl',mode:'lines',line:{color,width:1,dash:'dot'},opacity:.62});}
  return [traces,{paper_bgcolor:plotBg,plot_bgcolor:plotBg,font,margin:{l:48,r:12,t:12,b:35},xaxis:{title:'episode t',gridcolor:grid},yaxis:{title:'action',gridcolor:grid},
    legend:{orientation:'h',y:1.2,font:{size:9}},shapes:[{type:'line',x0:x[i],x1:x[i],y0:0,y1:1,yref:'paper',line:{color:'#fff',width:1}}]}];}
function updateCards(i){$('phase').textContent=episode.phase[i]; $('time').textContent=String(episode.t[i]);
  $('history').textContent=i>=episode.history_start?'ready':'warming '+(i+1)+'/'+config.sequence_length;
  $('gripper').textContent=fmt(episode.gripper[i],4)+' m'; const p=episode.predicted[i],a=episode.actual[i];
  $('frameRmse').textContent=p?fmt(Math.sqrt(p.reduce((s,v,j)=>s+(v-a[j])**2,0)/p.length),5):'—';
  $('latency').textContent=episode.latency_ms[i]==null?'—':fmt(episode.latency_ms[i],3)+' ms';
  $('episodeRmse').textContent=fmt(episode.metrics.overall_rmse,5); $('meanLatency').textContent=fmt(episode.metrics.mean_latency_ms,3)+' ms';
  $('frameLabel').textContent=`frame ${i+1} / ${episode.frame_count}`; $('timeline').value=i;}
async function render(full=false){if(!episode)return; updateCards(frame); const scene=$('scene');
  if(scene._fullLayout?.scene?.camera) camera=scene._fullLayout.scene.camera;
  await Plotly.react(scene,sceneTraces(frame),sceneLayout(),{responsive:true,displaylogo:false,scrollZoom:true});
  if(!scene.dataset.interactionBound){scene.dataset.interactionBound='1';scene.addEventListener('pointerdown',()=>{interacting=true;});
    document.addEventListener('pointerup',()=>{interacting=false;if(playing)schedule();});
    scene.on('plotly_relayout',change=>{if(change['scene.camera'])camera=change['scene.camera'];});}
  if(full){const [traces,layout]=actionPlot(frame); await Plotly.react('actions',traces,layout,{responsive:true,displaylogo:false});}
  else await Plotly.relayout('actions',{'shapes[0].x0':episode.t[frame],'shapes[0].x1':episode.t[frame]});
  await Plotly.react('joints',[{type:'bar',x:['q0','q1','q2','q3','q4','q5','q6'],y:episode.q[frame],marker:{color:episode.q[frame],colorscale:'Tealgrn'}}],
    {paper_bgcolor:plotBg,plot_bgcolor:plotBg,font,margin:{l:45,r:12,t:10,b:30},yaxis:{title:'rad',gridcolor:grid},xaxis:{gridcolor:grid}},{responsive:true,displaylogo:false});}
function schedule(){clearTimeout(timer);if(!playing)return;const delay=50/Number($('speed').value);timer=setTimeout(async()=>{
  if(interacting){schedule();return;} const stride=Number($('stride').value);
  if(frame<episode.frame_count-1){frame=Math.min(episode.frame_count-1,frame+stride);await render();schedule();return;} const mode=$('endMode').value;
  if(mode==='stop'){setPlaying(false);return;} if(mode==='loop'){frame=0;await render();schedule();return;}
  const opts=$('episode').options; $('episode').selectedIndex=($('episode').selectedIndex+1)%opts.length; await loadEpisode($('episode').value); schedule();},delay);}
function setPlaying(on){playing=on;$('play').textContent=on?'❚❚ Pause':'▶ Play';if(on)schedule();else clearTimeout(timer);}
async function loadEpisode(id){const wasPlaying=playing;clearTimeout(timer);$('loading').classList.add('show');
  try{episode=await getJson('/api/episode?id='+encodeURIComponent(id));frame=0;$('timeline').max=episode.frame_count-1;camera=null;await render(true);}
  catch(e){alert(e.message);setPlaying(false);}finally{$('loading').classList.remove('show');if(wasPlaying)schedule();}}
async function init(){try{config=await getJson('/api/config');$('modelStatus').textContent=`${config.model} · ${config.input.shape.join(' × ')}`;
  for(const e of config.episodes){const o=document.createElement('option');o.value=e.id;o.textContent=`${e.id} · ${e.frames} frames`;$('episode').appendChild(o);} await loadEpisode($('episode').value);
  $('play').onclick=()=>setPlaying(!playing);$('reset').onclick=()=>{frame=0;render();};$('back').onclick=()=>{frame=Math.max(0,frame-1);render();};
  $('forward').onclick=()=>{frame=Math.min(episode.frame_count-1,frame+1);render();};$('timeline').oninput=e=>{frame=Number(e.target.value);render();};
  $('episode').onchange=e=>loadEpisode(e.target.value);$('speed').onchange=()=>{if(playing)schedule();};}
  catch(e){$('modelStatus').textContent='startup failed';alert(e.message);}}
init();
</script>
</body></html>'''


def make_handler(engine: EpisodeInference):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlparse(self.path)
            try:
                if parsed.path == "/":
                    self._send(HTML.encode(), "text/html; charset=utf-8")
                elif parsed.path == "/api/config":
                    self._json(engine.config())
                elif parsed.path == "/api/episode":
                    raw_id = parse_qs(parsed.query).get("id", [None])[0]
                    if raw_id is None:
                        self._error(400, "Missing episode id")
                        return
                    episode_id = next(
                        (value for value in engine.episode_ids if str(value) == raw_id), None
                    )
                    if episode_id is None:
                        self._error(404, f"Unknown episode: {raw_id}")
                        return
                    self._json(engine.infer_episode(episode_id))
                elif parsed.path == "/favicon.ico":
                    self.send_response(204)
                    self.end_headers()
                else:
                    self._error(404, "Not found")
            except Exception as exc:
                self._error(500, str(exc))

        def _json(self, payload: dict):
            self._send(
                json.dumps(payload, separators=(",", ":"), allow_nan=False).encode(),
                "application/json",
            )

        def _error(self, status: int, message: str):
            body = json.dumps({"error": message}).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send(self, body: bytes, content_type: str):
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format_string, *args):
            print(f"[{self.log_date_time_string()}] {format_string % args}")

    return Handler


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--no-browser", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"Loading {MODEL_PATH.name} and {DATA_PATH.name}...")
    engine = EpisodeInference(MODEL_PATH, DATA_PATH)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(engine))
    display_host = "127.0.0.1" if args.host == "0.0.0.0" else args.host
    url = f"http://{display_host}:{args.port}"
    print(f"Robot BC viewer: {url}")
    print("Press Ctrl+C to stop.")
    if not args.no_browser:
        threading.Timer(0.7, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping viewer.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
