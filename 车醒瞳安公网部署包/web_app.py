"""车醒瞳安局域网测试网页。

默认监听所有本机网络接口，允许同一局域网内的设备访问；上传视频和分析产物
仍保存在运行本服务的电脑项目目录中。公网部署需要独立的 HTTPS 与访问控制。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
from threading import Lock

from flask import Flask, abort, jsonify, render_template_string, request, send_from_directory, url_for
from werkzeug.utils import secure_filename

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
from fatigue_agent import FatigueAgent
from video_adapter import _require_dependencies, frame_to_features
from review_agent import review_answer

BASE = Path(__file__).resolve().parent
UPLOADS = BASE / 'data' / 'uploads'
REPORTS = BASE / 'reports' / 'web'
SCRIPT = BASE / 'src' / 'render_analysis_video.py'
ALLOWED = {'.mp4', '.mov', '.avi', '.mkv'}
UPLOADS.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024
LIVE_AGENTS: dict[str, tuple[FatigueAgent, float]] = {}
CALIBRATION_SAMPLES: dict[str, dict] = {}
LIVE_SUMMARIES: dict[str, dict] = {}
LIVE_LOCK = Lock()
FACE_MESH = None


def live_mesh():
    """懒加载单个 Face Mesh 实例，避免实时请求重复加载模型。"""
    global FACE_MESH
    if FACE_MESH is None:
        _, mp = _require_dependencies()
        FACE_MESH = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    return FACE_MESH


def features_from_live_upload(uploaded, timestamp: float):
    """把浏览器 JPEG 帧解码为统一特征；只在内存中处理。"""
    raw = uploaded.read()
    if len(raw) > 1_500_000:
        raise ValueError('单帧图像过大，请降低摄像头分辨率')
    cv2, _ = _require_dependencies()
    import numpy as np
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('无法读取摄像头画面')
    height, width = image.shape[:2]
    # Face Mesh 的视频跟踪状态不是并发安全的。
    with LIVE_LOCK:
        result = live_mesh().process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return frame_to_features(result, timestamp, width, height)


def record_live_summary(session_id: str, timestamp: float, assessment) -> dict:
    """把结构化判断压缩成一次训练摘要，不保存实时图像。"""
    signals = dict(assessment.signals or {})
    state = LIVE_SUMMARIES.setdefault(session_id, {
        'started_at': timestamp,
        'last_timestamp': timestamp,
        'frames': 0,
        'last_level': assessment.level.value,
        'peak_level': assessment.level.value,
        'level_counts': {},
        'event_counts': {'fatigue': 0, 'monitoring_gap': 0, 'distraction_or_posture': 0},
        'events': [],
        'latest': {},
        'last_event_type': 'none',
        'last_event_level': 'normal',
    })
    state['frames'] += 1
    state['last_timestamp'] = timestamp
    level = assessment.level.value
    state['last_level'] = level
    state['level_counts'][level] = state['level_counts'].get(level, 0) + 1
    rank = {'normal': 0, 'unknown': 1, 'attention': 2, 'warning': 3, 'high': 4}
    if rank.get(level, 0) > rank.get(state['peak_level'], 0):
        state['peak_level'] = level
    event_type = signals.get('event_type', 'none')
    if event_type != 'none' and (event_type != state['last_event_type'] or level != state['last_event_level']):
        state['event_counts'][event_type] = state['event_counts'].get(event_type, 0) + 1
        state['events'].append({
            'timestamp_seconds': round(timestamp - state['started_at'], 1),
            'type': event_type,
            'level': level,
            'reasons': list(assessment.reasons),
            'signals': signals,
        })
        state['events'] = state['events'][-40:]
    state['last_event_type'] = event_type
    state['last_event_level'] = level
    state['latest'] = {
        'level': level,
        'confidence': assessment.confidence,
        'reasons': list(assessment.reasons),
        'recommendations': list(assessment.recommendations),
        'signals': signals,
    }
    return state


def public_summary(state: dict) -> dict:
    """只返回训练摘要和结构化事件，不返回图像或原始帧。"""
    return {
        'duration_seconds': round(max(0.0, state.get('last_timestamp', 0) - state.get('started_at', 0)), 1),
        'frames': state.get('frames', 0),
        'last_level': state.get('last_level', 'unknown'),
        'peak_level': state.get('peak_level', 'normal'),
        'level_counts': dict(state.get('level_counts', {})),
        'event_counts': dict(state.get('event_counts', {})),
        'events': list(state.get('events', [])),
        'latest': dict(state.get('latest', {})),
        'notice': '摘要仅用于本地训练复盘，不包含实时图像。',
    }

PAGE = '''<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>车醒瞳安 本地测试台</title>
<style>
body{margin:0;background:#f4f7fb;color:#1f2937;font-family:"Microsoft YaHei",Arial,sans-serif}
main{max-width:900px;margin:48px auto;padding:0 20px}.hero{background:#103b62;color:#fff;border-radius:18px;padding:30px 34px;margin-bottom:22px}
h1{margin:0 0 9px;font-size:30px}.sub{color:#dcecf7;line-height:1.7}.card{background:#fff;border-radius:15px;padding:28px 32px;box-shadow:0 6px 24px #16395d18;margin-bottom:20px}
input[type=file]{display:block;border:1px dashed #95abc0;padding:18px;border-radius:10px;width:calc(100% - 38px);margin:14px 0 18px;background:#fbfdff}
button{background:#1472b6;color:#fff;border:0;border-radius:9px;padding:12px 20px;font-size:16px;cursor:pointer}button:hover{background:#0e5f9d}
button:disabled{opacity:.55;cursor:not-allowed}.secondary{background:#6b7785}.secondary:hover{background:#53606d}code{background:#0b2b47;padding:2px 5px;border-radius:4px}
.notice{font-size:14px;line-height:1.65;color:#526170}.risk{font-size:28px;font-weight:bold}.normal{color:#16803d}.attention{color:#b67a00}.warning,.high{color:#c53929}.unknown{color:#5d6670}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px}.metric{background:#f5f9fc;padding:12px;border-radius:9px}.metric b{display:block;margin-top:5px}
video{width:100%;border-radius:10px;background:#000;margin-top:16px}a{color:#126ba8}.links{margin-top:16px;display:flex;gap:18px;flex-wrap:wrap}
.live-grid{display:grid;grid-template-columns:minmax(0,1.35fr) minmax(240px,.65fr);gap:18px;align-items:stretch}.live-grid video{margin:0;min-height:280px;object-fit:cover}.live-panel{border-radius:12px;padding:20px;background:#f1f4f7;border-left:6px solid #7a8591}.live-panel.normal{background:#eef9f1;border-color:#218344}.live-panel.attention{background:#fff8e6;border-color:#c48608}.live-panel.warning,.live-panel.high{background:#fff0ef;border-color:#c53b2c}.caption{font-size:14px;color:#53606d}.live-risk{font-size:28px;font-weight:bold;margin:10px 0}.live-panel.normal .live-risk{color:#16803d}.live-panel.attention .live-risk{color:#a87000}.live-panel.warning .live-risk,.live-panel.high .live-risk{color:#bd3023}.stats{display:grid;grid-template-columns:repeat(2,1fr);gap:9px;margin-top:20px}.stats span{background:#fff9;padding:10px;border-radius:8px;font-size:13px}.stats b{display:block;font-size:17px;margin-top:4px}.rest-advice{margin-top:12px;padding:10px;border-radius:8px;background:#eef4f8;font-size:14px}.rest-advice.plan{background:#fff3d9;color:#8a5c00}.rest-advice.urgent{background:#ffe5e3;color:#a3261b;font-weight:bold}@media(max-width:680px){.live-grid{grid-template-columns:1fr}.live-grid video{min-height:220px}}
.product-position{border-top:4px solid #1472b6}.product-position p{line-height:1.75}.product-position .metric b{font-size:14px;line-height:1.45}
.review-summary{background:#f5f9fc;border-radius:10px;padding:14px;line-height:1.7;white-space:pre-wrap}.review-input{display:flex;gap:10px;margin-top:14px}.review-input input{flex:1;border:1px solid #c5d2dc;border-radius:8px;padding:12px;font-size:15px}.review-answer{margin-top:14px;background:#eef7ff;border-left:4px solid #1472b6;border-radius:8px;padding:14px;line-height:1.7;min-height:28px}.review-answer.error{background:#fff0ef;border-color:#c53929}.review-card h2{margin-bottom:8px}
</style><body><main>{{ body|safe }}</main></body></html>'''

HOME = '''<section class="hero"><h1>车醒瞳安 局域网测试台</h1><div class="sub">疲劳驾驶风险识别与安全干预智能体<br>摄像头画面仅传送至局域网内的本机分析服务，不上传互联网，也不保存实时画面。</div></section>
<section class="card product-position"><h2>本项目的定位</h2><p>面向驾校、汽车实训基地和中小企业的本地化驾驶员状态分析工具。系统区分 <b>疲劳事件</b>、<b>分心/姿态异常</b> 和 <b>监测中断</b>，先判断“当前证据是否可靠”，再给出风险建议。</p><div class="grid"><div class="metric">疲劳事件<b>闭眼、PERCLOS、完整哈欠</b></div><div class="metric">分心/姿态异常<b>低头、偏头、视线偏离</b></div><div class="metric">监测可信度<b>墨镜、遮挡、人脸离开</b></div><div class="metric">隐私设计<b>本地处理、不保存实时画面</b></div></div></section>
<section class="card"><h2>实时摄像头检测</h2><p class="notice">点击后浏览器会请求摄像头权限。请在安全、静止环境中测试，切勿在真实驾驶过程中操作本网页。</p><div class="live-grid"><div><video id="camera" autoplay muted playsinline></video><canvas id="capture" hidden></canvas><div class="links"><button id="start">启动摄像头</button><button id="calibrate" class="secondary" disabled>个人校准 10 秒</button><button id="stop" class="secondary" disabled>停止检测</button></div><p id="calibration-note" class="notice">建议启动摄像头后，先面向镜头自然注视并完成个人校准。</p></div><div id="live-panel" class="live-panel unknown"><div class="caption">实时风险状态</div><div id="live-risk" class="live-risk">等待摄像头</div><div id="live-category" class="caption">尚未分类</div><div id="live-note">尚未开始分析</div><div class="stats"><span>可信度 <b id="confidence">--</b></span><span>闭眼 <b id="closed">--</b></span><span>哈欠 <b id="yawns">--</b></span><span>EAR <b id="ear">--</b></span><span>姿态偏离 <b id="pose">--</b></span><span>PERCLOS <b id="perclos">--</b></span><span>疲劳事件 5分 <b id="fatigue-events">0</b></span><span>监测中断 5分 <b id="monitoring-events">0</b></span></div><div id="rest-advice" class="rest-advice">暂无累计休息建议</div></div></div><canvas id="trend" width="820" height="150"></canvas></section>
<section class="card review-card"><h2>智能复盘 Agent</h2><p class="notice">停止检测后，系统会把结构化事件摘要交给本地复盘 Agent。它不接收或保存视频，只根据本次检测记录回答问题。</p><div id="review-summary" class="review-summary">完成一次实时检测后，这里会生成训练摘要。</div><div class="links"><button class="secondary quick-question" data-question="请总结本次训练">总结本次训练</button><button class="secondary quick-question" data-question="为什么这样判定？">为什么这样判定</button><button class="secondary quick-question" data-question="是否建议休息？">是否建议休息</button><button class="secondary quick-question" data-question="持续低头说明了什么？">分析持续低头</button></div><div class="review-input"><input id="review-question" type="text" placeholder="例如：刚才是疲劳还是监测中断？"><button id="review-ask">询问 Agent</button></div><div id="review-answer" class="review-answer">等待复盘问题。</div></section>
<section class="card"><h2>上传测试视频</h2><form method="post" action="/analyze" enctype="multipart/form-data"><input type="file" name="video" accept="video/mp4,video/quicktime,video/x-msvideo,video/x-matroska" required><button type="submit">开始本地分析</button></form><p class="notice">支持 MP4、MOV、AVI、MKV；单个文件不超过 100 MB。离线分析会生成标注视频和 JSON 报告。</p></section>
<section class="card"><h2>处理流程</h2><div class="grid"><div class="metric">1. 画面采集<b>浏览器摄像头</b></div><div class="metric">2. 证据检查<b>眼部是否可见</b></div><div class="metric">3. 多模态分析<b>眼睛、嘴部、头姿态</b></div><div class="metric">4. 事件分级<b>疲劳/分心/中断</b></div><div class="metric">5. 反馈建议<b>提醒、休息、复盘</b></div></div></section>
<section class="card"><h2>隐私与使用边界</h2><p class="notice">实时视频仅传送到运行本服务的局域网电脑内存中处理，不上传云端，不做人脸身份识别。局域网访问仅应向受信任设备开放。结果仅用于驾校、汽车实训和比赛原型验证，不作为真实道路驾驶安全系统或医疗判断。</p></section>
<script>
const $=id=>document.getElementById(id); let stream=null,timer=null,pending=false,lastAlert=0,calibrating=false,calibrationEnds=0,lastSummary=null; const history=[];
const sessionId=crypto.randomUUID ? crypto.randomUUID() : Math.random().toString(36).slice(2);
function drawTrend(){const c=$('trend'),x=c.getContext('2d'),w=c.width,h=c.height;x.clearRect(0,0,w,h);x.fillStyle='#f7fafc';x.fillRect(0,0,w,h);x.strokeStyle='#d7e1e8';x.beginPath();x.moveTo(0,h-25);x.lineTo(w,h-25);x.stroke();if(history.length<2)return;const max=history.length-1;x.beginPath();history.forEach((v,i)=>{const px=i/max*w,py=h-25-v*(h-40);i?x.lineTo(px,py):x.moveTo(px,py)});x.strokeStyle='#ce3b2d';x.lineWidth=3;x.stroke();x.fillStyle='#53606d';x.font='13px Arial';x.fillText('最近风险趋势（红色越高）',12,20)}
function setPanel(data){const a=data.assessment, sig=a.signals, level=a.level; const panel=$('live-panel'); panel.className='live-panel '+level; const missing=sig.face_unavailable_seconds||0; $('live-risk').textContent=missing>=4?'请抬头恢复监测':{normal:'正常',attention:'请注意',warning:'疲劳预警',high:'高风险疲劳',unknown:'证据不足'}[level]||level; $('live-category').textContent={fatigue:'疲劳事件',monitoring_gap:'监测中断',distraction_or_posture:'分心/姿态异常',none:'未发现明确事件'}[sig.event_type]||'未分类'; $('live-note').textContent=a.reasons.join('；'); $('confidence').textContent=a.confidence; $('closed').textContent=sig.closed_seconds+' 秒'; $('yawns').textContent=sig.yawn_count; $('ear').textContent=data.features.ear ?? '--'; $('pose').textContent=sig.head_pose_deviation===null?(missing>0?'面部不可见':'未校准'):sig.head_pose_deviation+'°'; $('perclos').textContent=(sig.perclos*100).toFixed(1)+'%'; $('fatigue-events').textContent=sig.fatigue_events_5min; $('monitoring-events').textContent=sig.monitoring_events_5min; const advice=$('rest-advice'); const adviseText={none:'暂无累计休息建议',plan_rest:'疲劳风险正在累积，建议在下一个安全地点休息',urgent_rest:'疲劳风险已多次累积，请尽快选择安全地点停车休息'}[sig.rest_recommendation]||'暂无累计休息建议'; advice.textContent=adviseText; advice.className='rest-advice '+({plan_rest:'plan',urgent_rest:'urgent'}[sig.rest_recommendation]||''); const score={normal:0,unknown:.05,attention:.35,warning:.7,high:1}[level]||0;history.push(score);if(history.length>180)history.shift();drawTrend(); if(['warning','high'].includes(level)&&Date.now()-lastAlert>5000){beep();lastAlert=Date.now();}}
function beep(){try{const ac=new (window.AudioContext||window.webkitAudioContext)();const o=ac.createOscillator(),g=ac.createGain();o.connect(g);g.connect(ac.destination);o.frequency.value=880;g.gain.setValueAtTime(.08,ac.currentTime);o.start();o.stop(ac.currentTime+.22)}catch(e){}}
async function sendFrame(){if(pending||!stream||$('camera').readyState<2)return; pending=true; const v=$('camera'),c=$('capture'); const width=480,height=Math.round(v.videoHeight*width/v.videoWidth);c.width=width;c.height=height;c.getContext('2d').drawImage(v,0,0,width,height); const blob=await new Promise(r=>c.toBlob(r,'image/jpeg',.75));const f=new FormData();f.append('frame',blob,'live.jpg');f.append('session_id',sessionId);const url=calibrating?'/api/calibrate-frame':'/api/analyze-frame'; try{const res=await fetch(url,{method:'POST',body:f});const data=await res.json();if(!res.ok)throw Error(data.error||'分析失败');if(calibrating){const remain=Math.max(0,Math.ceil((calibrationEnds-Date.now())/1000));$('calibration-note').textContent=data.status==='complete'?'校准完成：已应用个人阈值。':`正在校准，请自然注视镜头：剩余 ${remain} 秒；有效样本 ${data.valid_samples}`;if(data.status==='complete'||Date.now()>=calibrationEnds){calibrating=false;$('calibrate').disabled=false;}}else{setPanel(data)}}catch(e){$('live-panel').className='live-panel unknown';$('live-risk').textContent='连接异常';$('live-note').textContent=e.message}finally{pending=false}}
function renderReviewSummary(summary){lastSummary=summary;const c=summary.event_counts||{};const l=summary.latest||{};$('review-summary').textContent=`本次检测约 ${summary.duration_seconds} 秒，分析 ${summary.frames} 帧。最高风险：${summary.peak_level}；疲劳事件 ${c.fatigue||0} 次；监测中断 ${c.monitoring_gap||0} 次；姿态异常 ${c.distraction_or_posture||0} 次。\n最近状态：${l.level||'unknown'}。点击下方问题，或输入自己的问题。`;}
async function askReview(question){if(!lastSummary){$('review-answer').textContent='请先完成一次实时检测，再进行复盘。';return} question=(question||$('review-question').value||'请总结本次训练').trim();$('review-question').value=question;$('review-answer').textContent='Agent 正在根据结构化检测摘要复盘……';try{const res=await fetch('/api/review-agent',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({summary:lastSummary,question})});const data=await res.json();if(!res.ok)throw Error(data.error||'复盘失败');$('review-answer').textContent=data.answer+'\n\n'+data.boundary}catch(e){$('review-answer').className='review-answer error';$('review-answer').textContent=e.message}}
$('start').onclick=async()=>{try{stream=await navigator.mediaDevices.getUserMedia({video:{facingMode:'user',width:{ideal:640},height:{ideal:480}},audio:false});$('camera').srcObject=stream;$('start').disabled=true;$('stop').disabled=false;$('calibrate').disabled=false;timer=setInterval(sendFrame,350)}catch(e){$('live-note').textContent='未获得摄像头权限：'+e.message}};
$('calibrate').onclick=()=>{calibrating=true;calibrationEnds=Date.now()+10000;$('calibrate').disabled=true;$('calibration-note').textContent='正在校准：请自然睁眼、正视镜头 10 秒，不要说话或低头。'};
async function stopDetection(){clearInterval(timer);timer=null;calibrating=false;try{const summaryRes=await fetch('/api/session-summary',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({session_id:sessionId})});const summaryData=await summaryRes.json();if(summaryRes.ok)renderReviewSummary(summaryData.summary)}catch(e){$('review-summary').textContent='无法生成本次训练摘要：'+e.message}if(stream){stream.getTracks().forEach(t=>t.stop());stream=null}$('camera').srcObject=null;$('start').disabled=false;$('stop').disabled=true;$('calibrate').disabled=true;$('live-panel').className='live-panel unknown';$('live-risk').textContent='检测已停止';$('live-note').textContent='实时画面未保存';$('calibration-note').textContent='摘要已生成，可在下方询问本地复盘 Agent。';await fetch('/api/reset-session',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({session_id:sessionId})})}
$('stop').onclick=stopDetection;
document.querySelectorAll('.quick-question').forEach(b=>b.onclick=()=>askReview(b.dataset.question));$('review-ask').onclick=()=>askReview();$('review-question').addEventListener('keydown',e=>{if(e.key==='Enter')askReview()});
</script>'''


def page(body: str):
    return render_template_string(PAGE, body=body)


@app.get('/')
def index():
    return page(HOME)


@app.post('/api/analyze-frame')
def analyze_live_frame():
    """分析单张浏览器实时帧；帧仅在内存中解码，不写入磁盘。"""
    uploaded = request.files.get('frame')
    session_id = (request.form.get('session_id') or '').strip()
    if not uploaded or not session_id:
        return jsonify(error='缺少实时画面或会话标识'), 400
    try:
        timestamp = time.monotonic()
        features = features_from_live_upload(uploaded, timestamp)
        # Face Mesh 和同一会话的时序状态必须按顺序处理。
        with LIVE_LOCK:
            agent, _ = LIVE_AGENTS.get(session_id, (FatigueAgent(), timestamp))
            assessment = agent.update(features)
            LIVE_AGENTS[session_id] = (agent, timestamp)
            record_live_summary(session_id, timestamp, assessment)
            # 控制内存：清除十分钟前未活动的会话。
            expiry = timestamp - 600
            for key, (_, seen) in list(LIVE_AGENTS.items()):
                if seen < expiry:
                    LIVE_AGENTS.pop(key, None)
                    LIVE_SUMMARIES.pop(key, None)
    except Exception as exc:
        return jsonify(error=f'本地分析发生错误：{exc}'), 500
    return jsonify(
        assessment={
            'level': assessment.level.value,
            'confidence': assessment.confidence,
            'reasons': assessment.reasons,
            'recommendations': assessment.recommendations,
            'signals': assessment.signals,
        },
        features={
            'eye_visibility': round(features.eye_visibility, 2),
            'ear': None if features.ear is None else round(features.ear, 3),
            'mar': None if features.mar is None else round(features.mar, 3),
        },
    )


@app.post('/api/calibrate-frame')
def calibrate_live_frame():
    """收集个人正常状态样本，计算 EAR/MAR/头部姿态基线。"""
    uploaded = request.files.get('frame')
    session_id = (request.form.get('session_id') or '').strip()
    if not uploaded or not session_id:
        return jsonify(error='缺少实时画面或会话标识'), 400
    try:
        features = features_from_live_upload(uploaded, time.monotonic())
        with LIVE_LOCK:
            state = CALIBRATION_SAMPLES.setdefault(session_id, {'started': time.monotonic(), 'samples': []})
            samples = state['samples']
            if features.eye_visibility >= 0.65 and features.ear is not None:
                samples.append({'ear': features.ear, 'mar': features.mar, 'pitch': features.pitch})
            # 至少采样约十秒，避免几帧偶然状态影响个人基线。
            elapsed = time.monotonic() - state['started']
            if elapsed < 9.5 or len(samples) < 10:
                return jsonify(status='collecting', valid_samples=len(samples), elapsed_seconds=round(elapsed, 1))
            import statistics
            ear_values = [x['ear'] for x in samples]
            mar_values = [x['mar'] for x in samples if x['mar'] is not None]
            pitch_values = [x['pitch'] for x in samples if x['pitch'] is not None]
            baseline_ear = statistics.median(ear_values)
            baseline_mar = statistics.median(mar_values) if mar_values else 0.0
            baseline_pitch = statistics.median(pitch_values) if pitch_values else None
            # 个体阈值保守设置，至少保留较低的闭眼判定阈值。
            eye_threshold = max(0.12, min(0.24, baseline_ear * 0.70))
            agent = FatigueAgent(
                ear_closed_threshold=eye_threshold,
                mar_yawn_threshold=max(0.40, baseline_mar + 0.20),
                baseline_pitch=baseline_pitch,
            )
            LIVE_AGENTS[session_id] = (agent, time.monotonic())
            CALIBRATION_SAMPLES.pop(session_id, None)
        return jsonify(status='complete', valid_samples=len(samples), baseline={
            'ear': round(baseline_ear, 3), 'mar': round(baseline_mar, 3),
            'pitch': None if baseline_pitch is None else round(baseline_pitch, 2),
            'ear_closed_threshold': round(eye_threshold, 3),
            'mar_yawn_threshold': round(max(0.40, baseline_mar + 0.20), 3),
        })
    except Exception as exc:
        return jsonify(error=f'校准失败：{exc}'), 500


@app.post('/api/reset-session')
def reset_live_session():
    data = request.get_json(silent=True) or {}
    session_id = str(data.get('session_id') or '').strip()
    if session_id:
        with LIVE_LOCK:
            LIVE_AGENTS.pop(session_id, None)
            CALIBRATION_SAMPLES.pop(session_id, None)
            LIVE_SUMMARIES.pop(session_id, None)
    return jsonify(ok=True)


@app.post('/api/session-summary')
def session_summary():
    """返回一次实时会话的结构化摘要，供网页复盘 Agent 使用。"""
    data = request.get_json(silent=True) or {}
    session_id = str(data.get('session_id') or '').strip()
    if not session_id:
        return jsonify(error='缺少会话标识'), 400
    with LIVE_LOCK:
        state = LIVE_SUMMARIES.get(session_id)
        if state is None:
            return jsonify(error='当前会话没有可复盘的检测记录'), 404
        return jsonify(summary=public_summary(state))


@app.post('/api/review-agent')
def review_agent_api():
    """基于前端提交的结构化摘要回答复盘问题，不接收视频。"""
    data = request.get_json(silent=True) or {}
    summary = data.get('summary')
    question = str(data.get('question') or '').strip()
    if not isinstance(summary, dict):
        return jsonify(error='缺少结构化训练摘要'), 400
    if len(question) > 300:
        return jsonify(error='问题不能超过 300 个字符'), 400
    try:
        return jsonify(review_answer(summary, question))
    except Exception as exc:
        return jsonify(error=f'复盘生成失败：{exc}'), 500


@app.post('/analyze')
def analyze():
    uploaded = request.files.get('video')
    if not uploaded or not uploaded.filename:
        return page(HOME + '<section class="card">请选择一个视频文件。</section>'), 400
    suffix = Path(uploaded.filename).suffix.lower()
    if suffix not in ALLOWED:
        return page(HOME + '<section class="card">仅支持 MP4、MOV、AVI 或 MKV 视频。</section>'), 400
    job = uuid.uuid4().hex[:12]
    filename = f'{job}_{secure_filename(uploaded.filename)}'
    source = UPLOADS / filename
    uploaded.save(source)
    out_video = REPORTS / f'{job}_annotated.mp4'
    out_report = REPORTS / f'{job}_report.json'
    env = os.environ.copy()
    env['MPLCONFIGDIR'] = str(BASE / '.runtime' / 'matplotlib')
    command = [sys.executable, '-X', 'utf8', str(SCRIPT), str(source), '--out-video', str(out_video), '--out-report', str(out_report)]
    try:
        subprocess.run(command, cwd=BASE, env=env, check=True, timeout=300, capture_output=True, text=True, encoding='utf-8')
    except subprocess.TimeoutExpired:
        return page(HOME + '<section class="card">分析超时，请使用时长更短或文件更小的视频。</section>'), 408
    except subprocess.CalledProcessError as exc:
        message = (exc.stderr or exc.stdout or '视频处理失败').replace('<', '&lt;')[-1200:]
        return page(HOME + f'<section class="card"><h2>分析失败</h2><pre>{message}</pre></section>'), 500
    return result(job)


@app.get('/result/<job>')
def result(job: str):
    report_path = REPORTS / f'{job}_report.json'
    if not report_path.exists():
        abort(404)
    data = json.loads(report_path.read_text(encoding='utf-8'))
    peak = data['peak_assessment']; level = peak['level']
    reasons = '；'.join(peak['reasons']); recs = '；'.join(peak['recommendations'])
    video_url = url_for('artifact', filename=f'{job}_annotated.mp4')
    report_url = url_for('artifact', filename=f'{job}_report.json')
    body = f'''<section class="hero"><h1>分析完成</h1><div class="sub">已处理 {data['frames_analyzed']} 帧。本页与所有结果仅存在本机。</div></section>
<section class="card"><h2>整段视频最高风险</h2><div class="risk {level}">{level.upper()}</div><div class="grid"><div class="metric">触发时间<b>{peak['timestamp_seconds']} 秒</b></div><div class="metric">可信度<b>{peak['confidence']}</b></div><div class="metric">闭眼时长<b>{peak['signals']['closed_seconds']} 秒</b></div><div class="metric">打哈欠次数<b>{peak['signals']['yawn_count']}</b></div></div><p><b>判断依据：</b>{reasons}</p><p><b>建议：</b>{recs}</p><video controls src="{video_url}"></video><div class="links"><a href="{video_url}" download>下载标注视频</a><a href="{report_url}" download>下载 JSON 报告</a><a href="/">分析另一段视频</a></div></section>'''
    return page(body)


@app.get('/artifact/<path:filename>')
def artifact(filename: str):
    return send_from_directory(REPORTS, filename, as_attachment=False)


@app.errorhandler(413)
def too_large(_):
    return page(HOME + '<section class="card">文件超过 100 MB 限制。</section>'), 413


if __name__ == '__main__':
    # 0.0.0.0 让同一局域网设备可访问；需要仅限本机时设为 127.0.0.1。
    app.run(
        host=os.environ.get('CAR_AI_AGENT_HOST', '0.0.0.0'),
        port=int(os.environ.get('CAR_AI_AGENT_PORT', '5000')),
        debug=False,
    )
