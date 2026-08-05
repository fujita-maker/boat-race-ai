"""
ボートレース 妙味スコア予想 バックエンド
- 出走表・直前情報（展示/ST/コース）: boatraceopenapi の JSON（安定）
- 2連単オッズ: boatrace.jp を best-effort スクレイプ（取れなければ null → フロントは推定にフォールバック）
FastAPI / Render 対応。
"""
import os
import time
import datetime as dt
from typing import Any, Optional

import numpy as np
import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Boatrace Myoumi API")

UA = {"User-Agent": "Mozilla/5.0 (compatible; boatrace-myoumi/1.0)"}
OPENAPI = "https://boatraceopenapi.github.io/api/v1/{y}/{hd}.json"

# ---- openapi 日次ファイルの簡易キャッシュ（3分）----
_cache: dict[str, tuple[float, Any]] = {}

def fetch_openapi(hd: str) -> Any:
    now = time.time()
    if hd in _cache and now - _cache[hd][0] < 180:
        return _cache[hd][1]
    url = OPENAPI.format(y=hd[:4], hd=hd)
    r = requests.get(url, headers=UA, timeout=15)
    r.raise_for_status()
    data = r.json()
    _cache[hd] = (now, data)
    return data

# ---- 日次JSONから対象レースを取り出す（構造に多少の揺れがあっても拾えるように）----
def _as_items(node):
    if isinstance(node, dict):
        return list(node.values())
    if isinstance(node, list):
        return node
    return []

def find_race(data: Any, jcd: int, rno: int) -> Optional[dict]:
    progs = data.get("programs", data) if isinstance(data, dict) else data
    stadiums = progs.get("stadiums") if isinstance(progs, dict) else None
    if stadiums is None:
        return None
    stadium = None
    for s in _as_items(stadiums):
        if isinstance(s, dict) and int(s.get("stadium_number", -1)) == jcd:
            stadium = s
            break
    if stadium is None:
        # dict が番号キーの場合
        if isinstance(stadiums, dict) and str(jcd) in stadiums:
            stadium = stadiums[str(jcd)]
    if not isinstance(stadium, dict):
        return None
    races = stadium.get("races")
    for rc in _as_items(races):
        if isinstance(rc, dict) and int(rc.get("race_number", -1)) == rno:
            return rc
    if isinstance(races, dict) and str(rno) in races:
        return races[str(rno)]
    return None

def _collect_with_key(node, key, out):
    if isinstance(node, dict):
        if key in node:
            out.append(node)
        for v in node.values():
            _collect_with_key(v, key, out)
    elif isinstance(node, list):
        for v in node:
            _collect_with_key(v, key, out)

def extract_boats(race: dict) -> list[dict]:
    # 出走表エントリ（national_win_rate を持つ dict が各艇）
    entries: list[dict] = []
    _collect_with_key(race, "national_win_rate", entries)
    entries = {int(e.get("entry_number", 0)): e for e in entries if e.get("entry_number")}
    # 直前(preview: exhibition_time を持つ dict)
    prevs: list[dict] = []
    _collect_with_key(race, "exhibition_time", prevs)
    prevs = {int(p.get("entry_number", 0)): p for p in prevs if p.get("entry_number")}

    boats = []
    for n in range(1, 7):
        e = entries.get(n, {})
        p = prevs.get(n, {})
        course = p.get("course_number") or n
        st = p.get("start_timing")
        if st is None:
            st = e.get("average_start_timing")  # 直前が未確定なら平均STで代用
        boats.append({
            "frame": n,
            "course": int(course),
            "cls": e.get("rank_number_source") or "B1",
            "nat": e.get("national_win_rate"),
            "loc": e.get("local_win_rate"),
            "motor": e.get("motor_top_2_percent"),
            "ex": p.get("exhibition_time"),
            "st": st,
            "name": e.get("name"),
            "odds": None,  # 後で埋める
        })
    return boats

# ---- boatrace.jp から 2連単オッズ（best-effort）----
# 軸(1着) -> 各2着 のオッズを返す。取得/解析に失敗したら None を返す。
def fetch_exacta_odds(jcd: int, rno: int, hd: str) -> Optional[dict]:
    url = f"https://boatrace.jp/owpc/pc/race/odds2tf?rno={rno}&jcd={jcd:02d}&hd={hd}"
    try:
        r = requests.get(url, headers=UA, timeout=15)
        r.raise_for_status()
    except Exception:
        return None
    soup = BeautifulSoup(r.text, "lxml")
    cells = soup.select("td.oddsPoint")
    # 2連単は 30通り。
    vals = []
    for c in cells:
        t = c.get_text(strip=True).replace(",", "")
        try:
            vals.append(float(t))
        except ValueError:
            vals.append(None)
    if len(vals) < 30:
        return None
    vals = vals[:30]
    # boatrace.jp の2連単表は「1着1〜6の6列」を横に並べた1枚の表で、oddsPoint セルは
    # 行方向（各行=6列分）に並ぶ。各列(1着番号)内は「自分以外の艇番を昇順」で1行ずつ。
    # row-major で復元する（実オッズと数値一致を確認済み）。
    seconds_by_first = {f: [s for s in range(1, 7) if s != f] for f in range(1, 7)}
    odds = {}
    idx = 0
    for row_i in range(5):
        for first in range(1, 7):
            second = seconds_by_first[first][row_i]
            odds[(first, second)] = vals[idx]
            idx += 1
    return {f"{k[0]}-{k[1]}": v for k, v in odds.items()}

@app.get("/api/race")
def api_race(jcd: int = Query(...), rno: int = Query(...),
             hd: str = Query(...), with_odds: bool = Query(True)):
    try:
        data = fetch_openapi(hd)
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"openapi取得失敗: {e}"}, status_code=502)
    race = find_race(data, jcd, rno)
    if race is None:
        return JSONResponse({"ok": False, "error": "該当レースが見つかりません（日付/場/レース番号を確認、または未発売）"}, status_code=404)
    boats = extract_boats(race)

    axis_guess = min(boats, key=lambda b: b["course"])  # 進入1コース
    odds_map = None
    odds_status = "推定（オッズ未取得）"
    if with_odds:
        odds_map = fetch_exacta_odds(jcd, rno, hd)
        if odds_map:
            odds_status = "ライブ（boatrace.jp）"
            first = axis_guess["frame"]
            for b in boats:
                if b["frame"] != first:
                    b["odds"] = odds_map.get(f"{first}-{b['frame']}")
    now = dt.datetime.now(dt.timezone(dt.timedelta(hours=9)))
    return {
        "ok": True,
        "jcd": jcd, "rno": rno, "hd": hd,
        "boats": boats,
        "odds_status": odds_status,
        "odds_all": odds_map,
        "updated_at": now.strftime("%H:%M:%S"),
        "note": "出走表/直前はopenapi(約3分更新)、オッズはboatrace.jp best-effort",
    }

# ================= 過去データ学習（条件付きロジットで1着確率を較正）=================
_CLASS = {"A1": 1.0, "A2": 0.66, "B1": 0.33, "B2": 0.0}
_CBASE = {1: 0.55, 2: 0.15, 3: 0.12, 4: 0.10, 5: 0.06, 6: 0.02}

def _num(v, default):
    try:
        f = float(v)
        return f
    except (TypeError, ValueError):
        return default

def race_to_features(race: dict):
    """1レースを (6艇×7特徴, 勝者index) に変換。結果が無ければ None。"""
    boats = extract_boats(race)  # frame,course,cls,nat,loc,motor,ex,st
    # 勝者（place_number==1 の艇）
    places = []
    _collect_with_key(race, "place_number", places)
    winner = None
    for p in places:
        if p.get("place_number") == 1 and p.get("entry_number"):
            winner = int(p["entry_number"])
            break
    if winner is None:
        return None
    X = []
    for b in boats:
        X.append([
            _CBASE.get(b["course"], 0.05),
            _CLASS.get(b["cls"], 0.33),
            _num(b["nat"], 5.0),
            _num(b["loc"], 5.0),
            _num(b["motor"], 35.0),
            _num(b["st"], 0.16),
            _num(b["ex"], 6.80),
        ])
    return X, winner - 1  # index 0-5

def collect_dataset(days: int, step: int):
    """今日からdays日分をstep間隔でサンプルし、(X, winners) を集める。"""
    today = dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).date()
    Xs, ws, used_days = [], [], 0
    for d in range(1, days + 1, max(1, step)):
        day = today - dt.timedelta(days=d)
        hd = day.strftime("%Y%m%d")
        try:
            data = fetch_openapi(hd)
        except Exception:
            continue
        progs = data.get("programs", data) if isinstance(data, dict) else data
        stadiums = progs.get("stadiums") if isinstance(progs, dict) else None
        if not stadiums:
            continue
        got = 0
        for s in _as_items(stadiums):
            for rc in _as_items(s.get("races") if isinstance(s, dict) else None):
                if not isinstance(rc, dict):
                    continue
                fx = race_to_features(rc)
                if fx:
                    Xs.append(fx[0]); ws.append(fx[1]); got += 1
        if got:
            used_days += 1
    return Xs, ws, used_days

def train_conditional_logit(X, winners, iters=300, lr=0.4):
    X = np.asarray(X, float); winners = np.asarray(winners, int)
    R = len(X)
    flat = X.reshape(-1, X.shape[2])
    mu = flat.mean(0); sd = flat.std(0) + 1e-9
    Z = (X - mu) / sd
    cut = max(1, int(R * 0.8))
    beta = np.zeros(X.shape[2])
    for _ in range(iters):
        s = Z[:cut] @ beta; s -= s.max(1, keepdims=True)
        p = np.exp(s); p /= p.sum(1, keepdims=True)
        chosen = Z[:cut][np.arange(cut), winners[:cut]]
        exp_feat = (p[:, :, None] * Z[:cut]).sum(1)
        beta += lr * (chosen - exp_feat).mean(0)
    def probs(Zt):
        s = Zt @ beta; s -= s.max(1, keepdims=True)
        p = np.exp(s); p /= p.sum(1, keepdims=True); return p
    pte = probs(Z[cut:]); wte = winners[cut:]
    top = pte.argmax(1); ptop = pte.max(1)
    acc = float((top == wte).mean()) if len(wte) else 0.0
    base = float((wte == 0).mean()) if len(wte) else 0.0
    cal = {}
    for b in range(3, 10):
        m = (ptop >= b / 10) & (ptop < (b + 1) / 10)
        if m.sum() >= 10:
            cal[f"{b*10}-{b*10+10}%"] = round(float((top[m] == wte[m]).mean() * 100), 1)
    # 推奨・見送りしきい値: 予測トップ確率が高いほど買い。実的中率60%超えの帯の下限を目安に。
    thr = None
    for b in range(3, 10):
        key = f"{b*10}-{b*10+10}%"
        if cal.get(key, 0) >= 60:
            thr = round(b / 10, 2); break
    return {
        "beta": [round(x, 4) for x in beta.tolist()],
        "mu": [round(x, 4) for x in mu.tolist()],
        "sd": [round(x, 4) for x in sd.tolist()],
        "feature_order": ["course_base", "class", "nat", "loc", "motor", "st", "ex"],
        "test_acc": round(acc * 100, 1),
        "baseline_course1": round(base * 100, 1),
        "n_races": int(R),
        "calibration": cal,
        "suggest_buy_threshold": thr,
    }

@app.get("/api/train")
def api_train(days: int = Query(84), step: int = Query(3), iters: int = Query(300)):
    t0 = time.time()
    Xs, ws, used_days = collect_dataset(days, step)
    if len(Xs) < 200:
        return JSONResponse({"ok": False, "error": f"データ不足（{len(Xs)}レース）。daysを増やすか開催日を確認。"}, status_code=422)
    res = train_conditional_logit(Xs, ws, iters=iters)
    res.update({"ok": True, "used_days": used_days, "seconds": round(time.time() - t0, 1)})
    return res

# ---- フロント（HTMLを埋め込み：別ファイル不要で確実に配信）----
INDEX_HTML = r'''<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ボートレース 妙味スコア予想（自動更新版）</title>
<style>
  *{box-sizing:border-box;}
  body{font-family:"Hiragino Kaku Gothic ProN","Yu Gothic",Meiryo,Arial,sans-serif;margin:0;background:#0f1720;color:#1a2330;}
  .wrap{max-width:1080px;margin:0 auto;padding:0 16px 60px;}
  header{background:linear-gradient(135deg,#12263a,#1f3d5c);color:#fff;padding:20px 24px;}
  header h1{margin:0 0 4px;font-size:20px;} header p{margin:0;font-size:13px;color:#b9cbe0;}
  .card{background:#fff;border-radius:12px;padding:16px 20px;margin:14px 0;box-shadow:0 4px 18px rgba(0,0,0,.25);}
  .warn{background:#fff8e6;border-left:5px solid #e0a800;font-size:12.5px;line-height:1.6;color:#5a4600;}
  h2{font-size:15px;margin:2px 0 12px;color:#12263a;border-bottom:2px solid #e5ebf1;padding-bottom:6px;}
  table{border-collapse:collapse;width:100%;font-size:13px;} th,td{border:1px solid #dbe2ea;padding:5px 6px;text-align:center;}
  th{background:#eef3f8;font-weight:700;color:#33475b;} td input,td select{width:100%;border:1px solid #cfd8e3;border-radius:5px;padding:5px 4px;font-size:13px;text-align:center;background:#fbfdff;}
  td.frame{font-weight:800;}
  .chip.b1,td.frame.b1{background:#ffffff;color:#222;border:1.5px solid #b7bec6;}
  .chip.b2,td.frame.b2{background:#1a1a1a;color:#fff;}
  .chip.b3,td.frame.b3{background:#e6303a;color:#fff;}
  .chip.b4,td.frame.b4{background:#1f6fd0;color:#fff;}
  .chip.b5,td.frame.b5{background:#f4c430;color:#333;}
  .chip.b6,td.frame.b6{background:#3aa655;color:#fff;}
  .oddscol{background:#fff9ec!important;}
  button{background:#1f6fd0;color:#fff;border:none;border-radius:8px;padding:10px 20px;font-size:14px;font-weight:700;cursor:pointer;}
  button:hover{background:#175bb0;} .btn2{background:#5a6b7d;font-size:13px;padding:8px 14px;}
  .fetchbar{display:flex;gap:10px;align-items:center;flex-wrap:wrap;}
  .fetchbar select,.fetchbar input{padding:7px 8px;border:1px solid #cfd8e3;border-radius:6px;font-size:14px;}
  .status{font-size:12.5px;color:#33475b;background:#eef3f8;padding:6px 12px;border-radius:20px;}
  .status b{color:#12263a;} .live{color:#1c7a38;font-weight:800;} .est{color:#9a6a10;font-weight:800;}
  .controls{display:flex;align-items:center;gap:14px;flex-wrap:wrap;margin-top:10px;}
  .dial,.axisbox{font-size:13px;color:#33475b;background:#f4f7fa;padding:8px 12px;border-radius:8px;}
  .conf{font-size:14px;font-weight:800;padding:11px 14px;border-radius:9px;margin:2px 0 12px;}
  .cHi{background:#e3f6e8;color:#166a30;border:2px solid #3aa655;} .cMid{background:#fdf3e0;color:#8a5a00;border:2px solid #e0a800;} .cLow{background:#fdecec;color:#a02525;border:2px solid #e07272;}
  .rrow{display:grid;grid-template-columns:44px 1fr 76px;gap:10px;align-items:center;padding:9px 0;border-bottom:1px solid #eef2f6;}
  .chip{width:30px;height:30px;border-radius:6px;color:#fff;font-weight:800;display:flex;align-items:center;justify-content:center;}
  .barline{display:flex;align-items:center;gap:8px;margin:2px 0;font-size:11.5px;color:#556;} .barlabel{width:92px;text-align:right;color:#667;}
  .bar{height:12px;border-radius:3px;} .bJ{background:#1f6fd0;}.bM{background:#9aa7b4;}.bY{background:#3aa655;}.bYn{background:#e64c4c;}.bP{background:#7a5cd0;}
  .myo{font-size:19px;font-weight:800;text-align:right;}
  .flag{display:inline-block;font-size:11px;font-weight:800;padding:2px 8px;border-radius:20px;}
  .fAxis{background:#e7eefc;color:#1b4a9c;border:1px solid #9cb8e8;}.fHon{background:#e3f6e8;color:#1c7a38;border:1px solid #7fce9a;}.fSpec{background:#fdf3e0;color:#9a6a10;border:1px solid #e6c47a;}.fSkip{background:#f0f3f6;color:#7a8794;border:1px solid #cdd6df;}
  .verdict{font-size:17px;font-weight:800;padding:14px 16px;border-radius:10px;margin-top:6px;background:#e3f6e8;color:#166a30;border:2px solid #3aa655;}
  .roiNote{font-size:12.5px;margin-top:8px;padding:8px 12px;border-radius:8px;background:#f4f7fa;color:#455;line-height:1.6;}
  .expl{font-size:13px;line-height:1.7;color:#37485a;margin-top:10px;} .small{font-size:11.5px;color:#8794a2;}
</style>
</head>
<body>
<header>
  <h1>🚤 妙味スコア予想（自動更新版）</h1>
  <p>場・R・日付を選ぶと出走表と直前情報を自動取得。オッズは締切まで自動更新（取得できない時は推定にフォールバック）</p>
</header>
<div class="wrap">

  <div class="card warn">
    ⚠ 控除率25%は不変で、これは勝ちを保証しません。買い/見送りを一貫させ、結果を記録して<b>通算回収率</b>で検証する道具です。オッズは目安（取得タイミングで変動）。締切直前に最終確認を。
  </div>

  <div class="card">
    <h2>① レースを選んで取得</h2>
    <div class="fetchbar">
      <select id="jcd"></select>
      <select id="rno"></select>
      <input type="date" id="hd">
      <button onclick="fetchRace()">🔄 データ取得</button>
      <label class="status"><input type="checkbox" id="auto" onchange="toggleAuto()"> 自動更新(30秒)</label>
      <span class="status" id="status">未取得</span>
    </div>
    <p class="small">日付は開催日を選択。取得後は各セルを手で微調整もできます。オッズ列に値が入れば「軸→その艇」の実オッズで妙味を計算します。</p>
  </div>

  <div class="card">
    <h2>② 出走データ</h2>
    <table id="inputTable"><thead><tr>
      <th>艇</th><th>進入</th><th>級別</th><th>全国<br>勝率</th><th>当地<br>勝率</th><th>モーター<br>2連率</th><th>展示<br>タイム</th><th>展示<br>ST</th>
      <th class="oddscol">2連単<br>オッズ</th>
    </tr></thead><tbody id="tbody"></tbody></table>
    <div class="controls">
      <button onclick="calc()">▶ 計算する</button>
      <span class="axisbox">👑 1着軸：
        <select id="axisSel" onchange="if(lastData)calc()">
          <option value="auto">自動判定</option>
          <option value="1">1号艇</option><option value="2">2号艇</option><option value="3">3号艇</option>
          <option value="4">4号艇</option><option value="5">5号艇</option><option value="6">6号艇</option>
        </select></span>
      <span class="dial">🎯 カバー範囲：<input type="range" id="width" min="1" max="5" step="1" value="2" oninput="wv.textContent=this.value; if(lastData)calc()"><b id="wv">2</b> 艇</span>
    </div>
  </div>

  <div class="card" id="resultCard" style="display:none;">
    <h2>③ 判定結果</h2>
    <div id="conf"></div><div id="verdict"></div><div id="roiNote" class="roiNote"></div>
    <div id="rows" style="margin-top:14px;"></div><div class="expl" id="expl"></div>
  </div>
</div>

<script>
const VENUES={1:'桐生',2:'戸田',3:'江戸川',4:'平和島',5:'多摩川',6:'浜名湖',7:'蒲郡',8:'常滑',9:'津',10:'三国',11:'びわこ',12:'住之江',13:'尼崎',14:'鳴門',15:'丸亀',16:'児島',17:'宮島',18:'徳山',19:'下関',20:'若松',21:'芦屋',22:'福岡',23:'唐津',24:'大村'};
const CC=['b1','b2','b3','b4','b5','b6'];
const CLASS_SCORE={'A1':1.0,'A2':0.66,'B1':0.33,'B2':0.0};
const COURSE_BASE={1:0.55,2:0.15,3:0.12,4:0.10,5:0.06,6:0.02};
let lastData=null, autoTimer=null, lastOddsAll=null;

function initSelectors(){
  const j=document.getElementById('jcd');
  for(const[k,v]of Object.entries(VENUES)) j.insertAdjacentHTML('beforeend',`<option value="${k}">${String(k).padStart(2,'0')} ${v}</option>`);
  j.value='19';
  const r=document.getElementById('rno');
  for(let i=1;i<=12;i++) r.insertAdjacentHTML('beforeend',`<option value="${i}">${i}R</option>`);
  r.value='5';
  const d=new Date(); const jst=new Date(d.getTime()+ (d.getTimezoneOffset()+540)*60000);
  document.getElementById('hd').value=jst.toISOString().slice(0,10);
}
function hdCompact(){return document.getElementById('hd').value.replace(/-/g,'');}

function blankRow(c){return{course:c,cls:'B1',nat:'',loc:'',motor:'',ex:'',st:'',odds:''};}
function buildRows(data){const tb=document.getElementById('tbody');tb.innerHTML='';
 for(let i=0;i<6;i++){const d=data[i];const tr=document.createElement('tr');
  tr.innerHTML=`<td class="frame ${CC[i]}">${i+1}</td>
   <td><input type="number" min="1" max="6" value="${d.course}" data-k="course" data-i="${i}"></td>
   <td><select data-k="cls" data-i="${i}">${['A1','A2','B1','B2'].map(c=>`<option ${c===d.cls?'selected':''}>${c}</option>`).join('')}</select></td>
   <td><input type="number" step="0.01" value="${d.nat}" data-k="nat" data-i="${i}"></td>
   <td><input type="number" step="0.01" value="${d.loc}" data-k="loc" data-i="${i}"></td>
   <td><input type="number" step="0.01" value="${d.motor}" data-k="motor" data-i="${i}"></td>
   <td><input type="number" step="0.01" value="${d.ex}" data-k="ex" data-i="${i}"></td>
   <td><input type="number" step="0.01" value="${d.st}" data-k="st" data-i="${i}"></td>
   <td class="oddscol"><input type="number" step="0.1" value="${d.odds}" placeholder="—" data-k="odds" data-i="${i}"></td>`;
  tb.appendChild(tr);}}
function readInputs(){const data=[...Array(6)].map(()=>({}));
 document.querySelectorAll('#tbody [data-k]').forEach(el=>{const i=+el.dataset.i,k=el.dataset.k;
  if(k==='cls')data[i][k]=el.value;else if(k==='odds')data[i][k]=el.value===''?null:parseFloat(el.value);
  else data[i][k]=parseFloat(el.value);});return data;}

async function fetchRace(){
  const jcd=document.getElementById('jcd').value, rno=document.getElementById('rno').value, hd=hdCompact();
  const st=document.getElementById('status'); st.innerHTML='取得中…';
  try{
    const res=await fetch(`/api/race?jcd=${jcd}&rno=${rno}&hd=${hd}`);
    const j=await res.json();
    if(!j.ok){st.innerHTML='⚠ '+ (j.error||'取得失敗'); return;}
    lastOddsAll=j.odds_all||null;
    buildRows(j.boats.map(b=>({course:b.course, cls:b.cls||'B1',
      nat:b.nat??'', loc:b.loc??'', motor:b.motor??'', ex:b.ex??'', st:b.st??'', odds:b.odds??''})));
    const cls = j.odds_status.startsWith('ライブ')?'live':'est';
    st.innerHTML=`<b>${VENUES[jcd]} ${rno}R</b> 更新 ${j.updated_at} ／ オッズ:<span class="${cls}">${j.odds_status}</span>`;
    calc();
  }catch(e){ st.innerHTML='⚠ 通信エラー: '+e; }
}
function toggleAuto(){
  if(document.getElementById('auto').checked){ fetchRace(); autoTimer=setInterval(fetchRace,30000); }
  else { clearInterval(autoTimer); autoTimer=null; }
}

function norm(a,inv){const v=a.filter(x=>x!=null&&!isNaN(x));const mn=Math.min(...v),mx=Math.max(...v);
 if(!v.length||mx===mn)return a.map(()=>0.5);return a.map(x=>(x==null||isNaN(x))?0.5:(inv?(mx-x)/(mx-mn):(x-mn)/(mx-mn)));}

function calc(){
 const data=readInputs();lastData=data;
 const motorN=norm(data.map(d=>d.motor)),stN=norm(data.map(d=>d.st),true),exN=norm(data.map(d=>d.ex),true),
       locN=norm(data.map(d=>d.loc)),natN=norm(data.map(d=>d.nat));
 const clsN=data.map(d=>CLASS_SCORE[d.cls]??0.33),inner=data.map(d=>(6-d.course)/5);
 // ===== 学習済みモデル（過去約4530レースで較正した1着確率）=====
 // 特徴順: [course_base, class, nat, loc, motor, st, ex]
 const L_BETA=[0.6958,0.1224,0.728,0.0152,0.1583,-0.0137,-0.421];
 const L_MU=[0.1668,0.5436,5.2401,4.6892,32.7627,0.088,6.826];
 const L_SD=[0.1764,0.3059,1.3348,2.1123,11.0123,0.1053,0.1136];
 const scores=data.map(d=>{
   const f=[COURSE_BASE[d.course]??0.05, CLASS_SCORE[d.cls]??0.33, d.nat, d.loc, d.motor, d.st, d.ex];
   let s=0; for(let k=0;k<7;k++){const z=(f[k]-L_MU[k])/L_SD[k]; if(!isNaN(z)) s+=L_BETA[k]*z;} return s;
 });
 const _mx=Math.max(...scores),_e=scores.map(s=>Math.exp(s-_mx)),_sm=_e.reduce((a,b)=>a+b,0)||1;
 const power=_e.map(e=>e/_sm);   // 1着力 ＝ 学習済み1着確率（合計1）
 const pRank=[...power.keys()].sort((a,b)=>power[b]-power[a]);
 const sel=document.getElementById('axisSel').value;
 let axisIdx=(sel==='auto')?pRank[0]:(+sel-1);
 const relGap=(power[pRank[0]]-power[pRank[1]])/power[pRank[0]];
 const axisFrame=axisIdx+1;
 // 選択中の軸に対応する実オッズを全30通り(lastOddsAll)から都度引く（軸を変えると追従）
 const liveOdds=data.map((d,i)=>{
   if(i===axisIdx) return null;
   if(lastOddsAll){const v=lastOddsAll[`${axisFrame}-${i+1}`]; if(v!=null) return v;}
   return d.odds;
 });
 const impl=liveOdds.map(v=>v!=null?1/v:null),implN=norm(impl,false);
 const res=data.map((d,i)=>{const J=0.40*motorN[i]+0.25*stN[i]+0.20*exN[i]+0.15*locN[i];
   const oi=liveOdds[i];
   const M=oi!=null?implN[i]:0.45*clsN[i]+0.35*natN[i]+0.20*inner[i];
   return{i,frame:i+1,course:d.course,power:power[i],J,M,Y:J-M,byOdds:oi!=null,odds:oi};});
 const axis=res[axisIdx];
 const width=parseInt(document.getElementById('width').value);
 const cand=res.filter(r=>r.i!==axis.i).sort((a,b)=>b.Y-a.Y);
 const buys=cand.slice(0,width);const HON=0.15;
 // ---- 期待値(EV)判定：ライブオッズがある時だけ計算 ----
 const psum=power.reduce((a,b)=>a+b,0);
 const p1=power.map(x=>x/psum);                       // 各艇の1着確率(暫定)
 const others=[...Array(6).keys()].filter(i=>i!==axisIdx);
 const osum=others.reduce((s,i)=>s+power[i],0)||1;
 let evList=[];
 for(const i of others){
   const o=liveOdds[i]; if(o==null) continue;
   const p=p1[axisIdx]*(power[i]/osum);               // P(軸=1着 かつ i=2着)
   evList.push({frame:i+1, p, o, ev:p*o-1});
 }
 evList.sort((a,b)=>b.ev-a.ev);
 const evBuys=evList.filter(e=>e.ev>0);
 const confEl=document.getElementById('conf');let cCls;const autoTop=res[pRank[0]],second=res[pRank[1]];
 const headProb=power[pRank[0]]*100;  // 学習モデルの頭の1着確率
 if(headProb>=70){cCls='cHi';confEl.innerHTML=`1着信頼度：<b>高</b>（${autoTop.frame}号艇の1着確率 <b>${headProb.toFixed(0)}%</b>＝学習モデルで堅い）`;}
 else if(headProb>=60){cCls='cMid';confEl.innerHTML=`1着信頼度：<b>中</b>（${autoTop.frame}号艇の1着確率 <b>${headProb.toFixed(0)}%</b>）`;}
 else{cCls='cLow';confEl.innerHTML=`1着信頼度：<b>低 ⚠ 見送り推奨</b>（${autoTop.frame}号艇でも1着確率 <b>${headProb.toFixed(0)}%</b>＝学習モデルの推奨しきい値60%未満）`;}
 if(sel!=='auto'&&axisIdx!==pRank[0])confEl.innerHTML+=`　／ 自動推奨頭は${autoTop.frame}号艇（手動で${axis.frame}号艇指定中）`;
 confEl.className='conf '+cCls;
 const order=[axis,...cand];const rowsEl=document.getElementById('rows');rowsEl.innerHTML='';
 const maxAbs=Math.max(0.5,...res.map(r=>Math.abs(r.Y)));const maxP=Math.max(...power);
 order.forEach(r=>{const isAxis=r.i===axis.i,picked=buys.includes(r);let tag;
  if(isAxis)tag=`<span class="flag fAxis">1着 軸</span>`;
  else if(picked&&r.Y>=HON)tag=`<span class="flag fHon">◎本命妙味</span>`;
  else if(picked)tag=`<span class="flag fSpec">○投機カバー</span>`;else tag=`<span class="flag fSkip">見送り</span>`;
  const yPct=Math.round(Math.abs(r.Y)/maxAbs*100);
  const inner_html=isAxis
   ?`<div class="barline"><span class="barlabel">1着力</span><div class="bar bP" style="width:${Math.round(r.power/maxP*100)}%"></div><span>${(r.power/maxP*100).toFixed(0)}</span></div>`
   :`<div class="barline"><span class="barlabel">実力</span><div class="bar bJ" style="width:${Math.round(r.J*100)}%"></div><span>${(r.J*100).toFixed(0)}</span></div>
     <div class="barline"><span class="barlabel">市場評価</span><div class="bar bM" style="width:${Math.round(r.M*100)}%"></div><span>${(r.M*100).toFixed(0)}</span></div>
     <div class="barline"><span class="barlabel">妙味</span><div class="bar ${r.Y>=0?'bY':'bYn'}" style="width:${yPct}%"></div><span>${r.Y>=0?'+':''}${(r.Y*100).toFixed(0)}</span></div>`;
  rowsEl.insertAdjacentHTML('beforeend',`<div class="rrow"><div class="chip ${CC[r.i]}">${r.frame}</div>
    <div><div style="margin-bottom:3px;">${tag} <span class="small">${isAxis?'頭(1着)':'2着候補'}${r.byOdds?' ・実オッズ':''}</span></div>${inner_html}</div>
    <div class="myo" style="color:${isAxis?'#5a3fb0':(r.Y>=0?'#1c7a38':'#c0392b')}">${isAxis?(r.power/maxP*100).toFixed(0):(r.Y>=0?'+':'')+(r.Y*100).toFixed(0)}</div></div>`);});
 const legs=buys.map(b=>b.frame).join('・');const combos=buys.length*4,cost=combos*100;
 // ---- 期待値(EV)判定の表示 ----
 let evHtml='';
 if(evList.length){
   const lines=evList.map(e=>`<div style="font-size:12px;padding:1px 0;">${axisFrame}-${e.frame}：オッズ${e.o.toFixed(1)} × 推定${(e.p*100).toFixed(1)}% → EV <b style="color:${e.ev>=0?'#1c7a38':'#c0392b'}">${e.ev>=0?'+':''}${(e.ev*100).toFixed(0)}%</b></div>`).join('');
   const highOdds=evBuys.filter(e=>e.o>=15).length;
   const rec=evBuys.length
     ? `EV試算プラス：${evBuys.map(e=>axisFrame+'-'+e.frame).join('、')}`+(highOdds?`　⚠ <b style="color:#a02525;">高オッズ艇が含まれます＝モデルが穴を過大評価している可能性大。鵜呑み禁物。</b>`:'')
     : `EV試算プラスの買い目なし。`;
   evHtml=`<div style="background:#eef7ff;border:1px solid #b8d4ef;border-radius:8px;padding:10px;margin-bottom:10px;">
     <div style="font-weight:800;color:#12263a;margin-bottom:4px;">📊 期待値(EV)の試算 ＜実験中・未較正＞</div>
     ${lines}<div style="margin-top:5px;font-size:12.5px;">${rec}</div>
     <div style="font-size:11px;color:#a02525;margin-top:4px;">⚠ 確率は未較正の暫定モデル。EVプラスが高オッズ艇に偏るのは「穴の過大評価」の典型で、これは買い推奨ではありません。数百件記録して較正するまでは<b>参考値</b>として見てください。</div></div>`;
 } else {
   evHtml=`<div style="font-size:12px;color:#9a6a10;background:#fdf3e0;border-radius:8px;padding:8px 10px;margin-bottom:10px;">📊 EV試算：ライブオッズ未取得のため計算不可（下は推定妙味による目安）。</div>`;
 }
 document.getElementById('verdict').innerHTML=`${evHtml}✅ 【参考】3連単フォーメーション<br><span style="font-size:22px;">${axis.frame} → ${legs} → 全</span><br>
   <span style="font-size:13px;font-weight:600;">${combos}点＝${cost.toLocaleString()}円／2連単なら「${axis.frame}-${legs}」の${buys.length}点</span>`;
 const spec=buys.filter(b=>b.Y<HON).length;const roi=document.getElementById('roiNote');
 if(cCls==='cLow')roi.innerHTML=`⚠ <b>1着信頼度が低いレース。</b>頭自体が飛ぶ危険が高いので、買う前に「勝負するか見送るか」を先に判断してください。`;
 else if(spec>0)roi.innerHTML=`⚖️ 網に<b>「○投機カバー」が${spec}艇</b>（根拠薄・オッズ頼み）。的中率は上がるが回収率は75%側へ。`;
 else roi.innerHTML=`✅ 買い目は全て<b>「◎本命妙味」（裏付けあり）</b>。一番濃い狙い方です。`;
 document.getElementById('expl').innerHTML=`1着力順：${pRank.map(i=>res[i].frame+'号艇').join(' > ')}。頭＝${axis.frame}号艇、2着に ${legs} を流します。<br><span class="small">結果は記録シートへ。当たり外れ両方を残して通算回収率で検証を。</span>`;
 document.getElementById('resultCard').style.display='block';
}
initSelectors();
buildRows([1,2,3,4,5,6].map(blankRow));
</script>
</body>
</html>
'''

@app.get("/")
def index():
    return HTMLResponse(INDEX_HTML)
