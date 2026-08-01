
from __future__ import annotations
import json, math, os, re, sqlite3, time
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup

JST = timezone(timedelta(hours=9))
DB = Path("boat_race.db")
HEADERS = {"User-Agent":"Mozilla/5.0 Chrome/151 Safari/537.36"}
VENUES = {
"桐生":"01","戸田":"02","江戸川":"03","平和島":"04","多摩川":"05","浜名湖":"06",
"蒲郡":"07","常滑":"08","津":"09","三国":"10","びわこ":"11","住之江":"12",
"尼崎":"13","鳴門":"14","丸亀":"15","児島":"16","宮島":"17","徳山":"18",
"下関":"19","若松":"20","芦屋":"21","福岡":"22","唐津":"23","大村":"24"}

st.set_page_config(page_title="WDJ Boat Race AI", layout="wide")

def db():
    con=sqlite3.connect(DB)
    con.execute("""CREATE TABLE IF NOT EXISTS predictions(
      id INTEGER PRIMARY KEY AUTOINCREMENT, race_date TEXT, venue TEXT, race_no INTEGER,
      mode TEXT, fixed_at TEXT, deadline TEXT, classification TEXT, decision TEXT,
      confidence REAL, investment INTEGER, expected_value REAL, bets_json TEXT,
      weather_json TEXT, source_json TEXT, result_combo TEXT, payout INTEGER DEFAULT 0,
      profit INTEGER DEFAULT 0, settled INTEGER DEFAULT 0, season TEXT, wind_dir TEXT)""")
    con.commit(); return con

def fetch(url):
    r=requests.get(url,headers=HEADERS,timeout=18)
    r.raise_for_status(); r.encoding=r.apparent_encoding or "utf-8"
    return r.text

def text(html): return BeautifulSoup(html,"html.parser").get_text(" ",strip=True)

def official_urls(d,code,r):
    q=f"rno={r}&jcd={code}&hd={d}"
    b="https://www.boatrace.jp/owpc/pc/race"
    return {"list":f"{b}/racelist?{q}","before":f"{b}/beforeinfo?{q}",
            "odds":f"{b}/odds3t?{q}","result":f"{b}/raceresult?{q}"}

def official(d,code,r):
    urls=official_urls(d,code,r); out={"urls":urls,"errors":[]}
    for k in ("list","before","odds"):
        try: out[k]=text(fetch(urls[k]))
        except Exception as e: out["errors"].append(f"{k}:{e}")
    return out

def poseidon(d,code,r):
    u=f"https://poseidon-boatrace.net/race/{d}/{int(code)}/{r}R"
    try: return u,text(fetch(u)),None
    except Exception as e: return u,"",str(e)

def umepyon(d,venue,r):
    start="https://umepyon.com/"
    try:
        s=BeautifulSoup(fetch(start),"html.parser")
        venue_pages=[]
        for a in s.find_all("a",href=True):
            label=a.get_text(" ",strip=True); href=requests.compat.urljoin(start,a["href"])
            if venue in label or venue in href or (venue=="びわこ" and "琵琶湖" in label):
                venue_pages.append(href)
        for p in venue_pages[:8] or [start]:
            ss=BeautifulSoup(fetch(p),"html.parser")
            for a in ss.find_all("a",href=True):
                label=a.get_text(" ",strip=True); href=requests.compat.urljoin(p,a["href"])
                if re.search(rf"(^|\D){r}\s*R(\D|$)",label,re.I):
                    return href,text(fetch(href)),None
        return start,"","対象R未検出"
    except Exception as e: return start,"",str(e)

def parse_poseidon(t):
    rows=[]
    for c,p,o,idx in re.findall(r"\b([1-6]-[1-6]-[1-6])\s+([0-9.]+)%\s+([0-9.]+)\s+([0-9.]+)pt",t):
        if len(set(c.split("-")))==3:
            rows.append({"combo":c,"prob":float(p),"odds":float(o),"index":float(idx)})
    uniq={}
    for x in rows: uniq.setdefault(x["combo"],x)
    return list(uniq.values())

def parse_ume(t):
    out={}
    for a,b,c,p in re.findall(r"\b([1-6])[-－>]([1-6])[-－>]([1-6])\b[^%]{0,40}?([0-9.]+)\s*%",t):
        combo=f"{a}-{b}-{c}"
        if len({a,b,c})==3: out[combo]=max(out.get(combo,0),float(p))
    return out

def val(t,pattern,default="取得エラー"):
    m=re.search(pattern,t); return m.group(1).strip() if m else default

def extract_meta(t):
    wind=val(t,r"風速\s*([0-9.]+)\s*m")
    wave=val(t,r"波高\s*([0-9.]+)\s*cm")
    wdir=val(t,r"風向\s*([^\s]+)")
    a1=min(6,len(re.findall(r"\bA1\b",t)))
    exhibition_times=re.findall(r"\b([1-6])\s+([6-7]\.[0-9]{2})\b",t)
    exhibition_st=re.findall(r"\b([1-6])\s+F?([0-9]\.[0-9]{2})\b",t)
    return {"wind":wind,"wave":wave,"wind_dir":wdir,"a1":a1,
            "exhibition_time_count":len(set(x[0] for x in exhibition_times)),
            "exhibition_st_count":len(set(x[0] for x in exhibition_st))}

def season(month):
    return "春" if month in (3,4,5) else "夏" if month in (6,7,8) else "秋" if month in (9,10,11) else "冬"

def calculate(o,p,u,meta):
    """
    毎レース必ず参考予想を出す。
    「予想を出すこと」と「実際に買う判断」は分離する。
    """
    merged=[]
    all_combos=[f"{a}-{b}-{c}" for a in range(1,7) for b in range(1,7) for c in range(1,7) if len({a,b,c})==3]

    # ポセイドン中心
    for x in p:
        ume=u.get(x["combo"])
        prob=min(25.0, x["prob"]*0.8 + (ume if ume is not None else x["prob"])*0.2)
        ev=prob/100*x["odds"] if x.get("odds") else 0
        merged.append({**x,"ume_prob":ume,"model_prob":round(prob,2),"ev":round(ev,3)})

    # 梅吉だけ取得できた買い目も追加
    existing={x["combo"] for x in merged}
    for combo,up in u.items():
        if combo not in existing:
            merged.append({
                "combo":combo,"prob":up,"odds":0,"index":0,
                "ume_prob":up,"model_prob":round(up,2),"ev":0
            })

    # どちらも取れない場合の最低限フォールバック
    # 1号艇頭を基本に、次いで2号艇頭。これは参考予想として低評価にする。
    if not merged:
        fallback = [
            ("1-2-3",8.0),("1-3-2",7.5),("1-2-4",7.0),
            ("1-4-2",6.5),("2-1-3",5.5),("2-1-4",5.0),
            ("1-3-4",4.8),("3-1-2",4.2)
        ]
        for combo,prob in fallback:
            merged.append({
                "combo":combo,"prob":prob,"odds":0,"index":0,
                "ume_prob":None,"model_prob":prob,"ev":0
            })

    merged.sort(key=lambda x:(x.get("ev",0),x.get("model_prob",0)),reverse=True)

    try: wind=float(meta["wind"])
    except: wind=None
    official_ok = not o["errors"]
    exhibition_ok = meta["exhibition_time_count"]>=6 and meta["exhibition_st_count"]>=6
    middle = wind is not None and 2<=wind<=3 and meta["a1"]<=1

    source_count = int(official_ok) + int(bool(p)) + int(bool(u))
    data_score = min(3, source_count)
    top_prob = merged[0]["model_prob"] if merged else 0
    top_ev = max([x.get("ev",0) for x in merged], default=0)

    # 星評価
    stars = 1
    if source_count >= 1: stars += 1
    if source_count >= 2 and top_prob >= 6: stars += 1
    if official_ok and exhibition_ok and middle: stars += 1
    if top_ev >= 1.10 and len([x for x in merged if x.get("ev",0)>=1.10]) >= 6: stars += 1
    stars = max(1,min(5,stars))

    if stars >= 4:
        action = "買い"
    elif stars == 3:
        action = "少額"
    else:
        action = "見送り"

    classification = "中波乱" if middle else ("情報不足" if source_count < 2 else "堅い・一般")

    # 買い目は必ず上位表示
    predictions = merged[:8]

    # 実際の資金配分は買い判定時のみ
    selected=[x for x in merged if x.get("ev",0)>=1.10][:8]
    bets=[]
    if action=="買い" and len(selected)>=6 and official_ok and exhibition_ok and middle:
        total=5000
        weights=[max(.001,x["model_prob"]/max(x.get("odds",1),1)**.3) for x in selected]
        amounts=[max(100,round(total*w/sum(weights)/100)*100) for w in weights]
        diff=total-sum(amounts); i=0
        while diff and i<200:
            j=i%len(amounts); step=100 if diff>0 else -100
            if amounts[j]+step>=100:
                amounts[j]+=step; diff-=step
            i+=1
        for x,a in zip(selected,amounts):
            ret=int(a*x["odds"]) if x.get("odds") else 0
            bets.append({**x,"amount":a,"return":ret,"profit":ret-total})
        if any(x["return"]<=total for x in bets):
            bets=[]
            action="少額"
            stars=min(stars,3)

    if action=="買い":
        confidence=min(95,65+stars*5)
    elif action=="少額":
        confidence=55
    else:
        confidence=20 if source_count else 10

    avg_ev=round(sum(x.get("ev",0) for x in predictions)/len(predictions),3) if predictions else 0
    return classification,action,stars,confidence,avg_ev,merged,predictions,bets

def save_prediction(data):
    con=db()
    con.execute("""INSERT INTO predictions(
    race_date,venue,race_no,mode,fixed_at,deadline,classification,decision,confidence,
    investment,expected_value,bets_json,weather_json,source_json,season,wind_dir)
    VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",(
    data["race_date"],data["venue"],data["race_no"],data["mode"],data["fixed_at"],
    data["deadline"],data["classification"],data["decision"],data["confidence"],
    data["investment"],data["expected_value"],json.dumps(data["bets"],ensure_ascii=False),
    json.dumps(data["weather"],ensure_ascii=False),json.dumps(data["sources"],ensure_ascii=False),
    data["season"],data["weather"]["wind_dir"]))
    con.commit()

def settle_rows():
    con=db()
    rows=pd.read_sql_query("SELECT * FROM predictions ORDER BY id DESC",con)
    return rows

st.title("🚤 WDJ ボートレースAI Web版 v4")
tabs=st.tabs(["予想","締切3分前監視","成績ダッシュボード","結果登録"])

with tabs[0]:
    c1,c2,c3=st.columns(3)
    date=c1.date_input("開催日")
    venue=c2.selectbox("場",list(VENUES))
    race=c3.selectbox("R",range(1,13))
    deadline=st.time_input("締切時刻")
    real=st.checkbox("実購入として記録する",False)
    if st.button("今すぐ最新データ取得 → 予想固定",type="primary",use_container_width=True):
        d=date.strftime("%Y%m%d"); code=VENUES[venue]
        o=official(d,code,race); pu,pt,pe=poseidon(d,code,race); uu,ut,ue=umepyon(d,venue,race)
        alltxt=" ".join(o.get(k,"") for k in ("list","before","odds"))
        meta=extract_meta(alltxt); pr=parse_poseidon(pt); ur=parse_ume(ut)
        cls,dec,stars,conf,ev,rank,predictions,bets=calculate(o,pr,ur,meta)
        mode="real" if real else "virtual"
        if real and dec=="買い": dec="購入"
        fixed=datetime.now(JST)
        rec={"race_date":str(date),"venue":venue,"race_no":race,"mode":mode,
             "fixed_at":fixed.isoformat(),"deadline":f"{date} {deadline}",
             "classification":cls,"decision":dec,"confidence":conf,
             "investment":sum(x.get("amount",0) for x in bets),"expected_value":ev,
             "bets":bets,"weather":meta,"season":season(date.month),
             "sources":{"official":o["urls"],"official_errors":o["errors"],
                        "poseidon":pu,"poseidon_error":pe,"umepyon":uu,"umepyon_error":ue}}
        save_prediction(rec)
        star_text="★"*stars+"☆"*(5-stars)
        st.success(f"{venue}{race}R｜予想固定｜{dec}")
        st.markdown(f"## {star_text}　{dec}")
        a,b,c,dcol=st.columns(4)
        a.metric("分類",cls); b.metric("最終判断",dec); c.metric("自信度",star_text); dcol.metric("参考期待値",ev)

        st.subheader("毎回表示する参考予想")
        pdf=pd.DataFrame(predictions)
        pdf=pdf.rename(columns={"combo":"買い目","model_prob":"推定確率","odds":"直前オッズ","ev":"期待値","ume_prob":"梅吉確率","index":"海神指数"})
        showcols=[c for c in ["買い目","推定確率","直前オッズ","期待値","梅吉確率","海神指数"] if c in pdf.columns]
        st.dataframe(pdf[showcols],hide_index=True,use_container_width=True)

        if bets:
            st.subheader("購入対象の資金配分")
            df=pd.DataFrame(bets)
            df=df.rename(columns={"combo":"買い目","amount":"購入額","odds":"直前オッズ","return":"的中時払戻","profit":"純利益","model_prob":"推定確率","ev":"期待値"})
            st.dataframe(df[["買い目","購入額","直前オッズ","的中時払戻","純利益","推定確率","期待値"]],hide_index=True,use_container_width=True)
        elif dec=="少額":
            st.warning("参考予想は表示していますが、条件が弱いため少額判断です。自動資金配分は行いません。")
        else:
            st.warning("参考予想は表示していますが、実購入は見送りです。")

with tabs[1]:
    st.write("画面を開いたままにすると15秒ごとに確認し、締切3分前の範囲に入ると最新データで固定します。")
    c1,c2,c3=st.columns(3)
    mdate=c1.date_input("監視日",key="mdate")
    mvenue=c2.selectbox("監視する場",list(VENUES),key="mvenue")
    mrace=c3.selectbox("監視R",range(1,13),key="mrace")
    mdeadline=st.time_input("締切",key="mdeadline")
    active=st.toggle("自動監視ON",False)
    placeholder=st.empty()
    if active:
        target=datetime.combine(mdate,mdeadline,tzinfo=JST)
        now=datetime.now(JST); remain=(target-now).total_seconds()
        placeholder.info(f"締切まで {max(0,int(remain))} 秒")
        if 0 < remain <= 180:
            st.warning("締切3分前です。「予想」タブで最新取得・固定してください。サーバー常駐版ではここを自動実行します。")
        time.sleep(15)
        st.rerun()

with tabs[2]:
    rows=settle_rows()
    if rows.empty: st.info("予想データがまだありません")
    else:
        settled=rows[rows["settled"]==1]
        investment=int(settled["investment"].sum()) if not settled.empty else 0
        payout=int(settled["payout"].sum()) if not settled.empty else 0
        profit=payout-investment
        roi=(payout/investment*100) if investment else 0
        a,b,c,dcol=st.columns(4)
        a.metric("確定レース",len(settled)); b.metric("投資",f"{investment:,}円")
        c.metric("収支",f"{profit:,}円"); dcol.metric("回収率",f"{roi:.1f}%")
        if not settled.empty:
            for group,label in [("venue","場別"),("season","季節別"),("wind_dir","風向別")]:
                g=settled.groupby(group,dropna=False).agg(レース数=("id","count"),投資=("investment","sum"),払戻=("payout","sum")).reset_index()
                g["収支"]=g["払戻"]-g["投資"]; g["回収率"]=g.apply(lambda x:x["払戻"]/x["投資"]*100 if x["投資"] else 0,axis=1)
                st.subheader(label); st.dataframe(g,hide_index=True,use_container_width=True)
        st.subheader("過去の固定予想")
        st.dataframe(rows[["race_date","venue","race_no","mode","fixed_at","classification","decision","investment","expected_value","settled","profit"]],hide_index=True,use_container_width=True)

with tabs[3]:
    rows=settle_rows()
    if rows.empty: st.info("登録対象がありません")
    else:
        labels={int(x.id):f"#{x.id} {x.race_date} {x.venue}{x.race_no}R {x.decision}" for _,x in rows.iterrows()}
        pid=st.selectbox("対象予想",list(labels),format_func=lambda x:labels[x])
        combo=st.text_input("確定3連単（例 1-2-3）")
        payout=st.number_input("この予想の実払戻額",min_value=0,step=100)
        if st.button("結果を確定"):
            con=db(); inv=int(rows.loc[rows.id==pid,"investment"].iloc[0])
            con.execute("UPDATE predictions SET result_combo=?,payout=?,profit=?,settled=1 WHERE id=?",
                        (combo,int(payout),int(payout)-inv,pid)); con.commit()
            st.success("結果を確定しました。締切前予想は変更していません。")
