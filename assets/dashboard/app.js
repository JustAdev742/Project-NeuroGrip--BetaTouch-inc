"use strict";
const $=s=>document.querySelector(s),$$=s=>document.querySelectorAll(s);

// ── State colors ──
const SC={IDLE:{c:"#64748b",bg:"var(--slate-dim)"},DETECTING:{c:"#00d4ff",bg:"var(--cyan-dim)"},GRIPPING:{c:"#f59e0b",bg:"var(--amber-dim)"},HOLDING:{c:"#10b981",bg:"var(--emerald-dim)"},ERROR:{c:"#ef4444",bg:"var(--red-dim)"},MOCK_MODE:{c:"#8b5cf6",bg:"var(--violet-dim)"}};
const IC={REST:"#64748b",OPEN:"#10b981",CLOSE:"#ef4444",HOLD:"#f59e0b",CANCEL:"#8b5cf6"};

// ── DOM cache ──
const D={
	connDot:$("#conn-dot"),mockBadge:$("#mock-badge"),topbarMode:$("#topbar-mode"),topbarClock:$("#topbar-clock"),
	statePill:$("#state-pill"),stateDot:$("#state-dot-lg"),sysState:$("#sys-state-text"),ctrlFps:$("#ctrl-fps"),visFps:$("#vis-fps"),errorLine:$("#error-line"),
	ringFill:$("#ring-fill"),ringPct:$("#ring-pct"),detectLabel:$("#detect-label"),gripPill:$("#grip-pill"),gripReason:$("#grip-reason"),
	handGripTag:$("#hand-grip-tag"),servoBars:$("#servo-bars"),
	waveCanvas:$("#wave-canvas"),emgPill:$("#emg-intent-pill"),emgSig:$("#emg-sig"),emgQual:$("#emg-qual"),emgUptime:$("#emg-uptime"),qualFill:$("#qual-fill"),
	overrideBtns:$("#override-btns"),overrideSection:$("#override-section"),
	tlScroll:$("#tl-scroll"),
	watchTime:$("#watch-time"),watchSecs:$("#watch-secs"),watchDate:$("#watch-date"),wMode:$("#w-mode"),wState:$("#w-state"),wQuality:$("#w-quality"),wUptime:$("#w-uptime"),
	modeGrid:$("#mode-grid"),
	trainLabel:$("#train-label"),trainBtn:$("#train-capture-btn"),trainStatus:$("#train-status"),trainStatBody:$("#train-stat-body"),
	ghIssues:$("#gh-issues"),ghReleases:$("#gh-releases"),
	reactionCircle:$("#reaction-circle"),reactionResult:$("#reaction-result"),reactionStart:$("#reaction-start"),
	gripTarget:$("#grip-target-text"),gripCurrent:$("#grip-current-text"),gripScore:$("#grip-score"),gripStart:$("#grip-start"),
};

let isMock=false,lastEvtCount=0,currentGrip="OPEN";

// ══════════════════════════════════════════════════════════════
// SCREEN NAVIGATION
// ══════════════════════════════════════════════════════════════
$$(".nav-btn").forEach(btn=>{
	btn.addEventListener("click",()=>{
		const target=btn.dataset.screen;
		$$(".screen").forEach(s=>s.classList.remove("active"));
		$$(".nav-btn").forEach(b=>b.classList.remove("active"));
		$(`#screen-${target}`).classList.add("active");
		btn.classList.add("active");
		if(target==="github")loadGitHub();
		if(target==="training")loadTrainingStats();
	});
});

// ══════════════════════════════════════════════════════════════
// WAVEFORM CHART
// ══════════════════════════════════════════════════════════════
class WaveformChart{
	constructor(canvas){this.canvas=canvas;this.ctx=canvas.getContext("2d");this.data=[];this.max=200;this._resize();window.addEventListener("resize",()=>this._resize())}
	_resize(){const r=this.canvas.parentElement.getBoundingClientRect();const d=window.devicePixelRatio||1;this.canvas.width=r.width*d;this.canvas.height=70*d;this.ctx.setTransform(d,0,0,d,0,0);this.w=r.width;this.h=70}
	update(hist){if(!hist||!hist.length)return;this.data=hist.slice(-this.max);this._draw()}
	_draw(){const{ctx,w,h,data}=this;ctx.clearRect(0,0,w,h);if(data.length<2)return;
		ctx.strokeStyle="rgba(0,212,255,0.05)";ctx.lineWidth=.5;
		for(let y=0;y<=1;y+=.25){const py=h-y*h;ctx.beginPath();ctx.moveTo(0,py);ctx.lineTo(w,py);ctx.stroke()}
		const thresholds=[[.6,"rgba(239,68,68,0.2)"],[.4,"rgba(245,158,11,0.15)"],[.2,"rgba(100,116,139,0.1)"]];
		for(const[v,c]of thresholds){const py=h-v*h;ctx.strokeStyle=c;ctx.setLineDash([3,3]);ctx.beginPath();ctx.moveTo(0,py);ctx.lineTo(w,py);ctx.stroke();ctx.setLineDash([])}
		const step=w/(data.length-1);const grad=ctx.createLinearGradient(0,0,0,h);grad.addColorStop(0,"rgba(0,212,255,0.12)");grad.addColorStop(1,"rgba(0,212,255,0)");
		const fp=new Path2D();for(let i=0;i<data.length;i++){const x=i*step,y=h-data[i]*h;i===0?fp.moveTo(x,y):fp.lineTo(x,y)}fp.lineTo(w,h);fp.lineTo(0,h);fp.closePath();ctx.fillStyle=grad;ctx.fill(fp);
		ctx.beginPath();for(let i=0;i<data.length;i++){const x=i*step,y=h-data[i]*h;i===0?ctx.moveTo(x,y):ctx.lineTo(x,y)}ctx.strokeStyle="#00d4ff";ctx.lineWidth=1.5;ctx.lineJoin="round";ctx.stroke();
		const lx=(data.length-1)*step,ly=h-data[data.length-1]*h;ctx.beginPath();ctx.arc(lx,ly,3,0,Math.PI*2);ctx.fillStyle="#00d4ff";ctx.fill()}
}
const waveChart=new WaveformChart(D.waveCanvas);

// ══════════════════════════════════════════════════════════════
// HAND RENDERER
// ══════════════════════════════════════════════════════════════
const FINGERS={
	pinky:{base:[105,175],mid:[95,120],tip:[90,70]},
	ring:{base:[125,172],mid:[120,110],tip:[118,50]},
	middle:{base:[150,170],mid:[150,105],tip:[150,38]},
	index:{base:[175,172],mid:[180,108],tip:[183,45]},
	thumb:{base:[95,220],mid:[60,195],tip:[35,175]}
};

function updateHand(servo,grip){
	for(const name of["thumb","index","middle","ring","pinky"]){
		const g=$(`#finger-${name}`);if(!g)continue;
		const angle=servo[name]||0;const flex=Math.min(1,Math.max(0,angle/180));
		const s1=g.querySelector(".seg1"),s2=g.querySelector(".seg2"),tip=g.querySelector(".tip");
		const f=FINGERS[name];if(!f||!s1||!s2||!tip)continue;
		let cm,ct;
		if(name==="thumb"){cm=[f.mid[0]+flex*50,f.mid[1]+flex*30];ct=[f.tip[0]+flex*90,f.tip[1]+flex*55]}
		else{cm=[f.mid[0]+flex*(150-f.mid[0])*.4,f.mid[1]+flex*60];ct=[f.tip[0]+flex*(150-f.tip[0])*.6,f.tip[1]+flex*160]}
		s1.setAttribute("x2",cm[0]);s1.setAttribute("y2",cm[1]);
		s2.setAttribute("x1",cm[0]);s2.setAttribute("y1",cm[1]);s2.setAttribute("x2",ct[0]);s2.setAttribute("y2",ct[1]);
		tip.setAttribute("cx",ct[0]);tip.setAttribute("cy",ct[1]);
		const col=flex>.5?"#f59e0b":"#00d4ff",op=.4+flex*.6;
		s1.style.stroke=col;s2.style.stroke=col;tip.style.fill=col;s1.style.opacity=op;s2.style.opacity=op;tip.style.opacity=op;
	}
	if(D.handGripTag)D.handGripTag.textContent=grip||"OPEN";
}

function updateServoBars(pos){
	const order=["thumb","index","middle","ring","pinky"];
	D.servoBars.innerHTML=order.filter(k=>pos[k]!==undefined).map(n=>{
		const d=pos[n],p=Math.min(100,(d/180)*100);
		return `<div class="servo-row"><span class="servo-name">${n}</span><div class="servo-track"><div class="servo-fill" style="width:${p}%"></div></div><span class="servo-deg">${d.toFixed(0)}°</span></div>`;
	}).join("");
}

// ══════════════════════════════════════════════════════════════
// CONFIDENCE GAUGE
// ══════════════════════════════════════════════════════════════
function updateGauge(conf){
	const circ=2*Math.PI*40,off=circ*(1-conf);
	D.ringFill.style.strokeDashoffset=off;
	D.ringPct.textContent=`${Math.round(conf*100)}%`;
	D.ringFill.style.stroke=conf>=.7?"var(--emerald)":conf>=.4?"var(--amber)":"var(--red)";
}

// ══════════════════════════════════════════════════════════════
// TIMELINE
// ══════════════════════════════════════════════════════════════
function updateTimeline(events){
	if(!events||events.length===lastEvtCount)return;
	lastEvtCount=events.length;
	D.tlScroll.innerHTML=events.slice(-25).reverse().map(e=>{
		const t=new Date(e.timestamp*1000).toLocaleTimeString("en-US",{hour12:false});
		return `<div class="tl-row"><span class="tl-time">${t}</span><span class="tl-tag ${e.event_type}">${e.event_type.replace("_"," ")}</span><span class="tl-detail">${e.detail}</span></div>`;
	}).join("")||'<div class="tl-empty">No events yet</div>';
}

// ══════════════════════════════════════════════════════════════
// HELPERS
// ══════════════════════════════════════════════════════════════
function fmt(s){const h=Math.floor(s/3600),m=Math.floor(s%3600/60),sec=Math.floor(s%60);return[h,m,sec].map(v=>String(v).padStart(2,"0")).join(":")}
function fmtShort(s){const m=Math.floor(s/60),sec=Math.floor(s%60);return`${String(m).padStart(2,"0")}:${String(sec).padStart(2,"0")}`}

// ══════════════════════════════════════════════════════════════
// MAIN UI UPDATE
// ══════════════════════════════════════════════════════════════
function updateUI(data){
	const state=data.system_state||"IDLE",sc=SC[state]||SC.IDLE;
	D.sysState.textContent=state;D.sysState.style.color=sc.c;
	D.stateDot.style.background=sc.c;D.stateDot.style.boxShadow=`0 0 8px ${sc.c}`;D.stateDot.classList.toggle("glow",state!=="IDLE");
	D.statePill.textContent=state;D.statePill.style.background=sc.bg;D.statePill.style.color=sc.c;
	D.ctrlFps.textContent=data.control_fps||0;D.visFps.textContent=data.vision_fps||0;
	D.errorLine.textContent=data.last_error||"No errors";D.errorLine.style.color=data.last_error?"var(--red)":"var(--text-muted)";

	isMock=!!data.mock_mode_active;
	D.mockBadge.textContent=isMock?"MOCK":"LIVE";D.mockBadge.classList.toggle("mock",isMock);
	D.overrideBtns.querySelectorAll(".o-btn").forEach(b=>{b.disabled=!isMock});

	const dets=data.detected_objects||[],top=dets[0];
	D.detectLabel.textContent=top?top.label:"—";
	updateGauge(top?top.confidence:0);

	currentGrip=data.selected_grip||"OPEN";
	D.gripPill.textContent=currentGrip;D.gripReason.textContent=data.grip_selection_reason||"—";

	const intent=data.emg_intent||"REST";
	D.emgPill.textContent=intent;D.emgPill.style.background=(IC[intent]||"#64748b")+"1a";D.emgPill.style.color=IC[intent]||"#64748b";
	D.emgSig.textContent=(data.emg_signal_level||0).toFixed(2);
	const q=data.emg_quality||0;D.emgQual.textContent=`${Math.round(q*100)}%`;
	D.qualFill.style.width=`${q*100}%`;
	D.emgUptime.textContent=fmtShort(data.uptime_seconds||0);
	D.overrideBtns.querySelectorAll(".o-btn").forEach(b=>b.classList.toggle("active",b.dataset.intent===intent));

	waveChart.update(data.emg_signal_history||[]);

	const servo=data.servo_positions||{};
	updateHand(servo,currentGrip);updateServoBars(servo);

	// Mode display
	const mode=data.hand_mode||"NORMAL";
	const modeLabels={NORMAL:"Normal",KEYBOARD_TYPING:"Typing",KEYBOARD_GAMING:"Gaming",MOUSE:"Mouse",SPORT_BASKETBALL:"Basketball",SPORT_TENNIS:"Tennis",SPORT_BASEBALL:"Baseball",SPORT_CRICKET:"Cricket"};
	D.topbarMode.textContent=modeLabels[mode]||mode;
	D.modeGrid.querySelectorAll(".mode-card").forEach(c=>c.classList.toggle("active",c.dataset.mode===mode));

	// Watch screen
	D.wMode.textContent=modeLabels[mode]||mode;D.wState.textContent=state;D.wQuality.textContent=`${Math.round(q*100)}%`;
	D.wUptime.textContent=fmt(data.uptime_seconds||0);
}

// ══════════════════════════════════════════════════════════════
// CLOCK (watch screen)
// ══════════════════════════════════════════════════════════════
function tickClock(){
	const now=new Date();
	D.topbarClock.textContent=now.toLocaleTimeString("en-US",{hour:"2-digit",minute:"2-digit",hour12:false});
	D.watchTime.textContent=now.toLocaleTimeString("en-US",{hour:"2-digit",minute:"2-digit",hour12:false});
	D.watchSecs.textContent=":"+String(now.getSeconds()).padStart(2,"0");
	D.watchDate.textContent=now.toLocaleDateString("en-US",{weekday:"long",month:"long",day:"numeric"});
}
setInterval(tickClock,1000);tickClock();

// ══════════════════════════════════════════════════════════════
// EMG OVERRIDE
// ══════════════════════════════════════════════════════════════
D.overrideBtns.addEventListener("click",async e=>{
	const btn=e.target.closest(".o-btn");if(!btn||btn.disabled)return;
	try{await fetch("/api/emg-override",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({intent:btn.dataset.intent})})}catch(e){}
});

// ══════════════════════════════════════════════════════════════
// MODE SWITCHING
// ══════════════════════════════════════════════════════════════
D.modeGrid.addEventListener("click",async e=>{
	const card=e.target.closest(".mode-card");if(!card)return;
	try{const r=await fetch("/api/mode",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({mode:card.dataset.mode})});
		const d=await r.json();if(d.ok)D.modeGrid.querySelectorAll(".mode-card").forEach(c=>c.classList.toggle("active",c.dataset.mode===card.dataset.mode));
	}catch(e){}
});

// ══════════════════════════════════════════════════════════════
// TRAINING
// ══════════════════════════════════════════════════════════════
D.trainBtn.addEventListener("click",async()=>{
	const label=D.trainLabel.value.trim().toLowerCase().replace(/\s+/g,"_");
	if(!label){D.trainStatus.textContent="⚠️ Enter a label first";D.trainStatus.style.color="var(--amber)";return}
	D.trainStatus.textContent="📸 Capturing...";D.trainStatus.style.color="var(--cyan)";
	try{const r=await fetch("/api/training/capture",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({label})});
		const d=await r.json();
		if(d.ok){D.trainStatus.textContent=`✅ Saved "${label}" (${d.total_captures} total)`;D.trainStatus.style.color="var(--emerald)";D.trainLabel.value="";loadTrainingStats()}
		else{D.trainStatus.textContent=`❌ ${d.error}`;D.trainStatus.style.color="var(--red)"}
	}catch(e){D.trainStatus.textContent="❌ Request failed";D.trainStatus.style.color="var(--red)"}
});

async function loadTrainingStats(){
	try{const r=await fetch("/api/training/stats");const d=await r.json();
		if(d.total===0){D.trainStatBody.textContent="No captures yet.";return}
		D.trainStatBody.innerHTML=Object.entries(d.labels).sort((a,b)=>b[1]-a[1]).map(([k,v])=>
			`<div class="train-label-row"><span class="train-label-name">${k}</span><span class="train-label-count">${v} images</span></div>`
		).join("")+`<div style="margin-top:8px;font-size:11px;color:var(--cyan)">Total: ${d.total} captures</div>`;
	}catch(e){D.trainStatBody.textContent="Could not load stats."}
}

// ══════════════════════════════════════════════════════════════
// GITHUB
// ══════════════════════════════════════════════════════════════
async function loadGitHub(){
	try{const r=await fetch("/api/github/issues");const d=await r.json();
		if(d.ok&&d.issues.length){D.ghIssues.innerHTML=d.issues.map(i=>`<div class="gh-item"><div class="gh-item-title"><span class="gh-state ${i.state}">${i.state}</span><a href="${i.url}" target="_blank">#${i.number} ${i.title}</a></div><div class="gh-item-meta">${new Date(i.updated_at).toLocaleDateString()}</div></div>`).join("")}
		else{D.ghIssues.innerHTML='<div class="tl-empty">No issues found. Set your repo in handlers.py</div>'}
	}catch(e){D.ghIssues.innerHTML='<div class="tl-empty">Could not load issues</div>'}
	try{const r=await fetch("/api/github/releases");const d=await r.json();
		if(d.ok&&d.releases.length){D.ghReleases.innerHTML=d.releases.map(r=>`<div class="gh-item"><div class="gh-item-title"><a href="${r.url}" target="_blank">${r.tag} — ${r.name||"Release"}</a></div><div class="gh-item-meta">${new Date(r.published).toLocaleDateString()}</div></div>`).join("")}
		else{D.ghReleases.innerHTML='<div class="tl-empty">No releases found</div>'}
	}catch(e){D.ghReleases.innerHTML='<div class="tl-empty">Could not load releases</div>'}
}

// ══════════════════════════════════════════════════════════════
// GAMES
// ══════════════════════════════════════════════════════════════
// Reaction Time
let reactionTimer=null,reactionStart=0;
$("#game-reaction").addEventListener("click",()=>{$$("#screen-games .game-panel").forEach(p=>p.style.display="none");$("#game-panel-reaction").style.display="block"});
$("#game-grip-match").addEventListener("click",()=>{$$("#screen-games .game-panel").forEach(p=>p.style.display="none");$("#game-panel-grip").style.display="block"});

D.reactionStart.addEventListener("click",()=>{
	D.reactionCircle.className="reaction-circle waiting";D.reactionCircle.textContent="WAIT…";D.reactionResult.textContent="";
	D.reactionCircle.onclick=null;
	const delay=1500+Math.random()*4000;
	reactionTimer=setTimeout(()=>{
		D.reactionCircle.className="reaction-circle ready";D.reactionCircle.textContent="TAP!";
		reactionStart=performance.now();
		D.reactionCircle.onclick=()=>{
			const ms=Math.round(performance.now()-reactionStart);
			D.reactionCircle.className="reaction-circle result";D.reactionCircle.textContent=`${ms}ms`;
			D.reactionResult.textContent=ms<200?"⚡ Incredible!":ms<350?"🔥 Fast!":ms<500?"👍 Good":"Keep practicing!";
			D.reactionCircle.onclick=null;
		};
	},delay);
});

// Grip Match
const GRIPS=["PINCH","POWER","CYLINDRICAL","TRIPOD","HOOK"];
let gripMatchScore=0,gripMatchTarget="",gripMatchInterval=null;

D.gripStart.addEventListener("click",()=>{
	gripMatchScore=0;D.gripScore.textContent="0";nextGripTarget();
	if(gripMatchInterval)clearInterval(gripMatchInterval);
	gripMatchInterval=setInterval(()=>{
		if(currentGrip===gripMatchTarget){gripMatchScore++;D.gripScore.textContent=gripMatchScore;nextGripTarget()}
	},200);
});

function nextGripTarget(){
	gripMatchTarget=GRIPS[Math.floor(Math.random()*GRIPS.length)];
	D.gripTarget.textContent=`Match: ${gripMatchTarget}`;
	D.gripCurrent.textContent=`Your grip: ${currentGrip}`;
}

// Update grip current display
setInterval(()=>{if(D.gripCurrent)D.gripCurrent.textContent=`Your grip: ${currentGrip}`},300);

// ══════════════════════════════════════════════════════════════
// WEBSOCKET + POLLING
// ══════════════════════════════════════════════════════════════
let ws=null,reconTimer=null;
function setConn(on){D.connDot.classList.toggle("on",on)}

function connectWS(){
	const proto=location.protocol==="https:"?"wss":"ws";
	ws=new WebSocket(`${proto}://${location.host}/ws`);
	ws.onopen=()=>{setConn(true);if(reconTimer){clearTimeout(reconTimer);reconTimer=null}};
	ws.onmessage=e=>{try{updateUI(JSON.parse(e.data))}catch(e){}};
	ws.onclose=()=>{setConn(false);reconTimer=reconTimer||setTimeout(()=>{reconTimer=null;connectWS()},2000)};
	ws.onerror=()=>{setConn(false);ws.close()};
}

async function pollEvents(){
	try{const r=await fetch("/api/events");if(r.ok)updateTimeline(await r.json())}catch(e){}
	setTimeout(pollEvents,1500);
}

async function pollFallback(){
	try{const r=await fetch("/api/status");if(r.ok)updateUI(await r.json())}catch(e){}
	setTimeout(pollFallback,500);
}

connectWS();pollEvents();
setTimeout(()=>{if(!ws||ws.readyState!==WebSocket.OPEN)pollFallback()},5000);
