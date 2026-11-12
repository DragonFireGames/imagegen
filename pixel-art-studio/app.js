const $ = id => document.getElementById(id);
const canvas = $("pixelCanvas");
const ctx = canvas.getContext("2d", {willReadFrequently:true});
const overlay = $("overlayCanvas");
const octx = overlay.getContext("2d");
const wrap = $("canvasWrap");
let width = 32, height = 32, zoom = 8, tool = "pencil", color = "#25334a", brush = 1;
let pixels = new Uint8ClampedArray(width * height * 4);
let history = [], historyIndex = -1, drawing = false, startPoint = null, lastPoint = null, snapshot = null;
let palette = ["#25334a","#3f5b73","#638a80","#9dbb87","#e2d6a2","#f2a65a","#d76b55","#923f55","#52334e","#28263d","#f0e6d2","#ffffff"];
const presets = {
  gameboy:["#0f380f","#306230","#8bac0f","#9bbc0f"],
  nes:["#000000","#7c7c7c","#bcbcbc","#ffffff","#f83800","#d82800","#a80000","#7c0800","#0058f8","#0068a8","#00a800","#58d854","#f8b800","#f8d878"],
  pico:["#000000","#1d2b53","#7e2553","#008751","#ab5236","#5f574f","#c2c3c7","#fff1e8","#ff004d","#ffa300","#ffec27","#00e756","#29adff","#83769c","#ff77a8","#ffccaa"],
  mono:["#000000","#ffffff"]
};
function idx(x,y){return (y*width+x)*4}
function saveHistory(){
  history=history.slice(0,historyIndex+1);
  history.push(new Uint8ClampedArray(pixels));
  if(history.length>80)history.shift();
  historyIndex=history.length-1;
  updateHistory();
}
function updateHistory(){
  $("undoButton").disabled=historyIndex<=0;
  $("redoButton").disabled=historyIndex>=history.length-1;
  $("historyCount").textContent=history.length+" snapshots · "+(historyIndex+1);
}
function restore(i){if(i<0||i>=history.length)return;historyIndex=i;pixels=new Uint8ClampedArray(history[i]);render();updateHistory()}
function render(){
  const image=ctx.createImageData(width,height);image.data.set(pixels);ctx.putImageData(image,0,0);
  wrap.style.width=(width*zoom)+"px";wrap.style.height=(height*zoom)+"px";
  canvas.style.width=overlay.style.width=(width*zoom)+"px";
  canvas.style.height=overlay.style.height=(height*zoom)+"px";
  wrap.style.setProperty("--grid-w",width);wrap.style.setProperty("--grid-h",height);
  $("canvasWidth").value=width;$("canvasHeight").value=height;
  $("canvasInfo").textContent=width+" × "+height+" px";
  $("zoomLabel").textContent=Math.round(zoom*100)+"%";
  drawThumb();
}
function drawThumb(){
  const t=$("frameThumb");t.style.background="none";
  const c=document.createElement("canvas");c.width=width;c.height=height;c.getContext("2d").putImageData(ctx.getImageData(0,0,width,height),0,0);
  t.style.backgroundImage="url("+c.toDataURL()+")";t.style.backgroundSize="contain";t.style.backgroundRepeat="no-repeat";t.style.backgroundPosition="center";
}
function setColor(v){color=v;$("colorPicker").value=v;$("colorHex").textContent=v.toUpperCase();document.querySelectorAll(".swatch").forEach(s=>s.classList.toggle("selected",s.dataset.color.toLowerCase()===v.toLowerCase()))}
function renderPalette(){
  $("palette").replaceChildren();
  palette.forEach(c=>{const b=document.createElement("button");b.className="swatch";b.style.background=c;b.dataset.color=c;b.title=c;b.classList.toggle("selected",c.toLowerCase()===color.toLowerCase());b.addEventListener("click",()=>setColor(c));$("palette").append(b)});
}
function rgba(hex){const n=parseInt(hex.slice(1),16);return [(n>>16)&255,(n>>8)&255,n&255,255]}
function setPixel(x,y,c){if(x<0||y<0||x>=width||y>=height)return;pixels.set(c,idx(x,y))}
function point(e){const r=canvas.getBoundingClientRect();return {x:Math.floor((e.clientX-r.left)/zoom),y:Math.floor((e.clientY-r.top)/zoom)}}
function stamp(x,y,c){const rgb=typeof c==="string"?rgba(c):c;const r=Math.floor(brush/2);for(let yy=-r;yy<brush-r;yy++)for(let xx=-r;xx<brush-r;xx++)setPixel(x+xx,y+yy,rgb)}
function drawLine(a,b,c,paint=true){let x=a.x,y=a.y,dx=Math.abs(b.x-a.x),sx=a.x<b.x?1:-1,dy=-Math.abs(b.y-a.y),sy=a.y<b.y?1:-1,err=dx+dy;while(true){if(paint)stamp(x,y,c);else octx.fillRect(x,y,1,1);if(x===b.x&&y===b.y)break;const e=2*err;if(e>=dy){err+=dy;x+=sx}if(e<=dx){err+=dx;y+=sy}}}
function flood(x,y){
  const p=idx(x,y),target=Array.from(pixels.slice(p,p+4)),replacement=rgba(color);
  if(target.every((v,i)=>v===replacement[i]))return;
  const stack=[[x,y]],seen=new Uint8Array(width*height);
  while(stack.length){const [cx,cy]=stack.pop();if(cx<0||cy<0||cx>=width||cy>=height)continue;const n=cy*width+cx;if(seen[n])continue;seen[n]=1;const q=idx(cx,cy);if(pixels[q]!==target[0]||pixels[q+1]!==target[1]||pixels[q+2]!==target[2]||pixels[q+3]!==target[3])continue;pixels.set(replacement,q);stack.push([cx+1,cy],[cx-1,cy],[cx,cy+1],[cx,cy-1])}
}
function drawShape(a,b,mode,c){
  const minX=Math.min(a.x,b.x),maxX=Math.max(a.x,b.x),minY=Math.min(a.y,b.y),maxY=Math.max(a.y,b.y);
  if(mode==="line"){drawLine(a,b,c);return}
  if(mode==="rect"){for(let x=minX;x<=maxX;x++){stamp(x,minY,c);stamp(x,maxY,c)}for(let y=minY;y<=maxY;y++){stamp(minX,y,c);stamp(maxX,y,c)}return}
  const rx=(maxX-minX)/2,ry=(maxY-minY)/2,cx=(minX+maxX)/2,cy=(minY+maxY)/2;
  for(let y=minY;y<=maxY;y++)for(let x=minX;x<=maxX;x++){const d=((x-cx)*(x-cx))/(rx*rx||1)+((y-cy)*(y-cy))/(ry*ry||1);if(d<=1&&d>=.55)stamp(x,y,c)}
}
function paintPreview(a,b){
  octx.clearRect(0,0,width,height);octx.fillStyle=color;
  if(tool==="line")drawLine(a,b,color,false);else if(tool==="rect"||tool==="ellipse"){
    const minX=Math.min(a.x,b.x),maxX=Math.max(a.x,b.x),minY=Math.min(a.y,b.y),maxY=Math.max(a.y,b.y);
    if(tool==="rect"){octx.fillRect(minX,minY,maxX-minX+1,1);octx.fillRect(minX,maxY,maxX-minX+1,1);octx.fillRect(minX,minY,1,maxY-minY+1);octx.fillRect(maxX,minY,1,maxY-minY+1)}
    else {const rx=(maxX-minX)/2,ry=(maxY-minY)/2,cx=(minX+maxX)/2,cy=(minY+maxY)/2;for(let y=minY;y<=maxY;y++)for(let x=minX;x<=maxX;x++){const d=((x-cx)**2)/(rx*rx||1)+((y-cy)**2)/(ry*ry||1);if(d<=1&&d>=.55)octx.fillRect(x,y,1,1)}}
  }
}
function begin(e){
  if(e.button!==0)return;const p=point(e);if(p.x<0||p.y<0||p.x>=width||p.y>=height)return;
  canvas.setPointerCapture(e.pointerId);drawing=true;startPoint=p;lastPoint=p;snapshot=new Uint8ClampedArray(pixels);
  if(tool==="picker"){const i=idx(p.x,p.y);setColor("#"+[...pixels.slice(i,i+3)].map(v=>v.toString(16).padStart(2,"0")).join(""));drawing=false;return}
  if(tool==="fill"){flood(p.x,p.y);drawing=false;saveHistory();render();return}
  if(tool==="pencil"||tool==="eraser"){stamp(p.x,p.y,tool==="eraser"?[0,0,0,0]:color);render()}
}
function move(e){
  const p=point(e);$("cursorPosition").textContent="X: "+p.x+"  Y: "+p.y;
  if(!drawing)return;
  if(tool==="pencil"||tool==="eraser"){drawLine(lastPoint,p,tool==="eraser"?[0,0,0,0]:color);lastPoint=p;render()}
  else paintPreview(startPoint,p);
}
function end(e){
  if(!drawing)return;const p=point(e);
  if(tool==="line"||tool==="rect"||tool==="ellipse")drawShape(startPoint,p,tool,color);
  drawing=false;octx.clearRect(0,0,width,height);saveHistory();render();
}
canvas.addEventListener("pointerdown",begin);canvas.addEventListener("pointermove",move);canvas.addEventListener("pointerup",end);canvas.addEventListener("pointercancel",end);
document.querySelectorAll("[data-tool]").forEach(b=>b.addEventListener("click",()=>{tool=b.dataset.tool;document.querySelectorAll("[data-tool]").forEach(x=>x.classList.toggle("active",x===b))}));
$("brushSize").addEventListener("input",e=>{brush=+e.target.value;$("brushValue").textContent=brush+" px"});
$("colorPicker").addEventListener("input",e=>setColor(e.target.value));
$("addColor").addEventListener("click",()=>{if(!palette.includes(color)){palette.push(color);renderPalette()}});
$("removeColor").addEventListener("click",()=>{if(palette.length>1){palette=palette.filter(c=>c.toLowerCase()!==color.toLowerCase());renderPalette()}});
$("undoButton").addEventListener("click",()=>restore(historyIndex-1));$("redoButton").addEventListener("click",()=>restore(historyIndex+1));
function resize(w,h){w=Math.max(1,Math.min(256,Math.floor(w)));h=Math.max(1,Math.min(256,Math.floor(h)));const old=document.createElement("canvas");old.width=width;old.height=height;old.getContext("2d").putImageData(ctx.getImageData(0,0,width,height),0,0);width=w;height=h;pixels=new Uint8ClampedArray(w*h*4);const temp=document.createElement("canvas");temp.width=w;temp.height=h;temp.getContext("2d").drawImage(old,0,0,w,h);pixels.set(temp.getContext("2d").getImageData(0,0,w,h).data);canvas.width=overlay.width=w;canvas.height=overlay.height=h;history=[];historyIndex=-1;saveHistory();render()}
$("resizeButton").addEventListener("click",()=>resize(+$("canvasWidth").value||32,+$("canvasHeight").value||32));
function setZoom(z){zoom=Math.max(2,Math.min(24,z));render()}
$("zoomIn").addEventListener("click",()=>setZoom(zoom+2));$("zoomOut").addEventListener("click",()=>setZoom(zoom-2));$("centerCanvas").addEventListener("click",()=>{$("canvasStage").scrollTo({left:0,top:0,behavior:"smooth"})});
$("gridToggle").addEventListener("click",e=>{wrap.classList.toggle("grid-on");e.currentTarget.classList.toggle("active")});
function importImage(file){if(!file)return;const url=URL.createObjectURL(file),img=new Image();img.onload=()=>{const max=256,scale=Math.min(1,max/Math.max(img.width,img.height));const w=Math.max(1,Math.round(img.width*scale)),h=Math.max(1,Math.round(img.height*scale));width=w;height=h;canvas.width=overlay.width=w;canvas.height=overlay.height=h;pixels=new Uint8ClampedArray(w*h*4);const tmp=document.createElement("canvas");tmp.width=w;tmp.height=h;const tc=tmp.getContext("2d",{willReadFrequently:true});tc.clearRect(0,0,w,h);tc.drawImage(img,0,0,w,h);pixels.set(tc.getImageData(0,0,w,h).data);URL.revokeObjectURL(url);history=[];historyIndex=-1;saveHistory();render();$("projectName").value=file.name.replace(/\.[^.]+$/,"");$("processorStatus").textContent="Imported "+file.name+" · "+w+" × "+h+" pixels";};img.onerror=()=>{URL.revokeObjectURL(url);alert("Could not read this image.")};img.src=url}
$("importButton").addEventListener("click",()=>$("fileInput").click());$("fileInput").addEventListener("change",e=>importImage(e.target.files[0]));
$("exportButton").addEventListener("click",()=>{const out=document.createElement("canvas");out.width=width;out.height=height;out.getContext("2d").putImageData(ctx.getImageData(0,0,width,height),0,0);out.toBlob(blob=>{const a=document.createElement("a");a.href=URL.createObjectURL(blob);a.download=($("projectName").value||"sprite")+".png";a.click();URL.revokeObjectURL(a.href)},"image/png")});
$("newProject").addEventListener("click",()=>{if(!confirm("Start a new blank sprite?"))return;pixels=new Uint8ClampedArray(width*height*4);history=[];historyIndex=-1;saveHistory();render();$("projectName").value="Untitled sprite"});
function getSourceCanvas(){const c=document.createElement("canvas");c.width=width;c.height=height;c.getContext("2d").putImageData(ctx.getImageData(0,0,width,height),0,0);return c}
function quantize(data,count,preset=null,dither=false){
  const colors=preset||palette.slice(0,count);if(!colors.length)return data;const rgb=colors.map(rgba);
  const nearest=(r,g,b)=>{let best=rgb[0],dist=Infinity;for(const c of rgb){const d=(r-c[0])**2+(g-c[1])**2+(b-c[2])**2;if(d<dist){dist=d;best=c}}return best};
  const out=new Uint8ClampedArray(data);
  if(dither){const work=new Float32Array(data.length);for(let i=0;i<data.length;i++)work[i]=data[i];for(let y=0;y<height;y++)for(let x=0;x<width;x++){const i=idx(x,y);if(out[i+3]===0)continue;const old=[work[i],work[i+1],work[i+2]],n=nearest(...old);for(let k=0;k<3;k++){out[i+k]=n[k];const err=old[k]-n[k];for(const [dx,dy,f] of [[1,0,7/16],[-1,1,3/16],[0,1,5/16],[1,1,1/16]]){const xx=x+dx,yy=y+dy;if(xx>=0&&yy>=0&&xx<width&&yy<height)work[idx(xx,yy)+k]+=err*f}}}}
  else for(let i=0;i<out.length;i+=4){if(out[i+3]===0)continue;const c=nearest(out[i],out[i+1],out[i+2]);out[i]=c[0];out[i+1]=c[1];out[i+2]=c[2]}
  return out;
}
function applyPixels(next,msg){pixels=next;saveHistory();render();$("processorStatus").textContent=msg}
$("pixelScale").addEventListener("input",e=>$("pixelScaleValue").textContent=e.target.value);
$("colorCount").addEventListener("input",e=>$("colorCountValue").textContent=e.target.value);
$("pixelizeButton").addEventListener("click",()=>{const scale=+$("pixelScale").value;const small=document.createElement("canvas");small.width=Math.max(1,Math.round(width/scale));small.height=Math.max(1,Math.round(height/scale));const sc=small.getContext("2d");sc.imageSmoothingEnabled=false;sc.drawImage(getSourceCanvas(),0,0,small.width,small.height);const big=document.createElement("canvas");big.width=width;big.height=height;const bc=big.getContext("2d");bc.imageSmoothingEnabled=false;bc.drawImage(small,0,0,width,height);applyPixels(quantize(bc.getImageData(0,0,width,height).data,+$("colorCount").value,null,$("ditherToggle").checked),"Pixelated to "+scale+"px blocks and reduced to "+$("colorCount").value+" colors.")});
$("paletteButton").addEventListener("click",()=>applyPixels(quantize(pixels,+$("colorCount").value,null,$("ditherToggle").checked),"Reduced to "+$("colorCount").value+" palette colors."));
$("outlineButton").addEventListener("click",()=>{const src=new Uint8ClampedArray(pixels),out=new Uint8ClampedArray(pixels),edge=rgba("#171923");for(let y=0;y<height;y++)for(let x=0;x<width;x++){const i=idx(x,y);if(src[i+3])continue;let near=false;for(const [dx,dy] of [[1,0],[-1,0],[0,1],[0,-1]]){const xx=x+dx,yy=y+dy;if(xx>=0&&yy>=0&&xx<width&&yy<height&&src[idx(xx,yy)+3])near=true}if(near)out.set(edge,i)}applyPixels(out,"Added a 1px dark outline around opaque pixels.")});
document.querySelectorAll("[data-preset]").forEach(b=>b.addEventListener("click",()=>{const p=presets[b.dataset.preset];palette=p.slice();renderPalette();applyPixels(quantize(pixels,p.length,p,$("ditherToggle").checked),"Applied "+b.textContent+" palette.")}));
document.querySelectorAll("[data-tab]").forEach(b=>b.addEventListener("click",()=>{document.querySelectorAll("[data-tab]").forEach(x=>x.classList.toggle("active",x===b));$("tab-ai").hidden=b.dataset.tab!=="ai";$("tab-layers").hidden=b.dataset.tab!=="layers"}));
$("generateAI").addEventListener("click",()=>{const prompt=$("aiPrompt").value.trim();const size=$("aiSize").value.split(" ")[0];const url=new URL("../illustrious-experiment/index.html",location.href);url.searchParams.set("pixelArtPrompt",prompt);url.searchParams.set("pixelArtSize",size);window.open(url.href,"_blank","noopener")});
$("editAI").addEventListener("click",()=>window.open("../illustrious-experiment/infill.html","_blank","noopener"));
$("duplicateFrame").addEventListener("click",()=>{$("processorStatus").textContent="Frame duplication is queued for the animation timeline implementation."});
document.addEventListener("keydown",e=>{if(["INPUT","TEXTAREA","SELECT"].includes(document.activeElement.tagName))return;const keys={b:"pencil",e:"eraser",g:"fill",i:"picker",l:"line",u:"rect",o:"ellipse"};if(keys[e.key.toLowerCase()]){tool=keys[e.key.toLowerCase()];document.querySelectorAll("[data-tool]").forEach(x=>x.classList.toggle("active",x.dataset.tool===tool))}if((e.metaKey||e.ctrlKey)&&e.key.toLowerCase()==="z"){e.preventDefault();restore(historyIndex+(e.shiftKey?1:-1))}});
canvas.width=overlay.width=width;canvas.height=overlay.height=height;renderPalette();setColor(color);saveHistory();render();
