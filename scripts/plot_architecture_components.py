"""Compact source-grounded component diagram for the reproduction report."""
import argparse,pathlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib import font_manager

p=argparse.ArgumentParser()
p.add_argument('--out-dir',default='/root/autodl-tmp/manner-repro/reports/figures')
p.add_argument('--chinese-font')
a=p.parse_args();zh=bool(a.chinese_font)
if zh:
    font_manager.fontManager.addfont(a.chinese_font)
    plt.rcParams['font.family']=font_manager.FontProperties(fname=a.chinese_font).get_name()
R=pathlib.Path(a.out_dir)
R.mkdir(parents=True,exist_ok=True)
fig,ax=plt.subplots(figsize=(14,7.4),layout='constrained')
fig.patch.set_facecolor('#f8fafc');ax.set_facecolor('#f8fafc')
ax.set(xlim=(0,14),ylim=(0,7.4));ax.axis('off')

def box(x,y,w,h,title,detail,color):
    names={'7 camera images':'七路环视图像','BEV encoder':'鸟瞰特征编码器','Flow adapter':'流匹配适配器','AR decoder + LoRA':'自回归解码器','Lanes + connections':'中心线与连接关系','Pretrained geometry weights':'预训练几何权重','6 camera images':'六路环视图像','Image-to-BEV':'图像到鸟瞰特征','Prior fusion':'先验融合','Map decoder':'地图解码器','Road elements':'道路要素','HD + SD vectors':'高精图与标清图矢量','Satellite + raster SD':'卫星图与标清栅格'}
    if zh:
        english=title;title=names[title]
        detail='（'+english+'）'
        if english=='Flow adapter':detail='（Flow adapter / 6 steps）'
        if english=='Lanes + connections':detail='端点距离 0.5 m 构图'
        if english=='Road elements':detail='分隔线 / 人行横道 / 道路边界'
        if english=='Pretrained geometry weights':detail=''
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.04,rounding_size=0.12',facecolor=color,edgecolor='#cbd5e1',lw=.9))
    ax.text(x+w/2,y+h*.66,title,ha='center',va='center',fontsize=10.3,weight='bold',color='#0f172a')
    ax.text(x+w/2,y+h*.28,detail,ha='center',va='center',fontsize=9,color='#334155')

def arrow(a,b):
    ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='-|>',color='#64748b',lw=1.5,mutation_scale=13,shrinkA=4,shrinkB=4))

ax.text(.2,7.1,'TopoGPT：图像条件下的车道生成' if zh else 'TopoGPT / image-conditioned lane generation',fontsize=14,weight='bold',color='#0f172a')
box(.25,4.85,2.0,1.05,'7 camera images','OpenLane-V2 A','#dbeafe')
box(2.85,4.85,2.0,1.05,'BEV encoder','image features','#dbeafe')
box(5.45,4.85,2.0,1.05,'Flow adapter','6 inference steps','#ede9fe')
box(8.05,4.85,2.2,1.05,'AR decoder + LoRA','centerline tokens','#ffedd5')
box(10.85,4.85,2.8,1.05,'Lanes + connections','0.5 m endpoint rule','#dcfce7')
for x in [2.25,4.85,7.45,10.25]:arrow((x,5.375),(x+.6,5.375))
box(7.65,6.25,3,0.6,'Pretrained geometry weights','', '#ede9fe')
arrow((9.15,6.25),(9.15,5.90))
ax.text(.25,4.39,'OpenLane-V2 A：输出有向中心线图；分别评价几何、拓扑与图结构。' if zh else 'Output: directed centerline graph   |   Evaluation: geometry, topology and graph metrics',fontsize=10,color='#475569')

ax.plot([.2,13.8],[3.99,3.99],color='#cbd5e1',lw=1)
ax.text(.2,3.56,'UMPE：MapTRv2 融合四类对齐先验' if zh else 'UMPE / MapTRv2 with four aligned priors',fontsize=14,weight='bold',color='#0f172a')
box(.25,1.05,2.0,1.05,'6 camera images','nuScenes','#dbeafe')
box(2.85,1.05,2.0,1.05,'Image-to-BEV','MapTRv2 / LSS','#dbeafe')
box(5.45,1.05,2.3,1.05,'Prior fusion','gated / residual','#ede9fe')
box(8.35,1.05,2.0,1.05,'Map decoder','vector prediction','#ffedd5')
box(10.95,1.05,2.7,1.05,'Road elements','divider / crossing / edge','#dcfce7')
for a,b in [((2.25,1.575),(2.85,1.575)),((4.85,1.575),(5.45,1.575)),((7.75,1.575),(8.35,1.575)),((10.35,1.575),(10.95,1.575))]:arrow(a,b)
box(3.25,2.55,3.0,.65,'HD + SD vectors','AlignNet / graph encoder','#ede9fe')
box(6.95,2.55,3.15,.65,'Satellite + raster SD','FiLM raster encoder','#ede9fe')
arrow((5.3,2.55),(6.0,2.1));arrow((7.8,2.55),(7.1,2.1))
ax.text(.25,.6,'nuScenes：输出三类道路要素；评价 Chamfer AP，距离阈值为 0.5 / 1.0 / 1.5 m。' if zh else 'Output: 3 element classes   |   Evaluation: Chamfer AP @ 0.5 / 1.0 / 1.5 m',fontsize=10,color='#475569')
ax.text(.25,.18,'本次运行使用公开权重与完整验证集；验证单帧预测，未开展长时序 SLAM 评测。' if zh else 'This run: published checkpoints, complete validation sets; single-frame inference, no long-horizon SLAM evaluation.',fontsize=9.5,color='#64748b')
for ext in ['png','svg']:fig.savefig(R/('architecture_components.'+ext),dpi=170,facecolor=fig.get_facecolor())
print(R/'architecture_components.png')
