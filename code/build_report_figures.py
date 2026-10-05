"""Rebuild report charts from supplied saved aggregates and 36 case points.

This utility never fits a forecasting model or changes scientific metrics.
"""
from pathlib import Path
import argparse, json, hashlib

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--package',required=True)
    ap.add_argument('--output-dir',required=True)
    ap.add_argument('--temp-dir',required=True)
    ap.add_argument('--project-root')
    args=ap.parse_args()
    from v2_runtime import configure
    configure(args.temp_dir,args.project_root)
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(args.package).resolve();out=Path(args.output_dir).resolve()
    assert not out.exists(),'Choose a new output directory; never overwrite supplied figures.'
    out.mkdir(parents=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.prop_cycle':plt.cycler(color=['#8D9893','#537D84','#AC972D','#AFC6B5','#21A038'])})
    metrics=pd.read_csv(root/'results/METRICS.csv')
    methods=['Prophet','ProphetYearly','SharedSeasonal','ClippedEqualBlend','OnlineHorizonBlend']
    labels={'Prophet':'Prophet без сезонности','ProphetYearly':'Prophet с годовой сезонностью',
            'SharedSeasonal':'Общий календарь','ClippedEqualBlend':'Равная смесь','OnlineHorizonBlend':'Адаптивный ансамбль'}
    fig,ax=plt.subplots(figsize=(10,4.6))
    for j,m in enumerate(methods):
        part=metrics[(metrics.period=='JulDec')&(metrics.method==m)].sort_values('horizon')
        ax.bar([k+(j-2)*.15 for k in range(3)],part.native_macro_MAE,width=.145,label=labels[m])
    ax.set_xticks(range(3),['1 месяц','3 месяца','6 месяцев']);ax.set_ylabel('MAE, единицы value')
    ax.legend(fontsize=9,ncol=2);ax.grid(axis='y',alpha=.2);fig.tight_layout()
    fig.savefig(out/'forecast_errors.png',dpi=160);plt.close(fig)
    weight=pd.read_csv(root/'results/WEIGHTS_BY_ORIGIN.csv')
    fig,axes=plt.subplots(3,1,figsize=(10,6.8),sharex=True)
    for ax,h in zip(axes,[1,3,6]):
        part=weight[weight.horizon==h].sort_values('prefix_n')
        ax.stackplot(part.prefix_n,*[part[m] for m in ['LastValue','Prophet','SharedSeasonal']],
                     labels=['Последнее значение','Prophet','Общий календарь'],alpha=.85)
        ax.set_ylim(0,1);ax.set_ylabel(f'{h} мес.');ax.grid(alpha=.2)
    axes[0].legend(ncol=3,fontsize=9);axes[-1].set_xlabel('Число уже наблюдённых месяцев')
    fig.tight_layout();fig.savefig(out/'weights.png',dpi=160);plt.close(fig)
    casefile=root/'examples/CASE_POINTS.csv';names=json.loads((root/'examples/MUNICIPAL_NAMES.json').read_text(encoding='utf-8'))
    cases=pd.read_csv(casefile)
    assert len(cases)==36 and set(cases.territory_id)=={683,2145,1879}
    for tid,g in cases.groupby('territory_id',sort=False):
        g=g.sort_values('month');name=names['municipalities'][str(tid)]
        fig,ax=plt.subplots(figsize=(10,3.7))
        ax.plot(g.month,g.actual,label='Наблюдаемое значение',color='#20302b',marker='o',ms=3)
        for col,label in [('Prophet','Prophet'),('ClippedEqualBlend','Равная смесь'),('OnlineHorizonBlend','Адаптивный ансамбль')]:
            ax.plot(g.month,g[col],label=label)
        alarm=g[g.alarm==1]
        if len(alarm):ax.scatter(alarm.month,alarm.actual,marker='x',s=65,color='#ba432b',label='Сигнал после наблюдения')
        ax.set_title(name['name']+', '+name['region']);ax.set_xticks(range(1,13));ax.set_xlabel('Месяц 2024')
        ax.set_ylabel('value');ax.legend(fontsize=9,ncol=2);ax.grid(alpha=.2)
        fig.tight_layout();fig.savefig(out/f'case_{tid}.png',dpi=160);plt.close(fig)
    digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    receipt=dict(status='SAVED_INPUT_CHARTS_RENDERED',model_fits=0,new_metrics=0,
        inputs={str(p.relative_to(root)):digest(p) for p in [root/'results/METRICS.csv',root/'results/WEIGHTS_BY_ORIGIN.csv',casefile,root/'examples/MUNICIPAL_NAMES.json']},
        figures={p.name:digest(p) for p in out.glob('*.png')})
    (out/'PLOT_RECEIPT.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(status=receipt['status'],figures=len(receipt['figures']),model_fits=0)))
if __name__=='__main__':main()
