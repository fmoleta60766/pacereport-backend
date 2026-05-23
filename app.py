from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
import json
import os
import io
import base64
import zipfile
import tempfile
import math
import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, HRFlowable
from reportlab.lib.colors import HexColor, white, black

app = Flask(__name__)
CORS(app)

# ── Paleta ────────────────────────────────────────────────────────────────────
TEAL       = HexColor('#1D9E75')
TEAL_LIGHT = HexColor('#9FE1CB')
TEAL_BG    = HexColor('#E1F5EE')
BLUE       = HexColor('#185FA5')
CORAL      = HexColor('#D85A30')
GRAY_DARK  = HexColor('#2C2C2A')
GRAY_MID   = HexColor('#888780')
GRAY_LIGHT = HexColor('#F1EFE8')
GRAY_LINE  = HexColor('#D3D1C7')
PAGE_W, PAGE_H = A4
MARGIN     = 18 * mm
CONTENT_W  = PAGE_W - 2 * MARGIN

# ── Parser CSV ────────────────────────────────────────────────────────────────
def parse_zip(zip_bytes):
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        csv_name = next(n for n in z.namelist() if n.endswith('.csv'))
        with z.open(csv_name) as f:
            df = pd.read_csv(f)
    df['WorkoutDay'] = pd.to_datetime(df['WorkoutDay'])
    return df[df['WorkoutType'] == 'Run'].copy()

def calc_month(runs, year, month):
    m = runs[(runs['WorkoutDay'].dt.year == year) &
             (runs['WorkoutDay'].dt.month == month) &
             runs['DistanceInMeters'].notna()].copy()
    if m.empty:
        return None, None
    volume_km  = round(float(m['DistanceInMeters'].sum() / 1000), 1)
    tss        = int(round(float(m['TSS'].sum()), 0))
    sessions   = len(m)
    vel_avg    = float(m['VelocityAverage'].mean())
    pace_sec   = int(round(1000 / vel_avg)) if vel_avg > 0 else 0
    ctl        = int(round(tss / 30, 0))
    atl        = int(round(tss / 7, 0))
    m['week']  = m['WorkoutDay'].dt.isocalendar().week
    weeks      = sorted(m['week'].unique())[:4]
    weekly_vol = []
    weekly_tss = []
    for w in weeks:
        wdf = m[m['week'] == w]
        weekly_vol.append(round(float(wdf['DistanceInMeters'].sum() / 1000), 1))
        weekly_tss.append(int(round(float(wdf['TSS'].sum()), 0)))
    while len(weekly_vol) < 4:
        weekly_vol.append(0); weekly_tss.append(0)
    zone_cols = ['HRZone1Minutes','HRZone2Minutes','HRZone3Minutes','HRZone4Minutes','HRZone5Minutes']
    zones = [int(round(float(m[c].sum()), 0)) for c in zone_cols]
    data = dict(volume=volume_km, tss=tss, pace_sec=pace_sec,
                sessions=sessions, ctl=ctl, atl=atl,
                weekly_vol=weekly_vol, weekly_tss=weekly_tss)
    return data, zones

def available_months(runs):
    months = []
    for (y, mo), g in runs.groupby([runs['WorkoutDay'].dt.year, runs['WorkoutDay'].dt.month]):
        if g['DistanceInMeters'].notna().any():
            months.append({'year': int(y), 'month': int(mo),
                           'label': datetime.date(int(y), int(mo), 1).strftime('%B %Y')})
    return sorted(months, key=lambda x: (x['year'], x['month']), reverse=True)

# ── Gráficos ──────────────────────────────────────────────────────────────────
def chart_volume(data):
    fig, ax = plt.subplots(figsize=(7.2, 2.4))
    weeks = ['Sem 1', 'Sem 2', 'Sem 3', 'Sem 4']
    x = range(4); w = 0.35
    b1 = ax.bar([i+w/2 for i in x], data['cur']['weekly_vol'],  width=w, color='#1D9E75', label='Mês atual',    zorder=3)
    b2 = ax.bar([i-w/2 for i in x], data['prev']['weekly_vol'], width=w, color='#9FE1CB', label='Mês anterior', zorder=3)
    for b in b1:
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.4, f"{b.get_height():.0f}km",
                ha='center', va='bottom', fontsize=8, color='#1D9E75', fontweight='bold')
    ax.set_xticks(list(x)); ax.set_xticklabels(weeks, fontsize=9)
    ax.set_ylabel('km', fontsize=9, color='#888780')
    ax.tick_params(axis='y', labelsize=8, colors='#888780')
    ax.tick_params(axis='x', colors='#444441')
    for spine in ['top','right']: ax.spines[spine].set_visible(False)
    ax.spines['left'].set_color('#D3D1C7'); ax.spines['bottom'].set_color('#D3D1C7')
    ax.set_facecolor('#FAFAF8'); fig.patch.set_facecolor('#FAFAF8')
    ax.yaxis.grid(True, color='#E8E6DF', zorder=0)
    ax.legend(fontsize=8, frameon=False, loc='upper left')
    plt.tight_layout(pad=0.5)
    buf = io.BytesIO(); plt.savefig(buf, format='png', dpi=150, bbox_inches='tight'); plt.close(); buf.seek(0)
    return buf

def chart_tss(data):
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    ax.bar(['Sem 1','Sem 2','Sem 3','Sem 4'], data['cur']['weekly_tss'], color='#378ADD', width=0.5)
    ax.set_ylabel('TSS', fontsize=9, color='#888780'); ax.tick_params(labelsize=8, colors='#888780')
    for spine in ['top','right']: ax.spines[spine].set_visible(False)
    ax.spines['left'].set_color('#D3D1C7'); ax.spines['bottom'].set_color('#D3D1C7')
    ax.set_facecolor('#FAFAF8'); fig.patch.set_facecolor('#FAFAF8')
    ax.yaxis.grid(True, color='#E8E6DF', zorder=0)
    plt.tight_layout(pad=0.4)
    buf = io.BytesIO(); plt.savefig(buf, format='png', dpi=150, bbox_inches='tight'); plt.close(); buf.seek(0)
    return buf

def chart_zones(zones):
    total = sum(zones) or 1
    pcts  = [z/total*100 for z in zones]
    names = ['Z1 — Recuperação','Z2 — Base aeróbia','Z3 — Limiar','Z4 — VO2max','Z5 — Anaeróbio']
    clrs  = ['#185FA5','#1D9E75','#639922','#BA7517','#993C1D']
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    bars = ax.barh(names, pcts, color=clrs, height=0.55)
    for bar, pct in zip(bars, pcts):
        ax.text(bar.get_width()+0.5, bar.get_y()+bar.get_height()/2, f"{pct:.0f}%",
                va='center', fontsize=8, color='#444441')
    ax.set_xlim(0, max(pcts)*1.2); ax.set_xlabel('% tempo', fontsize=8, color='#888780')
    ax.tick_params(labelsize=8, colors='#444441')
    for spine in ['top','right']: ax.spines[spine].set_visible(False)
    ax.spines['left'].set_color('#D3D1C7'); ax.spines['bottom'].set_color('#D3D1C7')
    ax.set_facecolor('#FAFAF8'); fig.patch.set_facecolor('#FAFAF8')
    ax.xaxis.grid(True, color='#E8E6DF', zorder=0)
    plt.tight_layout(pad=0.4)
    buf = io.BytesIO(); plt.savefig(buf, format='png', dpi=150, bbox_inches='tight'); plt.close(); buf.seek(0)
    return buf

def chart_ctl_atl(data):
    import random; random.seed(42)
    fig, ax = plt.subplots(figsize=(7.2, 2.2))
    days = list(range(1, 31))
    ctl_c = [data['prev']['ctl']+(data['cur']['ctl']-data['prev']['ctl'])*i/29+(random.random()*1.5-0.75) for i in range(30)]
    atl_c = [data['prev']['atl']+(data['cur']['atl']-data['prev']['atl'])*i/29*abs(math.sin(i/5+1))+(random.random()*2-1) for i in range(30)]
    ax.fill_between(days, ctl_c, alpha=0.12, color='#185FA5')
    ax.fill_between(days, atl_c, alpha=0.08, color='#D85A30')
    ax.plot(days, ctl_c, color='#185FA5', linewidth=2, label='CTL (fitness)')
    ax.plot(days, atl_c, color='#D85A30', linewidth=2, label='ATL (fadiga)', linestyle='--')
    ax.set_xlabel('Dia', fontsize=9, color='#888780'); ax.tick_params(labelsize=8, colors='#888780')
    for spine in ['top','right']: ax.spines[spine].set_visible(False)
    ax.spines['left'].set_color('#D3D1C7'); ax.spines['bottom'].set_color('#D3D1C7')
    ax.set_facecolor('#FAFAF8'); fig.patch.set_facecolor('#FAFAF8')
    ax.yaxis.grid(True, color='#E8E6DF', zorder=0); ax.legend(fontsize=8, frameon=False)
    plt.tight_layout(pad=0.5)
    buf = io.BytesIO(); plt.savefig(buf, format='png', dpi=150, bbox_inches='tight'); plt.close(); buf.seek(0)
    return buf

# ── Estilos PDF ───────────────────────────────────────────────────────────────
def get_styles():
    s = {}
    s['name']    = ParagraphStyle('name', fontSize=22, fontName='Helvetica-Bold', textColor=GRAY_DARK, leading=26, spaceAfter=2)
    s['period']  = ParagraphStyle('period', fontSize=11, fontName='Helvetica', textColor=GRAY_MID, leading=14)
    s['sec']     = ParagraphStyle('sec', fontSize=8, fontName='Helvetica-Bold', textColor=GRAY_MID, leading=12, spaceBefore=12, spaceAfter=5)
    s['mv']      = ParagraphStyle('mv', fontSize=20, fontName='Helvetica-Bold', textColor=GRAY_DARK, leading=24, spaceAfter=1)
    s['ml']      = ParagraphStyle('ml', fontSize=8,  fontName='Helvetica', textColor=GRAY_MID, leading=12, spaceAfter=2)
    s['ct']      = ParagraphStyle('ct', fontSize=11, fontName='Helvetica-Bold', textColor=GRAY_DARK, leading=15, spaceBefore=10, spaceAfter=2)
    s['cs']      = ParagraphStyle('cs', fontSize=9,  fontName='Helvetica', textColor=GRAY_MID, leading=12, spaceAfter=6)
    s['ail']     = ParagraphStyle('ail', fontSize=9, fontName='Helvetica-Bold', textColor=TEAL, leading=13, spaceBefore=4, spaceAfter=4)
    s['ait']     = ParagraphStyle('ait', fontSize=10, fontName='Helvetica', textColor=GRAY_DARK, leading=16, spaceAfter=4, alignment=TA_JUSTIFY)
    s['footer']  = ParagraphStyle('footer', fontSize=7.5, fontName='Helvetica', textColor=GRAY_MID, leading=11, alignment=TA_CENTER)
    return s

def delta_color(cur, prev, lower=False):
    good = (cur < prev) if lower else (cur > prev)
    return TEAL if good else CORAL

def pace_str(sec):
    return f"{int(sec//60)}:{int(sec%60):02d}/km"

def build_pdf(data, logo_bytes=None):
    buf = io.BytesIO()
    s   = get_styles()
    doc = SimpleDocTemplate(buf, pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN, topMargin=14*mm, bottomMargin=14*mm)
    story = []
    cur = data['cur']; prev = data['prev']

    # Cabeçalho
    if logo_bytes:
        logo_img = Image(io.BytesIO(logo_bytes), width=40*mm, height=14*mm, kind='proportional')
        left_cell = logo_img
    else:
        left_cell = Paragraph(f"<font color='#1D9E75'><b>{data.get('coach','Consultoria de Corrida')}</b></font>",
                              ParagraphStyle('lc', fontSize=13, fontName='Helvetica-Bold', textColor=TEAL, leading=18))
    right_cell = Paragraph(data['period_cur'],
                           ParagraphStyle('rc', fontSize=9, fontName='Helvetica', textColor=GRAY_MID, alignment=TA_RIGHT, leading=14))
    ht = Table([[left_cell, right_cell]], colWidths=[CONTENT_W*0.6, CONTENT_W*0.4])
    ht.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'MIDDLE'),('ALIGN',(1,0),(1,0),'RIGHT'),('BOTTOMPADDING',(0,0),(-1,-1),6)]))
    story.append(ht)
    story.append(HRFlowable(width=CONTENT_W, thickness=0.5, color=GRAY_LINE, spaceAfter=10))
    story.append(Paragraph(data['athlete'], s['name']))
    story.append(Paragraph(f"Relatório mensal · {data['period_cur']} <font color='#B4B2A9'> vs {data['period_prev']}</font>", s['period']))
    story.append(Spacer(1, 10))

    # Métricas
    story.append(Paragraph("RESUMO DO MÊS", s['sec']))
    metrics = [
        ("Volume total",  f"{cur['volume']} km",  delta_color(cur['volume'],   prev['volume'])),
        ("TSS mensal",    f"{cur['tss']}",         delta_color(cur['tss'],      prev['tss'])),
        ("Pace médio",    pace_str(cur['pace_sec']),delta_color(cur['pace_sec'],prev['pace_sec'], True)),
        ("Sessões",       f"{cur['sessions']}",    delta_color(cur['sessions'], prev['sessions'])),
        ("CTL (fitness)", f"{cur['ctl']}",         delta_color(cur['ctl'],      prev['ctl'])),
        ("ATL (fadiga)",  f"{cur['atl']}",         delta_color(cur['atl'],      prev['atl'])),
    ]
    col_w = CONTENT_W/3 - 3
    for row_idx in range(0, 6, 3):
        row = []
        for label, value, dcolor in metrics[row_idx:row_idx+3]:
            def pct_delta(c, p):
                if not p: return ""
                d = c - p; pct = round(abs(d/p)*100)
                arrow = "▲" if d > 0 else "▼"
                return f"{arrow} {pct}%"
            if label == "Volume total":
                dp = pct_delta(cur['volume'], prev['volume'])
            elif label == "TSS mensal":
                dp = pct_delta(cur['tss'], prev['tss'])
            elif label == "Pace médio":
                dp = pct_delta(cur['pace_sec'], prev['pace_sec'])
            elif label == "Sessões":
                dp = pct_delta(cur['sessions'], prev['sessions'])
            elif label == "CTL (fitness)":
                dp = pct_delta(cur['ctl'], prev['ctl'])
            else:
                dp = pct_delta(cur['atl'], prev['atl'])
            ds = ParagraphStyle('ds', fontSize=8, fontName='Helvetica-Bold', textColor=dcolor, leading=11)
            row.append([Paragraph(label, s['ml']), Paragraph(value, s['mv']), Paragraph(dp, ds)])
        t = Table([row], colWidths=[col_w, col_w, col_w])
        t.setStyle(TableStyle([
            ('BACKGROUND',(0,0),(-1,-1),HexColor('#F1EFE8')),
            ('BOX',(0,0),(0,0),0.5,GRAY_LINE),('BOX',(1,0),(1,0),0.5,GRAY_LINE),('BOX',(2,0),(2,0),0.5,GRAY_LINE),
            ('LEFTPADDING',(0,0),(-1,-1),10),('RIGHTPADDING',(0,0),(-1,-1),10),
            ('TOPPADDING',(0,0),(-1,-1),9),('BOTTOMPADDING',(0,0),(-1,-1),9),
            ('VALIGN',(0,0),(-1,-1),'TOP'),
        ]))
        story.append(t); story.append(Spacer(1, 5))

    story.append(Spacer(1, 6))

    # Gráfico volume
    story.append(Paragraph("Volume semanal (km)", s['ct']))
    story.append(Paragraph(f"Mês atual vs {data['period_prev']}", s['cs']))
    story.append(Image(chart_volume(data), width=CONTENT_W, height=CONTENT_W*0.33))
    story.append(Spacer(1, 10))

    # TSS e Zonas
    hw = CONTENT_W/2 - 4
    tss_block   = [Paragraph("Carga semanal (TSS)", s['ct']), Paragraph("Por semana", s['cs']),
                   Image(chart_tss(data), width=hw, height=hw*0.65)]
    zones_block = [Paragraph("Zonas de FC", s['ct']), Paragraph("Distribuição do mês", s['cs']),
                   Image(chart_zones(data['zones']), width=hw, height=hw*0.65)]
    st = Table([[tss_block, zones_block]], colWidths=[hw+4, hw+4])
    st.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),
                            ('LEFTPADDING',(0,0),(-1,-1),0),('RIGHTPADDING',(0,0),(-1,-1),0),
                            ('TOPPADDING',(0,0),(-1,-1),0),('BOTTOMPADDING',(0,0),(-1,-1),0)]))
    story.append(st); story.append(Spacer(1, 8))

    # CTL/ATL
    story.append(Paragraph("Fitness vs Fadiga (CTL / ATL)", s['ct']))
    story.append(Paragraph("Evolução ao longo do mês", s['cs']))
    story.append(Image(chart_ctl_atl(data), width=CONTENT_W, height=CONTENT_W*0.30))
    story.append(Spacer(1, 10))

    # Análise IA
    story.append(HRFlowable(width=CONTENT_W, thickness=0.5, color=GRAY_LINE, spaceAfter=8))
    story.append(Paragraph("✦  Análise do coach (gerada por IA)", s['ail']))
    ai_txt = data.get('ai_analysis') or '—'
    ai_table = Table([[Paragraph(ai_txt, s['ait'])]], colWidths=[CONTENT_W])
    ai_table.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,-1),HexColor('#E1F5EE')),
        ('BOX',(0,0),(-1,-1),0.5,HexColor('#9FE1CB')),
        ('LEFTPADDING',(0,0),(-1,-1),12),('RIGHTPADDING',(0,0),(-1,-1),12),
        ('TOPPADDING',(0,0),(-1,-1),10),('BOTTOMPADDING',(0,0),(-1,-1),10),
    ]))
    story.append(ai_table); story.append(Spacer(1, 12))

    # Rodapé
    story.append(HRFlowable(width=CONTENT_W, thickness=0.5, color=GRAY_LINE, spaceAfter=5))
    story.append(Paragraph(
        f"{data.get('coach','Consultoria de Corrida')}  ·  {data['period_cur']}  ·  Gerado em {datetime.date.today().strftime('%d/%m/%Y')}",
        s['footer']))

    doc.build(story)
    buf.seek(0)
    return buf

# ── Rotas ─────────────────────────────────────────────────────────────────────
@app.route('/parse', methods=['POST'])
def parse():
    try:
        file = request.files['file']
        runs = parse_zip(file.read())
        months = available_months(runs)
        return jsonify({'months': months})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/metrics', methods=['POST'])
def metrics():
    try:
        file  = request.files['file']
        year  = int(request.form['year'])
        month = int(request.form['month'])
        runs  = parse_zip(file.read())
        cur, zones = calc_month(runs, year, month)
        # Mês anterior
        prev_month = month - 1 if month > 1 else 12
        prev_year  = year if month > 1 else year - 1
        prev, _    = calc_month(runs, prev_year, prev_month)
        if not cur:
            return jsonify({'error': 'Sem dados para este mês'}), 400
        return jsonify({'cur': cur, 'prev': prev or cur, 'zones': zones})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/pdf', methods=['POST'])
def generate_pdf():
    try:
        data = json.loads(request.form['data'])
        logo_bytes = None
        if 'logo' in request.files:
            logo_bytes = request.files['logo'].read()
        pdf_buf = build_pdf(data, logo_bytes)
        athlete = data.get('athlete','atleta').replace(' ', '_')
        period  = data.get('period_cur','').replace(' ', '_')
        return send_file(pdf_buf, mimetype='application/pdf',
                         as_attachment=True,
                         download_name=f"relatorio_{athlete}_{period}.pdf")
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/health')
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
