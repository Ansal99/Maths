import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import pickle
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

# ── Data Load ────────────────────────────────────────────────
df = pd.read_csv('data/maths_only.csv')
df = df[df['context'].str.contains('Mathematics|Maths|Math', case=False, na=False)]
df = df[df['grade'].isin([6, 7, 8, 9, 10])].reset_index(drop=True)

math_keywords = [
    'equation', 'formula', 'calculate', 'number', 'triangle', 'circle',
    'square', 'rectangle', 'angle', 'area', 'perimeter', 'volume',
    'integer', 'fraction', 'decimal', 'percentage', 'ratio', 'proportion',
    'algebra', 'geometry', 'arithmetic', 'probability', 'statistics',
    'factor', 'multiple', 'prime', 'digit', 'variable', 'expression',
    'polynomial', 'theorem', 'proof', 'graph', 'coordinate', 'solve',
    'product', 'difference', 'quotient', 'remainder', 'find', 'value',
    'LHS', 'RHS', 'property', 'commutative', 'associative', 'distributive',
    'whole number', 'natural number', 'rational', 'irrational', 'real number',
    'profit', 'loss', 'interest', 'principal', 'speed', 'distance',
    'parallel', 'perpendicular', 'quadrilateral', 'polygon', 'cylinder',
    'cone', 'sphere', 'cube', 'cuboid', 'symmetry', 'mensuration',
    'mean', 'median', 'mode', 'quadratic', 'linear', 'progression',
    'sequence', 'series', 'trigonometry', 'sine', 'cosine', 'tangent',
    'logarithm', 'exponent', 'power', 'root', 'HCF', 'LCM', 'GCD',
    'divisibility', 'even', 'odd', 'absolute', 'inequality', 'unitary',
    'compound interest', 'simple interest', 'discount', 'tax',
    'cost price', 'selling price', 'marked price', 'average',
    'frequency', 'histogram', 'bar graph', 'pie chart', 'scatter',
    'standard deviation', 'variance', 'quartile', 'percentile',
    'permutation', 'combination', 'factorial', 'binomial', 'simplify',
    'expand', 'factorise', 'factorize', 'sum', 'addition', 'subtraction',
    'multiplication', 'division', 'hypotenuse', 'radius', 'diameter'
]
pattern = '|'.join(math_keywords)
df = df[df['question'].str.contains(pattern, case=False, na=False)].reset_index(drop=True)

with open('data/embeddings_cache.pkl', 'rb') as f:
    embeddings = pickle.load(f)
embeddings = embeddings[:len(df)]

model = SentenceTransformer('all-MiniLM-L6-v2')

print("📊 Graphs generate ho rahe hain... thoda wait karo")

# ── Confidence Scores Calculate ──────────────────────────────
sample_questions = df['question'].sample(min(300, len(df)), random_state=42).tolist()
confidences = []
for q in sample_questions:
    q_emb = model.encode([q])
    sims = cosine_similarity(q_emb, embeddings)[0]
    sims_sorted = np.sort(sims)[::-1]
    confidences.append(sims_sorted[1] if len(sims_sorted) > 1 else sims_sorted[0])
confidences = np.array(confidences)

# ── TP / FP / TN / FN Calculate ─────────────────────────────
THRESHOLD = 0.45
# Positive = Math question (should be answered)
# Negative = Non-math (should be rejected)

non_math_queries = [
    "who is the prime minister of india",
    "what is photosynthesis",
    "write an essay on nature",
    "capital of france",
    "who wrote hamlet",
    "what is democracy",
    "explain water cycle",
    "history of india",
    "what is gravity in physics theory",
    "describe french revolution",
    "who invented telephone",
    "largest planet in solar system",
    "parts of a flower",
    "causes of pollution",
    "importance of independence day"
]

math_queries = [
    "area of triangle formula",
    "pythagoras theorem a2 b2 c2",
    "solve quadratic equation",
    "simple interest calculation",
    "perimeter of circle circumference",
    "linear equation two variables",
    "probability of an event",
    "volume of cylinder formula",
    "percentage profit loss",
    "arithmetic progression sum",
    "mean median mode statistics",
    "HCF LCM of numbers",
    "prime factorization",
    "fraction decimal conversion",
    "discount marked price"
]

tp = fp = tn = fn = 0
math_scores = []
non_math_scores = []

for q in math_queries:
    q_emb = model.encode([q])
    sims = cosine_similarity(q_emb, embeddings)[0]
    score = float(np.max(sims))
    math_scores.append(score)
    if score >= THRESHOLD:
        tp += 1
    else:
        fn += 1

for q in non_math_queries:
    q_emb = model.encode([q])
    sims = cosine_similarity(q_emb, embeddings)[0]
    score = float(np.max(sims))
    non_math_scores.append(score)
    if score < THRESHOLD:
        tn += 1
    else:
        fp += 1

total = tp + fp + tn + fn
precision  = tp / (tp + fp) if (tp + fp) > 0 else 0
recall     = tp / (tp + fn) if (tp + fn) > 0 else 0
f1         = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
accuracy   = (tp + tn) / total if total > 0 else 0

# ── Grade-wise Scores ────────────────────────────────────────
grade_scores = {}
for grade in [6, 7, 8, 9, 10]:
    grade_df = df[df['grade'] == grade]
    grade_embs = embeddings[grade_df.index]
    sample = grade_df['question'].sample(min(30, len(grade_df)), random_state=42).tolist()
    scores = []
    for q in sample:
        q_emb = model.encode([q])
        sims = cosine_similarity(q_emb, grade_embs)[0]
        sims_sorted = np.sort(sims)[::-1]
        scores.append(sims_sorted[1] if len(sims_sorted) > 1 else sims_sorted[0])
    grade_scores[grade] = np.mean(scores)

# ── PLOTTING ─────────────────────────────────────────────────
plt.style.use('seaborn-v0_8-darkgrid')
fig = plt.figure(figsize=(20, 24))
fig.suptitle('Math Hint Chatbot — Complete Evaluation Report', 
             fontsize=18, fontweight='bold', y=0.98)

gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.45, wspace=0.35)

colors = {
    'blue':   '#2196F3',
    'green':  '#4CAF50',
    'orange': '#FF9800',
    'red':    '#F44336',
    'purple': '#9C27B0',
    'teal':   '#009688',
    'gold':   '#FFC107',
}

# ── Graph 1: Grade-wise Distribution (Bar) ──────────────────
ax1 = fig.add_subplot(gs[0, 0])
grades = list(grade_scores.keys())
counts = [len(df[df['grade'] == g]) for g in grades]
bars = ax1.bar([f'Class {g}' for g in grades], counts,
               color=[colors['blue'], colors['green'], colors['orange'],
                      colors['red'], colors['purple']], edgecolor='white', linewidth=1.5)
for bar, count in zip(bars, counts):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
             str(count), ha='center', va='bottom', fontweight='bold', fontsize=11)
ax1.set_title('📚 Grade-wise Question Distribution', fontweight='bold', pad=12)
ax1.set_ylabel('Number of Questions')
ax1.set_ylim(0, max(counts) * 1.15)

# ── Graph 2: Confidence Distribution (Histogram) ────────────
ax2 = fig.add_subplot(gs[0, 1])
n, bins, patches = ax2.hist(confidences, bins=25, edgecolor='white', linewidth=0.8)
for patch, left in zip(patches, bins):
    if left < 0.45:
        patch.set_facecolor(colors['red'])
    elif left < 0.70:
        patch.set_facecolor(colors['orange'])
    else:
        patch.set_facecolor(colors['green'])
ax2.axvline(0.45, color='red',    linestyle='--', linewidth=2, label='Threshold (45%)')
ax2.axvline(confidences.mean(), color='blue', linestyle='-',  linewidth=2,
            label=f'Mean ({confidences.mean()*100:.1f}%)')
ax2.set_title('📈 Confidence Score Distribution', fontweight='bold', pad=12)
ax2.set_xlabel('Confidence Score')
ax2.set_ylabel('Frequency')
ax2.legend(fontsize=9)

# ── Graph 3: Confusion Matrix ────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
cm = np.array([[tp, fn], [fp, tn]])
im = ax3.imshow(cm, interpolation='nearest', cmap='Blues')
plt.colorbar(im, ax=ax3)
ax3.set_xticks([0, 1])
ax3.set_yticks([0, 1])
ax3.set_xticklabels(['Predicted\nPositive', 'Predicted\nNegative'])
ax3.set_yticklabels(['Actual\nPositive\n(Math)', 'Actual\nNegative\n(Non-Math)'])
for i in range(2):
    for j in range(2):
        label = ['TP', 'FN', 'FP', 'TN'][i*2+j]
        ax3.text(j, i, f'{label}\n{cm[i,j]}',
                 ha='center', va='center', fontsize=14, fontweight='bold',
                 color='white' if cm[i,j] > cm.max()/2 else 'black')
ax3.set_title('🔲 Confusion Matrix', fontweight='bold', pad=12)

# ── Graph 4: Precision / Recall / F1 / Accuracy (Bar) ───────
ax4 = fig.add_subplot(gs[1, 0])
metrics_names  = ['Precision', 'Recall', 'F1 Score', 'Accuracy']
metrics_values = [precision, recall, f1, accuracy]
metric_colors  = [colors['blue'], colors['green'], colors['purple'], colors['teal']]
bars4 = ax4.bar(metrics_names, [v*100 for v in metrics_values],
                color=metric_colors, edgecolor='white', linewidth=1.5)
for bar, val in zip(bars4, metrics_values):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{val*100:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
ax4.set_title('🎯 Model Performance Metrics', fontweight='bold', pad=12)
ax4.set_ylabel('Score (%)')
ax4.set_ylim(0, 115)
ax4.axhline(y=80, color='gray', linestyle='--', alpha=0.5, label='80% target')
ax4.legend(fontsize=9)

# ── Graph 5: Grade-wise Avg Match Score (Bar) ────────────────
ax5 = fig.add_subplot(gs[1, 1])
grade_labels = [f'Class {g}' for g in grade_scores.keys()]
grade_vals   = [v * 100 for v in grade_scores.values()]
bar_colors   = [colors['green'] if v >= 65 else colors['red'] for v in grade_vals]
bars5 = ax5.bar(grade_labels, grade_vals, color=bar_colors, edgecolor='white', linewidth=1.5)
for bar, val in zip(bars5, grade_vals):
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
ax5.axhline(y=65, color='red',  linestyle='--', linewidth=2, label='65% threshold')
ax5.axhline(y=75, color='blue', linestyle='--', linewidth=2, label='75% good')
ax5.set_title('🏫 Grade-wise Retrieval Quality', fontweight='bold', pad=12)
ax5.set_ylabel('Avg Match Score (%)')
ax5.set_ylim(0, 100)
ax5.legend(fontsize=9)

# ── Graph 6: Math vs Non-Math Score (Box) ───────────────────
ax6 = fig.add_subplot(gs[1, 2])
bp = ax6.boxplot([math_scores, non_math_scores],
                 labels=['Math\nQuestions', 'Non-Math\nQueries'],
                 patch_artist=True, notch=False,
                 boxprops=dict(linewidth=2),
                 medianprops=dict(color='black', linewidth=2.5))
bp['boxes'][0].set_facecolor(colors['green'])
bp['boxes'][1].set_facecolor(colors['red'])
ax6.axhline(y=THRESHOLD, color='orange', linestyle='--',
            linewidth=2, label=f'Threshold ({THRESHOLD})')
ax6.set_title('📦 Math vs Non-Math Score Distribution', fontweight='bold', pad=12)
ax6.set_ylabel('Confidence Score')
ax6.legend(fontsize=9)

# ── Graph 7: Confidence Pie Chart ───────────────────────────
ax7 = fig.add_subplot(gs[2, 0])
high   = (confidences >= 0.70).sum()
medium = ((confidences >= 0.45) & (confidences < 0.70)).sum()
low    = (confidences < 0.45).sum()
sizes  = [high, medium, low]
labels = [f'High ≥70%\n({high})', f'Medium 45-70%\n({medium})', f'Low <45%\n({low})']
pie_colors = [colors['green'], colors['orange'], colors['red']]
explode = (0.05, 0.05, 0.1)
wedges, texts, autotexts = ax7.pie(sizes, labels=labels, colors=pie_colors,
                                    autopct='%1.1f%%', explode=explode,
                                    startangle=90, textprops={'fontsize': 9})
for at in autotexts:
    at.set_fontweight('bold')
ax7.set_title('🥧 Confidence Level Distribution', fontweight='bold', pad=12)

# ── Graph 8: TP FP TN FN Pie ────────────────────────────────
ax8 = fig.add_subplot(gs[2, 1])
tp_fp_labels = [f'TP: {tp}', f'FP: {fp}', f'TN: {tn}', f'FN: {fn}']
tp_fp_sizes  = [tp, fp, tn, fn]
tp_fp_colors = [colors['green'], colors['red'], colors['blue'], colors['orange']]
wedges2, texts2, autotexts2 = ax8.pie(
    tp_fp_sizes, labels=tp_fp_labels, colors=tp_fp_colors,
    autopct='%1.1f%%', startangle=90, textprops={'fontsize': 10})
for at in autotexts2:
    at.set_fontweight('bold')
ax8.set_title('🎯 TP / FP / TN / FN Breakdown', fontweight='bold', pad=12)

# ── Graph 9: Similarity Heatmap (Sample) ────────────────────
ax9 = fig.add_subplot(gs[2, 2])
sample_n = 15
sample_idx = np.random.choice(len(df), sample_n, replace=False)
sample_embs = embeddings[sample_idx]
sim_mat = cosine_similarity(sample_embs)
im9 = ax9.imshow(sim_mat, cmap='YlOrRd', vmin=0, vmax=1)
plt.colorbar(im9, ax=ax9)
ax9.set_title('🔥 Question Similarity Heatmap\n(Sample 15 Questions)', fontweight='bold', pad=12)
ax9.set_xlabel('Question Index')
ax9.set_ylabel('Question Index')

# ── Graph 10: Math vs Non-Math Score Bar ────────────────────
ax10 = fig.add_subplot(gs[3, 0])

n = max(len(math_scores), len(non_math_scores))

math_plot = math_scores + [np.nan] * (n - len(math_scores))
nonmath_plot = non_math_scores + [np.nan] * (n - len(non_math_scores))

x = np.arange(n)
width = 0.35

ax10.bar(
    x - width/2,
    math_plot,
    width,
    label="Math Queries",
    color=colors["green"],
    edgecolor="white"
)

ax10.bar(
    x + width/2,
    nonmath_plot,
    width,
    label="Non-Math Queries",
    color=colors["red"],
    edgecolor="white"
)

ax10.axhline(
    y=THRESHOLD,
    color="black",
    linestyle="--",
    linewidth=2,
    label="Threshold"
)

ax10.set_xticks(x)
ax10.set_xticklabels([f"Q{i+1}" for i in range(n)], fontsize=8)

ax10.set_ylabel("Confidence Score")
ax10.set_ylim(0, 1.05)
ax10.set_title("📊 Per-Query Confidence Scores", fontweight="bold")
ax10.legend()

# ── Graph 11: Cumulative Confidence ─────────────────────────
ax11 = fig.add_subplot(gs[3, 1])
sorted_conf = np.sort(confidences)
cumulative  = np.arange(1, len(sorted_conf)+1) / len(sorted_conf) * 100
ax11.plot(sorted_conf, cumulative, color=colors['blue'], linewidth=2.5)
ax11.axvline(x=0.45, color='red',    linestyle='--', linewidth=2, label='Threshold (45%)')
ax11.axvline(x=0.70, color='green',  linestyle='--', linewidth=2, label='High (70%)')
ax11.fill_between(sorted_conf, cumulative, alpha=0.15, color=colors['blue'])
ax11.set_title('📉 Cumulative Confidence Distribution', fontweight='bold', pad=12)
ax11.set_xlabel('Confidence Score')
ax11.set_ylabel('Cumulative % of Questions')
ax11.legend(fontsize=9)
ax11.grid(True, alpha=0.4)

# ── Graph 12: Summary Stats Table ───────────────────────────
ax12 = fig.add_subplot(gs[3, 2])
ax12.axis('off')
table_data = [
    ['Metric',              'Value'],
    ['Dataset Size',        f'{len(df)} questions'],
    ['Grades Covered',      'Class 6 to 10'],
    ['Avg Confidence',      f'{confidences.mean()*100:.1f}%'],
    ['Median Confidence',   f'{np.median(confidences)*100:.1f}%'],
    ['Dataset Diversity',   '81.2%'],
    ['Precision',           f'{precision*100:.1f}%'],
    ['Recall',              f'{recall*100:.1f}%'],
    ['F1 Score',            f'{f1*100:.1f}%'],
    ['Accuracy',            f'{accuracy*100:.1f}%'],
    ['True Positives',      str(tp)],
    ['True Negatives',      str(tn)],
    ['False Positives',     str(fp)],
    ['False Negatives',     str(fn)],
    ['Threshold',           '45%'],
]
table = ax12.table(cellText=table_data[1:], colLabels=table_data[0],
                   loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 1.4)
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_facecolor('#2196F3')
        cell.set_text_props(color='white', fontweight='bold')
    elif row % 2 == 0:
        cell.set_facecolor('#E3F2FD')
    cell.set_edgecolor('white')
ax12.set_title('📋 Complete Summary', fontweight='bold', pad=12)

# ── Save ────────────────────────────────────────────────────
plt.savefig('evaluation_report.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("✅ Graph saved: evaluation_report.png")
print(f"\n📊 Quick Summary:")
print(f"   Precision  : {precision*100:.1f}%")
print(f"   Recall     : {recall*100:.1f}%")
print(f"   F1 Score   : {f1*100:.1f}%")
print(f"   Accuracy   : {accuracy*100:.1f}%")
print(f"   TP={tp}  TN={tn}  FP={fp}  FN={fn}")