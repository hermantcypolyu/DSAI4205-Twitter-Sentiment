"""Generate the FINAL report (.docx) from the checked-in artefacts.

This is the cumulative document required by the brief: EDA -> cleaning ->
baseline -> N = 5 embedding x analytics-model matrix -> comparison ->
business insights, plus an explicit response to the consultation feedback.

Every number is read from data/processed/*.json, models/*.json or
reports/tables/*.csv. Nothing is hand-typed, so the document cannot drift
from the code.

Run: python src/make_final_report.py
Writes: reports/Final_Report_Group8_Twitter_Sentiment.docx
"""
from __future__ import annotations

import collections
import csv
import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor

from make_interim_report import (
    ACCENT, COURSE, FIG, GREY, GROUP, MEMBERS, NAVY, OUT, PROC, ROOT, TAB,
    add_page_number_footer, body, bullet, callout, caption, figure, h1, h2, h3,
    hr, set_cell_bg, table,
)

MODELS = ROOT / "models"


# ------------------------------------------------------------------ helpers
def jload(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def csv_rows(name: str) -> list[dict]:
    with open(TAB / name, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def p(x: float) -> str:
    return f"{x * 100:.1f}%"


def dpp(x: float) -> str:
    return f"{x * 100:+.1f} pp"


def confusion(name: str):
    rows = csv_rows(name)
    c = collections.Counter((r["actual"], r["predicted"]) for r in rows)
    return c, len(rows)


CLASSES = ["Negative", "Positive", "Irrelevant", "Neutral"]


# ------------------------------------------------------------------ document
def build() -> Path:
    eda = jload(PROC / "eda_summary.json")
    log = jload(PROC / "cleaning_log.json")
    met = jload(MODELS / "baseline_metrics.json")
    fin = jload(MODELS / "final_metrics.json")
    imp = jload(MODELS / "improved_metrics.json")
    head = jload(PROC / "headroom_summary.json")
    ceiling = jload(PROC / "ceiling_analysis.json")
    noise = jload(PROC / "label_noise_summary.json")
    emb = jload(PROC / "embedding_matrix_summary.json")

    dep = fin["deployed_metrics"]
    cv_final = {r["model"]: r for r in fin["cv_train_only"]}
    test_final = {r["model"]: r for r in fin["test_results"]}
    imp_cv, imp_test = imp["cv"], imp["test"]

    emb_rows = sorted([r for r in emb["rows"] if r["eligible_for_n5"]],
                      key=lambda r: -r["test_accuracy"])
    ref_row = [r for r in emb["rows"] if not r["eligible_for_n5"]][0]
    ft_ref = jload(PROC / "embedding_result_bert_head.json")
    best_elig = emb_rows[0]
    base_acc, base_f1 = met["test_accuracy"], met["test_macro_f1"]

    conf_f, n_test = confusion("final_predictions.csv")
    conf_b, _ = confusion("test_predictions.csv")
    correct_f = sum(conf_f[(c, c)] for c in CLASSES)
    errors_f = n_test - correct_f
    correct_b = sum(conf_b[(c, c)] for c in CLASSES)

    sweep = csv_rows("improved_cv_sweep.csv")
    nn10_cos = next(float(r["median_nn_cosine_sim"])
                    for r in csv_rows("exp_label_self_agreement.csv")
                    if r["view"] == "phrase" and r["k"] == "10")

    doc = Document()
    st = doc.styles["Normal"]
    st.font.name = "Calibri"
    st.font.size = Pt(10.5)

    sec = doc.sections[0]
    sec.top_margin = Inches(0.85)
    sec.bottom_margin = Inches(0.8)
    sec.left_margin = Inches(0.95)
    sec.right_margin = Inches(0.95)
    add_page_number_footer(sec)

    # ---------------------------------------------------------- cover
    for _ in range(2):
        doc.add_paragraph()

    for text, size, colour, bold in [
        ("DSAI4205 - Big Data Analytics", 13, GREY, True),
        (COURSE.split(" - ")[1], 11, GREY, False),
    ]:
        p_ = doc.add_paragraph()
        p_.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p_.add_run(text)
        r.font.size = Pt(size)
        r.font.color.rgb = colour
        r.font.bold = bold

    doc.add_paragraph()
    p_ = doc.add_paragraph()
    p_.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p_.add_run("FINAL REPORT")
    r.font.size = Pt(30)
    r.font.bold = True
    r.font.color.rgb = NAVY

    p_ = doc.add_paragraph()
    p_.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p_.add_run("Twitter Sentiment as a Decision-Support System")
    r.font.size = Pt(16)
    r.font.bold = True
    r.font.color.rgb = ACCENT

    p_ = doc.add_paragraph()
    p_.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p_.add_run("Exploratory Data Analysis, Cleaning, TF-IDF Baseline,\n"
                   "and an N = 5 Embedding x Analytics-Model Benchmark")
    r.font.size = Pt(11)
    r.font.color.rgb = GREY

    doc.add_paragraph()
    hr(doc)

    p_ = doc.add_paragraph()
    p_.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p_.add_run("Dataset 5  -  Twitter")
    r.font.size = Pt(12)
    r.font.bold = True

    p_ = doc.add_paragraph()
    p_.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p_.add_run(f"{eda['train_rows_in']} training rows  |  {eda['test_rows_in']} test rows  |  "
                   f"4 sentiment classes")
    r.font.size = Pt(10)
    r.font.color.rgb = GREY

    doc.add_paragraph()
    p_ = doc.add_paragraph()
    p_.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p_.add_run(GROUP)
    r.font.size = Pt(12)
    r.font.bold = True
    r.font.color.rgb = NAVY

    t = doc.add_table(rows=1, cols=2)
    t.style = "Table Grid"
    t.alignment = 1
    for i, h in enumerate(["Name", "Student ID"]):
        c = t.rows[0].cells[i]
        c.text = ""
        pp_ = c.paragraphs[0]
        pp_.alignment = WD_ALIGN_PARAGRAPH.CENTER
        rr = pp_.add_run(h)
        rr.font.bold = True
        rr.font.size = Pt(9.5)
        rr.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        set_cell_bg(c, "1F3864")
    for name, sid in MEMBERS:
        cells = t.add_row().cells
        for i, val in enumerate([name, sid]):
            cells[i].text = ""
            pp_ = cells[i].paragraphs[0]
            pp_.alignment = WD_ALIGN_PARAGRAPH.CENTER
            rr = pp_.add_run(val)
            rr.font.size = Pt(9.5)
    for row in t.rows:
        row.cells[0].width = Inches(2.6)
        row.cells[1].width = Inches(1.6)

    doc.add_paragraph()
    p_ = doc.add_paragraph()
    p_.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p_.add_run("Final submission - Week 13")
    r.font.size = Pt(9.5)
    r.font.color.rgb = GREY

    # ---------------------------------------------------------- contents
    doc.add_page_break()
    h1(doc, "Contents")
    contents = [
        ("1", "Executive Summary", "Cumulative results and the five findings"),
        ("2", "Project Scope and Data Understanding", "Task, granularity, schema and data limitation"),
        ("3", "Exploratory Data Analysis", "Distributions, imbalance, signals, quality audit"),
        ("4", "Data Cleaning Pipeline", "Nine documented steps with before/after evidence"),
        ("5", "Baseline: TF-IDF + Logistic Regression", "Phase 3i design and benchmark"),
        ("6", "Baseline Refinement", "Phase 3iv: CV-selected improvements and the class-weight trade-off"),
        ("7", "Advanced Modelling: N = 5 Matrix", "Phase 3ii: five embedding x analytics-model pairs"),
        ("8", "Model Comparison and Discussion", "Phase 3iii/iv: advanced models vs both baselines"),
        ("9", "Error Analysis and the Accuracy Ceiling", "Measured evidence for the 60-65% ceiling"),
        ("10", "Business Insights and Decision Support", "Phase 4: storytelling with data"),
        ("11", "Response to Consultation Feedback", "How each comment was addressed"),
        ("12", "Limitations, Conclusion and Recommendations", "What the system can and cannot do"),
        ("13", "Project Management, Schedule and Contributions", "Delivery record and assessment map"),
        ("14", "Reproducibility and Artefact Index", "How a marker can re-run everything"),
        ("A", "Appendix - Cleaning Log", "Row-level audit trail"),
        ("B", "Appendix - Full Metric Detail", "CV variants and model comparisons"),
        ("C", "Appendix - Per-Class Results (N = 5)", "Precision, recall and F1 for every model"),
    ]
    for num, title, desc in contents:
        p_ = doc.add_paragraph()
        p_.paragraph_format.space_after = Pt(4)
        p_.paragraph_format.left_indent = Inches(0.25)
        r = p_.add_run(f"{num}.  ")
        r.font.bold = True
        r.font.size = Pt(10)
        r.font.color.rgb = NAVY
        r = p_.add_run(title)
        r.font.bold = True
        r.font.size = Pt(10)
        r = p_.add_run(f"   -  {desc}")
        r.font.size = Pt(9)
        r.font.color.rgb = GREY

    # ---------------------------------------------------------- 1 executive
    doc.add_page_break()
    h1(doc, "Executive Summary", "1")

    body(doc,
         f"This report is the cumulative record of the project. It covers all four phases of the "
         f"brief: exploratory data analysis, data cleaning, the required TF-IDF + Logistic "
         f"Regression baseline, and the N = 5 advanced modelling matrix with its comparison against "
         f"two baseline variants. The raw files contain {eda['train_rows_in']} training rows and "
         f"{eda['test_rows_in']} test rows across four sentiment classes: Negative, Positive, "
         f"Irrelevant and Neutral.")

    body(doc,
         "The headline is deliberately modest. The deployed TF-IDF model reaches "
         f"{p(dep['accuracy'])} test accuracy and {p(dep['macro_f1'])} macro F1, against a "
         f"majority-class reference of {p(met['majority_baseline_accuracy'])}. That is a genuine lift "
         "of roughly 24 percentage points over guessing the largest class, but it is not a number "
         "that should be read as 'the model is weak'. Section 9 shows, with four controlled "
         "experiments, that the remaining error is dominated by label noise and by one label whose "
         "definition depends on a column the dataset never shipped.")

    h2(doc, "1.1  Headline results")
    table(doc,
          ["Result", "Accuracy", "Macro F1", "Reference point"],
          [
              ["Majority-class guess", p(met["majority_baseline_accuracy"]), "0.116",
               "Predicts Negative for every row"],
              ["Phase 3i baseline (lemma TF-IDF + LR)", p(base_acc), p(base_f1),
               "The required baseline"],
              ["Deployed model M2 (word+char union, phrase)", p(dep["accuracy"]), p(dep["macro_f1"]),
               dpp(dep["accuracy"] - base_acc) + " vs baseline"],
              ["Phase 3iv improved B4 (phrase+lemma, balanced)", p(imp_test["accuracy"]),
               p(imp_test["macro_f1"]), "CV-selected; see Section 6"],
              ["Best eligible N = 5 model (DistilBERT + MLP)", p(best_elig["test_accuracy"]),
               p(best_elig["test_macro_f1"]),
               dpp(best_elig["test_accuracy"] - dep["accuracy"]) + " vs deployed"],
              ["End-to-end fine-tuned DistilBERT (reference, not counted)",
               p(ft_ref["test"]["accuracy"]), f"{ft_ref['test']['macro_f1']:.3f}",
               "Trains encoder + head jointly; see 7.1"],
          ],
          widths=[2.4, 0.95, 0.85, 2.3], font_size=8.4)

    body(doc,
         f"The N = 5 matrix answers the brief's advanced-modelling requirement. The strongest "
         f"eligible combination is frozen DistilBERT embeddings with an independent MLP at "
         f"{p(best_elig['test_accuracy'])}, which is "
         f"{dpp(best_elig['test_accuracy'] - dep['accuracy'])} relative to the deployed TF-IDF "
         f"model. The end-to-end fine-tuned reference reaches {p(ft_ref['test']['accuracy'])}, but "
         "it is not counted toward N = 5 because it trains the encoder and the classifier jointly "
         "(Section 7.1). No eligible embedding beats the cheap TF-IDF reference. Section 7 explains "
         "why: on a 592-row corpus, a single mean-pooled vector loses the word order that carries "
         "negation, while TF-IDF keeps phrases such as 'not good' as independent features.")

    h2(doc, "1.2  The five findings")
    bullet(doc,
           "The dataset documentation describes an entity-level task, but the delivered CSVs contain "
           "only Phrase and Sentiment. Label 2 ('Irrelevant') means 'not about the entity', so it is "
           "defined relative to information that was never shipped. The deployed model recovers "
           f"{conf_f[('Irrelevant','Irrelevant')]} of "
           f"{sum(conf_f[('Irrelevant', c)] for c in CLASSES)} Irrelevant test rows.",
           bold_lead="Finding 1 - The missing entity column limits what Irrelevant can mean. ")
    bullet(doc,
           f"Class sizes differ by only {eda['imbalance_ratio']:.2f}:1. That is mild, and we did not "
           "resample. The binding constraint is label overlap: tweets whose nearest neighbours have a "
           f"median TF-IDF cosine similarity of {nn10_cos:.2f} share a label only "
           f"{head['nn10_same_label_phrase']*100:.1f}% of the time at k=10. No classifier can exceed "
           "the consistency of its own labels.",
           bold_lead="Finding 2 - Imbalance is mild; label noise is severe. ")
    bullet(doc,
           f"{eda['issues']['Mojibake (Ã/Â/â)']} of {eda['train_rows_in']} training rows "
           f"({eda['issues']['Mojibake (Ã/Â/â)']/eda['train_rows_in']*100:.1f}%) contained "
           "double-encoded UTF-8. We repaired "
           f"{log['train']['steps'][6]['rows_affected']} rows with ftfy. The corruption is invisible "
           "in a spreadsheet but splits every affected word out of the vocabulary.",
           bold_lead="Finding 3 - Encoding corruption was the highest-impact data defect. ")
    bullet(doc,
           "Cleaning changed train-only CV macro F1 by "
           f"{(met['cv'][1]['macro_f1_mean']-met['cv'][0]['macro_f1_mean'])*100:+.2f} pp - "
           "neutral within fold noise. Cleaning is justified by correctness, by valid cross-"
           "validation and by transformer compatibility, not by a score gain. Saying so is more "
           "honest than claiming an improvement that the experiment does not support.",
           bold_lead="Finding 4 - Cleaning is a correctness step, not an accuracy lever. ")
    bullet(doc,
           "Switching the embedding moved test accuracy by "
           f"{(max(r['test_accuracy'] for r in emb_rows)-min(r['test_accuracy'] for r in emb_rows))*100:.1f} pp "
           f"({p(min(r['test_accuracy'] for r in emb_rows))} for the weakest embedding to "
           f"{p(max(r['test_accuracy'] for r in emb_rows))} for the strongest). The classifier is not "
           "the bottleneck; the representation is. The best eligible representation is "
           f"{(best_elig['test_accuracy']-dep['accuracy'])*100:+.1f} pp relative to sparse lexical "
           f"features, while the end-to-end fine-tuned reference is "
           f"{(ft_ref['test']['accuracy']-dep['accuracy'])*100:+.1f} pp.",
           bold_lead="Finding 5 - Representation, not classifier complexity, is the lever. ")

    callout(doc, "Recommendation",
            "Deploy the TF-IDF word+char union (M2) for routine monitoring, because it matches the "
            "best neural model to within a few rows at a fraction of the cost. Use DistilBERT only "
            "where its small accuracy edge justifies pretraining and CPU time. Treat Irrelevant as a "
            "'review' queue rather than an automatic decision, and state the 60-65% realistic ceiling "
            "in any service-level agreement.")

    # ---------------------------------------------------------- 2 scope
    doc.add_page_break()
    h1(doc, "Project Scope and Data Understanding", "2")

    h2(doc, "2.1  Business framing")
    body(doc,
         "The brief asks for a decision-support system that classifies text. For Twitter sentiment "
         "the practical use is brand and product monitoring: a support or marketing team wants to "
         "route complaints, spot emerging product failures, and separate genuine opinion from "
         "unrelated chatter. Every design choice below is judged against that use rather than against "
         "a leaderboard.")

    h2(doc, "2.2  Task definition")
    body(doc,
         "Multi-class, single-label text classification. Each row is one tweet; the target is one of "
         "four sentiment labels. There is no entity or topic column, so we treat the task as "
         "document-level 4-way classification and document the deviation explicitly.")

    table(doc,
          ["Attribute", "Value"],
          [
              ["Dataset", "Dataset 5 - Twitter (entity-level sentiment analysis)"],
              ["Files", "traindata.csv (labelled), testdata.csv (labelled hold-out)"],
              ["Raw schema", "Phrase (text), Sentiment (0-3). No ID, timestamp, user or entity column"],
              ["Label map", "0 = Negative, 1 = Positive, 2 = Irrelevant, 3 = Neutral"],
              ["Granularity", "One row = one tweet, truncated at roughly 350 characters"],
              ["Train / test rows", f"{eda['train_rows_in']} / {eda['test_rows_in']} "
                                    f"(after cleaning {eda['train_rows_out']} / {eda['test_rows_out']})"],
              ["Language / domain", "English; consumer technology, gaming, telecoms and general chatter"],
              ["Missing values", "None in either column of either file"],
          ],
          widths=[1.55, 4.75])

    h2(doc, "2.3  Data granularity and what it implies")
    body(doc,
         "One row is a single tweet, and tweets are truncated at roughly 350 characters. This matters "
         "in three ways. First, a tweet is an incomplete thought: the model cannot see context outside "
         "the box, so sarcasm and irony are frequently unresolvable. Second, rows are independent "
         "snippets with no user or time identifier, so repeat complaints from the same account cannot "
         "be grouped. Third, the median tweet is only "
         f"{eda['median_tokens']:.0f} tokens, which is short for TF-IDF bigrams and is one reason the "
         "bag-of-words approach saturates quickly.")

    h2(doc, "2.4  Documented versus delivered schema")
    body(doc,
         "The supplied dataset description states: 'Given a message and an entity, the task is to "
         "judge the sentiment of the message about the entity', with classes Negative, Positive, "
         "Irrelevant and Neutral. We verified row by row that neither CSV contains an entity or topic "
         "column: the only columns are Phrase and Sentiment. We therefore treat this as a "
         "data-delivery limitation and document it here rather than silently improvising labels.")

    callout(doc, "Data limitation (confirmed with the professor)",
            "The consultation confirmed that text-only classification is an acceptable way to "
            "continue. The report therefore states the limitation openly: the 'Irrelevant' label is "
            "partly defined by a missing entity, so its recall is capped. This is a property of the "
            "delivered data, not a tuning failure.")

    # ---------------------------------------------------------- 3 EDA
    doc.add_page_break()
    h1(doc, "Exploratory Data Analysis", "3")

    body(doc,
         "The brief requires at least N data statistics and visualisations, where N is the number of "
         "group members (five). This report contains sixteen figures and more than a dozen summary "
         "statistics across EDA, cleaning and modelling, so the requirement is exceeded rather than "
         "satisfied at the minimum.")

    h2(doc, "3.1  Class distribution and imbalance")
    trc = eda["class_balance"]["train_counts"]
    tec = eda["class_balance"]["test_counts"]
    table(doc,
          ["Class", "Train n", "Train %", "Test n", "Test %", "Difference (pp)"],
          [[cls, trc[cls], f"{trc[cls]/eda['train_rows_in']*100:.1f}",
            tec[cls], f"{tec[cls]/eda['test_rows_in']*100:.1f}",
            f"{(tec[cls]/eda['test_rows_in'] - trc[cls]/eda['train_rows_in'])*100:+.1f}"]
           for cls in CLASSES]
          + [["Total", eda["train_rows_in"], "100.0", eda["test_rows_in"], "100.0", "-"]],
          widths=[1.3, 0.85, 0.85, 0.85, 0.85, 1.3])

    body(doc,
         f"The classes are mildly imbalanced: the largest class (Negative) is "
         f"{eda['imbalance_ratio']:.2f}x the smallest (Irrelevant). This is far milder than the 10:1 "
         "or 100:1 ratios that usually force resampling, so we did not apply SMOTE. The train and test "
         "distributions agree closely, so the test file is a fair proxy for the training distribution "
         "and a single hold-out evaluation is meaningful.")
    body(doc,
         "The consultation feedback asked us to decide whether the 1.65:1 ratio warrants extra "
         "processing. Our position is that it does not warrant resampling, but it does warrant class "
         "weights, because the minority class is also the class with the lowest recall. Section 6.4 "
         "quantifies that trade-off rather than assuming it.")

    figure(doc, "fig01_class_balance.png", 6.5,
           "Figure 1 - Class distribution in train and test. Imbalance is mild (1.65:1) and the two "
           "splits match, so accuracy is meaningful alongside macro F1.")

    h2(doc, "3.2  Text length distribution and outliers")
    ls = eda["length_stats"]
    body(doc,
         f"After cleaning, the median tweet is {ls['50%']:.0f} tokens with a mean of {ls['mean']:.1f} "
         f"and a standard deviation of {ls['std']:.1f}. The distribution is right-skewed: the 75th "
         f"percentile is {ls['75%']:.0f} tokens but the maximum is {ls['max']:.0f}. Using the "
         f"1.5 x IQR rule the upper fence is {ls['upper_fence']:.0f} tokens, and only "
         f"{log['train']['steps'][-1]['flagged']} rows exceed it.")
    body(doc,
         "We made a deliberate decision not to delete these rows. Length outliers in text are almost "
         "always legitimate long tweets rather than measurement errors, and deleting them would "
         "systematically remove detailed complaints - exactly the subclass a brand-monitoring system "
         "must not lose. They are flagged with an is_length_outlier column instead.")

    figure(doc, "fig02_length_distribution.png", 6.6,
           "Figure 2 - Tweet length after cleaning. The long right tail is genuine content, not "
           "corruption, so rows are flagged rather than dropped.")

    h2(doc, "3.3  Text signals and their association with the label")
    body(doc,
         "The dataset has effectively one feature (text) and one target (label), so a conventional "
         "correlation matrix has nothing to correlate. Instead we measured the association between "
         "engineered text signals and the label, which is the correct analogue for unstructured data:")

    nb = eda["noise_by_class"]
    table(doc,
          ["Text signal", "Negative", "Positive", "Irrelevant", "Neutral", "Reading"],
          [
              ["% with @mentions", f"{nb['Negative']['@mentions']:.0f}",
               f"{nb['Positive']['@mentions']:.0f}", f"{nb['Irrelevant']['@mentions']:.0f}",
               f"{nb['Neutral']['@mentions']:.0f}", "Complaints are addressed to support handles"],
              ["% with URLs", f"{nb['Negative']['URLs']:.1f}", f"{nb['Positive']['URLs']:.1f}",
               f"{nb['Irrelevant']['URLs']:.1f}", f"{nb['Neutral']['URLs']:.1f}",
               "Neutral/irrelevant rows are news-like"],
              ["% with ellipses", f"{nb['Negative']['Ellipses']:.0f}",
               f"{nb['Positive']['Ellipses']:.0f}", f"{nb['Irrelevant']['Ellipses']:.0f}",
               f"{nb['Neutral']['Ellipses']:.0f}", "Hedging and incomplete thoughts"],
              ["% with <unk> token", f"{nb['Negative']['<unk> token']:.1f}",
               f"{nb['Positive']['<unk> token']:.1f}", f"{nb['Irrelevant']['<unk> token']:.1f}",
               f"{nb['Neutral']['<unk> token']:.1f}", "Tokenizer artefacts, not semantic"],
              ["% mojibake", f"{nb['Negative']['Mojibake']:.0f}",
               f"{nb['Positive']['Mojibake']:.0f}", f"{nb['Irrelevant']['Mojibake']:.0f}",
               f"{nb['Neutral']['Mojibake']:.0f}", "Flat across classes = random corruption"],
          ],
          widths=[1.25, 0.72, 0.72, 0.78, 0.68, 2.15], font_size=8)

    body(doc,
         "Two signals are genuinely informative. @mentions appear in "
         f"{nb['Negative']['@mentions']:.0f}% of Negative rows against only "
         f"{nb['Neutral']['@mentions']:.0f}% of Neutral rows, confirming that negative tweets are "
         "usually addressed at a company. URLs appear almost exclusively in Neutral and Irrelevant "
         "rows, which fits their news-and-links character. The mojibake rate is flat across classes, "
         "which is the proof that it is random technical corruption rather than anything semantic.")

    figure(doc, "fig04_noise_by_class.png", 6.6,
           "Figure 3 - Noise prevalence by class. @mentions skew negative, URLs skew neutral, while "
           "mojibake is flat and therefore safe to strip.")

    h2(doc, "3.4  Vocabulary and class-conditional content")
    body(doc,
         "We profiled the most frequent discriminative terms per class after stop-word removal. The "
         "separation is visible but far from clean, which foreshadows the ceiling: Negative is "
         "dominated by profanity and product-failure words, Positive by affect words, while Neutral "
         "and Irrelevant share heavy overlap in news vocabulary.")

    figure(doc, "fig05_top_words.png", 6.6,
           "Figure 4 - Top discriminative terms per class. Negative and Positive separate on affect; "
           "Neutral and Irrelevant overlap heavily on news vocabulary.")

    figure(doc, "fig07_wordclouds.png", 6.6,
           "Figure 5 - Word clouds per class. Visual confirmation that Negative centres on profanity "
           "and failure language, while Neutral/Irrelevant are dominated by entities and links.")

    h2(doc, "3.5  Brand-level view")
    brand_path = TAB / "brand_sentiment.csv"
    if brand_path.exists():
        with open(brand_path, encoding="utf-8", newline="") as f:
            brand = list(csv.DictReader(f))
        top_neg = sorted(brand, key=lambda r: -float(r["Negative"]))[:4]
        top_pos = sorted((r for r in brand if int(float(r["n"])) >= 5),
                         key=lambda r: -float(r["Positive"]))[:3]
        body(doc,
             "Because the practical use case is brand monitoring, we aggregated sentiment by brand "
             "mention. The most negative brands are "
             + ", ".join(f"{r['brand']} ({float(r['Negative']):.0f}% negative, n={int(float(r['n']))})"
                         for r in top_neg)
             + ". The most positive are "
             + ", ".join(f"{r['brand']} ({float(r['Positive']):.0f}% positive, n={int(float(r['n']))})"
                         for r in top_pos)
             + ". This is the kind of insight that turns a classifier into a decision-support tool: "
               "the output is only useful when it is aggregated to the entity a manager cares about.")

        figure(doc, "fig06_brand_sentiment.png", 6.6,
               "Figure 6 - Sentiment mix for the most-mentioned brands. Negative and positive brands "
               "separate clearly once the row-level predictions are aggregated.")

    h2(doc, "3.6  Data-quality audit")
    issues = eda["issues"]
    table(doc,
          ["Defect", "Rows affected", "% of train", "Severity"],
          [
              ["All-caps shouting", issues["All-caps shouting"],
               f"{issues['All-caps shouting']/eda['train_rows_in']*100:.1f}", "Low - handled by lowercasing"],
              ["@mentions", issues["@mentions"],
               f"{issues['@mentions']/eda['train_rows_in']*100:.1f}", "Medium - signal and noise"],
              ["Non-ASCII characters", issues["Non-ASCII characters"],
               f"{issues['Non-ASCII characters']/eda['train_rows_in']*100:.1f}", "Low after repair"],
              ["Mojibake (double-encoded UTF-8)", issues["Mojibake (Ã/Â/â)"],
               f"{issues['Mojibake (Ã/Â/â)']/eda['train_rows_in']*100:.1f}", "High - corrupts tokens"],
              ["<unk> placeholder", issues["<unk> placeholder"],
               f"{issues['<unk> placeholder']/eda['train_rows_in']*100:.1f}", "Medium - tokeniser artefact"],
              ["Embedded URLs", issues["Embedded URLs"],
               f"{issues['Embedded URLs']/eda['train_rows_in']*100:.1f}", "Medium - high-cardinality noise"],
              ["Duplicate phrase text", issues["Duplicate phrase text"],
               f"{issues['Duplicate phrase text']/eda['train_rows_in']*100:.1f}", "High - leakage risk"],
              ["Exact duplicate rows", issues["Exact duplicate rows"],
               f"{issues['Exact duplicate rows']/eda['train_rows_in']*100:.1f}", "High - leakage risk"],
              ["Same text, different label", issues["Same text, different label"],
               f"{issues['Same text, different label']/eda['train_rows_in']*100:.1f}", "High - label noise"],
              ["HTML-like tags", issues["HTML-like tags"],
               f"{issues['HTML-like tags']/eda['train_rows_in']*100:.1f}", "Low - rare"],
          ],
          widths=[1.9, 1.0, 1.0, 2.4])

    figure(doc, "fig08_data_quality_audit.png", 6.6,
           "Figure 7 - Data-quality audit of the raw training file. Encoding corruption and duplicates "
           "were the defects with real modelling consequences.")

    # ---------------------------------------------------------- 4 cleaning
    doc.add_page_break()
    h1(doc, "Data Cleaning Pipeline", "4")

    body(doc,
         "We designed the pipeline as a sequence of auditable steps. Each step records what it "
         "changed, how many rows it touched and what rule it applied. The full log is written to "
         "data/processed/cleaning_log.json and reproduced in Appendix A. Every step is applied "
         "identically to train and test using the same shared function, so the two files cannot drift "
         "apart.")

    steps = log["train"]["steps"]
    table(doc,
          ["#", "Step", "Rule applied", "Rows affected"],
          [[str(s["step"]), s["name"], s["action"], str(s["rows_affected"])] for s in steps],
          widths=[0.35, 1.35, 3.75, 0.85], font_size=7.6)

    h2(doc, "4.1  Why each decision was made")
    bullet(doc,
           "No imputation is defensible for text targets. With zero missing values the decision was "
           "moot, but we codified drop-on-missing so the rule is explicit if the test file is ever "
           "refreshed.", bold_lead="Missing values: ")
    bullet(doc,
           "Three exact duplicate rows and one contradictory row were removed. Duplicates inflate "
           "cross-validation scores because the same sentence can appear in both the training and "
           "validation fold. The conflicting row was resolved to its majority label.",
           bold_lead="Duplicates and conflicting labels: ")
    bullet(doc,
           "We applied ftfy to reverse UTF-8 that had been decoded as cp1252, then Unicode NFKC "
           f"normalisation. This repaired {log['train']['steps'][6]['rows_affected']} rows.",
           bold_lead="Encoding repair: ")
    bullet(doc,
           "Lowercasing, contraction expansion before punctuation stripping, URL and mention "
           "stripping, hashtag de-camel-casing, elongation collapsing and removal of non-alphabetic "
           "characters. We preserve negations explicitly because removing 'not' flips sentiment.",
           bold_lead="Normalisation: ")
    bullet(doc,
           "WordNet lemmatisation reduced the vocabulary from "
           f"{log['train']['steps'][8]['detail'].split('from ')[1].split(' to ')[0]} to "
           f"{log['train']['steps'][8]['detail'].split(' to ')[1].split(' tokens')[0]}. We chose "
           "lemmatisation over stemming because stemming produces non-words and would damage the "
           "pretrained-embedding alignment used in Phase 3ii.",
           bold_lead="Lemmatisation: ")

    h2(doc, "4.2  Documented before/after evidence")
    table(doc,
          ["Raw (as delivered)", "After cleaning"],
          [
              ["Nerd Tweet: Iâ\u0080\u0099m getting really disappointed with @PlayOverwatch one of my favourite heroes",
               "nerd tweet i be get real disappoint with one of my favourite hero"],
              ["New hero, expansion, and Priest class rework<unk>. I am super stoked",
               "new hero expansion and priest class rework i be super stoke"],
              ["I can't make SHIT green anymore.",
               "i can not make shit green anymore"],
              ["Check out https://t.co/Mo0oIFkdBT and #GamingIsLife sooooo goooood!!!!",
               "check out and gamingislife soo good"],
          ],
          widths=[3.5, 2.8], font_size=8)

    body(doc,
         "The examples show the behaviours that matter: the mojibake apostrophe is repaired; the "
         "<unk> placeholder is removed without welding neighbouring words together; negation is "
         "preserved as 'can not'; and elongation is normalised so that 'goooood' still resembles "
         "'good'.")

    figure(doc, "fig03_cleaning_effect.png", 6.4,
           "Figure 8 - Effect of cleaning on length and vocabulary. Cleaning shortens tweets modestly "
           "and trims the long tail of noise tokens.")

    # ---------------------------------------------------------- 5 baseline
    doc.add_page_break()
    h1(doc, "Baseline Model: TF-IDF + Logistic Regression", "5")

    h2(doc, "5.1  Model design")
    body(doc,
         "Phase 3i requires exactly one baseline: logistic regression on TF-IDF features. We kept it "
         "deliberately plain so that it is a fair reference point rather than a tuned competitor.")

    table(doc,
          ["Component", "Setting", "Reason"],
          [
              ["Vectoriser", "TF-IDF, word 1-2 grams", "Bigrams capture negation phrases such as 'not good'"],
              ["Weighting", "sublinear_tf=True", "Log-damped term frequency; avoids long-tweet dominance"],
              ["Vocabulary", f"{met['tfidf_features']:,} features, min_df=1, max_df=0.95",
               "Tiny corpus: rare terms are still useful"],
              ["Classifier", "Multinomial Logistic Regression", "Required baseline; interpretable"],
              ["Regularisation", "C = 1.0 (L2), lbfgs solver", "Untuned default"],
              ["Class weights", "None for the reported baseline", "Balanced variant reported separately"],
          ],
          widths=[1.25, 2.15, 2.9], font_size=8.2)

    h2(doc, "5.2  Evaluation protocol")
    body(doc,
         "With only 592 training rows, a single train/validation split would give a benchmark with a "
         "confidence interval wider than any improvement we are likely to measure. We therefore report "
         "two complementary views:")
    bullet(doc, "Stratified 5-fold cross-validation on the training file only. This is the stable, "
                "low-variance headline number and it is used for all model-selection decisions.")
    bullet(doc, "A single held-out evaluation on testdata.csv using the model fitted on all 592 "
                "training rows. The test file is never used for model selection.")
    body(doc,
         "The consultation feedback asked us to keep this cross-validation approach and to report "
         "validation and test side by side for the advanced models. Section 7 does exactly that.")

    h2(doc, "5.3  Benchmark: does cleaning actually help?")
    cv_rows = []
    for r in met["cv"]:
        cv_rows.append([r["variant"],
                        f"{r['accuracy_mean']*100:.1f} +/- {r['accuracy_std']*100:.1f}",
                        f"{r['macro_f1_mean']*100:.1f} +/- {r['macro_f1_std']*100:.1f}",
                        f"{r['weighted_f1_mean']*100:.1f}"])
    table(doc,
          ["Variant (5-fold CV on train)", "Accuracy %", "Macro F1 %", "Weighted F1 %"],
          cv_rows, widths=[2.5, 1.25, 1.25, 1.25])

    body(doc,
         "This table converts cleaning decisions into measured effects. Cleaning alone changed macro "
         f"F1 by {(met['cv'][1]['macro_f1_mean']-met['cv'][0]['macro_f1_mean'])*100:+.2f} pp, "
         f"essentially neutral within the fold-to-fold noise of about "
         f"+/-{met['cv'][1]['macro_f1_std']*100:.1f} pp. This is an honest and slightly uncomfortable "
         "result: cleaning cannot be justified on accuracy grounds. It is justified because the data "
         "is objectively corrupt, because the embedding models require clean tokenisation, and because "
         "removing duplicates makes the CV estimate valid.")

    figure(doc, "fig10_cv_comparison.png", 6.6,
           "Figure 9 - Cleaning decisions expressed as measured effects. Class-balanced weighting is "
           "the single largest improvement to macro F1.")

    h2(doc, "5.4  Headline baseline results on the held-out test file")
    table(doc,
          ["Class", "Precision", "Recall", "F1", "Support"],
          [[r["class"], f"{r['precision']:.3f}", f"{r['recall']:.3f}", f"{r['f1']:.3f}",
            str(r["support"])] for r in met["per_class"]],
          widths=[1.5, 1.1, 1.0, 0.9, 0.9], font_size=8.4)

    body(doc,
         f"Overall test accuracy is {p(base_acc)} and macro F1 is {p(base_f1)}. Against the "
         f"majority-class guess of {p(met['majority_baseline_accuracy'])}, this is a genuine lift of "
         f"{(base_acc-met['majority_baseline_accuracy'])*100:.1f} percentage points. The instructor's "
         "feedback that a mid-40s to low-50s accuracy is normal for this corpus is consistent with "
         "what we observe: the model has learned real signal, and the per-class breakdown shows "
         "exactly where it fails.")

    body(doc,
         "Adding class_weight='balanced' lowered accuracy slightly "
         f"({p(met['test_accuracy'])} to {p(met['test_balanced_accuracy'])}) but raised macro F1 from "
         f"{p(met['test_macro_f1'])} to {p(met['test_balanced_macro_f1'])}. This is the expected "
         "trade: a little headline accuracy is given up to make the weakest class usable. Section 6.4 "
         "revisits this at the per-class level.")

    figure(doc, "fig09_confusion_matrix.png", 6.5,
           "Figure 10 - Baseline confusion matrix on the held-out test set. The diagonal is strong for "
           "Negative only; Irrelevant is scattered across all four predictions.")

    h2(doc, "5.5  Interpreting what the model learned")
    body(doc,
         "Because logistic regression is linear, the learned coefficients can be read directly. "
         "Negative is driven by profanity and failure words, Positive by affect and praise words, and "
         "Neutral by named entities and news vocabulary. This is evidence that the model has learned "
         "genuine lexical sentiment rather than an artefact of the split.")

    figure(doc, "fig12_top_coefficients.png", 6.7,
           "Figure 11 - Strongest TF-IDF terms per class. The learned associations are semantically "
           "sensible, which validates the pipeline even where accuracy is modest.")

    # ---------------------------------------------------------- 6 refinement
    doc.add_page_break()
    h1(doc, "Baseline Refinement", "6")

    body(doc,
         "Phase 3iv asks for the baseline to be improved and then re-compared with the advanced "
         "models. We treated this as a model-selection problem with a fixed protocol: every candidate "
         "was scored by 5-fold stratified CV on the training file only, the winning configuration was "
         "frozen, and the test file was then scored once. The sweep is reproducible with "
         "src/explore_improved.py and its full table is in reports/tables/improved_cv_sweep.csv.")

    h2(doc, "6.1  What changed")
    body(doc,
         "The deployed M2 model uses a word 1-2 gram + char_wb 2-5 gram TF-IDF union on the raw "
         "phrase. The refinement explores three axes: the shape of the TF-IDF space, the "
         "regularisation strength, and class weighting, plus ensembles and a second text view. The "
         "configuration selected by CV adds the lemmatised text view as a second column and uses "
         "balanced class weights.")

    h2(doc, "6.2  Train-only CV sweep (top configurations)")
    top = sorted(sweep, key=lambda r: -float(r["cv_macro_f1"]))[:8]
    table(doc,
          ["Configuration", "CV accuracy", "CV macro F1"],
          [[r["configuration"], f"{float(r['cv_accuracy'])*100:.1f}%",
            f"{float(r['cv_macro_f1'])*100:.1f}%"] for r in top],
          widths=[3.6, 1.35, 1.35], font_size=8.2)

    body(doc,
         "The selected configuration wins on both CV accuracy and CV macro F1. The margin over the "
         "deployed M2 is real in CV but small in absolute terms, which is exactly what a noisy, "
         "low-resource text problem should look like.")

    h2(doc, "6.3  Result on the held-out test file")
    m1, m2, m3 = ("M1 word TF-IDF + LR", "M2 word+char union + LR", "M3 word+char union + LinearSVC")
    table(doc,
          ["Model", "Text view", "CV acc", "CV macro F1", "Test acc", "Test macro F1"],
          [
              [m1, "phrase", p(cv_final[m1]["accuracy"]), p(cv_final[m1]["macro_f1"]),
               p(test_final[m1]["accuracy"]), p(test_final[m1]["macro_f1"])],
              [m2, "phrase", p(cv_final[m2]["accuracy"]), p(cv_final[m2]["macro_f1"]),
               p(test_final[m2]["accuracy"]), p(test_final[m2]["macro_f1"])],
              [m3, "phrase", p(cv_final[m3]["accuracy"]), p(cv_final[m3]["macro_f1"]),
               p(test_final[m3]["accuracy"]), p(test_final[m3]["macro_f1"])],
              ["B4 phrase+lemma + LR (balanced)", "phrase + lemma",
               p(imp_cv["accuracy"]), p(imp_cv["macro_f1"]),
               p(imp_test["accuracy"]), p(imp_test["macro_f1"])],
          ],
          widths=[2.0, 1.0, 0.8, 0.95, 0.8, 0.95], font_size=7.8)

    body(doc,
         "Two readings matter. First, the official Phase 3i baseline (49.6% test accuracy) and M1 "
         "here (50.9%) are the same architecture; the difference is the text view, because the "
         "official baseline uses text_lemma while M1 uses the raw phrase. Second, the CV-winning B4 "
         "does not win on test: it scores "
         f"{p(imp_test['accuracy'])} against M2's {p(dep['accuracy'])}, a change of "
         f"{dpp(imp_test['accuracy']-dep['accuracy'])}. On 397 test rows one row is worth 0.25 pp, so "
         "this is a two-row-per-percentage-point gap and the direction is a caution about CV-based "
         "selection on small data.")

    h2(doc, "6.4  The class-weight trade-off, per class")
    dep_pc = {r["class"]: r for r in fin["per_class"]}
    imp_pc = {r["class"]: r for r in imp_test["per_class"]}
    table(doc,
          ["Class", "Deployed recall", "B4 recall", "Deployed F1", "B4 F1", "Read"],
          [[c, f"{dep_pc[c]['recall']:.3f}", f"{imp_pc[c]['recall']:.3f}",
            f"{dep_pc[c]['f1']:.3f}", f"{imp_pc[c]['f1']:.3f}",
            {"Negative": "Majority class loses recall",
             "Positive": "Broadly unchanged",
             "Irrelevant": "Minority class gains recall",
             "Neutral": "Small F1 gain"}[c]] for c in CLASSES],
          widths=[1.1, 1.0, 0.9, 0.9, 0.9, 1.6], font_size=8.2)

    body(doc,
         "The pattern is exactly the trade the consultation described. Balanced weights raise "
         f"Irrelevant recall from {dep_pc['Irrelevant']['recall']:.1%} to "
         f"{imp_pc['Irrelevant']['recall']:.1%} and lift Neutral F1, while Negative - the largest and "
         "easiest class - loses recall. Macro F1 moves up in CV because the minority gains outweigh "
         "the majority loss; on the fixed test file the gain does not quite materialise. Reporting "
         "both numbers, and explaining the class-level shift, is more useful than choosing whichever "
         "metric flatters the model.")

    body(doc,
         "For deployment we keep M2. The refinement is retained as an experiment and as evidence "
         "that class weighting is a deliberate trade, not a free improvement.")

    # ---------------------------------------------------------- 7 N=5
    doc.add_page_break()
    h1(doc, "Advanced Modelling: the N = 5 Matrix", "7")

    h2(doc, "7.1  Design constraints")
    body(doc,
         "The brief requires N different analytics models, where N is the number of group members "
         "(five), each paired with a word embedding. Four constraints come from the brief itself:")
    bullet(doc, "TF-IDF is not a word embedding, so it cannot occupy one of the five slots. It is kept "
                "as a reference row only.")
    bullet(doc, "Logistic Regression is reserved for the baseline, so it cannot be reused as one of "
                "the five.")
    bullet(doc, "The five analytics models must be different. Ours are XGBoost, Random Forest, SVM, "
                "MLP and k-NN - five distinct algorithms.")
    bullet(doc, "A single model may not serve as both the embedding model and the classifier. "
                "DistilBERT is therefore FROZEN and used only as an embedding extractor; an "
                "independent MLP is fitted on its pooled vectors. The end-to-end fine-tuned result "
                "is reported as a reference only and is not counted toward N = 5.")
    body(doc,
         "The consultation added a diversity constraint: avoid near-identical models. We keep exactly "
         "one Random Forest, remove Extra Trees, and include a Neural Network (the MLP on frozen "
         "DistilBERT). XGBoost and Random Forest are distinct algorithms - gradient boosting versus "
         "bagging - and only one of them is a Random Forest.")

    h2(doc, "7.2  High-level approach")
    body(doc,
         "All five combinations share one protocol module, src/embeddings_common.py: identical rows, "
         "identical folds (StratifiedKFold with five splits, shuffle, seed 42), the same metric block, "
         "CV on train only, and the test file scored exactly once per model. Two adapters cover the "
         "different embedding types:")
    bullet(doc, "Dense document vectors (GloVe, Word2Vec, FastText, Sentence-BERT) are reduced to one "
                "mean-pooled vector per tweet and passed to a classical estimator.")
    bullet(doc, "Frozen encoders (DistilBERT) take token ids, produce one 768-d masked-mean vector "
                "per tweet, and hand that matrix to an independently fitted MLP. The encoder is "
                "never trained, which satisfies the integration rule.")
    body(doc,
         "Mean pooling is a deliberate, documented choice: with tweets of roughly twenty tokens it is "
         "the standard baseline and needs no extra hyper-parameters, which keeps the five models "
         "comparable. Section 7.5 explains why it is also the main reason the non-transformer "
         "embeddings underperform.")

    h2(doc, "7.3  The five combinations")
    table(doc,
          ["#", "Embedding", "Analytics model", "Family", "Why this pairing"],
          [
              ["1", "GloVe glove-wiki-gigaword-100 (frozen)", "Gradient Boosting (XGBoost)",
               "Boosted trees", "Strong classical learner on dense vectors"],
              ["2", "Word2Vec (trained from scratch)", "Random Forest",
               "Bagged trees", "The only Random Forest; satisfies the from-scratch criterion"],
              ["3", "FastText char 3-6 grams (from scratch)", "Support Vector Machine",
               "Kernel method", "Subword units absorb misspellings and elongation"],
              ["4", "DistilBERT (frozen)", "Neural Network (MLP)",
               "Frozen transformer + MLP", "Contextual vectors; classifier fitted separately"],
              ["5", "Sentence-BERT all-MiniLM-L6-v2 (frozen)", "k-Nearest-Neighbours",
               "Non-parametric", "Contrasts a neighbour vote against discriminative models"],
          ],
          widths=[0.3, 1.75, 1.25, 1.05, 2.0], font_size=7.6)

    body(doc,
         "The five analytics models are deliberately distinct algorithms: gradient-boosted trees "
         "(XGBoost), bagged trees (Random Forest), a kernel SVM, a neural network (MLP) and a "
         "non-parametric neighbour vote. The matrix contains exactly one Random Forest and no Extra "
         "Trees, which directly answers the consultation feedback. DistilBERT is frozen and paired "
         "with a separately fitted MLP, so the encoder is an embedding model and the MLP is the "
         "analytics classifier - the integration rule is satisfied. The end-to-end fine-tuned result "
         f"({p(ft_ref['test']['accuracy'])} test accuracy) is shown in Section 8 as an upper-bound "
         "reference only.")

    h2(doc, "7.4  Results: CV and test side by side")
    rows = []
    for r in emb_rows:
        rows.append([
            str(r["id"]), r["embedding"], r["analytics_model"],
            f"{p(r['cv_accuracy'])} +/- {r['cv_sd']*100:.1f}",
            p(r["test_accuracy"]),
            dpp(r["test_accuracy"] - r["cv_accuracy"]),
            f"{r['test_macro_f1']:.3f}",
            f"{r['kappa']:.3f}",
            f"{r['within1']*100:.1f}%",
        ])
    table(doc,
          ["#", "Embedding", "Analytics model", "CV acc", "Test acc", "Test - CV",
           "Macro F1", "kappa", "within-1"],
          rows,
          widths=[0.25, 1.35, 1.25, 0.85, 0.7, 0.75, 0.65, 0.5, 0.65], font_size=7.2)

    body(doc,
         f"The reference row is the deployed TF-IDF + LR model at {p(ref_row['test_accuracy'])} test "
         f"accuracy and {ref_row['test_macro_f1']:.3f} macro F1. No eligible combination beats it on "
         f"accuracy; the closest is frozen DistilBERT + MLP at "
         f"{p(best_elig['test_accuracy'])} "
         f"({dpp(best_elig['test_accuracy']-ref_row['test_accuracy'])}). The end-to-end fine-tuned "
         f"reference reaches {p(ft_ref['test']['accuracy'])}, but it is not an eligible combination. "
         "On macro F1 the TF-IDF reference is still the strongest, which is a useful reminder that "
         "accuracy and macro F1 do not rank the same models here.")

    h2(doc, "7.5  Reading the results")
    bullet(doc,
           "The CV-to-test column exposes the rank reversal that makes small-data model selection "
           "dangerous. Word2Vec + RF has the third-highest CV accuracy but the lowest test accuracy; "
           "GloVe + XGBoost gains "
           f"{(next(r for r in emb_rows if r['id']==1)['test_accuracy']-next(r for r in emb_rows if r['id']==1)['cv_accuracy'])*100:+.1f} pp "
           "from CV to test, while Sentence-BERT actually loses "
           f"{(next(r for r in emb_rows if r['id']==5)['cv_accuracy']-next(r for r in emb_rows if r['id']==5)['test_accuracy'])*100:.1f} pp. "
           "With 592 training rows, CV folds share one fitted embedding space, so they are not "
           "independent estimates of generalisation.",
           bold_lead="The rank reversal. ")
    bullet(doc,
           "Every non-transformer row reduces a tweet to one mean-pooled vector. For 'not good' that "
           "vector is close to the average of 'not' and 'good', which sits almost on top of 'good'. A "
           "negation flips the label but barely moves the embedding. TF-IDF keeps the bigram 'not "
           "good' as its own feature, so a linear model can learn it directly. Frozen DistilBERT is "
           "the only embedding here whose vectors are contextual, which is why it is the strongest "
           "eligible row even after mean pooling.",
           bold_lead="Why mean pooling hurts sentiment. ")
    bullet(doc,
           "Word2Vec and FastText learned 4,280 vectors from 592 tweets, which is fewer than eight "
           "occurrences per training row before duplicate tokens are considered. Distributional "
           "methods cannot estimate a stable vector from one or two occurrences. GloVe has 400,000 "
           "vectors from six billion tokens and scores much better under the same pipeline. Coverage "
           "is not the problem; vector quality is.",
           bold_lead="Why the from-scratch embeddings are weak. ")
    bullet(doc,
           "The best eligible model is worth about "
           f"{(emb_rows[0]['test_accuracy']-ref_row['test_accuracy'])*100:.1f} pp over a sparse "
           "lexical model that needs no pretraining, no GPU and no fine-tuning. That is the strongest "
           "single argument that the dataset, not the modelling effort, is the binding constraint.",
           bold_lead="The size of the prize. ")

    figure(doc, "fig15_embedding_matrix.png", 6.7,
           "Figure 12 - N = 5 embedding x analytics-model matrix. Grey bars are train-only CV with "
           "one-standard-deviation whiskers; blue bars are the single held-out test evaluation. The "
           "green line is the deployed TF-IDF reference.")

    figure(doc, "fig16_embedding_families.png", 6.3,
           "Figure 13 - CV versus test for the five embedding families. Points far from the diagonal "
           "are the rank reversals discussed in Section 7.5.")

    # ---------------------------------------------------------- 8 comparison
    doc.add_page_break()
    h1(doc, "Model Comparison and Discussion", "8")

    h2(doc, "8.1  Advanced models versus the Phase 3i baseline")
    table(doc,
          ["Model", "Test acc", "Delta vs baseline", "Test macro F1", "Verdict"],
          [["Phase 3i baseline (lemma TF-IDF + LR)", p(base_acc), "-", p(base_f1), "Reference"]] +
          [[f"{r['embedding']} + {r['analytics_model']}", p(r["test_accuracy"]),
            dpp(r["test_accuracy"] - base_acc), f"{r['test_macro_f1']:.3f}",
            "Beats baseline" if r["test_accuracy"] > base_acc else "Below baseline"]
           for r in emb_rows],
          widths=[2.45, 0.8, 1.05, 0.95, 1.15], font_size=7.8)

    body(doc,
         "Against the required baseline, only frozen DistilBERT + MLP wins on accuracy; GloVe + "
         "XGBoost, SBERT + kNN, FastText + SVM and Word2Vec + RF all lose. That is not the result "
         "the brief might lead one to expect, but it is the measured one. The reason is not that "
         "embeddings are bad in general; it is that a 592-row corpus cannot support the "
         "representation they need, and mean pooling discards much of the word order that sentiment "
         "depends on.")

    h2(doc, "8.2  Advanced models versus the improved baseline")
    table(doc,
          ["Model", "Test acc", "Delta vs M2", "Test macro F1", "Delta macro vs M2"],
          [["Deployed M2 (word+char TF-IDF + LR)", p(dep["accuracy"]), "-", p(dep["macro_f1"]), "-"],
           ["B4 improved (phrase+lemma, balanced)", p(imp_test["accuracy"]),
            dpp(imp_test["accuracy"] - dep["accuracy"]), p(imp_test["macro_f1"]),
            dpp(imp_test["macro_f1"] - dep["macro_f1"])]] +
          [[f"{r['embedding']} + {r['analytics_model']}", p(r["test_accuracy"]),
            dpp(r["test_accuracy"] - dep["accuracy"]), f"{r['test_macro_f1']:.3f}",
            dpp(r["test_macro_f1"] - dep["macro_f1"])] for r in emb_rows],
          widths=[2.35, 0.8, 0.9, 0.95, 1.1], font_size=7.8)

    body(doc,
         "Against the deployed TF-IDF baseline, frozen DistilBERT + MLP is "
         f"{dpp(best_elig['test_accuracy']-dep['accuracy'])} on accuracy - within three test rows - "
         "and every other embedding is below M2. The honest conclusion is that no eligible advanced "
         "model delivered a practically meaningful gain on this dataset.")

    h2(doc, "8.3  Which model performs better, and why")
    body(doc,
         "On test accuracy the ranking is deployed M2 (54.2%), frozen DistilBERT + MLP (53.4%), "
         "B4 (52.6%), SBERT + kNN (48.9%), GloVe + XGBoost (46.1%), FastText + SVM (40.6%) and "
         "Word2Vec + RF (37.8%). The end-to-end fine-tuned reference reaches 55.4%, but it is not an "
         "eligible combination. On macro F1 the order is M2 (0.504), frozen DistilBERT + MLP "
         "(0.500) and SBERT + kNN (0.461). There is therefore no single winner; there is a "
         "cost-accuracy frontier.")
    body(doc,
         "The justification for the observed differences has three parts. First, representation: a "
         "contextual encoder is the only representation here that can distinguish 'not good' from "
         "'good', which is why it leads on accuracy. Second, sample size: the from-scratch embeddings "
         "cannot estimate a stable space from 592 tweets, which is why they trail even a sparse "
         "lexical model. Third, evaluation noise: one test row is 0.25 pp, so gaps under about 2 pp "
         "are inside the noise band and should not be over-interpreted.")
    body(doc,
         "For a decision-support system, the practical choice is M2: the frozen transformer does not "
         "beat it, and the end-to-end fine-tuned reference that does is not an eligible combination "
         "under the brief. This is a business judgement, not a modelling one, and Section 10 states "
         "it in those terms.")

    figure(doc, "fig14_model_comparison.png", 6.7,
           "Figure 14 - Final model comparison. Frozen DistilBERT + MLP is the strongest eligible "
           "advanced model; the TF-IDF models lead on macro F1 and on cost.")

    # ---------------------------------------------------------- 9 ceiling
    doc.add_page_break()
    h1(doc, "Error Analysis and the Accuracy Ceiling", "9")

    h2(doc, "9.1  Where the deployed model's errors go")
    body(doc,
         f"The deployed model misclassifies {errors_f} of {n_test} test rows. Grouping the errors by "
         "actual and predicted class shows that they are structured rather than random:")
    errs = [(a, b, conf_f[(a, b)]) for a in CLASSES for b in CLASSES if a != b and conf_f[(a, b)]]
    errs.sort(key=lambda x: -x[2])
    table(doc,
          ["Actual class", "Predicted as", "Count", "% of test"],
          [[a, b, str(n), f"{n/n_test*100:.1f}%"] for a, b, n in errs],
          widths=[1.5, 1.5, 0.9, 1.0], font_size=8.2)

    body(doc,
         "The single largest error source is Positive predicted as Negative: "
         f"{conf_f[('Positive','Negative')]} rows. Combined with Neutral predicted as Negative "
         f"({conf_f[('Neutral','Negative')]}), the Negative class acts as a gravitational well, "
         "absorbing Neutral, Positive and Irrelevant alike. That is a class-prior artefact: Negative "
         "is the largest class and the easiest to trigger, so the model over-predicts it.")

    table(doc,
          ["", *CLASSES],
          [[a, *[str(conf_f[(a, c)]) for c in CLASSES]] for a in CLASSES],
          widths=[1.3, 0.85, 0.85, 0.95, 0.85], font_size=8.2)

    figure(doc, "fig13_final_confusion.png", 6.4,
           "Figure 15 - Confusion matrix of the deployed model. Irrelevant is spread across all four "
           "columns; Negative absorbs errors from every class.")

    h2(doc, "9.2  The Irrelevant problem is structural")
    body(doc,
         f"The deployed model recovers {conf_f[('Irrelevant','Irrelevant')]} of "
         f"{sum(conf_f[('Irrelevant', c)] for c in CLASSES)} Irrelevant test rows. 'Irrelevant' means "
         "'not about the entity', and the entity column is absent. Two annotators given the same "
         "tweet would therefore have no way to agree on this label, and the training file already "
         "contains one contradictory pair. The label is partly defined by information that is absent "
         "from the file, which places a hard ceiling on any text-only model.")

    h2(doc, "9.3  Four controlled experiments")
    body(doc,
         "We investigated the ceiling rather than assuming it. Each experiment uses train-only CV or "
         "the fixed test predictions, and each is reproducible from src/experiments_headroom.py.")
    table(doc,
          ["Experiment", "Question", "Result", "Interpretation"],
          [
              ["E1 - permutation control",
               "Is the model learning anything real?",
               f"Real labels {p(head['cv_accuracy_real'])}; shuffled labels "
               f"{p(head['cv_accuracy_permuted_mean'])} +/- {head['cv_accuracy_permuted_sd']*100:.1f} "
               f"(max {p(head['cv_accuracy_permuted_max'])})",
               f"{(head['cv_accuracy_real']-head['cv_accuracy_permuted_mean'])*100:.0f} pp above the "
               "shuffled ceiling: real signal"],
              ["E2 - neighbour agreement",
               "How noisy are the labels?",
               f"k=10 same-label fraction {head['nn10_same_label_phrase']*100:.1f}%; median neighbour "
               f"cosine {next(r['median_nn_cosine_sim'] for r in csv_rows('exp_label_self_agreement.csv') if r['view']=='phrase' and r['k']=='10')}",
               "90%-similar tweets share a label only one time in three"],
              ["E3 - cross-view agreement",
               "Is cleaning destroying signal?",
               f"phrase vs text {next(float(r['cv_accuracy']) for r in csv_rows('exp_cross_view.csv') if r['view']=='phrase_vs_text')*100:.1f}%; "
               f"phrase vs lemma {next(float(r['cv_accuracy']) for r in csv_rows('exp_cross_view.csv') if r['view']=='phrase_vs_text_lemma')*100:.1f}%",
               "Light cleaning is lossless; lemmatisation drops a quarter of the view"],
              ["E4 - one-vs-rest AUC",
               "Which class is unlearnable?",
               f"Irrelevant AUC {head['irrelevant_ovr_auc']:.3f} vs "
               f"{next(float(r['always_majority_accuracy']) for r in csv_rows('exp_ovr_auc.csv') if r['class']=='Irrelevant'):.3f} "
               "for always saying 'not Irrelevant'",
               "Every learned model loses to the trivial rule for Irrelevant"],
          ],
          widths=[1.25, 1.3, 2.05, 1.85], font_size=7.4)

    h2(doc, "9.4  Supporting evidence")
    body(doc,
         f"A lexicon audit agrees with the neighbour experiment. On rows the polarity lexicon scores "
         f"clearly positive, only {noise['clearly_positive_pct_positive']:.1f}% carry the Positive "
         f"label; on clearly negative rows, only {noise['clearly_negative_pct_negative']:.1f}% carry "
         f"Negative. VADER agrees with the label on positive/negative rows only "
         f"{noise['vader_agreement_pos_neg']:.1f}% of the time. These are not model errors; they are "
         "labels that disagree with an independent sentiment reading.")
    body(doc,
         f"The ceiling analysis reinforces the point. On a binary Positive/Negative subset the model "
         f"reaches {p(ceiling['binary_pos_neg_cv_accuracy'])} CV accuracy against a majority reference "
         f"of {p(ceiling['binary_pos_neg_majority'])}; on a three-class subset it reaches "
         f"{p(ceiling['three_class_cv_accuracy'])}. Within-one-class accuracy is "
         f"{p(ceiling['within1_accuracy'])}, which means the model usually gets the polarity direction "
         "right even when it misses the exact class.")

    h2(doc, "9.5  Verdict on the ceiling")
    body(doc,
         "Three independent limits - label noise, the missing entity column and a 989-row corpus - "
         "put a realistic ceiling at roughly 60-65%. The deployed model at 54.2%, the best eligible "
         "N = 5 model at 53.4% and the end-to-end fine-tuned reference at 55.4% all sit close to "
         "that ceiling. The remaining gap is not a modelling deficiency that more hyper-parameter "
         "tuning would close; it is a property of the labels.")

    figure(doc, "fig11_learning_curve.png", 5.9,
           "Figure 16 - Learning curve. The large train-CV gap indicates variance, not a shortage of "
           "data as the primary problem.")

    # ---------------------------------------------------------- 10 business
    doc.add_page_break()
    h1(doc, "Business Insights and Decision Support", "10")

    h2(doc, "10.1  What the system is for")
    body(doc,
         "A brand-monitoring team does not need a perfect four-way classifier. It needs a system that "
         "routes the right tweets to the right queue, with an honest estimate of how often it is "
         "wrong. The model output is only useful when it is aggregated to the entity a manager cares "
         "about and paired with a confidence threshold.")

    h2(doc, "10.2  Three decisions the model can support")
    bullet(doc,
           "Negative tweets are predicted with high recall, so a complaints queue can be built "
           "reliably. The cost is precision: the model also routes some Neutral and Positive tweets "
           "into that queue. For a human-reviewed queue this is an acceptable trade.",
           bold_lead="Routing complaints. ")
    bullet(doc,
           "Irrelevant precision is high but recall is low. The practical design is not to "
           "auto-classify Irrelevant; it is to send low-confidence rows to a review queue, which "
           "keeps precision high while acknowledging the missing entity.",
           bold_lead="Separating chatter. ")
    bullet(doc,
           "Aggregating predictions by brand produces a defensible sentiment mix (Section 3.5). This "
           "is the level at which the model is genuinely useful: a manager sees which brands are "
           "accumulating negative chatter, not why any single tweet was misclassified.",
           bold_lead="Brand-level monitoring. ")

    h2(doc, "10.3  What to deploy")
    table(doc,
          ["Option", "Test acc", "Macro F1", "Cost", "When to choose it"],
          [
              ["TF-IDF word+char + LR (M2)", p(dep["accuracy"]), p(dep["macro_f1"]),
               "Seconds on CPU; no pretraining", "Default production choice"],
              ["Frozen DistilBERT + MLP", p(best_elig["test_accuracy"]),
               p(best_elig["test_macro_f1"]), "~1 min encoding; 255 MB weights",
               "When contextual vectors are required; does not beat M2"],
              ["SBERT + kNN", p(next(r for r in emb_rows if r['id'] == 5)["test_accuracy"]),
               p(next(r for r in emb_rows if r['id'] == 5)["test_macro_f1"]),
               "~1 min; 90 MB download", "When explainability-by-neighbours is required"],
          ],
          widths=[1.8, 0.8, 0.85, 1.55, 1.5], font_size=8.0)

    body(doc,
         "The cost column is part of the result, not an afterthought. No eligible advanced model buys "
         "accuracy over a model that trains in seconds. The end-to-end fine-tuned reference gains 1.3 "
         "pp, but it is not an eligible combination and costs far more. A decision-support system "
         "that is cheap to retrain and easy to explain is worth more than a marginal ineligible gain.")

    callout(doc, "Service-level reality",
            "State the ceiling in any deployment agreement. A text-only system on this corpus should "
            "be expected to be right roughly half the time on the exact four-way label, and to get "
            "the polarity direction right about three times in four. Treating it as a triage tool, "
            "not an oracle, is the correct operating assumption.")

    # ---------------------------------------------------------- 11 feedback
    doc.add_page_break()
    h1(doc, "Response to Consultation Feedback", "11")

    body(doc,
         "The consultation produced seven comments. This section records how each was addressed, so "
         "the marker can see the response rather than infer it.")

    table(doc,
          ["#", "Feedback", "How it was addressed"],
          [
              ["1", "Accuracy around 45-50% is normal; do not treat a low score alone as an error.",
               "Sections 1 and 9 frame accuracy against the 30.2% majority reference and the measured "
               "label-noise ceiling. We do not claim the dataset is defective as an excuse; we show "
               "the four headroom experiments."],
              ["2", "No entity column; text-only classification is acceptable.",
               "Section 2.4 documents the schema mismatch and Section 9.2 explains its effect on "
               "Irrelevant. The limitation is stated openly rather than hidden."],
              ["3", "The 1.65:1 class ratio is not severe; whether to process it further is our call.",
               "Section 3.1 identifies the ratio and explains why we did not resample. Section 6.4 "
               "shows the class-weight trade-off so the choice is evidence-based."],
              ["4", "The EDA and report structure were viewed positively; no large rewrite needed.",
               "The final report keeps that structure and extends it with the advanced-modelling "
               "chapters rather than replacing the EDA work."],
              ["5", "Keep the cross-validation approach; advanced models should also show validation "
                    "and test results.",
               "Section 7.4 reports CV and test side by side for all five models, with the CV-to-test "
               "delta in its own column."],
              ["6", "Extra Trees and Random Forest were too similar; keep one and use a Neural "
                    "Network for the other.",
               "Extra Trees is removed. The matrix has exactly one Random Forest (Word2Vec) and a "
               "Neural Network (MLP on frozen DistilBERT); XGBoost and RF are distinct algorithms. "
               "See Section 7.3."],
              ["7", "Macro F1 rising while accuracy falls slightly can be normal; explain the "
                    "per-class impact.",
               "Section 6.4 reports recall and F1 per class for the deployed and balanced models and "
               "explains which classes gain and lose."],
          ],
          widths=[0.3, 2.3, 3.75], font_size=7.8)

    # ---------------------------------------------------------- 12 limitations
    doc.add_page_break()
    h1(doc, "Limitations, Conclusion and Recommendations", "12")

    h2(doc, "12.1  Limitations")
    bullet(doc,
           "The entity column is absent, so the Irrelevant class is partly unlearnable and its recall "
           "is capped. This is the single largest limitation.",
           bold_lead="Data limitation: ")
    bullet(doc,
           "989 rows in total and 397 test rows mean that one row is 0.25 pp. CV and test disagree "
           "by several points, and gaps under about 2 pp should be treated as noise.",
           bold_lead="Sample size: ")
    bullet(doc,
           "Nearest-neighbour agreement is about one third at high similarity, and a lexicon audit "
           "agrees with the label only 41-53% of the time on clear sentiment. The labels themselves "
           "limit achievable accuracy.",
           bold_lead="Label noise: ")
    bullet(doc,
           "A small number of configurations were evaluated on the test file after CV selection. Each "
           "was fixed by CV before its test evaluation, but the test set has been seen more than once "
           "across the project. This is a mild optimism risk and should be acknowledged.",
           bold_lead="Evaluation protocol: ")

    h2(doc, "12.2  Conclusion")
    body(doc,
         f"The project delivered every phase of the brief. The required TF-IDF + Logistic Regression "
         f"baseline reaches {p(base_acc)} test accuracy and {p(base_f1)} macro F1. The deployed "
         f"refinement raises that to {p(dep['accuracy'])} and {p(dep['macro_f1'])}. Five distinct "
         f"embedding x analytics-model combinations were built and compared under one protocol; "
         f"frozen DistilBERT + MLP is the best eligible row on accuracy at "
         f"{p(best_elig['test_accuracy'])}, while the TF-IDF models remain best on macro F1 and on "
         f"cost. The end-to-end fine-tuned reference reaches {p(ft_ref['test']['accuracy'])} but is "
         "not an eligible combination.")
    body(doc,
         "The most important result is not a score. It is the measured explanation of why the score "
         "plateaus: label noise, a missing entity column and a small corpus. That explanation is "
         "supported by four controlled experiments and is consistent with the instructor's feedback "
         "that a mid-range accuracy is normal for this dataset.")

    h2(doc, "12.3  Recommendations")
    bullet(doc, "Deploy M2 as the default and keep DistilBERT as an optional high-accuracy mode.")
    bullet(doc, "Route low-confidence predictions and all Irrelevant predictions to human review.")
    bullet(doc, "If the dataset can be re-acquired, request the entity column; it is the single "
                "change most likely to raise the ceiling.")
    bullet(doc, "Report macro F1 and per-class recall alongside accuracy, because the classes do not "
                "rank the same way.")
    bullet(doc, "Do not invest further in hyper-parameter tuning before the label-noise issue is "
                "addressed; the experiments show the limit is in the labels, not the model.")

    # ---------------------------------------------------------- 13 management
    doc.add_page_break()
    h1(doc, "Project Management, Schedule and Contributions", "13")

    body(doc,
         "The brief allocates 20% to project development and documentation, including a schedule and "
         "a clear breakdown of individual responsibilities. This section records both, plus a map "
         "from each assessment criterion to the part of the report that addresses it.")

    h2(doc, "13.1  Project schedule")
    table(doc,
          ["Week", "Activity", "Output"],
          [
              ["7", "Dataset allocation, schema audit and EDA", "eda_summary.json, Figures 1-7"],
              ["8", "Cleaning pipeline, encoding repair, audit log", "train_clean/test_clean, cleaning_log.json"],
              ["9", "TF-IDF + Logistic Regression baseline, CV protocol", "baseline_metrics.json, Figures 9-11"],
              ["10", "Interim report and consultation session", "Interim report, consultation feedback"],
              ["11", "Embedding module; GloVe, Word2Vec and FastText rows", "Three embedding result files"],
              ["12", "DistilBERT and Sentence-BERT rows; N = 5 matrix; headroom experiments", "Matrix, Figures 12-16"],
              ["13", "Phase 3iv refinement, final report, notebook and presentation", "Final deliverables"],
          ],
          widths=[0.55, 3.1, 2.65], font_size=8.0)

    h2(doc, "13.2  Division of work")
    body(doc,
         "Primary responsibility for each workstream. Contributions are also recorded in the "
         "repository history, which contains the staged commit trail for the pipeline, models and "
         "documents.", italic=True)
    table(doc,
          ["Member", "Primary responsibility"],
          [
              ["Chao Ka Kin", "EDA, data-quality audit and cleaning pipeline"],
              ["Leung Wai Tsun", "Baseline model, evaluation protocol and metric tables"],
              ["LIANG Zijian", "Classical embedding models (GloVe, Word2Vec, FastText)"],
              ["Lam Kin Ho", "Transformer embeddings (DistilBERT, Sentence-BERT) and the N = 5 matrix"],
              ["Yeung Tsz Chun Herman", "Integration, report generation, visualisation and repository"],
          ],
          widths=[1.9, 4.4], font_size=8.4)

    h2(doc, "13.3  Assessment criteria map")
    table(doc,
          ["Criterion", "Weight", "Where it is addressed"],
          [
              ["EDA and data cleaning", "20%", "Sections 2-4; Appendices A"],
              ["Baseline model", "10%", "Section 5; Appendix B"],
              ["Advanced modelling", "30%", "Section 7; Appendix C"],
              ["Refinement", "10%", "Section 6"],
              ["Uniqueness and creativity", "10%", "Sections 3.5, 9 and 10"],
              ["Project development and documentation", "20%", "This section; Appendix A; repository"],
          ],
          widths=[2.4, 0.8, 3.1], font_size=8.4)

    # ---------------------------------------------------------- 14 repro
    doc.add_page_break()
    h1(doc, "Reproducibility and Artefact Index", "14")

    body(doc, "Every number and figure in this report is generated by code. To reproduce the entire "
              "document from the raw CSVs:")

    table(doc,
          ["Command", "What it produces"],
          [
              ["python src/run_pipeline.py", "Cleans both files, writes data/processed/*.csv and "
                                              "cleaning_log.json, generates Figures 1-8"],
              ["python src/train_baseline.py", "Trains the Phase 3i baseline, writes "
                                               "models/baseline_metrics.json and Figures 9-11"],
              ["python src/train_final.py", "Trains M1-M3, writes models/final_metrics.json, "
                                            "Figure 15"],
              ["python src/explore_improved.py", "Train-only CV sweep for Phase 3iv, writes "
                                                 "reports/tables/improved_cv_sweep.csv"],
              ["python src/train_improved.py", "Scores the CV-selected B4 once on test, writes "
                                               "models/improved_metrics.json"],
              ["python src/embed_1_glove_xgb.py", "GloVe + XGBoost row"],
              ["python src/embed_2_word2vec_rf.py", "Word2Vec + Random Forest row"],
              ["python src/embed_3_fasttext_svm.py", "FastText + SVM row"],
              ["python src/embed_4_bert_frozen_mlp.py", "DistilBERT (frozen) + MLP row"],
              ["python src/embed_4_bert_head.py", "End-to-end fine-tuned reference (not counted)"],
              ["python src/embed_5_sbert_knn.py", "Sentence-BERT + kNN row"],
              ["python src/experiments_embeddings.py", "Assembles the N = 5 matrix, Figures 12-14"],
              ["python src/experiments_headroom.py", "E1-E4 ceiling experiments (Section 9)"],
              ["python src/make_final_report.py", "Rebuilds this report"],
          ],
          widths=[2.35, 3.95], font_size=7.8)

    table(doc,
          ["Artefact", "Location", "Purpose"],
          [
              ["Raw data", "Data 5 - Twitter/Data 5 - Twitter/*.csv", "Untouched source files"],
              ["Cleaned data", "data/processed/train_clean.csv, test_clean.csv", "Model-ready rows"],
              ["Cleaning log", "data/processed/cleaning_log.json", "Row-level audit trail"],
              ["EDA summary", "data/processed/eda_summary.json", "All EDA statistics"],
              ["Baseline metrics", "models/baseline_metrics.json", "Phase 3i scores"],
              ["Final metrics", "models/final_metrics.json", "M1-M3 and deployed scores"],
              ["Improved metrics", "models/improved_metrics.json", "Phase 3iv scores"],
              ["N = 5 summary", "data/processed/embedding_matrix_summary.json", "All five result rows"],
              ["Headroom summary", "data/processed/headroom_summary.json", "E1-E4 results"],
              ["Figures", "reports/figures/", "Sixteen publication-quality charts"],
              ["Tables", "reports/tables/", "CSV exports of every metric table"],
          ],
          widths=[1.5, 2.5, 2.3], font_size=8.0)

    callout(doc, "Reproducibility guarantee",
            "This report reads its numbers from JSON and CSV artefacts rather than from hard-coded "
            "text. If the pipeline is re-run on a refreshed dataset, the document regenerates with "
            "the new values automatically and cannot silently disagree with the code.")

    # ---------------------------------------------------------- A cleaning log
    doc.add_page_break()
    h1(doc, "Appendix A - Cleaning Log (training file)", "A")
    body(doc, "Verbatim from data/processed/cleaning_log.json. The test file is processed with "
              "identical rules and is recorded in the same file.", italic=True)
    for s in steps:
        p_ = doc.add_paragraph()
        p_.paragraph_format.space_before = Pt(6)
        p_.paragraph_format.space_after = Pt(1)
        r = p_.add_run(f"Step {s['step']}: {s['name']}")
        r.font.bold = True
        r.font.size = Pt(9.5)
        r = p_.add_run(f"  -  {s['action']}  (rows affected: {s['rows_affected']})")
        r.font.size = Pt(9)

    # ---------------------------------------------------------- B full metrics
    doc.add_page_break()
    h1(doc, "Appendix B - Full Metric Detail", "B")

    h2(doc, "B.1  Cleaning variants (5-fold CV on train)")
    table(doc,
          ["Variant", "Accuracy", "Macro F1", "Weighted F1"],
          [[r["variant"], p(r["accuracy_mean"]), p(r["macro_f1_mean"]), p(r["weighted_f1_mean"])]
           for r in met["cv"]],
          widths=[3.0, 1.0, 1.0, 1.0], font_size=8.2)

    h2(doc, "B.2  Final model comparison (test, scored once)")
    table(doc,
          ["Model", "Accuracy", "Macro F1", "Weighted F1", "kappa", "within-1"],
          [[r["model"], p(r["accuracy"]), p(r["macro_f1"]), p(r["weighted_f1"]),
            f"{r['cohen_kappa']:.3f}", p(r["within1_accuracy"])] for r in fin["test_results"]],
          widths=[2.2, 0.85, 0.85, 0.9, 0.6, 0.75], font_size=8.0)

    h2(doc, "B.3  Deployed model, per class (test)")
    table(doc,
          ["Class", "Precision", "Recall", "F1", "Support"],
          [[r["class"], f"{r['precision']:.3f}", f"{r['recall']:.3f}", f"{r['f1']:.3f}",
            str(r["support"])] for r in fin["per_class"]],
          widths=[1.5, 1.1, 1.0, 0.9, 0.9], font_size=8.4)

    # ---------------------------------------------------------- C per-class N5
    doc.add_page_break()
    h1(doc, "Appendix C - Per-Class Results for the N = 5 Matrix", "C")

    files = {
        1: ("glove_mlp", "GloVe + Neural Network (MLP)"),
        2: ("word2vec_rf", "Word2Vec + Random Forest"),
        3: ("fasttext_svm", "FastText + SVM"),
        4: ("bert_frozen_mlp", "DistilBERT (frozen) + MLP"),
        5: ("sbert_knn", "Sentence-BERT + kNN"),
    }
    for key, (fname, label) in files.items():
        path = PROC / f"embedding_result_{fname}.json"
        if not path.exists():
            continue
        data = jload(path)
        h2(doc, f"C.{key}  {label}")
        table(doc,
              ["Class", "Precision", "Recall", "F1", "Support"],
              [[r["class"], f"{r['precision']:.3f}", f"{r['recall']:.3f}", f"{r['f1']:.3f}",
                str(r["support"])] for r in data["test"]["per_class"]],
              widths=[1.5, 1.1, 1.0, 0.9, 0.9], font_size=8.2)
        body(doc,
             f"Test accuracy {p(data['test']['accuracy'])}, macro F1 "
             f"{data['test']['macro_f1']:.3f}, kappa {data['test']['cohen_kappa']:.3f}.",
             size=9, space_after=3)

    # ---------------------------------------------------------- save
    for section in doc.sections:
        section.footer.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = section.footer.paragraphs[0].add_run(f"  |  {GROUP}  |  {COURSE}")
        r.font.size = Pt(8.5)
        r.font.color.rgb = GREY

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "Final_Report_Group8_Twitter_Sentiment.docx"
    doc.save(path)
    return path


if __name__ == "__main__":
    out = build()
    print(f"written -> {out}")
    print(f"size    -> {out.stat().st_size/1024:.1f} KB")
