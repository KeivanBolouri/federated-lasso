# Manuscript revision record

## Continuation of the submission-readiness revision — 11 September 2026

Resumed from committed version `2edff8b7f7f2f3a165445663650569763cf7d583`. The preceding scientific revision (`c697f93`) and the author's removal of the manuscript AI declaration (`2edff8b`) were already pushed. Their correct work and numerical experiments were preserved.

### Changes in this continuation

1. **Central claim and FedDualAvg.** Rewrote the 150-word abstract and connected the introduction, methods, budget results and discussion. The article diagnoses CD averaging. Adapted FedDualAvg is an established alternative for the pooled penalised objective; ST and P are modifications to existing CD workflows. The article does not claim that these retrofits outperform FedDualAvg.
2. **Tuning and budgets.** Kept the distinction between equal selected-path uploads and unequal total tuning costs. Added a supplementary comparison at a common total-upload cap using the existing paired runs, including all three FedDualAvg learning-rate paths and setup uploads. The largest affordable recorded checkpoint is used; unused budget is disclosed. No interpolation or equal-spending claim is made.
3. **Diabetes illustration.** Removed the low-dimensional diabetes section from the article. It does not directly validate the high-dimensional support-recovery question. Code, source data, split assignments, convergence diagnostics and results remain in the repository; they are excluded from the manuscript source archive.
4. **Fixed benchmark provenance.** Retained the explicit synthetic classification and missing generator, seed and true-coefficient information. Clarified that fixed-input reproduction is possible while new draws from the original generator are unavailable. Agreement with pooled support is not truth recovery.
5. **Matched-tuning interpretation.** Made explicit that ST versus FedAvg-1SE compares different tuning procedures and does not isolate a server-threshold effect at identical local penalties or tuning opportunity.
6. **Presentation.** Removed AI-production notes from captions and the visible title-page word count. Retained the author's prior declaration removal. Added an explicit supplementary-material statement and named the separate document Supplement S1. Updated repository descriptions to match the delivered article.
7. **Packaging and verification.** Regenerated outputs from stored results, checked cost and metric arithmetic, ran numerical tests, and rebuilt both PDFs and the source archive. The archive includes the transitive source dependencies of the two documents. Detailed evidence is in `VALIDATION_20260911.md`.

### Correct earlier work retained

- Corollary 1 concerns a fixed common penalty, fixed total sample size, independent Gaussian site statistics and feasible balanced orthogonal partitions. It does not establish monotonicity for tuned penalties, arbitrary repartitions or positive activity thresholds.
- Orthogonal support statements include the cancellation qualification; the general fixed-point characterization is not a convergence theorem.
- Precision and F1 for an empty selected model are zero, and all simulation replicates are included in means.
- Local one-SE controls, complete P candidate-fit costs, final-solver convergence checks and the documented deterministic FedDualAvg specialization remain intact.
- The 600 main simulation datasets, 150 paired budget-study datasets and original fixed benchmark inputs were reused. This continuation did not restart those completed experiments.

## Earlier revision

Commit `c697f93` corrected literature comparisons, orthogonal theory, tuning controls, convergence checks and candidate-fit cost accounting; reran the main simulations; added the budget and archived diabetes experiments; and supplied source, raw outputs and metadata. Commit `2edff8b` removed the specified AI declaration at the author's request. The full earlier text and artifacts remain available in git history.
