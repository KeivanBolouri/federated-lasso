# Revision record and required author review

Revision date: 11 September 2026. Starting GitHub commit: `c3a6470d21e0f7d8349c1b5b11d62e2c69d1e026`.

This revision responds to a scientific and reproducibility audit of the submitted 21-page manuscript. It is a revised research draft for the author's independent review, not a certification of acceptance or publisher-policy compliance.

## Scientific and computational changes

1. Corrected the literature comparison: Yuan et al. (2021) and Bao et al. (2022) already study sparse-feature/support recovery. The contribution is narrowed to the cyclic-CD calculation and the stated computational comparisons.
2. Corrected the orthogonal support and stopping statements, completed the cancellation argument, and gave explicit examples separating a sparse average from a pooled-Lasso solution. The general fixed-point statement is not presented as a convergence theorem.
3. Added local one-SE penalty controls for FedAvg and One-shot. Removed claims that all one-SE-labelled methods have identical search spaces or that fixed-penalty ADMM uses one-SE tuning.
4. Recomputed the 200-replicate studies with complete FedAvg-P candidate-fit accounting. Selected-fit and full-selection costs are both retained. Added paired epoch contrasts and a consistent zero-precision convention for empty models.
5. Replaced the former blanket suppression of final-solver warnings with strict final-solve convergence checks and adaptive iteration limits. Six of the 3,600 site/rule fits checked in the old study exceeded the former 10,000-iteration cap; the corresponding one-shot baselines were recomputed. Original inputs and earlier results remain in git history.
6. Added a deterministic full-participation/full-gradient specialization of the published FedDualAvg algorithm, tested independently against analytical and centralized recurrences. Fifty paired replicates per design compare fixed selected-path communication budgets. The additional learning-rate candidate paths and shared setup costs are explicitly exposed; this is not claimed as equal end-to-end tuning cost.
7. Reclassified the original node files as a fixed **synthetic** benchmark, as confirmed by the author. Their original generator, seed and true coefficient vector were not supplied. The existing files are preserved and hashed. They are not described as a real observational application.
8. Added a documented public diabetes-data example with 100 prespecified splits, training-only preprocessing, held-out test evaluation and explicit overlap/stability metrics. All capped iterates remain in the summaries; capped averaging/P runs are disclosed. This is an illustration on simulated sites, not a multicentre clinical validation.
9. Replaced local machine paths with repository-relative paths, recorded dependencies and metadata, added a single reproduction command, rebuilt the paper and prepared its LaTeX source archive.
10. Shortened the abstract to 150 words and included the requested author, funding, competing-interest, data-availability and AI-use information. The author confirmed no dedicated funding.
11. Kept the main article within the supplied word guide (approximately 9,800 words, including tables and references) and moved the complete 147-row simulation table to Supplementary Table S1. The source ZIP builds both documents.

## Author work before journal submission

The supplied JSCS instructions require an unstructured 150-word abstract, no more than 10 keywords, a word count, funding and disclosure statements, and a declaration of AI use. They allow initial format-free scholarly submissions with embedded figures/tables and require a PDF plus a LaTeX source ZIP when using LaTeX.

The current [Taylor & Francis AI policy](https://authorservices.taylorandfrancis.com/editorial-policies/using-ai-in-your-research-and-manuscript-preparations/) permits software coding and language assistance but restricts AI creation of first manuscript drafts/sections and scientific arguments. It also calls for tool/version disclosure and author confirmations of accuracy, originality, terms suitability and responsibility. This revision involved more than spelling correction: it includes proposed scientific revisions, new analysis code, additional experiments and proposed descriptions. It must not be described as language-only AI assistance.

Before submission, the author needs to independently verify the mathematics, literature, code and reported results, and author the new scientific arguments/sections in accordance with the journal's policy. In particular, review the new budget comparison and public-data section and the revised interpretations of tuning, convergence and selection. Obtain editorial clarification if the extent of assistance is incompatible with the policy; a generic statement of responsibility alone does not remove that restriction.

The AI-use declaration in `ms/main.tex` is deliberately candid and provisional. It does not claim the author has already completed independent review or checked tool terms. The exact model/version identifiers for this session and earlier AI-assisted preparation were not supplied; the author should complete the required tool/version details from their records rather than invent them. Confirm funding, competing interests, affiliation and all final submission-form declarations personally.

A separate optional mathematical note, `AUTHOR_REVIEW_orthogonal_gap.md`, contains a proposed elementary identity. It is excluded from the manuscript and submission ZIP. It is not established as a new contribution or approved for publication.

## Interpretation retained after correction

- Averaging can enlarge supports without making every coefficient nonzero.
- Improved penalized training objectives do not automatically imply better F1 or test MSE.
- Local penalty rules explain part, but not all, of the apparent benefit of server thresholding in the tested simulations.
- Post-fit ST and per-round P have distinct epoch responses and distinct tuning costs.
- A nearly fixed number selected does not establish stable variable identities.
- Iteration-cap outcomes and unknown benchmark provenance are reported rather than omitted.

All code, data descriptions, result files and the manuscript are supplied so these points can be checked. Additional robustness across sample sizes, signal strengths and site counts would still strengthen generalizability; no broad superiority or guaranteed acceptance claim is made.
