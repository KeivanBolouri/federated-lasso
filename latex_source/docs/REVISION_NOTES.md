# Revision following the editorial comments

- Clarified synthetic-data provenance and consistently used “not supplied”.
- Clarified that each design uses all three seed blocks, for 200 replicates.
- Completed the corresponding-author postal affiliation and removed date lines.
- Shortened the pooled-objective proof and removed repetitive caveats while retaining the qualifications needed for interpretation.
- Added an orthogonal-design illustration in Supplementary Table S3: 1,000 replicates for each of three site counts; E=1,3,20 fitted with the existing solver.
- Added the Table S1 citation for coefficient error, Test MSE MCSEs to Table 1, and E=2 rows to Table 4.
- Completed bibliography author metadata, corrected the Zhu–Han chapter entry, printed verified available DOIs, and harmonised “specialisation”.
- Set hyperlinks in both PDFs to black.
- Included data, recorded results, scripts, hashes and relative-path reproduction instructions in the source archive.

## Verification

Tables 3 and S2 and their budget summaries reproduced byte-for-byte from the archived replicate-level results. A fresh IID seed-5000 budget run matched 31 archived rows across 21 numerical columns to maximum absolute difference 1.78e-14. All 147 main method/schedule groups had the exact prescribed 200-seed set; all 93 budget groups had the prescribed 50 seeds. All 12 original node-file SHA-256 checksums matched. Eleven existing solver checks passed. The large main simulation study was not rerun during this revision; its complete archived results were re-aggregated and checked. The new orthogonal experiment was computed and independently rerun.

The original numerical findings and raw results are retained. The orthogonal experiment is additional evidence under its stated fixed-penalty assumptions. The detailed numerical audit is `numerical_verification.json`; orthogonal experiment checks are in `../code/orthogonal_metadata.json`.
