# Where the COSMOS-AI data lives, and why

This note records the measured size of the project, what GitHub can and cannot hold, and the
options for the remainder. It exists so that the choice is auditable rather than implicit.

Measured on 2026-11-02, on branch `q1-reentry-rebuild`.

## 1. What the project actually weighs

Excluding `.git` and `.venv`:

| Role | Files | Size |
|---|---|---|
| Source code, documentation, manuscripts, configuration | 762 | 0.13 GiB |
| `data/` — processed and derived datasets | 15,562 | 45.84 GiB |
| `data/` — authoritative raw archives | 36,934 | 21.26 GiB |
| `outputs/` — result tables, figures, manifests | 1,800 | 31.15 GiB |
| `outputs/` — out-of-fold score dumps and feature arrays | 138 | 72.90 GiB |
| `outputs/` — model checkpoints | 199 | 0.90 GiB |
| **Total** | **55,395** | **172.19 GiB** |

The two largest single items are `outputs/benchmarks/telemetry_esa_adb/` (66 GiB of
out-of-fold score arrays for the ESA-ADB companion study) and
`data/feature2_debris/spotgeo/` (45 GiB, the extracted SpotGEO v2 image sequences).

## 2. The limits that apply

| Limit | Value | Consequence here |
|---|---|---|
| Ordinary Git, per file | warning above 50 MB, **rejected above 100 MB** | 198 files on disk exceed 100 MB and cannot go in ordinary Git |
| Git LFS, per file (Free / Pro / Team) | **2 GB** | 4 archives are 3.5–4.0 GiB and cannot be stored in LFS at all without splitting |
| Git LFS storage, free tier | **1 GiB** | holds the committed dataset set (~0.7 GiB) and nothing larger |
| Git LFS bandwidth, free tier | **1 GiB / month** | about one full `git lfs pull` per month; this repository is **public**, so any visitor consumes the same quota |
| GitHub data pack | $5 / month per 50 GiB storage **and** 50 GiB bandwidth | 172 GiB would need 4 packs for storage alone, before any download traffic |
| Recommended repository size | under 1 GiB; warnings above 5 GiB | the committed set (~0.85 GiB total) sits just inside this |

Two of these are hard stops rather than costs: a file above 2 GB cannot enter GitHub LFS on
this plan at any price, and a file above 100 MB cannot enter ordinary Git.

## 3. What was committed, and what was not

Committed, in LFS: ESA Kelvins CAC (594 MiB, CC BY 4.0), NASA SMAP/MSL (179 MiB, BSD-style
redistribution licence), GCAT (12 MiB, CC BY 4.0), Kelvins Debris Origin (8.4 MiB, CC BY 4.0),
OPSSAT-AD (8.3 MiB, CC BY 4.0), CelesTrak snapshots (17 MiB), the re-entry cohort and NRLMSIS
density table (7 MiB), and the result arrays of the five benchmarks the manuscript reports.

Not committed, with the reason separated by kind:

**Blocked by the 2 GB per-file LFS limit** — SpotGEO v2 (`SpotGEOv2.zip`, 4.0 GiB) and
ESA-ADB missions 1–3 (3.6, 3.9, 3.5 GiB). These would have to be split into parts to enter
LFS, which would stop them being byte-identical to the checksums in
`data/DATASET_MANIFEST.csv`. Splitting is not done here; a checksum that no longer matches the
upstream artefact is worse than an honest pointer to the upstream source.

**Redistribution terms unclear or restrictive** — ESA-ADB (the Zenodo record states no licence;
CC BY 3.0 IGO is only asserted in the DMLR paper), SpaceSense-Bench (CC BY-NC 4.0 on the
Hugging Face card, no licence file in the source repository), SPARK (access by request).
On a **public** repository these should not be redistributed without checking with the
upstream authors.

**Regenerable derived artefacts** — `outputs/benchmarks/telemetry_esa_adb/` (66 GiB) and
`outputs/experiments/` (37 GiB). The metrics computed from these *are* committed; the raw
per-sample score dumps are not. They are reproducible from the recorded configurations, and
they are derived from a dataset that is itself not redistributed here.

Note that CelesTrak publishes under a usage policy rather than an explicit open licence. The
snapshots are small, carry no credentials, and were already present in this repository's
history before this upload; they are retained for reproducibility of the dated forecast. If
CelesTrak's terms are read as prohibiting redistribution, these are the files to remove.

## 4. If the full 172 GiB must be published

Ranked by how well each fits this project.

**a. DVC with cloud object storage — recommended.** Keep Git for code and small result
tables; keep DVC pointers in Git; keep the bytes in S3, Google Cloud Storage, Azure Blob or
Backblaze B2. There is no per-file size limit, storage is roughly $2–4 per TB-month on B2 and
$4 per TB-month on S3 standard, and egress is paid only when someone downloads. Reproducibility
improves rather than degrades, because `dvc.lock` pins content hashes.

```bash
pip install 'dvc[s3]'
dvc init
dvc remote add -d store s3://your-bucket/cosmos-ai
dvc add data/feature2_debris/spotgeo data/feature1_health/esa_adb
git add data/feature2_debris/spotgeo.dvc data/feature1_health/esa_adb.dvc .dvc/config
dvc push
```

**b. Zenodo deposition — best for the manuscript.** Zenodo accepts 50 GB per record by
default (more on request), mints a DOI, and is the citable archive a journal expects. Deposit
the derived cohort, the result arrays and the figures; cite the upstream datasets rather than
re-hosting them. This is the right home for the artefacts that support the paper.

**c. Hugging Face Hub datasets.** Free public hosting, LFS-backed, no 2 GB per-file ceiling in
practice. Suitable for the derived telemetry and cohort tables. Still bound by the upstream
licences, so it does not resolve the ESA-ADB or SpaceSense-Bench question.

**d. GitHub data packs.** 4 packs ($20/month) covers 172 GiB of storage and 200 GiB of
monthly bandwidth — but **does not raise the 2 GB per-file limit**, so the four large archives
remain impossible. Paying does not solve the blocking problem.

## 5. What does not work

`git rm --cached` is sometimes suggested for oversized files. It removes a file from the index
and from future commits, but the blob stays in every commit that already contained it, so the
repository still carries the bytes and GitHub still rejects the push. Only history rewriting
(`git lfs migrate import`, `git filter-repo`) removes a blob from past commits, and that
rewrites every commit hash after the earliest affected one.

That operation was **not needed here**: the largest blob in this repository's existing history
is 12.69 MB, and no blob exceeds 50 MB, so no migration was performed and no history was
rewritten.
