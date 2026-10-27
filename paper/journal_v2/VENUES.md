# Where to submit: Scopus-indexed journals related to AI (prepared 2026-10-25)

**What has been checked.** ISSN, publisher, open-access status, APC and 2-year mean citedness come from OpenAlex, fetched on 2026-10-25 and saved in `research_design/references/openalex_venues_2026-10-25.json`. The 2-year mean citedness is OpenAlex's analogue of an impact factor. It is not the Clarivate impact factor and not the SJR.

**What has not been checked.** Scopus indexing status, SJR quartile and review times could not be verified from this machine: scimagojr.com returns a Cloudflare challenge, and Scopus needs a login. Before submitting, check each journal in:

- the **Scopus Sources list** (scopus.com/sources);
- **SJR** (scimagojr.com): the quartile is per subject category and changes yearly;
- the journal's own page, for "review speed" or "time to first decision". Elsevier and Springer journals publish these.

Quartiles and speeds are therefore **not** stated below.

| Journal | Publisher | ISSN | Open access | APC (US$, if OA) | 2-yr cites | Fit for this paper |
|---|---|---|---|---|---|---|
| **Engineering Applications of Artificial Intelligence** | Elsevier | 0952-1976 | hybrid | 3,200 | 7.87 | **Best fit.** Applied ML with rigorous evaluation on real engineering data; negative results on deep models are acceptable if the protocol is strong |
| Expert Systems with Applications | Elsevier | 0957-4174 | hybrid | 3,500 | 7.93 | Broad applied-AI journal, very high volume |
| Knowledge-Based Systems | Elsevier | 0950-7051 | hybrid | 3,430 | 6.87 | Possible; prefers methodological novelty |
| Applied Soft Computing | Elsevier | 1568-4946 | hybrid | 3,730 | 6.47 | Possible; applied ML and decision systems |
| Neural Computing and Applications | Springer | 0941-0643 | hybrid | 3,290 | 6.60 | Possible; neural-network focus, so stress the TCN comparison |
| Applied Intelligence | Springer | 0924-669X | hybrid | 3,190 | 4.66 | Possible |
| IEEE Access | IEEE | 2169-3536 | gold OA only | 2,075 | 6.56 | Broad mega-journal known for quick decisions; Heydari et al. 2026 (ESA-ADB) appeared here |
| Machine Learning with Applications | Elsevier | 2666-8270 | gold OA | 2,340 | 5.31 | Newer OA journal; check its Scopus status first |
| Scientific Reports | Nature Portfolio | 2045-2322 | gold OA | 2,690 | 5.32 | Broad; judges soundness rather than novelty |
| Acta Astronautica | Elsevier | 0094-5765 | hybrid | 3,350 | 4.12 | Space-operations audience; ML for spacecraft operations appears regularly |
| Aerospace Science and Technology | Elsevier | 1270-9638 | hybrid | 4,360 | 4.45 | Published Brancato et al. on ESA-ADB, so you must discuss that paper in full |
| Reliability Engineering & System Safety | Elsevier | 0951-8320 | hybrid | 4,830 | 11.14 | Possible if framed as alarm management / PHM; expect comparison with TC-CAD |
| Advances in Space Research | Elsevier (COSPAR) | 0273-1177 | hybrid | 2,780 | 3.28 | Fallback with a space audience |

"Hybrid" means a subscription journal with an optional OA fee. **Publishing on the subscription route costs nothing**, while gold-OA journals always charge the APC.

## Recommended order

1. **Engineering Applications of Artificial Intelligence.** It is AI-focused and matches the paper's content: a deep model against strong baselines, pre-registration and an operational alarm layer.
2. **IEEE Access**, if a quick decision matters more than prestige. It is gold OA, so the APC applies.
3. **Acta Astronautica**, if you prefer a space-systems readership.

## About "submitting early"

No legitimate journal guarantees a fast acceptance. Avoid any journal that promises publication within days, or that charges before peer review. Also check that the journal is not in Scopus' list of discontinued titles. Submit to one journal at a time: simultaneous submission is not allowed.

## Before submitting

- **Generative-AI disclosure.** Elsevier, Springer Nature and IEEE all require authors to disclose the use of generative AI in preparing a manuscript, and AI tools may not be listed as authors. The manuscript contains a disclosure statement to complete. Keep it, and name the tool honestly.
- **Complete the placeholders**: authors, affiliations, CRediT, funding and competing interests.
- **Read the closest papers yourself** (Brancato et al.; TC-CAD), so that you can answer reviewers in your own words.
