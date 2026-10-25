# Candidate venues for the telemetry paper (Scopus-indexed, aiming at Q1)

Written 2026-10-24. Scopus quartiles come from SJR (scimagojr.com). That site is behind
a Cloudflare challenge from this machine, so the **quartile column comes from general
knowledge and has not been verified here**. Check every row on scimagojr.com (or Scopus
Sources) on the day you submit. Quartiles are assigned per subject category, and they
change every year.

The citation figures were fetched from OpenAlex on 2026-10-24
(`references/openalex_venues_2026-10-24.json`). "2-yr cites" is OpenAlex's 2-year mean
citedness, a proxy similar to the impact factor.

| Journal | Publisher | Expected SJR quartile (verify) | 2-yr cites | h-index | APC (OA option) | Fit for this paper |
|---|---|---|---|---|---|---|
| **Engineering Applications of Artificial Intelligence** | Elsevier | Q1 (AI; Control & Systems Eng.; Electrical Eng.) | 7.87 | 187 | US$3,200 (subscription route free) | **best fit**: an applied deep-learning method with rigorous evaluation on real industrial or space data |
| Acta Astronautica | Elsevier | Q1 (Aerospace Eng.) | 4.12 | 133 | US$3,350 | good: space operations audience; ML for spacecraft operations appears regularly |
| Aerospace Science and Technology | Elsevier | Q1 (Aerospace Eng.) | 4.45 | 137 | US$4,360 | good: published Brancato et al. 2026 on ESA-ADB, so the topic fits. That group may well review the paper |
| Reliability Engineering & System Safety | Elsevier | Q1 (Safety, Risk, Reliability; Industrial Eng.) | 11.14 | 226 | US$4,830 | possible if framed as PHM / alarm management. TC-CAD's keywords suggest it was submitted here: expect a direct comparison |
| Expert Systems with Applications | Elsevier | Q1 (AI; Engineering) | 7.93 | 352 | US$3,500 | broad, high volume |
| IEEE Transactions on Aerospace and Electronic Systems | IEEE | Q1 (Aerospace Eng.; Electrical Eng.) | 4.99 | 229 | US$2,645 | demanding; expects signal-processing depth |
| Advances in Space Research | Elsevier (COSPAR) | Q1/Q2 depending on category | 3.28 | 151 | US$2,780 | acceptable fallback |
| IEEE Access | IEEE | Q1/Q2 depending on category | 6.56 | 379 | US$2,075 (gold OA only) | fast, but a mega-journal; Heydari et al. 2026 is here |
| Chinese Journal of Aeronautics | Elsevier | Q1 (Aerospace Eng.) | 5.44 | 106 | US$2,000 | fit is weaker for operations and ML |
| Journal of Space Safety Engineering | Elsevier | probably Q2/Q3 | 1.80 | 26 | US$2,450 | not Q1 |
| Journal of Aerospace Information Systems | AIAA | probably Q2/Q3 | 1.78 | 40 | — | not Q1 |

## Recommendation

1. **Engineering Applications of Artificial Intelligence** first. Its readers expect what
   the paper offers: a deep model, strong classical and tree baselines, a statistical
   protocol, and a deployable alarm layer.
2. **Acta Astronautica** if the confirmatory result is positive on Mission 1 but modest.
3. **Aerospace Science and Technology** as an alternative. Cite and discuss Brancato et
   al. 2026 carefully: their Mission 1 number (0.902) could not be checked here.

None of these journals accepts a paper only because it is "Q1-worthy". Acceptance depends on
the confirmatory result (H4–H6) and on positioning against TC-CAD, Brancato et al. and
Heydari et al. (`NOVELTY_NOTE.md`).
