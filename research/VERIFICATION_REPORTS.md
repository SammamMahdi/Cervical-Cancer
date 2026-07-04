# Verification — ChatGPT *CerviWISE* Research Report (cervical cancer)

**Method:** 9 independent web-grounded fact-checkers re-fetched each cited paper/dataset from its primary source (arXiv / publisher / PMC / Mendeley / CrossRef) and confirmed existence, attribution, and the specific numbers. Verified 2026-06-21.

**Bottom line:** The report is **well-grounded — no fabricated papers and no wrong numbers.** Two items must be fixed before relying on it: one fabricated *citation wrapper* and one false absolute claim.

---

## Claim-by-claim verdict — 9 claims: **7 confirmed · 1 misattributed · 1 overstated**

| # | Claim | Verdict | Note |
|---|-------|---------|------|
| C1 | UniCAS, Cell Rep Med 2026, 48,532 WSI pretrain, cancer AUC ~0.926 | ✅ confirmed | AUC 92.60% / 48,532 WSIs exact |
| C2 | Smart-CCS, ~127K slides, internal AUC 0.965, external 0.950 / 6 sites | ✅ confirmed | corpus is **CCS-127K = 127,471 WSIs, 48 centers** |
| C3 | CausalCervixNet, Taghados et al., **BMC Cancer 2025**, 99.14% SIPaKMeD / 97.31% Herlev | ✅ confirmed | venue + author + numbers all correct (BMC Cancer 25:607) |
| C4 | CCanNet, Mehedi et al., Sci Rep 14:29446 (2024), 98.53%, 1,274,663 params | ✅ confirmed | exact |
| C5 | CerviSegNet-DistillPlus, Kang & Li, IEEE Access 2024 | ✅ confirmed | Jie Kang & Ning Li, IEEE Access v12 |
| C6 | **HybridKANViTAE, "Zhang et al., Digital Health, 2024"**, 99.68% SIPaKMeD / 91.69% APCData | ⚠️ **misattributed (HIGH)** | model + numbers + APCData are **real**, but the **citation is fabricated** — see below |
| C7 | CRIC Cervix Collection: 11,534 cells / 400 smears | ✅ confirmed | Rezende et al., Sci Data 2021 |
| C8 | SIPaKMeD 4,049 (5-cls) · Mendeley LBC 963 (4-cls) · Herlev 917 | ✅ confirmed | all three exact |
| C9 | **"No cervical AI study reports reliability diagrams / ECE"** | ❌ **overstated/false (HIGH)** | counterexamples exist — see below |

---

## Must-fix

- **C6 — fabricated citation wrapper.** The result is real but the citation is invented. The true source is a **Research Square preprint** *"Federated Learning-Based Cervical Cancer Classification Using a Novel Hybrid KAN-ViT-Autoencoder Architecture"* (rs-8457292/v1, posted **17 Feb 2026, not peer-reviewed**). There is **no "Zhang et al." author**, **no "Digital Health" journal**, and **not 2024**; "HybridKANViTAE" is the model's nickname, not the paper title. Keep the metrics (99.68% SIPaKMeD, 91.69% APCData — both real; APCData is a real Mendeley dataset, 425 images / 3,619 cells), but re-cite it as a 2026 preprint.

- **C9 — false absolute.** Calibration is **not** absent in cervical AI. Counterexamples: **TopotiCerv** (temperature scaling, 2026), a 2026 evidential-DL study reporting ECE/Brier/NLL on SPKMED, a fuzzy-ensemble model (ECE 0.030, Brier 0.078), and SMART-HPV (reliability curves). Soften to: *"calibration reporting remains rare/under-used"* — which is true and defensible.

---

## Takeaways
1. **Trust the numbers** — every metric, parameter count, dataset size, and corpus size checked out exactly against primary sources.
2. **The weakness is provenance, not data:** an invented author/venue wrapper (C6) and a false absolute (C9). Adopt a rule: distinguish *"peer-reviewed / in official proceedings"* from *"self-reported arXiv metadata or preprint."*
3. **One useful correction to our own earlier review:** CerviWISE correctly pins CausalCervixNet to **BMC Cancer 2025 (Taghados et al.)** — more complete than the venue we'd left blank.
