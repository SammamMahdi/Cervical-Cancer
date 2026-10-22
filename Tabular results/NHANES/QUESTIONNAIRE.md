# NHANES Cervical HPV-Risk Questionnaire (female respondents)

Administered items pooled across NHANES 2003-2016. **n = 12,634** women with a valid high-risk-HPV swab result; **24.6%** positive. Each item shows its question, answer coding, source NHANES variable, and % answered in this cohort.

> **Outcome (predicted, not asked):** high-risk HPV DNA on the self-collected vaginal swab (any oncogenic type 16/18/31/33/35/39/45/51/52/56/58/59/66/68).

> **Deliberately EXCLUDED:** `SXQ753` *"Ever told by a doctor you had HPV"* — a self-report of the outcome itself; using it is the target-leakage behind the published 0.90-0.95 AUC claims.


## Demographics

**Q1. What is your age?**  
*Answers:* age in years (18-59)  
*Variable:* `age` (DEMO) · *answered:* 100%

**Q2. Race / Hispanic origin**  
*Answers:* 1=MexicanAmerican 2=OtherHispanic 3=White 4=Black 5=Other  
*Variable:* `race` (DEMO) · *answered:* 100%

**Q3. Highest education completed**  
*Answers:* 1=<9th 2=9-11th 3=HS/GED 4=SomeCollege 5=CollegeGrad+  
*Variable:* `education` (DEMO) · *answered:* 91%

**Q4. Marital status**  
*Answers:* 1=Married 2=Widowed 3=Divorced 4=Separated 5=NeverMarried 6=LivingWithPartner  
*Variable:* `marital` (DEMO) · *answered:* 95%

**Q5. Ratio of family income to the poverty threshold**  
*Answers:* 0.0-5.0 (higher = wealthier)  
*Variable:* `poverty_ratio` (DEMO) · *answered:* 93%


## Sexual behaviour

**Q6. Ever had vaginal, anal, or oral sex?**  
*Answers:* 1=Yes 2=No  
*Variable:* `ever_sex` (SXQ) · *answered:* 76%

**Q7. How old were you when you had sex for the first time?**  
*Answers:* age in years  
*Variable:* `age_first_sex` (SXQ) · *answered:* 72%

**Q8. How old were you when you first performed oral sex?**  
*Answers:* age in years (asked 2009+)  
*Variable:* `age_first_oral` (SXQ) · *answered:* 41%

**Q9. Ever performed oral sex on a man?**  
*Answers:* 1=Yes 2=No (asked 2009+)  
*Variable:* `ever_oral_man` (SXQ) · *answered:* 53%

**Q10. Ever had anal sex with a man?**  
*Answers:* 1=Yes 2=No (asked 2009+)  
*Variable:* `ever_anal_man` (SXQ) · *answered:* 53%

**Q11. How many male sex partners have you had in your lifetime?**  
*Answers:* count  
*Variable:* `n_partners_life` (SXQ) · *answered:* 74%

**Q12. How many male sex partners in the past 12 months?**  
*Answers:* count  
*Variable:* `n_partners_year` (SXQ) · *answered:* 53%

**Q13. How many male vaginal-sex partners in your lifetime?**  
*Answers:* count (asked 2009+)  
*Variable:* `n_vaginal_partners_life` (SXQ) · *answered:* 49%

**Q14. How many male oral-sex partners in your lifetime?**  
*Answers:* count (asked 2009+)  
*Variable:* `n_oral_partners_life` (SXQ) · *answered:* 41%

**Q15. In the past year, how many times did you have sex without a condom?**  
*Answers:* count  
*Variable:* `condomless_year` (SXQ) · *answered:* 61%

**Q16. Have you had sex with a new partner in the past year?**  
*Answers:* 1=Yes 2=No (asked 2009+)  
*Variable:* `new_partner_year` (SXQ) · *answered:* 43%

**Q17. Which best describes your sexual orientation?**  
*Answers:* 1=Heterosexual 2=Homosexual 3=Bisexual 4=Other  
*Variable:* `orientation` (SXQ) · *answered:* 72%


## STI history

**Q18. Has a doctor ever told you that you had genital herpes?**  
*Answers:* 1=Yes 2=No  
*Variable:* `herpes_hx` (SXQ) · *answered:* 81%

**Q19. Has a doctor ever told you that you had gonorrhea?**  
*Answers:* 1=Yes 2=No  
*Variable:* `gonorrhea_hx` (SXQ) · *answered:* 81%

**Q20. Has a doctor ever told you that you had chlamydia?**  
*Answers:* 1=Yes 2=No  
*Variable:* `chlamydia_hx` (SXQ) · *answered:* 81%

**Q21. Has a doctor ever told you that you had genital warts?  [HPV-ADJACENT — reported with/without]**  ⚠️_HPV-adjacent_  
*Answers:* 1=Yes 2=No  (warts = low-risk HPV 6/11; target = high-risk types)  
*Variable:* `genital_warts_hx` (SXQ) · *answered:* 81%


## Reproductive

**Q22. How old were you when your periods started?**  
*Answers:* age in years  
*Variable:* `age_menarche` (RHQ) · *answered:* 90%

**Q23. Have you ever been pregnant?**  
*Answers:* 1=Yes 2=No  
*Variable:* `ever_pregnant` (RHQ) · *answered:* 86%

**Q24. How many times have you been pregnant?**  
*Answers:* count (0 if never pregnant)  
*Variable:* `n_pregnancies` (RHQ) · *answered:* 86%


## Smoking

**Q25. Have you smoked at least 100 cigarettes in your entire life?**  
*Answers:* 1=Yes 2=No  
*Variable:* `smoked_100` (SMQ) · *answered:* 93%

**Q26. Current smoking status (derived from SMQ020 + SMQ040)**  
*Answers:* 0=Never 1=Former 2=Current  
*Variable:* `smoking_status` (SMQ) · *answered:* 93%


## Vaccination

**Q27. Have you ever received the HPV vaccine (Cervarix or Gardasil)?**  
*Answers:* 1=Yes 2=No  (question introduced 2007; set 2=No for pre-2007 cycles)  
*Variable:* `hpv_vaccine` (IMQ) · *answered:* 98%


## Biomarker

**Q28. Herpes Simplex Virus type-2 antibody (laboratory blood test)**  
*Answers:* 1=Positive 2=Negative  (different virus; marks shared sexual-transmission risk)  
*Variable:* `hsv2` (HSV) · *answered:* 65%

## Engineered (derived from the answers above)

- `years_active`
- `partners_per_active_year`
- `pregnancies_per_active_year`