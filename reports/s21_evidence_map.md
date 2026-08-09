# Evidence map -- review document
- corpus manifest `62b9e550fc137800`, 189 chunks
- dense channel: ncbi/MedCPT-Query-Encoder + ncbi/MedCPT-Article-Encoder on cpu + ncbi/MedCPT-Cross-Encoder
- **41 accepted**, **7 weak**, **9 with no admissible passage**, 57 keys total
- built in 48 s

## Needs your eye

A key with no admissible passage is emitted with its passage **suppressed**, not dropped -- a missing key has to be visible in the output rather than absent from it.

**No admissible passage** (nothing in the corpus mentions this parameter in this channel):

- `context:etco2` -- pool 189 chunks
- `action:LOW:flow_rate` -- pool 45 chunks
- `action:LOW:etco2` -- pool 45 chunks
- `action:MEDIUM:flow_rate` -- pool 45 chunks
- `action:MEDIUM:etco2` -- pool 45 chunks
- `action:HIGH:flow_rate` -- pool 45 chunks
- `action:HIGH:etco2` -- pool 45 chunks
- `action:CRITICAL:flow_rate` -- pool 45 chunks
- `action:CRITICAL:etco2` -- pool 45 chunks

**Weak match** (passage names the parameter fewer than 2 times, so it is probably mentioning it in passing):

- `context:flow_rate` -- 1 mention(s) -- AARC CPG, Respir Care 2024;69(8):1042-1054, p.5 -- "meta-analyses with 4,549 subjects and found that all 3 parameters were associated with mortality. The effect s..."
- `context:pip` -- 1 mention(s) -- AARC Adult Mechanical Ventilator Protocols v1.0a (2003), p.5 -- "2. For a pH < 7.30, evaluate to ensure the cause is respiratory. If appropriate, increase rate to a maximum of..."
- `context:respiratory_rate_total` -- 1 mention(s) -- AARC Adult Mechanical Ventilator Protocols v1.0a (2003), p.4 -- "D. Rate: 8 to 26 breaths/minute adjusted to achieve optimum total cycle time and maintain desired minute venti..."
- `action:LOW:pip` -- 1 mention(s) -- AARC Adult Mechanical Ventilator Protocols v1.0a (2003), p.5 -- "2. For a pH < 7.30, evaluate to ensure the cause is respiratory. If appropriate, increase rate to a maximum of..."
- `action:MEDIUM:pip` -- 1 mention(s) -- AARC Adult Mechanical Ventilator Protocols v1.0a (2003), p.5 -- "2. For a pH < 7.30, evaluate to ensure the cause is respiratory. If appropriate, increase rate to a maximum of..."
- `action:HIGH:pip` -- 1 mention(s) -- AARC Adult Mechanical Ventilator Protocols v1.0a (2003), p.5 -- "2. For a pH < 7.30, evaluate to ensure the cause is respiratory. If appropriate, increase rate to a maximum of..."
- `action:CRITICAL:pip` -- 1 mention(s) -- AARC Adult Mechanical Ventilator Protocols v1.0a (2003), p.5 -- "2. For a pH < 7.30, evaluate to ensure the cause is respiratory. If appropriate, increase rate to a maximum of..."

## Definitional context

### `context:spo2`

query: `SpO2 % SpO2 oxygen saturation pulse oximetry oximetry saturation hypoxemia normoxemia` | pool: 189 | status: **ok**

- **AARC Adult Mechanical Ventilator Protocols v1.0a (2003), p.4** (recommendation, rrf 0.04918, ranks {'lexical': 1, 'dense': 1, 'cross': 1}, 10 mention(s))
  - section: I. Adult Invasive Ventilation Protocol Initial Parameters and Goals
  - > 2. Pulse oximetry should be correlated with initial ABG and the patient subsequently monitored with continuous pulse oximetry to maintain SpO2 at or above patient’s normal or >90% SpO2 (Oxygen saturation by pulse oximetry).

### `context:fio2`

query: `FiO2 % FiO2 FIO2 oxygen concentration fraction of inspired oxygen inspired oxygen` | pool: 189 | status: **ok**

- **NHSN VAE protocol (Jan 2026), p.11** (table, rrf 0.0484, ranks {'lexical': 3, 'dense': 1, 'cross': 2}, 3 mention(s))
  - section: Ventilator-Associated Event (VAE) > Daily minimum
  - > FiO2 (oxygen concentration, %) VAE 1 8 1.0 (100%) - 2 6 0.50 (50%) - 3 5 0.35 (35%) - 4 5 0.40 (40%) - 5 6 0.70 (70%) No event 6 6 0.70 (70%) -

### `context:flow_rate`

query: `flow rate L/min flow rate inspiratory flow peak flow` | pool: 189 | status: **weak**

- **AARC CPG, Respir Care 2024;69(8):1042-1054, p.5** (prose, rrf 0.03279, ranks {'lexical': 1, 'cross': 1}, 1 mention(s))  :warning: weak
  - section: AARC Clinical Practice Guideline: Patient-Ventilator Assessment > Lung-protective ventilation in patients with ARDS. Petrucci
  - > meta-analyses with 4,549 subjects and found that all 3 parameters were associated with mortality. The effect size for each cm H2O increase in driving pressure was about 4 times greater than that of each 1 breath/min increase in f, leading to higher mortality. In children, the PALICC guidelines recommended that driving pressure be limited to < 15 cm H2O.23 No studies evaluating driving pressure were found in neonates. FIO2 . In 2022, an AARC CPG recommended an SpO2 range of 94–98% for general critically ill patients and an SpO2 range of 88–93% for patients with ARDS, especially when the FIO2 is > 0.7.24 Cumpstey et al,25 in a systematic review and meta-analysis, found that normoxemia versus hyperoxemia was associated with decreased mortality, (odds ratio 0.73 [95% CI 0.57–0.97]). In children, the PALICC guidelines recommend an SpO2 range of 92–97% with mild or moderate ARDS, and an SpO2 < 92% can be accepted in children with severe ARDS.23 However, an ungraded good practice statement states to avoid hypoxemia (SpO2 < 88%) or hyperoxemia (SpO2 > 97%) in mechanically ventilated children.14 There were no recommendations in the AARC pediatric oxygen therapy CPG regarding FIO2 26 No studies were found evaluating FIO2 in neonates. Furthermore, no studies evaluating f, inspiratory-expiratory (I-E) ratio, inspiratory flow, and inspiratory time were found.

### `context:peep`

query: `PEEP cmH2O PEEP positive end-expiratory pressure end-expiratory pressure auto-PEEP CPAP` | pool: 189 | status: **ok**

- **AARC CPG, Respir Care 2024;69(8):1042-1054, p.1** (recommendation, rrf 0.04649, ranks {'lexical': 9, 'dense': 2, 'cross': 3}, 3 mention(s)) -- **strong recommendation, high certainty**
  - section: AARC Clinical Practice Guideline: Patient-Ventilator Assessment > Joel T Glogowski, and Dean R Hess
  - > We recommend an assessment of PEEP and auto-PEEP (strong recommendation, high certainty)

### `context:pip`

query: `PIP cmH2O PIP peak inspiratory pressure peak pressure peak airway pressure` | pool: 189 | status: **weak**

- **AARC Adult Mechanical Ventilator Protocols v1.0a (2003), p.5** (recommendation, rrf 0.04892, ranks {'lexical': 2, 'dense': 1, 'cross': 1}, 1 mention(s))  :warning: weak
  - section: II. Ventilator Adjustments Based on Patient Assessment
  - > 2. For a pH < 7.30, evaluate to ensure the cause is respiratory. If appropriate, increase rate to a maximum of 24 breaths/min until pH is > 7.30. If further adjustment is needed increase VT until PIP > 40 cm H2O or Pplateau > 30 cm H2O. If unable to maintain these parameters, consider allowing permissive hypercapnia.

### `context:respiratory_rate_total`

query: `respiratory rate /min respiratory rate breathing frequency breaths/minute breaths per minute tachypnea` | pool: 189 | status: **weak**

- **AARC Adult Mechanical Ventilator Protocols v1.0a (2003), p.4** (recommendation, rrf 0.04892, ranks {'lexical': 2, 'dense': 1, 'cross': 1}, 1 mention(s))  :warning: weak
  - section: I. Adult Invasive Ventilation Protocol Initial Parameters and Goals
  - > D. Rate: 8 to 26 breaths/minute adjusted to achieve optimum total cycle time and maintain desired minute ventilation, while maintaining plateau pressure < 30 cm H2O and delta P < 20 cm H2O.

### `context:minute_volume`

query: `minute volume L/min minute ventilation minute volume VE` | pool: 189 | status: **ok**

- **AARC Adult Mechanical Ventilator Protocols v1.0a (2003), p.4** (recommendation, rrf 0.04892, ranks {'lexical': 1, 'dense': 2, 'cross': 1}, 3 mention(s))
  - section: I. Adult Invasive Ventilation Protocol Initial Parameters and Goals
  - > C. Minute ventilation: 4.0 x BSA (Body Surface Area) = VE (L/min) for males and 3.5 x BSA = VE (L/min) for females adjusted for altitude and body temperature ( DuBois BSA Nomogram) while maintaining plateau pressure < 30 cm H2O and delta P <20 cm H2O.

### `context:tidal_volume_observed`

query: `tidal volume mL tidal volume VT mL/kg lung-protective lung protective predicted body weight` | pool: 189 | status: **ok**

- **AARC CPG, Respir Care 2024;69(8):1042-1054, p.1** (recommendation, rrf 0.04817, ranks {'lexical': 1, 'dense': 1, 'cross': 5}, 5 mention(s)) -- **strong recommendation, high certainty**
  - section: AARC Clinical Practice Guideline: Patient-Ventilator Assessment > Joel T Glogowski, and Dean R Hess
  - > We recommend an assessment of tidal volume (VT) to ensure lung-protective ventilation (4–8 mL/kg/predicted body weight) (strong recommendation, high certainty)

### `context:etco2`

query: `EtCO2 mmHg EtCO2 end-tidal capnography carbon dioxide` | pool: 189 | status: **missing**

_no admissible passage -- suppressed in output_

### `context:inspiratory_ratio`

query: `I:E inspiratory part  I:E I:E ratio inspiratory time inspiratory-to-expiratory` | pool: 189 | status: **ok**

- **AARC Adult Mechanical Ventilator Protocols v1.0a (2003), p.3** (recommendation, rrf 0.04813, ranks {'lexical': 3, 'dense': 2, 'cross': 2}, 2 mention(s))
  - section: III. Suggestions for Use of Modes
  - > 3. Pressure support by itself may be effective in patients who have an adequate respiratory drive and who might tolerate mechanical ventilation better when a variable I:E ratio is available.

### `context:expiratory_ratio`

query: `I:E expiratory part  I:E I:E ratio expiratory time inspiratory-to-expiratory` | pool: 189 | status: **ok**

- **AARC Adult Mechanical Ventilator Protocols v1.0a (2003), p.3** (recommendation, rrf 0.04839, ranks {'lexical': 2, 'dense': 2, 'cross': 2}, 2 mention(s))
  - section: III. Suggestions for Use of Modes
  - > 3. Pressure support by itself may be effective in patients who have an adequate respiratory drive and who might tolerate mechanical ventilation better when a variable I:E ratio is available.

### `context:kind:physiology`

query: `patient-ventilator assessment assessment physiologic clinical assessment` | pool: 189 | status: **ok**

- **AARC CPG, Respir Care 2024;69(8):1042-1054, p.1** (recommendation, rrf 0.04546, ranks {'lexical': 7, 'dense': 6, 'cross': 5}, 3 mention(s)) -- **strong recommendation, high certainty**
  - section: AARC Clinical Practice Guideline: Patient-Ventilator Assessment > Joel T Glogowski, and Dean R Hess
  - > We recommend assessment of plateau pressure to ensure lung-protective ventilator settings (strong recommendation, high certainty)

### `context:kind:documentation`

query: `documentation documenting recorded charting surveillance data collection reporting` | pool: 189 | status: **ok**

- **NHSN VAE protocol (Jan 2026), p.22** (prose, rrf 0.04841, ranks {'lexical': 1, 'dense': 1, 'cross': 4}, 5 mention(s))
  - section: Ventilator-Associated Event (VAE) > Numerator and Denominator Data
  - > Numerator Data: The Ventilator-Associated Event (VAE) form (CDC 57.112) is used to collect and report each VAE that is identified during the month selected for surveillance. The Instructions for Completion of Ventilator-Associated Event Form includes brief instructions for collection and entry of each data element on the form. The VAE form includes patient demographic information and information on the start date and location of initiation of mechanical ventilation. Additional data include the specific criteria met for identifying VAE, whether the patient developed a secondary bloodstream infection, whether the patient died, and, where applicable, the organisms detected and their antimicrobial susceptibilities. Reporting Instruction: If no VAEs are identified during the month of surveillance, the “Report No Events” box must be checked on the appropriate denominator summary screen, for example, Denominators for Intensive Care Unit (ICU)/Other Locations (Not NICU or SCA), etc. Denominator Data: Device days and patient days are used for denominators (see Chapter 16 General Key Terms). Ventilator days, which are the numbers of patients managed with ventilatory devices, are collected daily, at the same time each day, according to the chosen location using the appropriate form (CDC 57.117 [Specialty Care Areas] or 57.118 [ICU/Other Locations]). These daily counts are summed and only the total for the month is entered into NHSN. Ventilator and patient days are collected for each of the locations monitored. When denominator data are available from electronic sources, these sources may be used as long as the counts are within +/- 5% of manually collected counts, validated for a minimum of 3 consecutive months. Validation of electronic counts should be performed separately for each location conducting VAE surveillance. All ventilator days are counted, including ventilator days for patients on mechanical ventilation for < 3 days, and patients on high frequency ventilation and other therapies excluded from VAE surveillance. Patients with tracheostomies who are undergoing weaning from mechanical ventilation using tracheostomy collar trials are included in ventilator day counts if they are on mechanical ventilation at the

## Suggested actions

### `action:LOW:spo2`

query: `SpO2 oxygen saturation pulse oximetry oximetry saturation hypoxemia normoxemia weaning liberation spontaneous breathing trial extubation readiness reduce support` | pool: 45 | status: **ok**

- **AARC Adult Mechanical Ventilator Protocols v1.0a (2003), p.4** (recommendation, rrf 0.04918, ranks {'lexical': 1, 'dense': 1, 'cross': 1}, 10 mention(s))
  - section: I. Adult Invasive Ventilation Protocol Initial Parameters and Goals
  - > 2. Pulse oximetry should be correlated with initial ABG and the patient subsequently monitored with continuous pulse oximetry to maintain SpO2 at or above patient’s normal or >90% SpO2 (Oxygen saturation by pulse oximetry).

- **AARC CPG, Respir Care 2024;69(8):1042-1054, p.1** (recommendation, rrf 0.03226, ranks {'dense': 2, 'cross': 2}, 1 mention(s)) -- **conditional recommendation, very low certainty**  :warning: weak
  - section: AARC Clinical Practice Guideline: Patient-Ventilator Assessment > Joel T Glogowski, and Dean R Hess
  - > We suggest assessing FIO2 to ensure normoxemia (conditional recommendation, very low certainty)

### `action:LOW:fio2`

query: `FiO2 FIO2 oxygen concentration fraction of inspired oxygen inspired oxygen weaning liberation spontaneous breathing trial extubation readiness reduce support` | pool: 45 | status: **ok**

- **AARC CPG, Respir Care 2024;69(8):1042-1054, p.1** (recommendation, rrf 0.03279, ranks {'dense': 1, 'cross': 1}, 2 mention(s)) -- **conditional recommendation, very low certainty**
  - section: AARC Clinical Practice Guideline: Patient-Ventilator Assessment > Joel T Glogowski, and Dean R Hess
  - > We suggest assessing FIO2 to ensure normoxemia (conditional recommendation, very low certainty)

### `action:LOW:flow_rate`

query: `flow rate inspiratory flow peak flow weaning liberation spontaneous breathing trial extubation readiness reduce support` | pool: 45 | status: **missing**

_no admissible passage -- suppressed in output_

### `action:LOW:peep`

query: `PEEP positive end-expiratory pressure end-expiratory pressure auto-PEEP CPAP weaning liberation spontaneous breathing trial extubation readiness reduce support` | pool: 45 | status: **ok**

- **AARC CPG, Respir Care 2024;69(8):1042-1054, p.1** (recommendation, rrf 0.03279, ranks {'dense': 1, 'cross': 1}, 3 mention(s)) -- **strong recommendation, high certainty**
  - section: AARC Clinical Practice Guideline: Patient-Ventilator Assessment > Joel T Glogowski, and Dean R Hess
  - > We recommend an assessment of PEEP and auto-PEEP (strong recommendation, high certainty)

- **AARC Adult Mechanical Ventilator Protocols v1.0a (2003), p.4** (recommendation, rrf 0.03226, ranks {'dense': 2, 'cross': 2}, 2 mention(s))
  - section: I. Adult Invasive Ventilation Protocol Initial Parameters and Goals
  - > 3. PEEP 5 to 15 cm H2O. Set initial PEEP at 5 cm H2O, unless otherwise indicated. Higher PEEPs may be required with acute lung injury (ALI) or acute respiratory distress syndrome (ARDS). [Note: See ALI/ARDS Protocol]

### `action:LOW:pip`

query: `PIP peak inspiratory pressure peak pressure peak airway pressure weaning liberation spontaneous breathing trial extubation readiness reduce support` | pool: 45 | status: **weak**

- **AARC Adult Mechanical Ventilator Protocols v1.0a (2003), p.5** (recommendation, rrf 0.03279, ranks {'lexical': 1, 'cross': 1}, 1 mention(s))  :warning: weak
  - section: II. Ventilator Adjustments Based on Patient Assessment
  - > 2. For a pH < 7.30, evaluate to ensure the cause is respiratory. If appropriate, increase rate to a maximum of 24 breaths/min until pH is > 7.30. If further adjustment is needed increase VT until PIP > 40 cm H2O or Pplateau > 30 cm H2O. If unable to maintain these parameters, consider allowing permissive hypercapnia.

### `action:LOW:respiratory_rate_total`

query: `respiratory rate breathing frequency breaths/minute breaths per minute tachypnea weaning liberation spontaneous breathing trial extubation readiness reduce support` | pool: 45 | status: **ok**

- **AARC Adult Mechanical Ventilator Protocols v1.0a (2003), p.4** (recommendation, rrf 0.04865, ranks {'lexical': 2, 'dense': 1, 'cross': 2}, 1 mention(s))  :warning: weak
  - section: I. Adult Invasive Ventilation Protocol Initial Parameters and Goals
  - > D. Rate: 8 to 26 breaths/minute adjusted to achieve optimum total cycle time and maintain desired minute ventilation, while maintaining plateau pressure < 30 cm H2O and delta P < 20 cm H2O.

- **AARC Adult Mechanical Ventilator Protocols v1.0a (2003), p.5** (recommendation, rrf 0.03279, ranks {'lexical': 1, 'cross': 1}, 2 mention(s))
  - section: II. Ventilator Adjustments Based on Patient Assessment
  - > 3. For a pH > 7.45, evaluate to ensure the cause is respiratory. If appropriate, reduce rate to a minimum of 8 breaths/minute or until pH is < 7.45. After rate is decreased to 8 breaths/minute, if pH is still > 7.45, reduce volume to a minimum of 4 mL/Kg (IBW).

### `action:LOW:minute_volume`

query: `minute ventilation minute volume VE weaning liberation spontaneous breathing trial extubation readiness reduce support` | pool: 45 | status: **ok**

- **AARC Adult Mechanical Ventilator Protocols v1.0a (2003), p.4** (recommendation, rrf 0.04892, ranks {'lexical': 2, 'dense': 1, 'cross': 1}, 1 mention(s))  :warning: weak
  - section: I. Adult Invasive Ventilation Protocol Initial Parameters and Goals
  - > D. Rate: 8 to 26 breaths/minute adjusted to achieve optimum total cycle time and maintain desired minute ventilation, while maintaining plateau pressure < 30 cm H2O and delta P < 20 cm H2O.

- **AARC Adult Mechanical Ventilator Protocols v1.0a (2003), p.4** (recommendation, rrf 0.03252, ranks {'lexical': 1, 'cross': 2}, 3 mention(s))
  - section: I. Adult Invasive Ventilation Protocol Initial Parameters and Goals
  - > C. Minute ventilation: 4.0 x BSA (Body Surface Area) = VE (L/min) for males and 3.5 x BSA = VE (L/min) for females adjusted for altitude and body temperature ( DuBois BSA Nomogram) while maintaining plateau pressure < 30 cm H2O and delta P <20 cm H2O.

### `action:LOW:tidal_volume_observed`

query: `tidal volume VT mL/kg lung-protective lung protective predicted body weight weaning liberation spontaneous breathing trial extubation readiness reduce support` | pool: 45 | status: **ok**

- **AARC CPG, Respir Care 2024;69(8):1042-1054, p.1** (recommendation, rrf 0.04918, ranks {'lexical': 1, 'dense': 1, 'cross': 1}, 5 mention(s)) -- **strong recommendation, high certainty**
  - section: AARC Clinical Practice Guideline: Patient-Ventilator Assessment > Joel T Glogowski, and Dean R Hess
  - > We recommend an assessment of tidal volume (VT) to ensure lung-protective ventilation (4–8 mL/kg/predicted body weight) (strong recommendation, high certainty)

- **AARC CPG, Respir Care 2024;69(8):1042-1054, p.6** (recommendation, rrf 0.04788, ranks {'lexical': 2, 'dense': 2, 'cross': 4}, 5 mention(s)) -- **strong recommendation, high certainty**
  - section: AARC Clinical Practice Guideline: Patient-Ventilator Assessment > Lung-protective ventilation in patients with ARDS. Petrucci
  - > We recommend an assessment of VT to ensure lung-protective ventilation (4–8 mL/kg/predicted body weight) (strong recommendation, high certainty)

### `action:LOW:etco2`

query: `EtCO2 end-tidal capnography carbon dioxide weaning liberation spontaneous breathing trial extubation readiness reduce support` | pool: 45 | status: **missing**

_no admissible passage -- suppressed in output_

### `action:LOW:inspiratory_ratio`

query: `I:E I:E ratio inspiratory time inspiratory-to-expiratory weaning liberation spontaneous breathing trial extubation readiness reduce support` | pool: 45 | status: **ok**

- **AARC Adult Mechanical Ventilator Protocols v1.0a (2003), p.3** (recommendation, rrf 0.04918, ranks {'lexical': 1, 'dense': 1, 'cross': 1}, 2 mention(s))
  - section: III. Suggestions for Use of Modes
  - > 3. Pressure support by itself may be effective in patients who have an adequate respiratory drive and who might tolerate mechanical ventilation better when a variable I:E ratio is available.

### `action:LOW:expiratory_ratio`

query: `I:E I:E ratio expiratory time inspiratory-to-expiratory weaning liberation spontaneous breathing trial extubation readiness reduce support` | pool: 45 | status: **ok**

- **AARC Adult Mechanical Ventilator Protocols v1.0a (2003), p.3** (recommendation, rrf 0.04918, ranks {'lexical': 1, 'dense': 1, 'cross': 1}, 2 mention(s))
  - section: III. Suggestions for Use of Modes
  - > 3. Pressure support by itself may be effective in patients who have an adequate respiratory drive and who might tolerate mechanical ventilation better when a variable I:E ratio is available.

### `action:MEDIUM:spo2`

query: `SpO2 oxygen saturation pulse oximetry oximetry saturation hypoxemia normoxemia assessment assess monitoring evaluate documenting` | pool: 45 | status: **ok**

- **AARC Adult Mechanical Ventilator Protocols v1.0a (2003), p.4** (recommendation, rrf 0.04918, ranks {'lexical': 1, 'dense': 1, 'cross': 1}, 10 mention(s))
  - section: I. Adult Invasive Ventilation Protocol Initial Parameters and Goals
  - > 2. Pulse oximetry should be correlated with initial ABG and the patient subsequently monitored with continuous pulse oximetry to maintain SpO2 at or above patient’s normal or >90% SpO2 (Oxygen saturation by pulse oximetry).

- **AARC CPG, Respir Care 2024;69(8):1042-1054, p.1** (recommendation, rrf 0.04839, ranks {'lexical': 2, 'dense': 2, 'cross': 2}, 1 mention(s)) -- **conditional recommendation, very low certainty**  :warning: weak
  - section: AARC Clinical Practice Guideline: Patient-Ventilator Assessment > Joel T Glogowski, and Dean R Hess
  - > We suggest assessing FIO2 to ensure normoxemia (conditional recommendation, very low certainty)

### `action:MEDIUM:fio2`

query: `FiO2 FIO2 oxygen concentration fraction of inspired oxygen inspired oxygen assessment assess monitoring evaluate documenting` | pool: 45 | status: **ok**

- **AARC CPG, Respir Care 2024;69(8):1042-1054, p.1** (recommendation, rrf 0.03279, ranks {'dense': 1, 'cross': 1}, 2 mention(s)) -- **conditional recommendation, very low certainty**
  - section: AARC Clinical Practice Guideline: Patient-Ventilator Assessment > Joel T Glogowski, and Dean R Hess
  - > We suggest assessing FIO2 to ensure normoxemia (conditional recommendation, very low certainty)

### `action:MEDIUM:flow_rate`

query: `flow rate inspiratory flow peak flow assessment assess monitoring evaluate documenting` | pool: 45 | status: **missing**

_no admissible passage -- suppressed in output_

### `action:MEDIUM:peep`

query: `PEEP positive end-expiratory pressure end-expiratory pressure auto-PEEP CPAP assessment assess monitoring evaluate documenting` | pool: 45 | status: **ok**

- **AARC CPG, Respir Care 2024;69(8):1042-1054, p.1** (recommendation, rrf 0.04918, ranks {'lexical': 1, 'dense': 1, 'cross': 1}, 3 mention(s)) -- **strong recommendation, high certainty**
  - section: AARC Clinical Practice Guideline: Patient-Ventilator Assessment > Joel T Glogowski, and Dean R Hess
  - > We recommend an assessment of PEEP and auto-PEEP (strong recommendation, high certainty)

- **AARC Adult Mechanical Ventilator Protocols v1.0a (2003), p.4** (recommendation, rrf 0.03226, ranks {'dense': 2, 'cross': 2}, 2 mention(s))
  - section: I. Adult Invasive Ventilation Protocol Initial Parameters and Goals
  - > 3. PEEP 5 to 15 cm H2O. Set initial PEEP at 5 cm H2O, unless otherwise indicated. Higher PEEPs may be required with acute lung injury (ALI) or acute respiratory distress syndrome (ARDS). [Note: See ALI/ARDS Protocol]

### `action:MEDIUM:pip`

query: `PIP peak inspiratory pressure peak pressure peak airway pressure assessment assess monitoring evaluate documenting` | pool: 45 | status: **weak**

- **AARC Adult Mechanical Ventilator Protocols v1.0a (2003), p.5** (recommendation, rrf 0.04918, ranks {'lexical': 1, 'dense': 1, 'cross': 1}, 1 mention(s))  :warning: weak
  - section: II. Ventilator Adjustments Based on Patient Assessment
  - > 2. For a pH < 7.30, evaluate to ensure the cause is respiratory. If appropriate, increase rate to a maximum of 24 breaths/min until pH is > 7.30. If further adjustment is needed increase VT until PIP > 40 cm H2O or Pplateau > 30 cm H2O. If unable to maintain these parameters, consider allowing permissive hypercapnia.

### `action:MEDIUM:respiratory_rate_total`

query: `respiratory rate breathing frequency breaths/minute breaths per minute tachypnea assessment assess monitoring evaluate documenting` | pool: 45 | status: **ok**

- **AARC Adult Mechanical Ventilator Protocols v1.0a (2003), p.5** (recommendation, rrf 0.04918, ranks {'lexical': 1, 'dense': 1, 'cross': 1}, 2 mention(s))
  - section: II. Ventilator Adjustments Based on Patient Assessment
  - > 3. For a pH > 7.45, evaluate to ensure the cause is respiratory. If appropriate, reduce rate to a minimum of 8 breaths/minute or until pH is < 7.45. After rate is decreased to 8 breaths/minute, if pH is still > 7.45, reduce volume to a minimum of 4 mL/Kg (IBW).

- **AARC Adult Mechanical Ventilator Protocols v1.0a (2003), p.4** (recommendation, rrf 0.04839, ranks {'lexical': 2, 'dense': 2, 'cross': 2}, 1 mention(s))  :warning: weak
  - section: I. Adult Invasive Ventilation Protocol Initial Parameters and Goals
  - > D. Rate: 8 to 26 breaths/minute adjusted to achieve optimum total cycle time and maintain desired minute ventilation, while maintaining plateau pressure < 30 cm H2O and delta P < 20 cm H2O.

### `action:MEDIUM:minute_volume`

query: `minute ventilation minute volume VE assessment assess monitoring evaluate documenting` | pool: 45 | status: **ok**

- **AARC Adult Mechanical Ventilator Protocols v1.0a (2003), p.4** (recommendation, rrf 0.04865, ranks {'lexical': 2, 'dense': 1, 'cross': 2}, 1 mention(s))  :warning: weak
  - section: I. Adult Invasive Ventilation Protocol Initial Parameters and Goals
  - > D. Rate: 8 to 26 breaths/minute adjusted to achieve optimum total cycle time and maintain desired minute ventilation, while maintaining plateau pressure < 30 cm H2O and delta P < 20 cm H2O.

- **AARC Adult Mechanical Ventilator Protocols v1.0a (2003), p.4** (recommendation, rrf 0.03279, ranks {'lexical': 1, 'cross': 1}, 3 mention(s))
  - section: I. Adult Invasive Ventilation Protocol Initial Parameters and Goals
  - > C. Minute ventilation: 4.0 x BSA (Body Surface Area) = VE (L/min) for males and 3.5 x BSA = VE (L/min) for females adjusted for altitude and body temperature ( DuBois BSA Nomogram) while maintaining plateau pressure < 30 cm H2O and delta P <20 cm H2O.

### `action:MEDIUM:tidal_volume_observed`

query: `tidal volume VT mL/kg lung-protective lung protective predicted body weight assessment assess monitoring evaluate documenting` | pool: 45 | status: **ok**

- **AARC CPG, Respir Care 2024;69(8):1042-1054, p.1** (recommendation, rrf 0.04892, ranks {'lexical': 1, 'dense': 1, 'cross': 2}, 5 mention(s)) -- **strong recommendation, high certainty**
  - section: AARC Clinical Practice Guideline: Patient-Ventilator Assessment > Joel T Glogowski, and Dean R Hess
  - > We recommend an assessment of tidal volume (VT) to ensure lung-protective ventilation (4–8 mL/kg/predicted body weight) (strong recommendation, high certainty)

- **AARC CPG, Respir Care 2024;69(8):1042-1054, p.6** (recommendation, rrf 0.04865, ranks {'lexical': 2, 'dense': 2, 'cross': 1}, 5 mention(s)) -- **strong recommendation, high certainty**
  - section: AARC Clinical Practice Guideline: Patient-Ventilator Assessment > Lung-protective ventilation in patients with ARDS. Petrucci
  - > We recommend an assessment of VT to ensure lung-protective ventilation (4–8 mL/kg/predicted body weight) (strong recommendation, high certainty)

### `action:MEDIUM:etco2`

query: `EtCO2 end-tidal capnography carbon dioxide assessment assess monitoring evaluate documenting` | pool: 45 | status: **missing**

_no admissible passage -- suppressed in output_

### `action:MEDIUM:inspiratory_ratio`

query: `I:E I:E ratio inspiratory time inspiratory-to-expiratory assessment assess monitoring evaluate documenting` | pool: 45 | status: **ok**

- **AARC Adult Mechanical Ventilator Protocols v1.0a (2003), p.3** (recommendation, rrf 0.04918, ranks {'lexical': 1, 'dense': 1, 'cross': 1}, 2 mention(s))
  - section: III. Suggestions for Use of Modes
  - > 3. Pressure support by itself may be effective in patients who have an adequate respiratory drive and who might tolerate mechanical ventilation better when a variable I:E ratio is available.

### `action:MEDIUM:expiratory_ratio`

query: `I:E I:E ratio expiratory time inspiratory-to-expiratory assessment assess monitoring evaluate documenting` | pool: 45 | status: **ok**

- **AARC Adult Mechanical Ventilator Protocols v1.0a (2003), p.3** (recommendation, rrf 0.04918, ranks {'lexical': 1, 'dense': 1, 'cross': 1}, 2 mention(s))
  - section: III. Suggestions for Use of Modes
  - > 3. Pressure support by itself may be effective in patients who have an adequate respiratory drive and who might tolerate mechanical ventilation better when a variable I:E ratio is available.

### `action:HIGH:spo2`

query: `SpO2 oxygen saturation pulse oximetry oximetry saturation hypoxemia normoxemia lung-protective adjust escalation increase support ventilator adjustments assess` | pool: 45 | status: **ok**

- **AARC Adult Mechanical Ventilator Protocols v1.0a (2003), p.4** (recommendation, rrf 0.04892, ranks {'lexical': 1, 'dense': 1, 'cross': 2}, 10 mention(s))
  - section: I. Adult Invasive Ventilation Protocol Initial Parameters and Goals
  - > 2. Pulse oximetry should be correlated with initial ABG and the patient subsequently monitored with continuous pulse oximetry to maintain SpO2 at or above patient’s normal or >90% SpO2 (Oxygen saturation by pulse oximetry).

- **AARC Adult Mechanical Ventilator Protocols v1.0a (2003), p.5** (recommendation, rrf 0.0484, ranks {'lexical': 2, 'dense': 3, 'cross': 1}, 1 mention(s))  :warning: weak
  - section: II. Ventilator Adjustments Based on Patient Assessment
  - > D. Adjust the ventilator settings so that ABG results are acceptable. Patient Category pH PaCO2 PaO2 SpO2 Normal 7.35-7.45 35-45 mmHg > 80 mm Hg 92-97% Chronic CO2 Retention 7.30-7.45 45-55 mmHg adjust to pH range 55-75 mmHg >89% Open Heart Patients 7.35-7.50 35-50 mmHg > 65 mm Hg 90-95% ARDS* 7.25-7.45 Adjust to pH range > 60 mmHg 90-95% *See ALI/ARDS Protocol

### `action:HIGH:fio2`

query: `FiO2 FIO2 oxygen concentration fraction of inspired oxygen inspired oxygen lung-protective adjust escalation increase support ventilator adjustments assess` | pool: 45 | status: **ok**

- **AARC CPG, Respir Care 2024;69(8):1042-1054, p.1** (recommendation, rrf 0.03279, ranks {'dense': 1, 'cross': 1}, 2 mention(s)) -- **conditional recommendation, very low certainty**
  - section: AARC Clinical Practice Guideline: Patient-Ventilator Assessment > Joel T Glogowski, and Dean R Hess
  - > We suggest assessing FIO2 to ensure normoxemia (conditional recommendation, very low certainty)

### `action:HIGH:flow_rate`

query: `flow rate inspiratory flow peak flow lung-protective adjust escalation increase support ventilator adjustments assess` | pool: 45 | status: **missing**

_no admissible passage -- suppressed in output_

### `action:HIGH:peep`

query: `PEEP positive end-expiratory pressure end-expiratory pressure auto-PEEP CPAP lung-protective adjust escalation increase support ventilator adjustments assess` | pool: 45 | status: **ok**

- **AARC CPG, Respir Care 2024;69(8):1042-1054, p.1** (recommendation, rrf 0.03279, ranks {'dense': 1, 'cross': 1}, 3 mention(s)) -- **strong recommendation, high certainty**
  - section: AARC Clinical Practice Guideline: Patient-Ventilator Assessment > Joel T Glogowski, and Dean R Hess
  - > We recommend an assessment of PEEP and auto-PEEP (strong recommendation, high certainty)

- **AARC Adult Mechanical Ventilator Protocols v1.0a (2003), p.4** (recommendation, rrf 0.03226, ranks {'dense': 2, 'cross': 2}, 2 mention(s))
  - section: I. Adult Invasive Ventilation Protocol Initial Parameters and Goals
  - > 3. PEEP 5 to 15 cm H2O. Set initial PEEP at 5 cm H2O, unless otherwise indicated. Higher PEEPs may be required with acute lung injury (ALI) or acute respiratory distress syndrome (ARDS). [Note: See ALI/ARDS Protocol]

### `action:HIGH:pip`

query: `PIP peak inspiratory pressure peak pressure peak airway pressure lung-protective adjust escalation increase support ventilator adjustments assess` | pool: 45 | status: **weak**

- **AARC Adult Mechanical Ventilator Protocols v1.0a (2003), p.5** (recommendation, rrf 0.04918, ranks {'lexical': 1, 'dense': 1, 'cross': 1}, 1 mention(s))  :warning: weak
  - section: II. Ventilator Adjustments Based on Patient Assessment
  - > 2. For a pH < 7.30, evaluate to ensure the cause is respiratory. If appropriate, increase rate to a maximum of 24 breaths/min until pH is > 7.30. If further adjustment is needed increase VT until PIP > 40 cm H2O or Pplateau > 30 cm H2O. If unable to maintain these parameters, consider allowing permissive hypercapnia.

### `action:HIGH:respiratory_rate_total`

query: `respiratory rate breathing frequency breaths/minute breaths per minute tachypnea lung-protective adjust escalation increase support ventilator adjustments assess` | pool: 45 | status: **ok**

- **AARC Adult Mechanical Ventilator Protocols v1.0a (2003), p.5** (recommendation, rrf 0.04918, ranks {'lexical': 1, 'dense': 1, 'cross': 1}, 2 mention(s))
  - section: II. Ventilator Adjustments Based on Patient Assessment
  - > 3. For a pH > 7.45, evaluate to ensure the cause is respiratory. If appropriate, reduce rate to a minimum of 8 breaths/minute or until pH is < 7.45. After rate is decreased to 8 breaths/minute, if pH is still > 7.45, reduce volume to a minimum of 4 mL/Kg (IBW).

- **AARC Adult Mechanical Ventilator Protocols v1.0a (2003), p.4** (recommendation, rrf 0.04839, ranks {'lexical': 2, 'dense': 2, 'cross': 2}, 1 mention(s))  :warning: weak
  - section: I. Adult Invasive Ventilation Protocol Initial Parameters and Goals
  - > D. Rate: 8 to 26 breaths/minute adjusted to achieve optimum total cycle time and maintain desired minute ventilation, while maintaining plateau pressure < 30 cm H2O and delta P < 20 cm H2O.

### `action:HIGH:minute_volume`

query: `minute ventilation minute volume VE lung-protective adjust escalation increase support ventilator adjustments assess` | pool: 45 | status: **ok**

- **AARC Adult Mechanical Ventilator Protocols v1.0a (2003), p.4** (recommendation, rrf 0.04865, ranks {'lexical': 2, 'dense': 1, 'cross': 2}, 1 mention(s))  :warning: weak
  - section: I. Adult Invasive Ventilation Protocol Initial Parameters and Goals
  - > D. Rate: 8 to 26 breaths/minute adjusted to achieve optimum total cycle time and maintain desired minute ventilation, while maintaining plateau pressure < 30 cm H2O and delta P < 20 cm H2O.

- **AARC Adult Mechanical Ventilator Protocols v1.0a (2003), p.4** (recommendation, rrf 0.03279, ranks {'lexical': 1, 'cross': 1}, 3 mention(s))
  - section: I. Adult Invasive Ventilation Protocol Initial Parameters and Goals
  - > C. Minute ventilation: 4.0 x BSA (Body Surface Area) = VE (L/min) for males and 3.5 x BSA = VE (L/min) for females adjusted for altitude and body temperature ( DuBois BSA Nomogram) while maintaining plateau pressure < 30 cm H2O and delta P <20 cm H2O.

### `action:HIGH:tidal_volume_observed`

query: `tidal volume VT mL/kg lung-protective lung protective predicted body weight lung-protective adjust escalation increase support ventilator adjustments assess` | pool: 45 | status: **ok**

- **AARC CPG, Respir Care 2024;69(8):1042-1054, p.1** (recommendation, rrf 0.04892, ranks {'lexical': 1, 'dense': 1, 'cross': 2}, 5 mention(s)) -- **strong recommendation, high certainty**
  - section: AARC Clinical Practice Guideline: Patient-Ventilator Assessment > Joel T Glogowski, and Dean R Hess
  - > We recommend an assessment of tidal volume (VT) to ensure lung-protective ventilation (4–8 mL/kg/predicted body weight) (strong recommendation, high certainty)

- **AARC CPG, Respir Care 2024;69(8):1042-1054, p.6** (recommendation, rrf 0.04865, ranks {'lexical': 2, 'dense': 2, 'cross': 1}, 5 mention(s)) -- **strong recommendation, high certainty**
  - section: AARC Clinical Practice Guideline: Patient-Ventilator Assessment > Lung-protective ventilation in patients with ARDS. Petrucci
  - > We recommend an assessment of VT to ensure lung-protective ventilation (4–8 mL/kg/predicted body weight) (strong recommendation, high certainty)

### `action:HIGH:etco2`

query: `EtCO2 end-tidal capnography carbon dioxide lung-protective adjust escalation increase support ventilator adjustments assess` | pool: 45 | status: **missing**

_no admissible passage -- suppressed in output_

### `action:HIGH:inspiratory_ratio`

query: `I:E I:E ratio inspiratory time inspiratory-to-expiratory lung-protective adjust escalation increase support ventilator adjustments assess` | pool: 45 | status: **ok**

- **AARC Adult Mechanical Ventilator Protocols v1.0a (2003), p.3** (recommendation, rrf 0.04918, ranks {'lexical': 1, 'dense': 1, 'cross': 1}, 2 mention(s))
  - section: III. Suggestions for Use of Modes
  - > 3. Pressure support by itself may be effective in patients who have an adequate respiratory drive and who might tolerate mechanical ventilation better when a variable I:E ratio is available.

### `action:HIGH:expiratory_ratio`

query: `I:E I:E ratio expiratory time inspiratory-to-expiratory lung-protective adjust escalation increase support ventilator adjustments assess` | pool: 45 | status: **ok**

- **AARC Adult Mechanical Ventilator Protocols v1.0a (2003), p.3** (recommendation, rrf 0.04918, ranks {'lexical': 1, 'dense': 1, 'cross': 1}, 2 mention(s))
  - section: III. Suggestions for Use of Modes
  - > 3. Pressure support by itself may be effective in patients who have an adequate respiratory drive and who might tolerate mechanical ventilation better when a variable I:E ratio is available.

### `action:CRITICAL:spo2`

query: `SpO2 oxygen saturation pulse oximetry oximetry saturation hypoxemia normoxemia escalation urgent deterioration physician lung-protective assess` | pool: 45 | status: **ok**

- **AARC Adult Mechanical Ventilator Protocols v1.0a (2003), p.4** (recommendation, rrf 0.04918, ranks {'lexical': 1, 'dense': 1, 'cross': 1}, 10 mention(s))
  - section: I. Adult Invasive Ventilation Protocol Initial Parameters and Goals
  - > 2. Pulse oximetry should be correlated with initial ABG and the patient subsequently monitored with continuous pulse oximetry to maintain SpO2 at or above patient’s normal or >90% SpO2 (Oxygen saturation by pulse oximetry).

- **AARC CPG, Respir Care 2024;69(8):1042-1054, p.1** (recommendation, rrf 0.04839, ranks {'lexical': 2, 'dense': 2, 'cross': 2}, 1 mention(s)) -- **conditional recommendation, very low certainty**  :warning: weak
  - section: AARC Clinical Practice Guideline: Patient-Ventilator Assessment > Joel T Glogowski, and Dean R Hess
  - > We suggest assessing FIO2 to ensure normoxemia (conditional recommendation, very low certainty)

### `action:CRITICAL:fio2`

query: `FiO2 FIO2 oxygen concentration fraction of inspired oxygen inspired oxygen escalation urgent deterioration physician lung-protective assess` | pool: 45 | status: **ok**

- **AARC CPG, Respir Care 2024;69(8):1042-1054, p.1** (recommendation, rrf 0.03279, ranks {'dense': 1, 'cross': 1}, 2 mention(s)) -- **conditional recommendation, very low certainty**
  - section: AARC Clinical Practice Guideline: Patient-Ventilator Assessment > Joel T Glogowski, and Dean R Hess
  - > We suggest assessing FIO2 to ensure normoxemia (conditional recommendation, very low certainty)

### `action:CRITICAL:flow_rate`

query: `flow rate inspiratory flow peak flow escalation urgent deterioration physician lung-protective assess` | pool: 45 | status: **missing**

_no admissible passage -- suppressed in output_

### `action:CRITICAL:peep`

query: `PEEP positive end-expiratory pressure end-expiratory pressure auto-PEEP CPAP escalation urgent deterioration physician lung-protective assess` | pool: 45 | status: **ok**

- **AARC CPG, Respir Care 2024;69(8):1042-1054, p.1** (recommendation, rrf 0.04918, ranks {'lexical': 1, 'dense': 1, 'cross': 1}, 3 mention(s)) -- **strong recommendation, high certainty**
  - section: AARC Clinical Practice Guideline: Patient-Ventilator Assessment > Joel T Glogowski, and Dean R Hess
  - > We recommend an assessment of PEEP and auto-PEEP (strong recommendation, high certainty)

- **AARC Adult Mechanical Ventilator Protocols v1.0a (2003), p.4** (recommendation, rrf 0.03226, ranks {'dense': 2, 'cross': 2}, 2 mention(s))
  - section: I. Adult Invasive Ventilation Protocol Initial Parameters and Goals
  - > 3. PEEP 5 to 15 cm H2O. Set initial PEEP at 5 cm H2O, unless otherwise indicated. Higher PEEPs may be required with acute lung injury (ALI) or acute respiratory distress syndrome (ARDS). [Note: See ALI/ARDS Protocol]

### `action:CRITICAL:pip`

query: `PIP peak inspiratory pressure peak pressure peak airway pressure escalation urgent deterioration physician lung-protective assess` | pool: 45 | status: **weak**

- **AARC Adult Mechanical Ventilator Protocols v1.0a (2003), p.5** (recommendation, rrf 0.03279, ranks {'lexical': 1, 'cross': 1}, 1 mention(s))  :warning: weak
  - section: II. Ventilator Adjustments Based on Patient Assessment
  - > 2. For a pH < 7.30, evaluate to ensure the cause is respiratory. If appropriate, increase rate to a maximum of 24 breaths/min until pH is > 7.30. If further adjustment is needed increase VT until PIP > 40 cm H2O or Pplateau > 30 cm H2O. If unable to maintain these parameters, consider allowing permissive hypercapnia.

### `action:CRITICAL:respiratory_rate_total`

query: `respiratory rate breathing frequency breaths/minute breaths per minute tachypnea escalation urgent deterioration physician lung-protective assess` | pool: 45 | status: **ok**

- **AARC Adult Mechanical Ventilator Protocols v1.0a (2003), p.5** (recommendation, rrf 0.04918, ranks {'lexical': 1, 'dense': 1, 'cross': 1}, 2 mention(s))
  - section: II. Ventilator Adjustments Based on Patient Assessment
  - > 3. For a pH > 7.45, evaluate to ensure the cause is respiratory. If appropriate, reduce rate to a minimum of 8 breaths/minute or until pH is < 7.45. After rate is decreased to 8 breaths/minute, if pH is still > 7.45, reduce volume to a minimum of 4 mL/Kg (IBW).

- **AARC Adult Mechanical Ventilator Protocols v1.0a (2003), p.4** (recommendation, rrf 0.04839, ranks {'lexical': 2, 'dense': 2, 'cross': 2}, 1 mention(s))  :warning: weak
  - section: I. Adult Invasive Ventilation Protocol Initial Parameters and Goals
  - > D. Rate: 8 to 26 breaths/minute adjusted to achieve optimum total cycle time and maintain desired minute ventilation, while maintaining plateau pressure < 30 cm H2O and delta P < 20 cm H2O.

### `action:CRITICAL:minute_volume`

query: `minute ventilation minute volume VE escalation urgent deterioration physician lung-protective assess` | pool: 45 | status: **ok**

- **AARC Adult Mechanical Ventilator Protocols v1.0a (2003), p.4** (recommendation, rrf 0.04865, ranks {'lexical': 2, 'dense': 1, 'cross': 2}, 1 mention(s))  :warning: weak
  - section: I. Adult Invasive Ventilation Protocol Initial Parameters and Goals
  - > D. Rate: 8 to 26 breaths/minute adjusted to achieve optimum total cycle time and maintain desired minute ventilation, while maintaining plateau pressure < 30 cm H2O and delta P < 20 cm H2O.

- **AARC Adult Mechanical Ventilator Protocols v1.0a (2003), p.4** (recommendation, rrf 0.03279, ranks {'lexical': 1, 'cross': 1}, 3 mention(s))
  - section: I. Adult Invasive Ventilation Protocol Initial Parameters and Goals
  - > C. Minute ventilation: 4.0 x BSA (Body Surface Area) = VE (L/min) for males and 3.5 x BSA = VE (L/min) for females adjusted for altitude and body temperature ( DuBois BSA Nomogram) while maintaining plateau pressure < 30 cm H2O and delta P <20 cm H2O.

### `action:CRITICAL:tidal_volume_observed`

query: `tidal volume VT mL/kg lung-protective lung protective predicted body weight escalation urgent deterioration physician lung-protective assess` | pool: 45 | status: **ok**

- **AARC CPG, Respir Care 2024;69(8):1042-1054, p.1** (recommendation, rrf 0.04892, ranks {'lexical': 1, 'dense': 1, 'cross': 2}, 5 mention(s)) -- **strong recommendation, high certainty**
  - section: AARC Clinical Practice Guideline: Patient-Ventilator Assessment > Joel T Glogowski, and Dean R Hess
  - > We recommend an assessment of tidal volume (VT) to ensure lung-protective ventilation (4–8 mL/kg/predicted body weight) (strong recommendation, high certainty)

- **AARC CPG, Respir Care 2024;69(8):1042-1054, p.6** (recommendation, rrf 0.04865, ranks {'lexical': 2, 'dense': 2, 'cross': 1}, 5 mention(s)) -- **strong recommendation, high certainty**
  - section: AARC Clinical Practice Guideline: Patient-Ventilator Assessment > Lung-protective ventilation in patients with ARDS. Petrucci
  - > We recommend an assessment of VT to ensure lung-protective ventilation (4–8 mL/kg/predicted body weight) (strong recommendation, high certainty)

### `action:CRITICAL:etco2`

query: `EtCO2 end-tidal capnography carbon dioxide escalation urgent deterioration physician lung-protective assess` | pool: 45 | status: **missing**

_no admissible passage -- suppressed in output_

### `action:CRITICAL:inspiratory_ratio`

query: `I:E I:E ratio inspiratory time inspiratory-to-expiratory escalation urgent deterioration physician lung-protective assess` | pool: 45 | status: **ok**

- **AARC Adult Mechanical Ventilator Protocols v1.0a (2003), p.3** (recommendation, rrf 0.04918, ranks {'lexical': 1, 'dense': 1, 'cross': 1}, 2 mention(s))
  - section: III. Suggestions for Use of Modes
  - > 3. Pressure support by itself may be effective in patients who have an adequate respiratory drive and who might tolerate mechanical ventilation better when a variable I:E ratio is available.

### `action:CRITICAL:expiratory_ratio`

query: `I:E I:E ratio expiratory time inspiratory-to-expiratory escalation urgent deterioration physician lung-protective assess` | pool: 45 | status: **ok**

- **AARC Adult Mechanical Ventilator Protocols v1.0a (2003), p.3** (recommendation, rrf 0.04918, ranks {'lexical': 1, 'dense': 1, 'cross': 1}, 2 mention(s))
  - section: III. Suggestions for Use of Modes
  - > 3. Pressure support by itself may be effective in patients who have an adequate respiratory drive and who might tolerate mechanical ventilation better when a variable I:E ratio is available.

