# Risk bands and hysteresis, in plain language

**Who this is for:** clinicians, stakeholders and reviewers who want to know what the badge on
the board means and why it behaves the way it does. No maths beyond percentages is assumed.

**What the system does:** it reads ventilator telemetry, scores it, and shows one of four risk
bands per patient. It is read-only. It never controls a ventilator and never recommends
treatment.

**What is being predicted:** whether the patient's breathing deteriorates within the next six
hours, judged against a fixed, written-down definition. Not a diagnosis, and not a decision. A
prioritisation aid.

Every figure below is measured on held-out patient data (MIMIC-IV 3.1) and read from the build
artifacts, not from anyone's judgement.

---

## 1. The risk score

> **Risk score:** a number between 0 and 1 giving the chance the patient deteriorates within six
> hours. 0.25 means roughly a 1-in-4 chance.

Most machine-learning models do not naturally produce a chance. They produce a *ranking*: a
number reliably higher for sicker patients, but whose actual value means nothing. A league table
tells you who is top. It does not tell you whether the top team is any good.

So the score is put through a step called **calibration**, which turns the ranking number into a
real probability. Usefully, calibration changes nothing about the order: the same patient is
still top. It only relabels the scale, the way converting Celsius to Fahrenheit does not change
which day was hotter.

This matters practically. Before calibration, a rule like "alert if the score is above 0.70" is
meaningless, because 0.70 on a ranking scale is not 70% of anything. After calibration, it is.

**Everything below is measured on the calibrated score.**

---

## 2. Bands, and why a bare number is not enough

> **Band:** one of four named risk levels: LOW, MEDIUM, HIGH, CRITICAL.

Telling a clinician "0.31" is not useful. It looks precise and communicates nothing. Telling them
"HIGH, and readings in HIGH were followed by deterioration about 25% of the time, roughly three
times the ward average" is a statement that can be checked, argued with and acted on.

Bands work like triage categories: a few named levels with known meanings, so everybody in the
unit means the same thing by the same word. The names are final, because a downstream LLM writes
its rationale from the band and its measured rate.

### The vocabulary

> **Cut:** the exact score where one band ends and the next begins. The sign on the road where a
> 30 zone becomes a 50 zone.
>
> **Share of readings:** what fraction of all readings sat in that band.
>
> **Observed rate:** of the readings in that band, how often the predicted event actually
> happened afterwards. The band's real-world meaning, not a claim about it.
>
> **Base rate:** the same measurement across every reading from every patient. The house average.
> Here, **8.3%**.
>
> **Lift:** observed rate divided by base rate, i.e. how many times more likely than average.
> Lift 3 means three times the usual chance. Lift below 1 means safer than average.

### The band table

Measured on held-out test data. Base rate 8.3%.

| Band | Cut (band floor) | Share of readings | Observed rate | Lift | Envelope |
|---|---|---|---|---|---|
| LOW | (below 0.1253) | 71.5% | 3.8% | 0.46x | 1.7% to 6.5% |
| MEDIUM | 0.1253 | 17.7% | 11.0% | 1.32x | 8.9% to 14.0% |
| HIGH | 0.2556 | 7.2% | 24.8% | 2.98x | 21.9% to 28.3% |
| CRITICAL | 0.5430 | 3.6% | 51.9% | 6.23x | 49.4% to 57.2% |

**Reading one row.** Take CRITICAL: only 3.6% of readings ever land there, and of those, 51.9%
were followed by respiratory deterioration within six hours, which is 6.23 times the base rate.
Rare and meaningful, which is what a top band has to be. A top band that is not rare is not a
monitor.

**And LOW** is not "no risk". At 3.8% it is well below the 8.3% average, so it means "less than
half the usual risk".

One honesty note: these rates are measured on what a clinician actually **sees** after the
smoothing in section 4, not on the raw score. That is deliberate. The badge is what gets acted
on, so the badge is what should carry a measured meaning.

---

## 3. The cuts were calibrated, not chosen by hand

Nobody decided "HIGH starts at 0.2556". Each cut was **solved** against an **alert budget**.

> **Alert budget:** the maximum number of times per patient per day a patient may be escalated
> into that band.

| Band | Budget (escalations per ventilated patient-day) | Roughly |
|---|---|---|
| MEDIUM | 0.70 | once every day and a half |
| HIGH | 0.45 | once every two days |
| CRITICAL | 0.20 | once every five days |

Each cut is the value that just fits its budget. The unanswerable question "where should HIGH
start?" became the answerable one: "what is the most sensitive line we can afford?"

### Escalations, not hours

> **Promotion event:** the moment a patient moves *up* into a band. One event per crossing.

A patient sitting at HIGH for six hours is **one** alert, not six. The doorbell rings when the
visitor arrives; it does not ring once a minute while they stand on the step.

This is the whole point. Alarm fatigue is caused by interruptions, not by a badge being a colour.
Counting hours would penalise the system for correctly and quietly continuing to flag someone who
is genuinely unwell.

### A counter-intuitive detail

Lowering a cut does not always produce more alerts. Push it very low and every patient crosses once
at the start and never comes back down: one alert per admission, then permanent silence. So the
search runs downwards from the top, and the code refuses to return a near-zero cut that "meets" the
budget by parking everyone at the top band.

---

## 4. Hysteresis: stopping the badge from flickering

### The problem

A patient whose score hovers around 0.2556 would, with a plain threshold, flip HIGH, MEDIUM,
HIGH, MEDIUM all afternoon while nothing clinically meaningful happens. Clinicians stop trusting
a badge that does that, and rightly so. A twitching indicator is noise dressed as information.

> **Hysteresis:** the umbrella term for the two mechanisms below that make the badge stickier than
> the raw number. It is what a thermostat does so the boiler does not click on and off every
> thirty seconds.

### Mechanism 1: the deadband

> **Deadband (or demote margin):** to come *down* out of a band, the score must fall clearly below
> the line it crossed, not merely back across it.

The margin is **25%** of the gap down to the band below:

| To leave | Score had to rise above | ...but only drops below | Dead zone |
|---|---|---|---|
| MEDIUM | 0.1253 | 0.0940 | 0.0940 to 0.1253 |
| HIGH | 0.2556 | 0.2230 | 0.2230 to 0.2556 |
| CRITICAL | 0.5430 | 0.4712 | 0.4712 to 0.5430 |

Inside the dead zone nothing happens and the badge stays put. Exactly a thermostat set to 20
degrees that fires the boiler at 19.5 and stops it at 20.5: one line to go up, a different, lower
line to come back down. The margin is a fraction rather than a fixed amount because a flat 0.03
would be trivial at the top of the scale and overwhelming at the bottom.

### Mechanism 2: dwell

> **Dwell:** the score must stay on the far side of the line for a set time before the badge
> changes.

The fitted settings are deliberately **asymmetric**:

- **Promotion: immediate.** The first reading above the cut promotes. No waiting.
- **Demotion: 120 minutes.** The score must sit below the deadband for two hours first.

A fire alarm sounds the instant it detects smoke, and does not switch itself off the moment the
smoke thins. Waiting before escalating would cost warning time, which is the one thing this
system exists to provide, so the promote side waits for nothing.

One guard: if data stops arriving for more than **four hours**, any pending change is cancelled,
because "sustained for two hours" cannot honestly be claimed across a gap in the record. It does
not drop a badge a patient already has. Dropping someone's badge because nobody charted would be
its own kind of lie.

### The words for what is happening

> **Promotion:** the badge moving up a band. **Demotion:** the badge moving down.
>
> **Latch:** a single on/off switch at one boundary. There are three, one per boundary, and the
> displayed band is simply **how many are on**: none is LOW, one MEDIUM, two HIGH, three CRITICAL.
>
> **Band state:** whether the badge currently agrees with the raw score.
> **confirmed** (they agree, normal), **provisional** (the score has just crossed upward and the
> badge is catching up), **demoting** (the score has dropped but the badge is held up, waiting out
> the two hours). Like a pending transaction on a bank statement: real, and labelled so you know
> it is still settling.

Three independent switches rather than one four-way dial is what let each cut be solved against
its own budget separately.

### The measured effect, which is the surprising part

A fair comparison holds the cuts fixed, so it measures the machine and not a moved goalpost:

| At cuts 0.1253 / 0.2556 / 0.5430 | Band changes per patient-day | Detection at HIGH or above |
|---|---|---|
| No hysteresis | 3.170 | 32.7% |
| With hysteresis | 1.760 | 43.3% |

**44.5% fewer badge changes, and better detection at the same time.**

That is the quotable result, because it is not the trade-off anyone expects. Smoothing usually
costs sensitivity. Here it does not: the promote side has no waiting, so nothing is detected
late, while the demote side stops the badge falling off a patient about to bounce straight back
up. Median warning time lost to hysteresis is **0 minutes**.

("Detection at HIGH or above" means: of all readings really followed by deterioration within six
hours, the share where the badge showed HIGH or CRITICAL at the time.)

Holding the *budget* fixed instead and letting the cuts move gives the same verdict from the other
direction: hysteresis buys a lower, more sensitive MEDIUM line (0.2352 down to 0.1310) and lifts
detection from 27.0% to 45.0%, for about 12% more badge changes.

### One thing checked because it could have gone wrong

Holding badges up means patients spend about 1.90 times longer at the top band than the raw score
alone implies. That is the mechanism working. The real risk is that effect *growing* with length of
stay, quietly gluing long-stay patients to CRITICAL. Across four groups by stay length (median 5,
21, 66 and 229 hours) the ratios were 1.53, 1.69, 1.91 and 1.99: only 1.31 times growth end to end,
against an agreed ceiling of 1.5. Re-checked on every retrain.

---

## 5. The envelope: what stops a retrain quietly changing the meaning

> **Envelope:** the range each band's observed rate is permitted to occupy. If a retrained model
> moves a band's real rate outside its envelope, **the build fails.**

HIGH currently means 24.8%, with an envelope of 21.9% to 28.3%. Retrain on new data and HIGH might
drift to 26%: fine. Drift to 15% and the build stops so someone has to look.

Without this, HIGH could come to mean something materially different while the word on the screen,
the LLM prompt built on it and the clinician's learned instinct all stayed the same. The failure
mode the envelope prevents is not a wrong number, it is a **stale meaning**.

The ranges are not guesses. They come from resampling the calibration data 400 times at the
patient level (so many readings from one patient cannot masquerade as independent evidence) plus a
2 percentage point tolerance. Think of it as the calibration certificate on ward equipment: it
must still read within its stated range against a reference, or it is out of service rather than
quietly used anyway.

---

## 6. What these numbers do not cover

- **Measured offline, not at a bedside.** All of this comes from retrospective ICU data, not a
  live ventilator feed.
- **The two-hour wait belongs to this data's rhythm.** In the source data, ventilator settings are
  charted roughly once an hour. A bedside feed is far faster, so the timing would have to be
  re-expressed for it, and this dataset can neither confirm nor rule out a much shorter setting: it
  cannot see that finely. The system stores both a time and a reading count for exactly this reason.
- **Some of the model's skill comes from charting patterns, not physiology.** About 18% comes from
  *which* measurements were taken and how recently, rather than their values. That is reported
  alongside every result rather than hidden, and generated explanations must keep the two apart.
- **The cuts are tied to how the score is computed.** They were fitted on scores produced on the
  GPU, and the same model scored on the CPU disagrees enough to shift every band.

---

## Quick reference

| Term | One line |
|---|---|
| Risk score | A calibrated probability, 0 to 1, that deterioration follows within six hours. |
| Band | One of LOW, MEDIUM, HIGH, CRITICAL. |
| Cut | The score value where a band boundary sits. |
| Share of readings | What fraction of all readings land in that band. |
| Observed rate | How often the predicted event actually happened, for readings in that band. |
| Base rate | The same figure across all patients: 8.3%. |
| Lift | Observed rate divided by base rate. How many times more likely than average. |
| Envelope | The range a band's observed rate may occupy. Outside it, the build fails. |
| Deadband | The score must fall clearly below the line it crossed, by 25% of the band below. |
| Dwell | And it must stay there: two hours to come down, no wait at all to go up. |
| Promotion / demotion | The badge moving up / moving down. |
| Latch | One on/off switch per boundary. The band is how many are on. |
| Band state | confirmed, provisional or demoting: whether the badge agrees with the score. |
| Hysteresis | Deadband plus dwell together. The reason the badge does not flicker. |
| Alert budget | The cap on escalations per patient per day each cut was solved against. |
| Promotion event | One escalation. Six hours held at HIGH is one alert, not six. |

---

*Sources: `models/risk_bands_y_resp_6h.json`, `bki/reports/s16_bands.json`, `bki/pipeline/core/bands.py`,
`bki/pipeline/stages/s16_bands.py`. Every figure is read from the build artifacts. Aggregate statistics
only.*
