"""Charlson comorbidity mapping, Quan et al. (2005) enhanced coding algorithm.

MIMIC-IV stores ICD codes without dots and mixes ICD-9 (45.7%) with ICD-10
(54.3%), so both code sets are required -- an ICD-9-only mapping would silently
drop half the diagnosis records.
"""
from __future__ import annotations


def _num_range(lo: int, hi: int, width: int = 3) -> list[str]:
    """Expand an inclusive numeric ICD-9 range into zero-padded prefixes."""
    return [str(i).zfill(width) for i in range(lo, hi + 1)]


def _alpha_range(letter: str, lo: int, hi: int) -> list[str]:
    """Expand an inclusive ICD-10 range such as C00-C26."""
    return [f"{letter}{str(i).zfill(2)}" for i in range(lo, hi + 1)]


# category -> (icd9 prefixes, icd10 prefixes, Charlson weight)
CHARLSON: dict[str, tuple[list[str], list[str], int]] = {
    "myocardial_infarction": (
        ["410", "412"],
        ["I21", "I22", "I252"], 1),
    "congestive_heart_failure": (
        ["39891", "40201", "40211", "40291", "40401", "40403", "40411", "40413",
         "40491", "40493", "4254", "4255", "4256", "4257", "4258", "4259", "428"],
        ["I099", "I110", "I130", "I132", "I255", "I420", "I425", "I426", "I427",
         "I428", "I429", "I43", "I50", "P290"], 1),
    "peripheral_vascular": (
        ["0930", "4373", "440", "441", "4431", "4432", "4433", "4434", "4435",
         "4436", "4437", "4438", "4439", "4471", "5571", "5579", "V434"],
        ["I70", "I71", "I731", "I738", "I739", "I771", "I790", "I792", "K551",
         "K558", "K559", "Z958", "Z959"], 1),
    "cerebrovascular": (
        ["36234"] + _num_range(430, 438),
        ["G45", "G46", "H340"] + _alpha_range("I", 60, 69), 1),
    "dementia": (
        ["290", "2941", "3312"],
        ["F00", "F01", "F02", "F03", "F051", "G30", "G311"], 1),
    "chronic_pulmonary": (
        ["4168", "4169"] + _num_range(490, 505) + ["5064", "5081", "5088"],
        ["I278", "I279"] + _alpha_range("J", 40, 47) + _alpha_range("J", 60, 67)
        + ["J684", "J701", "J703"], 1),
    "rheumatic": (
        ["4465", "7100", "7101", "7102", "7103", "7104", "7140", "7141", "7142",
         "7148", "725"],
        ["M05", "M06", "M315", "M32", "M33", "M34", "M351", "M353", "M360"], 1),
    "peptic_ulcer": (
        ["531", "532", "533", "534"],
        ["K25", "K26", "K27", "K28"], 1),
    "mild_liver_disease": (
        ["07022", "07023", "07032", "07033", "07044", "07054", "0706", "0709",
         "570", "571", "5733", "5734", "5738", "5739", "V427"],
        ["B18", "K700", "K701", "K702", "K703", "K709", "K713", "K714", "K715",
         "K717", "K73", "K74", "K760", "K762", "K763", "K764", "K768", "K769",
         "Z944"], 1),
    "diabetes_uncomplicated": (
        ["2500", "2501", "2502", "2503", "2508", "2509"],
        [f"E{d}{s}" for d in ("10", "11", "12", "13", "14")
         for s in ("0", "1", "6", "8", "9")], 1),
    "diabetes_complicated": (
        ["2504", "2505", "2506", "2507"],
        [f"E{d}{s}" for d in ("10", "11", "12", "13", "14")
         for s in ("2", "3", "4", "5", "7")], 2),
    "paraplegia_hemiplegia": (
        ["3341", "342", "343", "3440", "3441", "3442", "3443", "3444", "3445",
         "3446", "3449"],
        ["G041", "G114", "G801", "G802", "G81", "G82", "G830", "G831", "G832",
         "G833", "G834", "G839"], 2),
    "renal_disease": (
        ["40301", "40311", "40391", "40402", "40403", "40412", "40413", "40492",
         "40493", "582", "5830", "5831", "5832", "5833", "5834", "5835", "5836",
         "5837", "585", "586", "5880", "V420", "V451", "V56"],
        ["I120", "I131", "N032", "N033", "N034", "N035", "N036", "N037", "N052",
         "N053", "N054", "N055", "N056", "N057", "N18", "N19", "N250", "Z490",
         "Z491", "Z492", "Z940", "Z992"], 2),
    "malignancy": (
        _num_range(140, 172) + _num_range(174, 195) + _num_range(200, 208) + ["2386"],
        _alpha_range("C", 0, 26) + _alpha_range("C", 30, 34) + _alpha_range("C", 37, 41)
        + ["C43"] + _alpha_range("C", 45, 58) + _alpha_range("C", 60, 76)
        + _alpha_range("C", 81, 85) + ["C88"] + _alpha_range("C", 90, 97), 2),
    "severe_liver_disease": (
        ["4560", "4561", "4562", "5722", "5723", "5724", "5725", "5726", "5727", "5728"],
        ["I850", "I859", "I864", "I982", "K704", "K711", "K721", "K729", "K765",
         "K766", "K767"], 3),
    "metastatic_tumor": (
        ["196", "197", "198", "199"],
        ["C77", "C78", "C79", "C80"], 6),
    "aids_hiv": (
        ["042", "043", "044"],
        ["B20", "B21", "B22", "B24"], 6),
}


def category_sql(name: str) -> str:
    """A boolean SQL expression selecting one Charlson category."""
    icd9, icd10, _ = CHARLSON[name]
    p9 = " OR ".join(f"starts_with(code, '{p}')" for p in icd9)
    p10 = " OR ".join(f"starts_with(code, '{p}')" for p in icd10)
    return f"((icd_version = 9 AND ({p9})) OR (icd_version = 10 AND ({p10})))"


def score_sql(prefix: str = "cci_") -> str:
    """Weighted Charlson comorbidity index across the per-category flags."""
    return " + ".join(f"{w} * COALESCE({prefix}{n}, 0)"
                      for n, (_, _, w) in CHARLSON.items())


def age_points_sql(age_col: str = "age_at_icu") -> str:
    """Standard Charlson age adjustment."""
    return (f"CASE WHEN {age_col} < 50 THEN 0 WHEN {age_col} < 60 THEN 1 "
            f"WHEN {age_col} < 70 THEN 2 WHEN {age_col} < 80 THEN 3 ELSE 4 END")
