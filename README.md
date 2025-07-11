# 🔢 Weighted Model Counting (WMC)

This repository provides an implementation and evaluation of **Weighted Model Counting (WMC)** methods for propositional logic formulas. The project explores both **exact** and **approximate** approaches, and includes formula generation via the **Google Gemini API** to test on diverse cases.

---

## 🎯 Objective

- Implement and evaluate algorithms for computing the **Weighted Model Count** of propositional formulas.
- Provide a unified framework for testing and comparing multiple WMC techniques.

---

## 🎯 Goal

- Compare **exact methods**:
  - **Knowledge Compilation** (e.g., d-DNNF, BDDs)
  - **Truth Table Enumeration**
- With **approximate methods**:
  - **SampleSAT**-based probabilistic sampling
- Use **Google Gemini API** to generate diverse and scalable test formulas.

---

## 🧱 Features

- ✅ Exact WMC via truth table enumeration  
- ✅ Exact WMC via knowledge compilation (using existing libraries)  
- ✅ Approximate WMC via SampleSAT  
- ✅ Integration with Google Gemini API for formula generation  
- ✅ Performance benchmarking and comparison metrics (accuracy, time)



