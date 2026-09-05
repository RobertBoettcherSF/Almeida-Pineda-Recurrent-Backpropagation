# Almeida-Pineda Recurrent Backpropagation in Ada 2023

---

## Project Overview

This repository contains a complete, strongly-typed Ada 2023 implementation of the **Almeida–Pineda recurrent backpropagation algorithm**. This supervised learning algorithm extends standard backpropagation to Continuous-Time Recurrent Neural Networks (CTRNNs) or discrete iterated networks by running the network dynamics forward to a stable fixed point, and then calculating error gradients by running a reciprocal, transposed linear error network backward to its own fixed point.

---

## Features

- **Variants Supported:** Includes both Synchronous and Asynchronous (Gauss-Seidel) update modes for the relaxation steps. Asynchronous mode updates states in-place, which typically stabilizes oscillating dynamics faster.
- **Strong Typing:** Uses custom strictly-typed indices (`Node_Count`, `Node_Index`) instead of bare integers to prevent domain confusion.
- **Contract Driven:** Employs Ada 2012/2023 aspects (`Pre`, `Global`) and runtime exception verification (`Dimension_Error`) for robust boundaries validation.
- **Independent Phases:** Public subprograms for `Forward_Relaxation`, `Backward_Relaxation`, and `Update_Weights` exist independent of the unified `Train_Pattern` routine to support custom network loop integrations.
- **No External Dependencies:** Works purely with Ada standard libraries (Numerics, Text\_IO).

---

## Usage

The algorithm is demonstrated explicitly through the standalone test suite (`tests.adb`). Building the project will produce a binary that exercises all edge cases and API features.

To build and run tests:

```bash
make test
```

**Expected Output:**

```plaintext
Running tests...
========================================
Almeida-Pineda Algorithm Test Suite
========================================
TEST 1 — Sigmoid Math Edge Cases
  PASS — 1.1 Sigmoid(0) equals 0.5
  PASS — 1.2 Sigmoid(100) clamps to 1.0
...
===  39 passed,  0 failed ===
```

---

## Testing

The test suite ensures total correctness through 13 discrete categories, each with multiple assertions:

- **Functional Correctness:** Validates that network weights adapt correctly relative to defined errors and that forward states correctly lock onto target behaviors.
- **Math Edge Cases:** Checks saturation extremes for sigmoidal activations, averting floating-point explosion (underflow/overflow bounds testing).
- **Error Handling:** Uses deliberate matrix mismatches to catch `Dimension_Error` exceptions, fulfilling constraints robustly before risking memory corruption.
- **Invariants:** Validates that state gradients calculate transposed weights independently between Asynchronous and Synchronous approaches.

---

## Building

**Prerequisites:** GNAT (GNU NYU Ada Translator), or an equivalent Ada 2022/2023 capable compiler. Make sure `gnatmake` is installed.

**Language:** Standard Ada 2023 (ISO/IEC 8652:2023). Uses 2012+ standard features like `Pre`, `Global` aspects and conditional expressions.
