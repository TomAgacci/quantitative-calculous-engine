# 6‑STEP QUANTITATIVE CALCULOUS BREAKDOWN
# (Full formula form with quantitative check)

1.  Start with two numbers:
        a ≥ b > 0

2.  Initialize:
        r₀ = a
        r₁ = b

3.  First division:
        r₀ = q₁ r₁ + r₂
        0 ≤ r₂ < r₁

4.  Second division:
        r₁ = q₂ r₂ + r₃
        0 ≤ r₃ < r₂

5.  General iterative step:
        r₍k−1₎ = qₖ rₖ + r₍k+1₎
        0 ≤ r₍k+1₎ < rₖ

6.  Termination:
        rₙ = 0
        r₍n−1₎ ≠ 0

7.  Quantitative outcome:
        R = r₍n−1₎

# QUANTITATIVE CHECK

8.  Verify:
        R ∣ a   and   R ∣ b

9.  Equivalent form:
        a = k₁ R
        b = k₂ R
        (k₁, k₂ are integers)

# End of full breakdown
