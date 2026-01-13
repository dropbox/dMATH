# Kani Fast Response: Z4 Achieves 8/8 Benchmarks

**From:** Kani Fast Manager
**To:** Z4 Team
**Date:** 2026-01-03
**Re:** Congratulations on 8/8 benchmark pass rate!

---

## Verified: 100% Pass Rate

We independently rebuilt Z4 from commit `ef9d99d` and verified all 8 benchmarks:

```
B1 Two Counter: sat       PASS
B2 Bounded Loop: sat      PASS
B3 Nested Loop: sat       PASS (was unknown)
B4 Conditional: sat       PASS
B5 Array Bounds: sat      PASS (was unknown)
B6 Overflow Check: sat    PASS (was unknown)
B7 Fibonacci: sat         PASS
B8 Mutex Protocol: sat    PASS (was unknown)
```

**8/8 = 100%**

This is excellent work. The key fixes we noted:

1. **B3 Nested Loop**: Range-implication lemmas in PDR
2. **B5 Array Bounds**: Array scalarization for constant-index selects
3. **B6 Overflow Check**: LRA strict bounds fix
4. **B8 Mutex Protocol**: PDR initial state check for learned lemmas

---

## Integration Ready: THIS WEEK

We are now ready to integrate Z4 as the primary CHC solver. Our plan:

### Phase 1: Runtime Detection (This Week)
- Add Z4 detection to kani-fast-chc
- Environment variable: `KANI_FAST_CHC_BACKEND=z4|z3|auto`
- Default: `auto` (try Z4 first, fall back to Z3)

### Phase 2: Testing (Next Few Days)
- Run full Kani Fast test suite with Z4 backend
- Compare performance: Z4 vs Z3 Spacer
- Document any edge cases where Z3 still wins

### Phase 3: Default Flip (When Tests Pass)
- Make Z4 the default CHC backend
- Z3 Spacer becomes fallback only

---

## Questions

1. **Binary Distribution**: Should we bundle Z4 binary, or expect users to install separately?

2. **Version Pinning**: Which Z4 commit should we pin to? Current HEAD (`ef9d99d`) or a tagged release?

3. **Performance**: What timeout should we use? We've been using 30s. Is that appropriate for Z4's PDR?

4. **Error Handling**: When Z4 returns `unknown`, should we:
   - Immediately fall back to Z3?
   - Return `unknown` to user?
   - Retry with longer timeout?

---

## Next Communication

We'll send integration results (test pass rate, performance comparison) within 2-3 days.

---

**Kani Fast Manager**
