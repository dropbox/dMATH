# How This Was Built

**TLA2 was built by AI, directed by a human who doesn't deeply understand TLA+.**

This is not an apology. It's the point.

---

## The Thesis

Domain expertise is no longer a prerequisite for building domain-specific tooling.

TLA2 is a ~25,000 line Rust codebase that reimplements the TLA+ model checker (TLC) and proof manager (TLAPM). The original implementations are:

- **TLC**: ~93,000 lines of Java
- **TLAPM**: ~67,000 lines of OCaml
- **SANY** (parser): ~52,000 lines of Java

The human directing this project has a grad school background and builds distributed systems. He recognized that TLA+ tooling was painful to use and decided to build something better. He did not spend a year becoming a TLC expert. He hired one.

His expert happens to be AI, and it costs electricity.

---

## The Bulldozer Analogy

The person operating a bulldozer doesn't need to be able to lift rocks. The bulldozer does the lifting. The operator provides direction, judgment, and decides when the job is done.

The person directing AI workers doesn't need to understand temporal logic, parser combinators, or tableau provers. The AI reads the 160,000 lines of baseline implementations. The AI writes the Rust code. The AI iterates until the tests pass.

The human provides:
- **Direction**: "Make it equivalent to TLC"
- **Judgment**: "Memory usage is blocking, fix it"
- **Taste**: "The error messages should be beautiful"
- **Will**: "Ship it when 16/16 comparison tests pass"

The human does not provide:
- Implementation
- Deep domain expertise in formal methods
- The ability to read SANY's Java or TLAPM's OCaml
- Years of TLC debugging experience

---

## The Steve Jobs A/B Test

Steve Jobs couldn't write code. Steve Wozniak could. But Wozniak didn't build Apple into a trillion-dollar company.

Apple's history is as close to a controlled experiment as you'll find:

| Period | Jobs Present | Result |
|--------|--------------|--------|
| 1976-1985 | Yes | Created Apple, Macintosh, changed computing |
| 1985-1997 | No | Near bankruptcy, 90 days from insolvent in 1997 |
| 1997-2011 | Yes | iPod, iPhone, iPad, most valuable company on Earth |

Jobs provided direction, taste, judgment, and will. He didn't provide implementation. The division of labor was clean: engineers built, Jobs directed.

This project applies the same division. AI builds. Human directs.

---

## How It Actually Works

### The Process

1. **Baseline exists**: TLC (Java) and TLAPM (OCaml) are working implementations
2. **AI reads baseline**: The AI studies the existing code to understand semantics
3. **AI writes Rust**: Port the logic to Rust, maintaining semantic equivalence
4. **Tests verify correctness**: Run both TLC and TLA2 on the same specs, compare results
5. **Iterate**: When tests fail, AI debugs and fixes

### The Iteration Count

At time of writing, there have been ~400 AI worker iterations on this codebase. Each iteration is roughly 12 minutes of AI work. The AI commits its work with structured messages so the next AI session can continue.

The human reads the commit logs, runs the test suite, and provides direction. The human does not write code.

### What Tests Prove

The human doesn't need to understand TLA+ semantics to verify correctness. The test suite does that:

```
tests/tlc_comparison/
├── test_core.py      # Core operator semantics
├── test_mutex.py     # Mutual exclusion specs
├── test_distributed.py # Distributed system specs
└── ...
```

Each test runs the same spec through both TLC and TLA2, then compares:
- Number of states found
- Error detection (both find the same bugs or both verify)
- Counterexample structure

If TLA2 produces the same results as TLC across a comprehensive test suite, the implementation is correct. The human verifies that "16/16 tests passing" means correctness. The human doesn't need to verify the temporal logic.

---

## The Economics

### Old Model

To build TLA+ tooling, you needed:
- Deep understanding of temporal logic (1+ years to acquire)
- Ability to read and modify 160K lines of Java/OCaml
- Hire formal methods experts ($200-400K/year, scarce talent)

### New Model

- AI reads the baseline implementations
- AI writes the port
- Human provides direction and runs tests
- Cost: electricity + API fees

The expertise is rented, not acquired. The rental is cheap and scales with compute.

---

## The Uncomfortable Implication

Leslie Lamport spent decades developing TLA+ so engineers could think more precisely about systems. The value of TLA+ is in *writing* specifications—the act of precise specification catches bugs before any tool runs.

This project uses that work to build tools so engineers don't have to think at all. The AI writes the formal methods tooling. Soon, AI will write the specifications. Eventually, AI writes the systems and proves them correct.

The irony is sharp: formal verification—the pinnacle of rigorous human reasoning—is being commoditized by machines that don't reason. They pattern-match against working implementations.

Whether this is a good future is not the project's concern. The project's concern is whether TLA2 produces the same results as TLC. It does. 16/16 tests pass.

---

## What This Demonstrates

TLA2 is a proof of concept that:

1. **AI can comprehend large legacy codebases** (93K Java + 67K OCaml)
2. **AI can port semantics across languages** while maintaining correctness
3. **AI can iterate toward equivalence** through testing, not understanding
4. **Humans can direct without implementing** when verification is automated

This is not about TLA+. TLA+ is the test case. The pattern generalizes.

---

## For Skeptics

**"But you must understand the domain to direct effectively."**

You need to understand the domain enough to:
- Recognize the problem exists (painful tooling)
- Evaluate whether solutions work (test suite passes)
- Set direction (make it faster, use less memory)
- Know when to ship (comparison tests pass)

You do not need to understand the domain enough to implement solutions. That's what the AI is for.

**"What about subtle bugs the tests don't catch?"**

The same risk exists with human implementers. The mitigation is the same: comprehensive testing, property-based testing, comparison against baseline, and production monitoring.

TLA2 has ~1,175 unit tests, 64 validated specs, and a comparison test suite against TLC. This is more rigorous than most human-written software.

**"Isn't this just outsourcing?"**

Yes. It's outsourcing to an entity that:
- Costs electricity instead of salary
- Is available immediately
- Works 24/7
- Doesn't quit
- Can be parallelized
- Has read more code than any human

The unusual part is the economics, not the structure.

---

## Conclusion

TLA2 was built by trading electricity for expertise. The human provided direction, taste, judgment, and the will to ship. The AI provided implementation.

This is not the future of software engineering. It is the present. The only question is whether you're building the bulldozer or still lifting rocks by hand.
