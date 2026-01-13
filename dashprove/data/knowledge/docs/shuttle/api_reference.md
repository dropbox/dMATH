# Crate shuttle Copy item path

[Source][1]
Expand description

Shuttle is a library for testing concurrent Rust code, heavily inspired by [Loom][2].

Shuttle focuses on randomized testing, rather than the exhaustive testing that Loom offers. This is
a soundness—scalability trade-off: Shuttle is not sound (a passing Shuttle test does not prove the
code is correct), but it scales to much larger test cases than Loom. Empirically, randomized testing
is successful at finding most concurrency bugs, which tend not to be adversarial.

### [§][3]Testing concurrent code

Consider this simple piece of concurrent code:

`use std::sync::{Arc, Mutex};
use std::thread;

let lock = Arc::new(Mutex::new(0u64));
let lock2 = lock.clone();

thread::spawn(move || {
    *lock.lock().unwrap() = 1;
});

assert_eq!(0, *lock2.lock().unwrap());`

There is an obvious race condition here: if the spawned thread runs before the assertion, the
assertion will fail. But writing a unit test that finds this execution is tricky. We could run the
test many times and try to “get lucky” by finding a failing execution, but that’s not a very
reliable testing approach. Even if the test does fail, it will be difficult to debug: we won’t be
able to easily catch the failure in a debugger, and every time we make a change, we will need to run
the test many times to decide whether we fixed the issue.

#### [§][4]Randomly testing concurrent code with Shuttle

Shuttle avoids this issue by controlling the scheduling of each thread in the program, and
scheduling those threads *randomly*. By controlling the scheduling, Shuttle allows us to reproduce
failing tests deterministically. By using random scheduling, with appropriate heuristics, Shuttle
can still catch most (non-adversarial) concurrency bugs even though it is not an exhaustive checker.

A Shuttle version of the above test just wraps the test body in a call to Shuttle’s
[check_random][5] function, and replaces the concurrency-related imports from `std` with imports
from `shuttle`:

[ⓘ][6]
`use shuttle::sync::{Arc, Mutex};
use shuttle::thread;

shuttle::check_random(|| {
    let lock = Arc::new(Mutex::new(0u64));
    let lock2 = lock.clone();

    thread::spawn(move || {
        *lock.lock().unwrap() = 1;
    });

    assert_eq!(0, *lock2.lock().unwrap());
}, 100);`

This test detects the assertion failure with extremely high probability (over 99.9999%).

### [§][7]Testing non-deterministic code

Shuttle supports testing code that uses *data non-determinism* (random number generation). For
example, this test uses the [`rand`][8] crate to generate a random number:

`use rand::{thread_rng, Rng};

let x = thread_rng().gen::<u64>();
assert_eq!(x % 10, 7);`

Shuttle provides its own implementation of [`rand`][9] that is a drop-in replacement:

[ⓘ][10]
`use shuttle::rand::{thread_rng, Rng};

shuttle::check_random(|| {
    let x = thread_rng().gen::<u64>();
    assert_ne!(x % 10, 7);
}, 100);`

This test will run the body 100 times, and fail if any of those executions fails; the test therefore
fails with probability 1-(9/10)^100, or 99.997%. We can increase the `100` parameter to run more
executions and increase the probability of finding the failure. Note that Shuttle isn’t doing
anything special to increase the probability of this test failing other than running the body
multiple times.

When this test fails, Shuttle provides output that can be used to **deterministically** reproduce
the failure:

`test panicked in task "task-0" with schedule: "910102ccdedf9592aba2afd70104"
pass that schedule string into `shuttle::replay` to reproduce the failure`

We can use Shuttle’s [`replay`][11] function to replay the execution that causes the failure:

[ⓘ][12]
`use shuttle::rand::{thread_rng, Rng};

shuttle::replay(|| {
    let x = thread_rng().gen::<u64>();
    assert_ne!(x % 10, 7);
}, "910102ccdedf9592aba2afd70104");`

This runs the test only once, and is guaranteed to reproduce the failure.

Support for data non-determinism is most useful when *combined* with support for schedule
non-determinism (i.e., concurrency). For example, an integration test might spawn several threads,
and within each thread perform a random sequence of actions determined by `thread_rng` (this style
of testing is often referred to as a “stress test”). By using Shuttle to implement the stress test,
we can both increase the coverage of the test by exploring more thread interleavings and allow test
failures to be deterministically reproducible for debugging.

### [§][13]Writing Shuttle tests

To test concurrent code with Shuttle, all uses of synchronization primitives from `std` must be
replaced by their Shuttle equivalents. The simplest way to do this is via `cfg` flags. Specifically,
if you enforce that all synchronization primitives are imported from a single `sync` module in your
code, and implement that module like this:

`#[cfg(all(feature = "shuttle", test))]
use shuttle::{sync::*, thread};
#[cfg(not(all(feature = "shuttle", test)))]
use std::{sync::*, thread};`

Then a Shuttle test can be written like this:

`#[cfg(feature = "shuttle")]
#[test]
fn concurrency_test_shuttle() {
    use my_crate::*;
    // ...
}`

and be executed by running `cargo test --features shuttle`.

#### [§][14]Choosing a scheduler and running a test

Shuttle tests need to choose a *scheduler* to use to direct the execution. The scheduler determines
the order in which threads are scheduled. Different scheduling policies can increase the probability
of detecting certain classes of bugs (e.g., race conditions), but at the cost of needing to test
more executions.

Shuttle has a number of built-in schedulers, which implement the [`Scheduler`][15] trait. They are
most easily accessed via convenience methods:

* [`check_random`][16] runs a test using a random scheduler for a chosen number of executions.
* [`check_pct`][17] runs a test using the [Probabilistic Concurrency Testing][18] (PCT) algorithm.
  PCT bounds the number of preemptions a test explores; empirically, most concurrency bugs can be
  detected with very few preemptions, and so PCT increases the probability of finding such bugs. The
  PCT scheduler can be configured with a “bug depth” (the number of preemptions) and a number of
  executions.
* [`check_dfs`][19] runs a test with an *exhaustive* scheduler using depth-first search. Exhaustive
  testing is intractable for all but the very simplest programs, and so using this scheduler is not
  recommended, but it can be useful to thoroughly test small concurrency primitives. The DFS
  scheduler can be configured with a bound on the depth of schedules to explore.

When these convenience methods do not provide enough control, Shuttle provides a [`Runner`][20]
object for executing a test. A runner is constructed from a chosen [scheduler][21], and then invoked
with the [`Runner::run`][22] method. Shuttle also provides a [`PortfolioRunner`][23] object for
running multiple schedulers, using parallelism to increase the number of test executions explored.

## Modules[§][24]

*[annotations][25]*
  Annotated schedules. When an execution is scheduled using the
  [`crate::scheduler::AnnotationScheduler`][26], Shuttle will produce a file that contains
  additional information about the execution, such as the kind of step that was taken (was a task
  created, were permits acquired from a semaphore, etc) as well as the task’s vector clocks and thus
  any causal dependence between the tasks. The resulting file can be visualized using the Shuttle
  Explorer IDE extension.
*[current][27]*
  Information about the current thread and current Shuttle execution.
*[future][28]*
  Shuttle’s implementation of an async executor, roughly equivalent to [`futures::executor`][29].
*[hint][30]*
  Shuttle’s implementation of [`std::hint`][31].
*[lazy_static][32]*
  Shuttle’s implementation of the [`lazy_static`][33] crate, v1.4.0.
*[rand][34]*
  Shuttle’s implementation of the [`rand`][35] crate, v0.8.
*[scheduler][36]*
  Implementations of different scheduling strategies for concurrency testing.
*[sync][37]*
  Shuttle’s implementation of [`std::sync`][38].
*[thread][39]*
  Shuttle’s implementation of [`std::thread`][40].

## Macros[§][41]

*[lazy_static][42]*
  Declare a new [lazy static value][43], like the `lazy_static` crate.
*[thread_local][44]*
  Declare a new thread local storage key of type [`LocalKey`][45].

## Structs[§][46]

*[Config][47]*
  Configuration parameters for Shuttle
*[PortfolioRunner][48]*
  A `PortfolioRunner` is the same as a `Runner`, except that it can run multiple different
  schedulers (a “portfolio” of schedulers) in parallel. If any of the schedulers finds a failing
  execution of the test, the entire run fails.
*[Runner][49]*
  A `Runner` is the entry-point for testing concurrent code.

## Enums[§][50]

*[FailurePersistence][51]*
  Specifies how to persist schedules when a Shuttle test fails
*[MaxSteps][52]*
  Specifies an upper bound on the number of steps a single iteration of a Shuttle test can take, and
  how to react when the bound is reached.

## Functions[§][53]

*[annotate_replay][54]*
  Run the given function according to a given encoded schedule, usually produced as the output of a
  failing Shuttle test case, while recording an annotated schedule, for use with the Shuttle
  Explorer extension.
*[check_dfs][55]*
  Run the given function under a depth-first-search scheduler until all interleavings have been
  explored (but if the max_iterations bound is provided, stop after that many iterations).
*[check_pct][56]*
  Run the given function under a PCT concurrency scheduler for some number of iterations at the
  given depth. Each iteration will run a (potentially) different randomized schedule.
*[check_random][57]*
  Run the given function under a randomized concurrency scheduler for some number of iterations.
  Each iteration will run a (potentially) different randomized schedule.
*[check_random_with_seed][58]*
  Run function `f` using `RandomScheduler` initialized with the provided `seed` for the given
  `iterations`. This makes generating the random seed for each execution independent from
  `RandomScheduler`. Therefore, this can be used with a library (like proptest) that takes care of
  generating the random seeds.
*[check_uncontrolled_nondeterminism][59]*
  Run the given function under a scheduler that checks whether the function contains randomness
  which is not controlled by Shuttle. Each iteration will check a different random schedule and
  replay that schedule once.
*[replay][60]*
  Run the given function according to a given encoded schedule, usually produced as the output of a
  failing Shuttle test case.
*[replay_from_file][61]*
  Run the given function according to a schedule saved in the given file, usually produced as the
  output of a failing Shuttle test case.

[1]: ../src/shuttle/lib.rs.html#1-560
[2]: https://github.com/tokio-rs/loom
[3]: #testing-concurrent-code
[4]: #randomly-testing-concurrent-code-with-shuttle
[5]: fn.check_random.html
[6]: #
[7]: #testing-non-deterministic-code
[8]: https://crates.io/crates/rand
[9]: rand/index.html
[10]: #
[11]: fn.replay.html
[12]: #
[13]: #writing-shuttle-tests
[14]: #choosing-a-scheduler-and-running-a-test
[15]: scheduler/trait.Scheduler.html
[16]: fn.check_random.html
[17]: fn.check_pct.html
[18]: https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/asplos277-pct.pdf
[19]: fn.check_dfs.html
[20]: struct.Runner.html
[21]: scheduler/index.html
[22]: struct.Runner.html#method.run
[23]: struct.PortfolioRunner.html
[24]: #modules
[25]: annotations/index.html
[26]: scheduler/struct.AnnotationScheduler.html
[27]: current/index.html
[28]: future/index.html
[29]: https://docs.rs/futures/0.3.13/futures/executor/index.html
[30]: hint/index.html
[31]: https://doc.rust-lang.org/nightly/core/hint/index.html
[32]: lazy_static/index.html
[33]: https://crates.io/crates/lazy_static
[34]: rand/index.html
[35]: https://docs.rs/rand/0.8.5/rand/index.html
[36]: scheduler/index.html
[37]: sync/index.html
[38]: https://doc.rust-lang.org/nightly/std/sync/index.html
[39]: thread/index.html
[40]: https://doc.rust-lang.org/nightly/std/thread/index.html
[41]: #macros
[42]: macro.lazy_static.html
[43]: lazy_static/struct.Lazy.html
[44]: macro.thread_local.html
[45]: thread/struct.LocalKey.html
[46]: #structs
[47]: struct.Config.html
[48]: struct.PortfolioRunner.html
[49]: struct.Runner.html
[50]: #enums
[51]: enum.FailurePersistence.html
[52]: enum.MaxSteps.html
[53]: #functions
[54]: fn.annotate_replay.html
[55]: fn.check_dfs.html
[56]: fn.check_pct.html
[57]: fn.check_random.html
[58]: fn.check_random_with_seed.html
[59]: fn.check_uncontrolled_nondeterminism.html
[60]: fn.replay.html
[61]: fn.replay_from_file.html
