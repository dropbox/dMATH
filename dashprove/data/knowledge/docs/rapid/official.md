## README [¶][1]

### rapid [[PkgGoDev]][2] [[CI]][3]

Rapid is a Go library for property-based testing.

Rapid checks that properties you define hold for a large number of automatically generated test
cases. If a failure is found, rapid automatically minimizes the failing test case before presenting
it.

#### Features

* Imperative Go API with type-safe data generation using generics
* Data generation biased to explore "small" values and edge cases more thoroughly
* Fully automatic minimization of failing test cases
* Persistence and automatic re-running of minimized failing test cases
* Support for state machine ("stateful" or "model-based") testing
* No dependencies outside the Go standard library

#### Examples

Here is what a trivial test using rapid looks like ([playground][4]):

`package rapid_test

import (
        "sort"
        "testing"

        "pgregory.net/rapid"
)

func TestSortStrings(t *testing.T) {
        rapid.Check(t, func(t *rapid.T) {
                s := rapid.SliceOf(rapid.String()).Draw(t, "s")
                sort.Strings(s)
                if !sort.StringsAreSorted(s) {
                        t.Fatalf("unsorted after sort: %v", s)
                }
        })
}
`

More complete examples:

* `ParseDate` function test: [source code][5], [playground][6]
* `Queue` state machine test: [source code][7], [playground][8]

#### Comparison

Rapid aims to bring to Go the power and convenience [Hypothesis][9] brings to Python.

Compared to [testing.F.Fuzz][10], rapid shines in generating complex structured data, including
state machine tests, but lacks coverage-guided feedback and mutations. Note that with
[`MakeFuzz`][11], any rapid test can be used as a fuzz target for the standard fuzzer.

Compared to [gopter][12], rapid provides a much simpler API (queue test in [rapid][13] vs
[gopter][14]), is much smarter about data generation and is able to minimize failing test cases
fully automatically, without any user code.

As for [testing/quick][15], it lacks both convenient data generation facilities and any form of test
case minimization, which are two main things to look for in a property-based testing library.

#### FAQ

What is property-based testing?

Suppose we've written arithmetic functions `add`, `subtract` and `multiply` and want to test them.
Traditional testing approach is example-based — we come up with example inputs and outputs, and
verify that the system behavior matches the examples:

`func TestArithmetic_Example(t *testing.T) {
        t.Run("add", func(t *testing.T) {
                examples := [][3]int{
                        {0, 0, 0},
                        {0, 1, 1},
                        {2, 2, 4},
                        // ...
                }
                for _, e := range examples {
                        if add(e[0], e[1]) != e[2] {
                                t.Fatalf("add(%v, %v) != %v", e[0], e[1], e[2])
                        }
                }
        })
        t.Run("subtract", func(t *testing.T) { /* ... */ })
        t.Run("multiply", func(t *testing.T) { /* ... */ })
}
`

In comparison, with property-based testing we define higher-level properties that should hold for
arbitrary input. Each time we run a property-based test, properties are checked on a new set of
pseudo-random data:

`func TestArithmetic_Property(t *testing.T) {
        rapid.Check(t, func(t *rapid.T) {
                var (
                        a = rapid.Int().Draw(t, "a")
                        b = rapid.Int().Draw(t, "b")
                        c = rapid.Int().Draw(t, "c")
                )
                if add(a, 0) != a {
                        t.Fatalf("add() does not have 0 as identity")
                }
                if add(a, b) != add(b, a) {
                        t.Fatalf("add() is not commutative")
                }
                if add(a, add(b, c)) != add(add(a, b), c) {
                        t.Fatalf("add() is not associative")
                }
                if multiply(a, add(b, c)) != add(multiply(a, b), multiply(a, c)) {
                        t.Fatalf("multiply() is not distributive over add()")
                }
                // ...
        })
}
`

Property-based tests are more powerful and concise than example-based ones — and are also much more
fun to write. As an additional benefit, coming up with general properties of the system often
improves the design of the system itself.

What properties should I test?

As you've seen from the examples above, it depends on the system you are testing. Usually a good
place to start is to put yourself in the shoes of your user and ask what are the properties the user
will rely on (often unknowingly or implicitly) when building on top of your system. That said, here
are some broadly applicable and often encountered properties to keep in mind:

* function does not panic on valid input data
* behavior of two algorithms or data structures is identical
* all variants of the `decode(encode(x)) == x` roundtrip
How does rapid work?

At its core, rapid does a fairly simple thing: generates pseudo-random data based on the
specification you provide, and check properties that you define on the generated data.

Checking is easy: you simply write `if` statements and call something like `t.Fatalf` when things
look wrong.

Generating is a bit more involved. When you construct a `Generator`, nothing happens: `Generator` is
just a specification of how to `Draw` the data you want. When you call `Draw`, rapid will take some
bytes from its internal random bitstream, use them to construct the value based on the `Generator`
specification, and track how the random bytes used correspond to the value (and its subparts). This
knowledge about the structure of the values being generated, as well as their relationship with the
parts of the bitstream allows rapid to intelligently and automatically minify any failure found.

What about fuzzing?

Property-based testing focuses on quick feedback loop: checking the properties on a small but
diverse set of pseudo-random inputs in a fractions of a second.

In comparison, fuzzing focuses on slow, often multi-day, brute force input generation that maximizes
the coverage.

Both approaches are useful. Property-based tests are used alongside regular example-based tests
during development, and fuzzing is used to search for edge cases and security vulnerabilities. With
[`MakeFuzz`][16], any rapid test can be used as a fuzz target.

#### Usage

Just run `go test` as usual, it will pick up also all `rapid` tests.

There are a number of optional flags to influence rapid behavior, run `go test -args -h` and look at
the flags with the `-rapid.` prefix. You can then pass such flags as usual. For example:

`go test -rapid.checks=10_000
`

#### Status

Rapid is stable: tests using rapid should continue to work with all future rapid releases with the
same major version. Possible exceptions to this rule are API changes that replace the concrete type
of parameter with an interface type, or other similar mostly non-breaking changes.

#### License

Rapid is licensed under the [Mozilla Public License Version 2.0][17].

Expand ▾ Collapse ▴

## Documentation [¶][18]

### Overview [¶][19]

* [Generators][20]

Package rapid implements utilities for property-based testing.

[Check][21] verifies that properties you define hold for a large number of automatically generated
test cases. If a failure is found, rapid fails the current test and presents an automatically
minimized version of the failing test case.

[T.Repeat][22] is used to construct state machine (sometimes called "stateful" or "model-based")
tests.

#### Generators [¶][23]

Primitives:

* [Bool][24]
* [Rune][25], [RuneFrom][26]
* [Byte][27], [ByteMin][28], [ByteMax][29], [ByteRange][30]
* [Int][31], [IntMin][32], [IntMax][33], [IntRange][34]
* [Int8][35], [Int8Min][36], [Int8Max][37], [Int8Range][38]
* [Int16][39], [Int16Min][40], [Int16Max][41], [Int16Range][42]
* [Int32][43], [Int32Min][44], [Int32Max][45], [Int32Range][46]
* [Int64][47], [Int64Min][48], [Int64Max][49], [Int64Range][50]
* [Uint][51], [UintMin][52], [UintMax][53], [UintRange][54]
* [Uint8][55], [Uint8Min][56], [Uint8Max][57], [Uint8Range][58]
* [Uint16][59], [Uint16Min][60], [Uint16Max][61], [Uint16Range][62]
* [Uint32][63], [Uint32Min][64], [Uint32Max][65], [Uint32Range][66]
* [Uint64][67], [Uint64Min][68], [Uint64Max][69], [Uint64Range][70]
* [Uintptr][71], [UintptrMin][72], [UintptrMax][73], [UintptrRange][74]
* [Float32][75], [Float32Min][76], [Float32Max][77], [Float32Range][78]
* [Float64][79], [Float64Min][80], [Float64Max][81], [Float64Range][82]

Collections:

* [String][83], [StringMatching][84], [StringOf][85], [StringOfN][86], [StringN][87]
* [SliceOfBytesMatching][88]
* [SliceOf][89], [SliceOfN][90], [SliceOfDistinct][91], [SliceOfNDistinct][92]
* [Permutation][93]
* [MapOf][94], [MapOfN][95], [MapOfValues][96], [MapOfNValues][97]

User-defined types:

* [Custom][98]
* [Make][99]

Other:

* [Map][100],
* [Generator.Filter][101]
* [SampledFrom][102], [Just][103]
* [OneOf][104]
* [Deferred][105]
* [Ptr][106]

### Index [¶][107]

* [func Check(t TB, prop func(*T))][108]
* [func ID[V any](v V) V][109]
* [func MakeCheck(prop func(*T)) func(*testing.T)][110]
* [func MakeFuzz(prop func(*T)) func(*testing.T, []byte)][111]
* [func StateMachineActions(sm StateMachine) map[string]func(*T)][112]
* [type Generator][113]
* * [func Bool() *Generator[bool]][114]
  * [func Byte() *Generator[byte]][115]
  * [func ByteMax(max byte) *Generator[byte]][116]
  * [func ByteMin(min byte) *Generator[byte]][117]
  * [func ByteRange(min byte, max byte) *Generator[byte]][118]
  * [func Custom[V any](fn func(*T) V) *Generator[V]][119]
  * [func Deferred[V any](fn func() *Generator[V]) *Generator[V]][120]
  * [func Float32() *Generator[float32]][121]
  * [func Float32Max(max float32) *Generator[float32]][122]
  * [func Float32Min(min float32) *Generator[float32]][123]
  * [func Float32Range(min float32, max float32) *Generator[float32]][124]
  * [func Float64() *Generator[float64]][125]
  * [func Float64Max(max float64) *Generator[float64]][126]
  * [func Float64Min(min float64) *Generator[float64]][127]
  * [func Float64Range(min float64, max float64) *Generator[float64]][128]
  * [func Int() *Generator[int]][129]
  * [func Int16() *Generator[int16]][130]
  * [func Int16Max(max int16) *Generator[int16]][131]
  * [func Int16Min(min int16) *Generator[int16]][132]
  * [func Int16Range(min int16, max int16) *Generator[int16]][133]
  * [func Int32() *Generator[int32]][134]
  * [func Int32Max(max int32) *Generator[int32]][135]
  * [func Int32Min(min int32) *Generator[int32]][136]
  * [func Int32Range(min int32, max int32) *Generator[int32]][137]
  * [func Int64() *Generator[int64]][138]
  * [func Int64Max(max int64) *Generator[int64]][139]
  * [func Int64Min(min int64) *Generator[int64]][140]
  * [func Int64Range(min int64, max int64) *Generator[int64]][141]
  * [func Int8() *Generator[int8]][142]
  * [func Int8Max(max int8) *Generator[int8]][143]
  * [func Int8Min(min int8) *Generator[int8]][144]
  * [func Int8Range(min int8, max int8) *Generator[int8]][145]
  * [func IntMax(max int) *Generator[int]][146]
  * [func IntMin(min int) *Generator[int]][147]
  * [func IntRange(min int, max int) *Generator[int]][148]
  * [func Just[V any](val V) *Generator[V]][149]
  * [func Make[V any]() *Generator[V]][150]
  * [func Map[U any, V any](g *Generator[U], fn func(U) V) *Generator[V]][151]
  * [func MapOf[K comparable, V any](key *Generator[K], val *Generator[V]) *Generator[map[K]V]][152]
  * [func MapOfN[K comparable, V any](key *Generator[K], val *Generator[V], minLen int, maxLen int)
    *Generator[map[K]V]][153]
  * [func MapOfNValues[K comparable, V any](val *Generator[V], minLen int, maxLen int, keyFn func(V)
    K) *Generator[map[K]V]][154]
  * [func MapOfValues[K comparable, V any](val *Generator[V], keyFn func(V) K)
    *Generator[map[K]V]][155]
  * [func OneOf[V any](gens ...*Generator[V]) *Generator[V]][156]
  * [func Permutation[S ~[]E, E any](slice S) *Generator[S]][157]
  * [func Ptr[E any](elem *Generator[E], allowNil bool) *Generator[*E]][158]
  * [func Rune() *Generator[rune]][159]
  * [func RuneFrom(runes []rune, tables ...*unicode.RangeTable) *Generator[rune]][160]
  * [func SampledFrom[S ~[]E, E any](slice S) *Generator[E]][161]
  * [func SliceOf[E any](elem *Generator[E]) *Generator[[]E]][162]
  * [func SliceOfBytesMatching(expr string) *Generator[[]byte]][163]
  * [func SliceOfDistinct[E any, K comparable](elem *Generator[E], keyFn func(E) K)
    *Generator[[]E]][164]
  * [func SliceOfN[E any](elem *Generator[E], minLen int, maxLen int) *Generator[[]E]][165]
  * [func SliceOfNDistinct[E any, K comparable](elem *Generator[E], minLen int, maxLen int, keyFn
    func(E) K) *Generator[[]E]][166]
  * [func String() *Generator[string]][167]
  * [func StringMatching(expr string) *Generator[string]][168]
  * [func StringN(minRunes int, maxRunes int, maxLen int) *Generator[string]][169]
  * [func StringOf(elem *Generator[rune]) *Generator[string]][170]
  * [func StringOfN(elem *Generator[rune], minRunes int, maxRunes int, maxLen int)
    *Generator[string]][171]
  * [func Uint() *Generator[uint]][172]
  * [func Uint16() *Generator[uint16]][173]
  * [func Uint16Max(max uint16) *Generator[uint16]][174]
  * [func Uint16Min(min uint16) *Generator[uint16]][175]
  * [func Uint16Range(min uint16, max uint16) *Generator[uint16]][176]
  * [func Uint32() *Generator[uint32]][177]
  * [func Uint32Max(max uint32) *Generator[uint32]][178]
  * [func Uint32Min(min uint32) *Generator[uint32]][179]
  * [func Uint32Range(min uint32, max uint32) *Generator[uint32]][180]
  * [func Uint64() *Generator[uint64]][181]
  * [func Uint64Max(max uint64) *Generator[uint64]][182]
  * [func Uint64Min(min uint64) *Generator[uint64]][183]
  * [func Uint64Range(min uint64, max uint64) *Generator[uint64]][184]
  * [func Uint8() *Generator[uint8]][185]
  * [func Uint8Max(max uint8) *Generator[uint8]][186]
  * [func Uint8Min(min uint8) *Generator[uint8]][187]
  * [func Uint8Range(min uint8, max uint8) *Generator[uint8]][188]
  * [func UintMax(max uint) *Generator[uint]][189]
  * [func UintMin(min uint) *Generator[uint]][190]
  * [func UintRange(min uint, max uint) *Generator[uint]][191]
  * [func Uintptr() *Generator[uintptr]][192]
  * [func UintptrMax(max uintptr) *Generator[uintptr]][193]
  * [func UintptrMin(min uintptr) *Generator[uintptr]][194]
  * [func UintptrRange(min uintptr, max uintptr) *Generator[uintptr]][195]
* * [func (g *Generator[V]) AsAny() *Generator[any]][196]
  * [func (g *Generator[V]) Draw(t *T, label string) V][197]
  * [func (g *Generator[V]) Example(seed ...int) V][198]
  * [func (g *Generator[V]) Filter(fn func(V) bool) *Generator[V]][199]
  * [func (g *Generator[V]) String() string][200]
* [type StateMachine][201]
* [type T][202]
* * [func (t *T) Cleanup(f func())][203]
  * [func (t *T) Context() context.Context][204]
  * [func (t *T) Error(args ...any)][205]
  * [func (t *T) Errorf(format string, args ...any)][206]
  * [func (t *T) Fail()][207]
  * [func (t *T) FailNow()][208]
  * [func (t *T) Failed() bool][209]
  * [func (t *T) Fatal(args ...any)][210]
  * [func (t *T) Fatalf(format string, args ...any)][211]
  * [func (t *T) Log(args ...any)][212]
  * [func (t *T) Logf(format string, args ...any)][213]
  * [func (t *T) Repeat(actions map[string]func(*T))][214]
  * [func (t *T) Skip(args ...any)][215]
  * [func (t *T) SkipNow()][216]
  * [func (t *T) Skipf(format string, args ...any)][217]
* [type TB][218]

### Examples [¶][219]

* [Check (ParseDate)][220]
* [Custom][221]
* [Deferred][222]
* [Just][223]
* [Make][224]
* [Make (Tree)][225]
* [Map][226]
* [MapOf][227]
* [MapOfN][228]
* [MapOfNValues][229]
* [MapOfValues][230]
* [OneOf][231]
* [Permutation][232]
* [Ptr][233]
* [Rune][234]
* [RuneFrom][235]
* [SampledFrom][236]
* [SliceOf][237]
* [SliceOfBytesMatching][238]
* [SliceOfDistinct][239]
* [SliceOfN][240]
* [SliceOfNDistinct][241]
* [String][242]
* [StringMatching][243]
* [StringN][244]
* [StringOf][245]
* [StringOfN][246]
* [T.Repeat (Queue)][247]

### Constants [¶][248]

This section is empty.

### Variables [¶][249]

This section is empty.

### Functions [¶][250]

#### func [Check][251] [¶][252]

func Check(t [TB][253], prop func(*[T][254]))

Check fails the current test if rapid can find a test case which falsifies prop.

Property is falsified in case of a panic or a call to [*T.Fatalf][255], [*T.Fatal][256],
[*T.Errorf][257], [*T.Error][258], [*T.FailNow][259] or [*T.Fail][260].

Example (ParseDate) [¶][261]

Rename to TestParseDate(t *testing.T) to make an actual (failing) test.

package main

import (
        "fmt"
        "strconv"
        "testing"

        "pgregory.net/rapid"
)

// ParseDate parses dates in the YYYY-MM-DD format.
func ParseDate(s string) (int, int, int, error) {
        if len(s) != 10 {
                return 0, 0, 0, fmt.Errorf("%q has wrong length: %v instead of 10", s, len(s))
        }

        if s[4] != '-' || s[7] != '-' {
                return 0, 0, 0, fmt.Errorf("'-' separators expected in %q", s)
        }

        y, err := strconv.Atoi(s[0:4])
        if err != nil {
                return 0, 0, 0, fmt.Errorf("failed to parse year: %v", err)
        }

        m, err := strconv.Atoi(s[6:7])
        if err != nil {
                return 0, 0, 0, fmt.Errorf("failed to parse month: %v", err)
        }

        d, err := strconv.Atoi(s[8:10])
        if err != nil {
                return 0, 0, 0, fmt.Errorf("failed to parse day: %v", err)
        }

        return y, m, d, nil
}

func testParseDate(t *rapid.T) {
        y := rapid.IntRange(0, 9999).Draw(t, "y")
        m := rapid.IntRange(1, 12).Draw(t, "m")
        d := rapid.IntRange(1, 31).Draw(t, "d")

        s := fmt.Sprintf("%04d-%02d-%02d", y, m, d)

        y_, m_, d_, err := ParseDate(s)
        if err != nil {
                t.Fatalf("failed to parse date %q: %v", s, err)
        }

        if y_ != y || m_ != m || d_ != d {
                t.Fatalf("got back wrong date: (%d, %d, %d)", y_, m_, d_)
        }
}

// Rename to TestParseDate(t *testing.T) to make an actual (failing) test.
func main() {
        var t *testing.T
        rapid.Check(t, testParseDate)
}
Share Format Run

#### func [ID][262] [¶][263] added in v0.5.0

func ID[V [any][264]](v V) V

ID returns its argument as is. ID is a helper for use with [SliceOfDistinct][265] and similar
functions.

#### func [MakeCheck][266] [¶][267]

func MakeCheck(prop func(*[T][268])) func(*[testing][269].[T][270])

MakeCheck is a convenience function for defining subtests suitable for [*testing.T.Run][271]. It
allows you to write this:

t.Run("subtest name", rapid.MakeCheck(func(t *rapid.T) {
    // test code
}))

instead of this:

t.Run("subtest name", func(t *testing.T) {
    rapid.Check(t, func(t *rapid.T) {
        // test code
    })
})

#### func [MakeFuzz][272] [¶][273] added in v0.5.2

func MakeFuzz(prop func(*[T][274])) func(*[testing][275].[T][276], [][byte][277])

MakeFuzz creates a fuzz target for [*testing.F.Fuzz][278]:

func FuzzFoo(f *testing.F) {
    f.Fuzz(rapid.MakeFuzz(func(t *rapid.T) {
        // test code
    }))
}

#### func [StateMachineActions][279] [¶][280] added in v0.6.0

func StateMachineActions(sm [StateMachine][281]) map[[string][282]]func(*[T][283])

StateMachineActions creates an actions map for [*T.Repeat][284] from methods of a
[StateMachine][285] type instance using reflection.

### Types [¶][286]

#### type [Generator][287] [¶][288]

type Generator[V [any][289]] struct {
        // contains filtered or unexported fields
}

Generator describes a generator of values of type V.

#### func [Bool][290] [¶][291] added in v0.3.6

func Bool() *[Generator][292][[bool][293]]

#### func [Byte][294] [¶][295]

func Byte() *[Generator][296][[byte][297]]

#### func [ByteMax][298] [¶][299]

func ByteMax(max [byte][300]) *[Generator][301][[byte][302]]

#### func [ByteMin][303] [¶][304]

func ByteMin(min [byte][305]) *[Generator][306][[byte][307]]

#### func [ByteRange][308] [¶][309]

func ByteRange(min [byte][310], max [byte][311]) *[Generator][312][[byte][313]]

#### func [Custom][314] [¶][315]

func Custom[V [any][316]](fn func(*[T][317]) V) *[Generator][318][V]

Custom creates a generator which produces results of calling fn. In fn, values should be generated
by calling other generators; it is invalid to return a value from fn without using any other
generator. Custom is a primary way of creating user-defined generators.

Example [¶][319]
package main

import (
        "fmt"

        "pgregory.net/rapid"
)

func main() {
        type point struct {
                x int
                y int
        }

        gen := rapid.Custom(func(t *rapid.T) point {
                return point{
                        x: rapid.IntRange(-100, 100).Draw(t, "x"),
                        y: rapid.IntRange(-100, 100).Draw(t, "y"),
                }
        })

        for i := 0; i < 5; i++ {
                fmt.Println(gen.Example(i))
        }
}
Output:

{-1 23}
{-3 -50}
{0 94}
{-2 -50}
{11 -57}
Share Format Run

#### func [Deferred][320] [¶][321] added in v0.5.0

func Deferred[V [any][322]](fn func() *[Generator][323][V]) *[Generator][324][V]

Deferred creates a generator which defers calling fn until attempting to produce a value. This
allows to define recursive generators.

Example [¶][325]
package main

import (
        "fmt"

        "pgregory.net/rapid"
)

func recursive() *rapid.Generator[any] {
        return rapid.OneOf(
                rapid.Bool().AsAny(),
                rapid.SliceOfN(rapid.Deferred(recursive), 1, 2).AsAny(),
        )
}

func main() {
        gen := recursive()
        for i := 0; i < 5; i++ {
                fmt.Println(gen.Example(i))
        }
}
Output:

[[[[false] false]]]
false
[[true [[[true]]]]]
true
true
Share Format Run

#### func [Float32][326] [¶][327]

func Float32() *[Generator][328][[float32][329]]

Float32 is a shorthand for [Float32Range][330](-[math.MaxFloat32][331], [math.MaxFloat32][332]).

#### func [Float32Max][333] [¶][334]

func Float32Max(max [float32][335]) *[Generator][336][[float32][337]]

Float32Max is a shorthand for [Float32Range][338](-[math.MaxFloat32][339], max).

#### func [Float32Min][340] [¶][341]

func Float32Min(min [float32][342]) *[Generator][343][[float32][344]]

Float32Min is a shorthand for [Float32Range][345](min, [math.MaxFloat32][346]).

#### func [Float32Range][347] [¶][348]

func Float32Range(min [float32][349], max [float32][350]) *[Generator][351][[float32][352]]

Float32Range creates a generator of 32-bit floating-point numbers in range [min, max]. Both min and
max can be infinite.

#### func [Float64][353] [¶][354]

func Float64() *[Generator][355][[float64][356]]

Float64 is a shorthand for [Float64Range][357](-[math.MaxFloat64][358], [math.MaxFloat64][359]).

#### func [Float64Max][360] [¶][361]

func Float64Max(max [float64][362]) *[Generator][363][[float64][364]]

Float64Max is a shorthand for [Float64Range][365](-[math.MaxFloat64][366], max).

#### func [Float64Min][367] [¶][368]

func Float64Min(min [float64][369]) *[Generator][370][[float64][371]]

Float64Min is a shorthand for [Float64Range][372](min, [math.MaxFloat64][373]).

#### func [Float64Range][374] [¶][375]

func Float64Range(min [float64][376], max [float64][377]) *[Generator][378][[float64][379]]

Float64Range creates a generator of 64-bit floating-point numbers in range [min, max]. Both min and
max can be infinite.

#### func [Int][380] [¶][381]

func Int() *[Generator][382][[int][383]]

#### func [Int16][384] [¶][385]

func Int16() *[Generator][386][[int16][387]]

#### func [Int16Max][388] [¶][389]

func Int16Max(max [int16][390]) *[Generator][391][[int16][392]]

#### func [Int16Min][393] [¶][394]

func Int16Min(min [int16][395]) *[Generator][396][[int16][397]]

#### func [Int16Range][398] [¶][399]

func Int16Range(min [int16][400], max [int16][401]) *[Generator][402][[int16][403]]

#### func [Int32][404] [¶][405]

func Int32() *[Generator][406][[int32][407]]

#### func [Int32Max][408] [¶][409]

func Int32Max(max [int32][410]) *[Generator][411][[int32][412]]

#### func [Int32Min][413] [¶][414]

func Int32Min(min [int32][415]) *[Generator][416][[int32][417]]

#### func [Int32Range][418] [¶][419]

func Int32Range(min [int32][420], max [int32][421]) *[Generator][422][[int32][423]]

#### func [Int64][424] [¶][425]

func Int64() *[Generator][426][[int64][427]]

#### func [Int64Max][428] [¶][429]

func Int64Max(max [int64][430]) *[Generator][431][[int64][432]]

#### func [Int64Min][433] [¶][434]

func Int64Min(min [int64][435]) *[Generator][436][[int64][437]]

#### func [Int64Range][438] [¶][439]

func Int64Range(min [int64][440], max [int64][441]) *[Generator][442][[int64][443]]

#### func [Int8][444] [¶][445]

func Int8() *[Generator][446][[int8][447]]

#### func [Int8Max][448] [¶][449]

func Int8Max(max [int8][450]) *[Generator][451][[int8][452]]

#### func [Int8Min][453] [¶][454]

func Int8Min(min [int8][455]) *[Generator][456][[int8][457]]

#### func [Int8Range][458] [¶][459]

func Int8Range(min [int8][460], max [int8][461]) *[Generator][462][[int8][463]]

#### func [IntMax][464] [¶][465]

func IntMax(max [int][466]) *[Generator][467][[int][468]]

#### func [IntMin][469] [¶][470]

func IntMin(min [int][471]) *[Generator][472][[int][473]]

#### func [IntRange][474] [¶][475]

func IntRange(min [int][476], max [int][477]) *[Generator][478][[int][479]]

#### func [Just][480] [¶][481]

func Just[V [any][482]](val V) *[Generator][483][V]

Just creates a generator which always produces the given value. Just(val) is a shorthand for
[SampledFrom][484]([]V{val}).

Example [¶][485]
package main

import (
        "fmt"

        "pgregory.net/rapid"
)

func main() {
        gen := rapid.Just(42)

        for i := 0; i < 5; i++ {
                fmt.Println(gen.Example(i))
        }
}
Output:

42
42
42
42
42
Share Format Run

#### func [Make][486] [¶][487] added in v0.5.0

func Make[V [any][488]]() *[Generator][489][V]

Make creates a generator of values of type V, using reflection to infer the required structure.
Currently, Make may be unable to terminate generation of values of some recursive types, thus using
Make with recursive types requires extra care.

Example [¶][490]
package main

import (
        "fmt"

        "pgregory.net/rapid"
)

func main() {
        gen := rapid.Make[map[int]bool]()

        for i := 0; i < 5; i++ {
                fmt.Println(gen.Example(i))
        }
}
Output:

map[-433:true -261:false -53:false -23:false 1:true 184:false]
map[-3:true 0:true]
map[4:true]
map[-359:true -154:true -71:true -17:false -1:false 590:false 22973756520:true]
map[]
Share Format Run
Example (Tree) [¶][491]
package main

import (
        "fmt"

        "pgregory.net/rapid"
)

type nodeValue int

type tree struct {
        Value       nodeValue
        Left, Right *tree
}

func (t *tree) String() string {
        if t == nil {
                return "nil"
        }
        return fmt.Sprintf("(%s %v %s)", t.Left.String(), t.Value, t.Right.String())
}

func main() {
        gen := rapid.Make[*tree]()

        for i := 0; i < 5; i++ {
                fmt.Println(gen.Example(i))
        }
}
Output:

(nil 1 (nil 184 nil))
(((nil -1 (((((nil -485 ((nil -2 ((((nil -5 nil) -9898554875447 nil) -34709387 ((nil 50440 nil) 113 
(((((nil -442 nil) -66090341586 nil) 179745 nil) 494 (((nil -2 nil) 543360606020 nil) 15261837 nil))
 -1778 nil))) -21034573818 nil)) -5 nil)) 15606609 nil) 882666 (nil 3 nil)) -12 (nil -2 ((nil 1 nil)
 -2 (((nil 11 nil) -187307 ((nil -198 (nil -6895 nil)) 12027 (nil -539313 nil))) 1532 (nil 6 nil))))
) 1745354 nil)) -2 nil) -3 nil)
nil
(((nil -15 (nil 6598 nil)) -131 (nil 317121006373596 ((nil 14 ((nil -9223372036854775808 nil) 1 nil)
) 14668 nil))) 590 nil)
nil
Share Format Run

#### func [Map][492] [¶][493] added in v0.5.4

func Map[U [any][494], V [any][495]](g *[Generator][496][U], fn func(U) V) *[Generator][497][V]

Map creates a generator producing fn(u) for each u produced by g.

Example [¶][498]
package main

import (
        "fmt"
        "strconv"

        "pgregory.net/rapid"
)

func main() {
        gen := rapid.Map(rapid.Int(), strconv.Itoa)
        for i := 0; i < 5; i++ {
                fmt.Printf("%#v\n", gen.Example(i))
        }
}
Output:

"-3"
"-186981"
"4"
"-2"
"43"
Share Format Run

#### func [MapOf][499] [¶][500]

func MapOf[K [comparable][501], V [any][502]](key *[Generator][503][K], val *[Generator][504][V]) *[
Generator][505][map[K]V]

MapOf is a shorthand for [MapOfN][506](key, val, -1, -1).

Example [¶][507]
package main

import (
        "fmt"

        "pgregory.net/rapid"
)

func main() {
        gen := rapid.MapOf(rapid.Int(), rapid.StringMatching(`[a-z]+`))

        for i := 0; i < 5; i++ {
                fmt.Println(gen.Example(i))
        }
}
Output:

map[1:nhlgqwasbggbaociac 561860:r]
map[-3752:pizpv -3:bacuabp 0:bi]
map[-33086515648293:gewf -264276:b -1313:a -258:v -4:b -2:fdhbzcz 4:ubfsdbowrja 1775:tcozav 8334:lvc
prss 376914:braigey]
map[-350:h 590:coaaamcasnapgaad]
map[]
Share Format Run

#### func [MapOfN][508] [¶][509]

func MapOfN[K [comparable][510], V [any][511]](key *[Generator][512][K], val *[Generator][513][V], m
inLen [int][514], maxLen [int][515]) *[Generator][516][map[K]V]

MapOfN creates a map[K]V generator. If minLen >= 0, generated maps have minimum length of minLen. If
maxLen >= 0, generated maps have maximum length of maxLen. MapOfN panics if maxLen >= 0 and minLen >
maxLen.

Example [¶][517]
package main

import (
        "fmt"

        "pgregory.net/rapid"
)

func main() {
        gen := rapid.MapOfN(rapid.Int(), rapid.StringMatching(`[a-z]+`), 5, 5)

        for i := 0; i < 5; i++ {
                fmt.Println(gen.Example(i))
        }
}
Output:

map[-130450326583:bd -2983:bbdbcs 1:nhlgqwasbggbaociac 31:kmdnpmcbuagzr 561860:r]
map[-82024404:d -3752:pizpv -3:bacuabp 0:bi 179745:rzkneb]
map[-33086515648293:gewf -258:v 4:ubfsdbowrja 1775:tcozav 8334:lvcprss]
map[-4280678227:j -25651:aafmd -3308:o -350:h 590:coaaamcasnapgaad]
map[-9614404661322:gsb -378:y 2:paai 4629136912:otg 1476419818092:qign]
Share Format Run

#### func [MapOfNValues][518] [¶][519]

func MapOfNValues[K [comparable][520], V [any][521]](val *[Generator][522][V], minLen [int][523], ma
xLen [int][524], keyFn func(V) K) *[Generator][525][map[K]V]

MapOfNValues creates a map[K]V generator, where keys are generated by applying keyFn to values. If
minLen >= 0, generated maps have minimum length of minLen. If maxLen >= 0, generated maps have
maximum length of maxLen. MapOfNValues panics if maxLen >= 0 and minLen > maxLen.

Example [¶][526]
package main

import (
        "fmt"

        "pgregory.net/rapid"
)

func main() {
        gen := rapid.MapOfNValues(rapid.StringMatching(`[a-z]+`), 5, 5, func(s string) int { return 
len(s) })

        for i := 0; i < 5; i++ {
                fmt.Println(gen.Example(i))
        }
}
Output:

map[1:s 2:dr 3:anc 7:xguehfc 11:sbggbaociac]
map[1:b 2:bp 4:ydag 5:jarxz 6:ebzkwa]
map[1:j 3:gjl 5:eeeqa 7:stcozav 9:fxmcadagf]
map[2:ub 8:waraafmd 10:bfiqcaxazu 16:rjgqimcasnapgaad 17:gckfbljafcedhcvfc]
map[1:k 2:ay 3:wzb 4:dign 7:faabhcb]
Share Format Run

#### func [MapOfValues][527] [¶][528]

func MapOfValues[K [comparable][529], V [any][530]](val *[Generator][531][V], keyFn func(V) K) *[Gen
erator][532][map[K]V]

MapOfValues is a shorthand for [MapOfNValues][533](val, -1, -1, keyFn).

Example [¶][534]
package main

import (
        "fmt"

        "pgregory.net/rapid"
)

func main() {
        gen := rapid.MapOfValues(rapid.StringMatching(`[a-z]+`), func(s string) int { return len(s) 
})

        for i := 0; i < 5; i++ {
                fmt.Println(gen.Example(i))
        }
}
Output:

map[2:dr 7:xguehfc 11:sbggbaociac]
map[2:bp 5:jarxz 6:ebzkwa]
map[1:j 2:aj 3:gjl 4:vayt 5:eeeqa 6:riacaa 7:stcozav 8:mfdhbzcz 9:fxmcadagf 10:bgsbraigey 15:gxongyg
nxqlovib]
map[2:ub 8:waraafmd 10:bfiqcaxazu 16:rjgqimcasnapgaad 17:gckfbljafcedhcvfc]
map[]
Share Format Run

#### func [OneOf][535] [¶][536]

func OneOf[V [any][537]](gens ...*[Generator][538][V]) *[Generator][539][V]

OneOf creates a generator which produces each value by selecting one of gens and producing a value
from it. OneOf panics if gens is empty.

Example [¶][540]
package main

import (
        "fmt"

        "pgregory.net/rapid"
)

func main() {
        gen := rapid.OneOf(rapid.Int32Range(1, 10).AsAny(), rapid.Float32Range(100, 1000).AsAny())

        for i := 0; i < 5; i++ {
                fmt.Println(gen.Example(i))
        }
}
Output:

997.0737
10
475.3125
2
9
Share Format Run

#### func [Permutation][541] [¶][542] added in v0.5.3

func Permutation[S ~[]E, E [any][543]](slice S) *[Generator][544][S]

Permutation creates a generator which produces permutations of the given slice.

Example [¶][545]
package main

import (
        "fmt"

        "pgregory.net/rapid"
)

func main() {
        gen := rapid.Permutation([]int{1, 2, 3})

        for i := 0; i < 5; i++ {
                fmt.Println(gen.Example(i))
        }
}
Output:

[2 3 1]
[3 2 1]
[2 1 3]
[3 2 1]
[1 2 3]
Share Format Run

#### func [Ptr][546] [¶][547]

func Ptr[E [any][548]](elem *[Generator][549][E], allowNil [bool][550]) *[Generator][551][*E]

Ptr creates a *E generator. If allowNil is true, Ptr can return nil pointers.

Example [¶][552]
package main

import (
        "fmt"

        "pgregory.net/rapid"
)

func main() {
        gen := rapid.Ptr(rapid.Int(), true)

        for i := 0; i < 5; i++ {
                v := gen.Example(i)
                if v == nil {
                        fmt.Println("<nil>")
                } else {
                        fmt.Println("(*int)", *v)
                }
        }
}
Output:

(*int) 1
(*int) -3
<nil>
(*int) 590
<nil>
Share Format Run

#### func [Rune][553] [¶][554]

func Rune() *[Generator][555][[rune][556]]

Rune creates a rune generator. Rune is equivalent to [RuneFrom][557] with default set of runes and
tables.

Example [¶][558]
package main

import (
        "fmt"

        "pgregory.net/rapid"
)

func main() {
        gen := rapid.Rune()

        for i := 0; i < 25; i++ {
                if i%5 == 0 {
                        fmt.Println()
                } else {
                        fmt.Print(" ")
                }
                fmt.Printf("%q", gen.Example(i))
        }
}
Output:

'\n' '\x1b' 'A' 'a' '*'
'0' '@' '?' '\'' '\ue05d'
'<' '%' '!' '\u0604' 'A'
'%' '╷' '~' '!' '/'
'\u00ad' '𝅾' '@' '҈' ' '
Share Format Run

#### func [RuneFrom][559] [¶][560]

func RuneFrom(runes [][rune][561], tables ...*[unicode][562].[RangeTable][563]) *[Generator][564][[r
une][565]]

RuneFrom creates a rune generator from provided runes and tables. RuneFrom panics if both runes and
tables are empty. RuneFrom panics if tables contain an empty table.

Example [¶][566]
package main

import (
        "fmt"
        "unicode"

        "pgregory.net/rapid"
)

func main() {
        gens := []*rapid.Generator[rune]{
                rapid.RuneFrom([]rune{'A', 'B', 'C'}),
                rapid.RuneFrom(nil, unicode.Cyrillic, unicode.Greek),
                rapid.RuneFrom([]rune{'⌘'}, &unicode.RangeTable{
                        R32: []unicode.Range32{{0x1F600, 0x1F64F, 1}},
                }),
        }

        for _, gen := range gens {
                for i := 0; i < 5; i++ {
                        if i > 0 {
                                fmt.Print(" ")
                        }
                        fmt.Printf("%q", gen.Example(i))
                }
                fmt.Println()
        }
}
Output:

'A' 'A' 'A' 'B' 'A'
'Ͱ' 'Ѥ' 'Ͱ' 'ͱ' 'Ϳ'
'😀' '⌘' '😀' '😁' '😋'
Share Format Run

#### func [SampledFrom][567] [¶][568]

func SampledFrom[S ~[]E, E [any][569]](slice S) *[Generator][570][E]

SampledFrom creates a generator which produces values from the given slice. SampledFrom panics if
slice is empty.

Example [¶][571]
package main

import (
        "fmt"

        "pgregory.net/rapid"
)

func main() {
        gen := rapid.SampledFrom([]int{1, 2, 3})

        for i := 0; i < 5; i++ {
                fmt.Println(gen.Example(i))
        }
}
Output:

2
3
2
3
1
Share Format Run

#### func [SliceOf][572] [¶][573]

func SliceOf[E [any][574]](elem *[Generator][575][E]) *[Generator][576][[]E]

SliceOf is a shorthand for [SliceOfN][577](elem, -1, -1).

Example [¶][578]
package main

import (
        "fmt"

        "pgregory.net/rapid"
)

func main() {
        gen := rapid.SliceOf(rapid.Int())

        for i := 0; i < 5; i++ {
                fmt.Println(gen.Example(i))
        }
}
Output:

[1 -1902 7 -236 14 -433 -1572631 -1 4219826 -50 1414 -3890044391133 -9223372036854775808 5755498240 
-10 680558 10 -80458281 0 -27]
[-3 -2 -1 -3 -2172865589 -5 -2 -2503553836720]
[4 308 -2 21 -5843 3 1 78 6129321692 -59]
[590 -131 -15 -769 16 -1 14668 14 -1 -58784]
[]
Share Format Run

#### func [SliceOfBytesMatching][579] [¶][580]

func SliceOfBytesMatching(expr [string][581]) *[Generator][582][[][byte][583]]

SliceOfBytesMatching creates a UTF-8 byte slice generator matching the provided [syntax.Perl][584]
regular expression.

Example [¶][585]
package main

import (
        "fmt"

        "pgregory.net/rapid"
)

func main() {
        gen := rapid.SliceOfBytesMatching(`[CAGT]+`)

        for i := 0; i < 5; i++ {
                fmt.Printf("%q\n", gen.Example(i))
        }
}
Output:

"CCTTGAGAGCGATACGGAAG"
"GCAGAACT"
"AACCGTCGAG"
"GGGAAAAGAT"
"AGTG"
Share Format Run

#### func [SliceOfDistinct][586] [¶][587]

func SliceOfDistinct[E [any][588], K [comparable][589]](elem *[Generator][590][E], keyFn func(E) K) 
*[Generator][591][[]E]

SliceOfDistinct is a shorthand for [SliceOfNDistinct][592](elem, -1, -1, keyFn).

Example [¶][593]
package main

import (
        "fmt"

        "pgregory.net/rapid"
)

func main() {
        gen := rapid.SliceOfDistinct(rapid.IntMin(0), func(i int) int { return i % 2 })

        for i := 0; i < 5; i++ {
                fmt.Println(gen.Example(i))
        }
}
Output:

[1]
[2 1]
[4 1]
[590]
[]
Share Format Run

#### func [SliceOfN][594] [¶][595]

func SliceOfN[E [any][596]](elem *[Generator][597][E], minLen [int][598], maxLen [int][599]) *[Gener
ator][600][[]E]

SliceOfN creates a []E generator. If minLen >= 0, generated slices have minimum length of minLen. If
maxLen >= 0, generated slices have maximum length of maxLen. SliceOfN panics if maxLen >= 0 and
minLen > maxLen.

Example [¶][601]
package main

import (
        "fmt"

        "pgregory.net/rapid"
)

func main() {
        gen := rapid.SliceOfN(rapid.Int(), 5, 5)

        for i := 0; i < 5; i++ {
                fmt.Println(gen.Example(i))
        }
}
Output:

[1 -1902 7 -236 14]
[-3 -2 -1 -3 -2172865589]
[4 308 -2 21 -5843]
[590 -131 -15 -769 16]
[4629136912 270 141395 -129322425838843911 -7]
Share Format Run

#### func [SliceOfNDistinct][602] [¶][603]

func SliceOfNDistinct[E [any][604], K [comparable][605]](elem *[Generator][606][E], minLen [int][607
], maxLen [int][608], keyFn func(E) K) *[Generator][609][[]E]

SliceOfNDistinct creates a []E generator. Elements of each generated slice are distinct according to
keyFn. If minLen >= 0, generated slices have minimum length of minLen. If maxLen >= 0, generated
slices have maximum length of maxLen. SliceOfNDistinct panics if maxLen >= 0 and minLen > maxLen.
[ID][610] helper can be used as keyFn to generate slices of distinct comparable elements.

Example [¶][611]
package main

import (
        "fmt"

        "pgregory.net/rapid"
)

func main() {
        gen := rapid.SliceOfNDistinct(rapid.IntMin(0), 2, 2, func(i int) int { return i % 2 })

        for i := 0; i < 5; i++ {
                fmt.Println(gen.Example(i))
        }
}
Output:

[4219826 49]
[2 1]
[4 1]
[0 58783]
[4629136912 141395]
Share Format Run

#### func [String][612] [¶][613]

func String() *[Generator][614][[string][615]]

String is a shorthand for [StringOf][616]([Rune][617]()).

Example [¶][618]
package main

import (
        "fmt"

        "pgregory.net/rapid"
)

func main() {
        gen := rapid.String()

        for i := 0; i < 5; i++ {
                fmt.Printf("%q\n", gen.Example(i))
        }
}
Output:

"\n߾⃝?\rA�֍"
"\u2006𑨳"
"A＄\u0603ᾢ"
"+^#.[#৲"
""
Share Format Run

#### func [StringMatching][619] [¶][620]

func StringMatching(expr [string][621]) *[Generator][622][[string][623]]

StringMatching creates a UTF-8 string generator matching the provided [syntax.Perl][624] regular
expression.

Example [¶][625]
package main

import (
        "fmt"

        "pgregory.net/rapid"
)

func main() {
        gen := rapid.StringMatching(`\(?([0-9]{3})\)?([ .-]?)([0-9]{3})([ .-]?)([0-9]{4})`)

        for i := 0; i < 5; i++ {
                fmt.Printf("%q\n", gen.Example(i))
        }
}
Output:

"(532) 649-9610"
"901)-5783983"
"914.444.1575"
"(316 696.3584"
"816)0861080"
Share Format Run

#### func [StringN][626] [¶][627]

func StringN(minRunes [int][628], maxRunes [int][629], maxLen [int][630]) *[Generator][631][[string]
[632]]

StringN is a shorthand for [StringOfN][633]([Rune][634](), minRunes, maxRunes, maxLen).

Example [¶][635]
package main

import (
        "fmt"

        "pgregory.net/rapid"
)

func main() {
        gen := rapid.StringN(5, 5, -1)

        for i := 0; i < 5; i++ {
                fmt.Printf("%q\n", gen.Example(i))
        }
}
Output:

"\n߾⃝?\r"
"\u2006𑨳#`\x1b"
"A＄\u0603ᾢÉ"
"+^#.["
".A<a¤"
Share Format Run

#### func [StringOf][636] [¶][637]

func StringOf(elem *[Generator][638][[rune][639]]) *[Generator][640][[string][641]]

StringOf is a shorthand for [StringOfN][642](elem, -1, -1, -1).

Example [¶][643]
package main

import (
        "fmt"
        "unicode"

        "pgregory.net/rapid"
)

func main() {
        gen := rapid.StringOf(rapid.RuneFrom(nil, unicode.Tibetan))

        for i := 0; i < 5; i++ {
                fmt.Printf("%q\n", gen.Example(i))
        }
}
Output:

"༁༭༇ཬ༆༐༖ༀྸ༁༆༎ༀ༁ཱི༂༨ༀ༂"
"༂༁ༀ༂༴ༀ༁ྵ"
"ༀ༴༁༅ན༃༁༎ྼ༄༽"
"༎༂༎ༀༀༀཌྷ༂ༀྥ"
""
Share Format Run

#### func [StringOfN][644] [¶][645]

func StringOfN(elem *[Generator][646][[rune][647]], minRunes [int][648], maxRunes [int][649], maxLen
 [int][650]) *[Generator][651][[string][652]]

StringOfN creates a UTF-8 string generator. If minRunes >= 0, generated strings have minimum
minRunes runes. If maxRunes >= 0, generated strings have maximum maxRunes runes. If maxLen >= 0,
generates strings have maximum length of maxLen. StringOfN panics if maxRunes >= 0 and minRunes >
maxRunes. StringOfN panics if maxLen >= 0 and maxLen < maxRunes.

Example [¶][653]
package main

import (
        "fmt"
        "unicode"

        "pgregory.net/rapid"
)

func main() {
        gen := rapid.StringOfN(rapid.RuneFrom(nil, unicode.ASCII_Hex_Digit), 6, 6, -1)

        for i := 0; i < 5; i++ {
                fmt.Printf("%q\n", gen.Example(i))
        }
}
Output:

"1D7B6a"
"2102e0"
"0e15c3"
"E2E000"
"aEd623"
Share Format Run

#### func [Uint][654] [¶][655]

func Uint() *[Generator][656][[uint][657]]

#### func [Uint16][658] [¶][659]

func Uint16() *[Generator][660][[uint16][661]]

#### func [Uint16Max][662] [¶][663]

func Uint16Max(max [uint16][664]) *[Generator][665][[uint16][666]]

#### func [Uint16Min][667] [¶][668]

func Uint16Min(min [uint16][669]) *[Generator][670][[uint16][671]]

#### func [Uint16Range][672] [¶][673]

func Uint16Range(min [uint16][674], max [uint16][675]) *[Generator][676][[uint16][677]]

#### func [Uint32][678] [¶][679]

func Uint32() *[Generator][680][[uint32][681]]

#### func [Uint32Max][682] [¶][683]

func Uint32Max(max [uint32][684]) *[Generator][685][[uint32][686]]

#### func [Uint32Min][687] [¶][688]

func Uint32Min(min [uint32][689]) *[Generator][690][[uint32][691]]

#### func [Uint32Range][692] [¶][693]

func Uint32Range(min [uint32][694], max [uint32][695]) *[Generator][696][[uint32][697]]

#### func [Uint64][698] [¶][699]

func Uint64() *[Generator][700][[uint64][701]]

#### func [Uint64Max][702] [¶][703]

func Uint64Max(max [uint64][704]) *[Generator][705][[uint64][706]]

#### func [Uint64Min][707] [¶][708]

func Uint64Min(min [uint64][709]) *[Generator][710][[uint64][711]]

#### func [Uint64Range][712] [¶][713]

func Uint64Range(min [uint64][714], max [uint64][715]) *[Generator][716][[uint64][717]]

#### func [Uint8][718] [¶][719]

func Uint8() *[Generator][720][[uint8][721]]

#### func [Uint8Max][722] [¶][723]

func Uint8Max(max [uint8][724]) *[Generator][725][[uint8][726]]

#### func [Uint8Min][727] [¶][728]

func Uint8Min(min [uint8][729]) *[Generator][730][[uint8][731]]

#### func [Uint8Range][732] [¶][733]

func Uint8Range(min [uint8][734], max [uint8][735]) *[Generator][736][[uint8][737]]

#### func [UintMax][738] [¶][739]

func UintMax(max [uint][740]) *[Generator][741][[uint][742]]

#### func [UintMin][743] [¶][744]

func UintMin(min [uint][745]) *[Generator][746][[uint][747]]

#### func [UintRange][748] [¶][749]

func UintRange(min [uint][750], max [uint][751]) *[Generator][752][[uint][753]]

#### func [Uintptr][754] [¶][755]

func Uintptr() *[Generator][756][[uintptr][757]]

#### func [UintptrMax][758] [¶][759]

func UintptrMax(max [uintptr][760]) *[Generator][761][[uintptr][762]]

#### func [UintptrMin][763] [¶][764]

func UintptrMin(min [uintptr][765]) *[Generator][766][[uintptr][767]]

#### func [UintptrRange][768] [¶][769]

func UintptrRange(min [uintptr][770], max [uintptr][771]) *[Generator][772][[uintptr][773]]

#### func (*Generator[V]) [AsAny][774] [¶][775] added in v0.5.0

func (g *[Generator][776][V]) AsAny() *[Generator][777][[any][778]]

AsAny creates a generator producing values from g converted to any.

#### func (*Generator[V]) [Draw][779] [¶][780]

func (g *[Generator][781][V]) Draw(t *[T][782], label [string][783]) V

Draw produces a value from the generator.

#### func (*Generator[V]) [Example][784] [¶][785]

func (g *[Generator][786][V]) Example(seed ...[int][787]) V

Example produces an example value from the generator. If seed is provided, value is produced
deterministically based on seed. Example should only be used for examples; always use
*Generator.Draw in property-based tests.

#### func (*Generator[V]) [Filter][788] [¶][789]

func (g *[Generator][790][V]) Filter(fn func(V) [bool][791]) *[Generator][792][V]

Filter creates a generator producing only values from g for which fn returns true.

#### func (*Generator[V]) [String][793] [¶][794]

func (g *[Generator][795][V]) String() [string][796]

#### type [StateMachine][797] [¶][798]

type StateMachine interface {
        // Check is ran after every action and should contain invariant checks.
        //
        // All other public methods should have a form ActionName(t *rapid.T)
        // or ActionName(t rapid.TB) and are used as possible actions.
        // At least one action has to be specified.
        Check(*[T][799])
}

#### type [T][800] [¶][801]

type T struct {
        // contains filtered or unexported fields
}

T is similar to [testing.T][802], but with extra bookkeeping for property-based tests.

For tests to be reproducible, they should generally run in a single goroutine. If concurrency is
unavoidable, methods on *T, such as [*testing.T.Helper][803] and [*T.Errorf][804], are safe for
concurrent calls, but *Generator.Draw from a given *T is not.

#### func (*T) [Cleanup][805] [¶][806] added in v1.2.0

func (t *[T][807]) Cleanup(f func())

Cleanup registers a function to be called when a property function finishes running.

For [Check][808], [MakeFuzz][809], and similar functions, each call to the property function
registers its cleanup functions, which are called after the property function exits.

For [Custom][810], each time a new value is generated, the generator function registers its cleanup
functions, which are called after the generator function exits.

Cleanup functions are called in last-in, first-out order.

If [T.Context][811] is used, the context is canceled before the Cleanup functions are executed.

#### func (*T) [Context][812] [¶][813] added in v1.2.0

func (t *[T][814]) Context() [context][815].[Context][816]

Context returns a context.Context that is canceled after the property function exits, before
Cleanup-registered functions are run.

For [Check][817], [MakeFuzz][818], and similar functions, each call to the property function gets a
unique context that is canceled after that property function exits.

For [Custom][819], each time a new value is generated, the generator function gets a unique context
that is canceled after the generator function exits.

#### func (*T) [Error][820] [¶][821]

func (t *[T][822]) Error(args ...[any][823])

Error is equivalent to [T.Log][824] followed by [T.Fail][825].

#### func (*T) [Errorf][826] [¶][827]

func (t *[T][828]) Errorf(format [string][829], args ...[any][830])

Errorf is equivalent to [T.Logf][831] followed by [T.Fail][832].

#### func (*T) [Fail][833] [¶][834]

func (t *[T][835]) Fail()

#### func (*T) [FailNow][836] [¶][837]

func (t *[T][838]) FailNow()

#### func (*T) [Failed][839] [¶][840]

func (t *[T][841]) Failed() [bool][842]

#### func (*T) [Fatal][843] [¶][844]

func (t *[T][845]) Fatal(args ...[any][846])

Fatal is equivalent to [T.Log][847] followed by [T.FailNow][848].

#### func (*T) [Fatalf][849] [¶][850]

func (t *[T][851]) Fatalf(format [string][852], args ...[any][853])

Fatalf is equivalent to [T.Logf][854] followed by [T.FailNow][855].

#### func (*T) [Log][856] [¶][857]

func (t *[T][858]) Log(args ...[any][859])

#### func (*T) [Logf][860] [¶][861]

func (t *[T][862]) Logf(format [string][863], args ...[any][864])

#### func (*T) [Repeat][865] [¶][866] added in v0.7.0

func (t *[T][867]) Repeat(actions map[[string][868]]func(*[T][869]))

Repeat executes a random sequence of actions (often called a "state machine" test). actions[""], if
set, is executed before/after every other action invocation and should only contain invariant
checking code.

For complex state machines, it can be more convenient to specify actions as methods of a special
state machine type. In this case, [StateMachineActions][870] can be used to create an actions map
from state machine methods using reflection.

Example (Queue) [¶][871]

Rename to TestQueue(t *testing.T) to make an actual (failing) test.

package main

import (
        "testing"

        "pgregory.net/rapid"
)

// Queue implements integer queue with a fixed maximum size.
type Queue struct {
        buf []int
        in  int
        out int
}

func NewQueue(n int) *Queue {
        return &Queue{
                buf: make([]int, n+1),
        }
}

// Precondition: Size() > 0.
func (q *Queue) Get() int {
        i := q.buf[q.out]
        q.out = (q.out + 1) % len(q.buf)
        return i
}

// Precondition: Size() < n.
func (q *Queue) Put(i int) {
        q.buf[q.in] = i
        q.in = (q.in + 1) % len(q.buf)
}

func (q *Queue) Size() int {
        return (q.in - q.out) % len(q.buf)
}

func testQueue(t *rapid.T) {
        n := rapid.IntRange(1, 1000).Draw(t, "n") // maximum queue size
        q := NewQueue(n)                          // queue being tested
        var state []int                           // model of the queue

        t.Repeat(map[string]func(*rapid.T){
                "get": func(t *rapid.T) {
                        if q.Size() == 0 {
                                t.Skip("queue empty")
                        }

                        i := q.Get()
                        if i != state[0] {
                                t.Fatalf("got invalid value: %v vs expected %v", i, state[0])
                        }
                        state = state[1:]
                },
                "put": func(t *rapid.T) {
                        if q.Size() == n {
                                t.Skip("queue full")
                        }

                        i := rapid.Int().Draw(t, "i")
                        q.Put(i)
                        state = append(state, i)
                },
                "": func(t *rapid.T) {
                        if q.Size() != len(state) {
                                t.Fatalf("queue size mismatch: %v vs expected %v", q.Size(), len(sta
te))
                        }
                },
        })
}

// Rename to TestQueue(t *testing.T) to make an actual (failing) test.
func main() {
        var t *testing.T
        rapid.Check(t, testQueue)
}
Share Format Run

#### func (*T) [Skip][872] [¶][873]

func (t *[T][874]) Skip(args ...[any][875])

Skip is equivalent to [T.Log][876] followed by [T.SkipNow][877].

#### func (*T) [SkipNow][878] [¶][879]

func (t *[T][880]) SkipNow()

SkipNow marks the current test case as invalid (except in [T.Repeat][881] actions, where it marks
current action as non-applicable instead). If too many test cases are skipped, rapid will mark the
test as failing due to inability to generate enough valid test cases.

Prefer *Generator.Filter to SkipNow, and prefer generators that always produce valid test cases to
Filter.

#### func (*T) [Skipf][882] [¶][883]

func (t *[T][884]) Skipf(format [string][885], args ...[any][886])

Skipf is equivalent to [T.Logf][887] followed by [T.SkipNow][888].

#### type [TB][889] [¶][890] added in v0.4.8

type TB interface {
        Helper()
        Name() [string][891]
        Logf(format [string][892], args ...[any][893])
        Log(args ...[any][894])
        Skipf(format [string][895], args ...[any][896])
        Skip(args ...[any][897])
        SkipNow()
        Errorf(format [string][898], args ...[any][899])
        Error(args ...[any][900])
        Fatalf(format [string][901], args ...[any][902])
        Fatal(args ...[any][903])
        FailNow()
        Fail()
        Failed() [bool][904]
}

TB is a common interface between [*testing.T][905], [*testing.B][906] and [*T][907].

## Source Files [¶][908]

[View all Source files][909]

* [collections.go][910]
* [combinators.go][911]
* [data.go][912]
* [doc.go][913]
* [engine.go][914]
* [floats.go][915]
* [generator.go][916]
* [integers.go][917]
* [make.go][918]
* [persist.go][919]
* [shrink.go][920]
* [statemachine.go][921]
* [strings.go][922]
* [utils.go][923]
* [vis.go][924]
Click to show internal directories.
Click to hide internal directories.

[1]: #section-readme
[2]: https://pkg.go.dev/pgregory.net/rapid
[3]: https://github.com/flyingmutant/rapid/actions
[4]: https://go.dev/play/p/QJhOzo_BByz
[5]: https://github.com/flyingmutant/rapid/blob/v1.2.0/example_function_test.go
[6]: https://go.dev/play/p/tZFU8zv8AUl
[7]: https://github.com/flyingmutant/rapid/blob/v1.2.0/example_statemachine_test.go
[8]: https://go.dev/play/p/cxEh4deG-4n
[9]: https://github.com/HypothesisWorks/hypothesis
[10]: https://pkg.go.dev/testing#F.Fuzz
[11]: https://pkg.go.dev/pgregory.net/rapid#MakeFuzz
[12]: https://pkg.go.dev/github.com/leanovate/gopter
[13]: https://github.com/flyingmutant/rapid/blob/v1.2.0/example_statemachine_test.go
[14]: https://github.com/leanovate/gopter/raw/90cc76d7f1b21637b4b912a7c19dea3efe145bb2/commands/exam
ple_circularqueue_test.go
[15]: https://pkg.go.dev/testing/quick
[16]: https://pkg.go.dev/pgregory.net/rapid#MakeFuzz
[17]: https://github.com/flyingmutant/rapid/blob/v1.2.0/LICENSE
[18]: #section-documentation
[19]: #pkg-overview
[20]: #hdr-Generators
[21]: #Check
[22]: #T.Repeat
[23]: #hdr-Generators
[24]: #Bool
[25]: #Rune
[26]: #RuneFrom
[27]: #Byte
[28]: #ByteMin
[29]: #ByteMax
[30]: #ByteRange
[31]: #Int
[32]: #IntMin
[33]: #IntMax
[34]: #IntRange
[35]: #Int8
[36]: #Int8Min
[37]: #Int8Max
[38]: #Int8Range
[39]: #Int16
[40]: #Int16Min
[41]: #Int16Max
[42]: #Int16Range
[43]: #Int32
[44]: #Int32Min
[45]: #Int32Max
[46]: #Int32Range
[47]: #Int64
[48]: #Int64Min
[49]: #Int64Max
[50]: #Int64Range
[51]: #Uint
[52]: #UintMin
[53]: #UintMax
[54]: #UintRange
[55]: #Uint8
[56]: #Uint8Min
[57]: #Uint8Max
[58]: #Uint8Range
[59]: #Uint16
[60]: #Uint16Min
[61]: #Uint16Max
[62]: #Uint16Range
[63]: #Uint32
[64]: #Uint32Min
[65]: #Uint32Max
[66]: #Uint32Range
[67]: #Uint64
[68]: #Uint64Min
[69]: #Uint64Max
[70]: #Uint64Range
[71]: #Uintptr
[72]: #UintptrMin
[73]: #UintptrMax
[74]: #UintptrRange
[75]: #Float32
[76]: #Float32Min
[77]: #Float32Max
[78]: #Float32Range
[79]: #Float64
[80]: #Float64Min
[81]: #Float64Max
[82]: #Float64Range
[83]: #String
[84]: #StringMatching
[85]: #StringOf
[86]: #StringOfN
[87]: #StringN
[88]: #SliceOfBytesMatching
[89]: #SliceOf
[90]: #SliceOfN
[91]: #SliceOfDistinct
[92]: #SliceOfNDistinct
[93]: #Permutation
[94]: #MapOf
[95]: #MapOfN
[96]: #MapOfValues
[97]: #MapOfNValues
[98]: #Custom
[99]: #Make
[100]: #Map
[101]: #Generator.Filter
[102]: #SampledFrom
[103]: #Just
[104]: #OneOf
[105]: #Deferred
[106]: #Ptr
[107]: #pkg-index
[108]: #Check
[109]: #ID
[110]: #MakeCheck
[111]: #MakeFuzz
[112]: #StateMachineActions
[113]: #Generator
[114]: #Bool
[115]: #Byte
[116]: #ByteMax
[117]: #ByteMin
[118]: #ByteRange
[119]: #Custom
[120]: #Deferred
[121]: #Float32
[122]: #Float32Max
[123]: #Float32Min
[124]: #Float32Range
[125]: #Float64
[126]: #Float64Max
[127]: #Float64Min
[128]: #Float64Range
[129]: #Int
[130]: #Int16
[131]: #Int16Max
[132]: #Int16Min
[133]: #Int16Range
[134]: #Int32
[135]: #Int32Max
[136]: #Int32Min
[137]: #Int32Range
[138]: #Int64
[139]: #Int64Max
[140]: #Int64Min
[141]: #Int64Range
[142]: #Int8
[143]: #Int8Max
[144]: #Int8Min
[145]: #Int8Range
[146]: #IntMax
[147]: #IntMin
[148]: #IntRange
[149]: #Just
[150]: #Make
[151]: #Map
[152]: #MapOf
[153]: #MapOfN
[154]: #MapOfNValues
[155]: #MapOfValues
[156]: #OneOf
[157]: #Permutation
[158]: #Ptr
[159]: #Rune
[160]: #RuneFrom
[161]: #SampledFrom
[162]: #SliceOf
[163]: #SliceOfBytesMatching
[164]: #SliceOfDistinct
[165]: #SliceOfN
[166]: #SliceOfNDistinct
[167]: #String
[168]: #StringMatching
[169]: #StringN
[170]: #StringOf
[171]: #StringOfN
[172]: #Uint
[173]: #Uint16
[174]: #Uint16Max
[175]: #Uint16Min
[176]: #Uint16Range
[177]: #Uint32
[178]: #Uint32Max
[179]: #Uint32Min
[180]: #Uint32Range
[181]: #Uint64
[182]: #Uint64Max
[183]: #Uint64Min
[184]: #Uint64Range
[185]: #Uint8
[186]: #Uint8Max
[187]: #Uint8Min
[188]: #Uint8Range
[189]: #UintMax
[190]: #UintMin
[191]: #UintRange
[192]: #Uintptr
[193]: #UintptrMax
[194]: #UintptrMin
[195]: #UintptrRange
[196]: #Generator.AsAny
[197]: #Generator.Draw
[198]: #Generator.Example
[199]: #Generator.Filter
[200]: #Generator.String
[201]: #StateMachine
[202]: #T
[203]: #T.Cleanup
[204]: #T.Context
[205]: #T.Error
[206]: #T.Errorf
[207]: #T.Fail
[208]: #T.FailNow
[209]: #T.Failed
[210]: #T.Fatal
[211]: #T.Fatalf
[212]: #T.Log
[213]: #T.Logf
[214]: #T.Repeat
[215]: #T.Skip
[216]: #T.SkipNow
[217]: #T.Skipf
[218]: #TB
[219]: #pkg-examples
[220]: #example-Check-ParseDate
[221]: #example-Custom
[222]: #example-Deferred
[223]: #example-Just
[224]: #example-Make
[225]: #example-Make-Tree
[226]: #example-Map
[227]: #example-MapOf
[228]: #example-MapOfN
[229]: #example-MapOfNValues
[230]: #example-MapOfValues
[231]: #example-OneOf
[232]: #example-Permutation
[233]: #example-Ptr
[234]: #example-Rune
[235]: #example-RuneFrom
[236]: #example-SampledFrom
[237]: #example-SliceOf
[238]: #example-SliceOfBytesMatching
[239]: #example-SliceOfDistinct
[240]: #example-SliceOfN
[241]: #example-SliceOfNDistinct
[242]: #example-String
[243]: #example-StringMatching
[244]: #example-StringN
[245]: #example-StringOf
[246]: #example-StringOfN
[247]: #example-T.Repeat-Queue
[248]: #pkg-constants
[249]: #pkg-variables
[250]: #pkg-functions
[251]: https://github.com/flyingmutant/rapid/blob/v1.2.0/engine.go#L118
[252]: #Check
[253]: #TB
[254]: #T
[255]: #T.Fatalf
[256]: #T.Fatal
[257]: #T.Errorf
[258]: #T.Error
[259]: #T.FailNow
[260]: #T.Fail
[261]: #example-Check-ParseDate
[262]: https://github.com/flyingmutant/rapid/blob/v1.2.0/collections.go#L12
[263]: #ID
[264]: /builtin#any
[265]: #SliceOfDistinct
[266]: https://github.com/flyingmutant/rapid/blob/v1.2.0/engine.go#L137
[267]: #MakeCheck
[268]: #T
[269]: /testing
[270]: /testing#T
[271]: /testing#T.Run
[272]: https://github.com/flyingmutant/rapid/blob/v1.2.0/engine.go#L151
[273]: #MakeFuzz
[274]: #T
[275]: /testing
[276]: /testing#T
[277]: /builtin#byte
[278]: /testing#F.Fuzz
[279]: https://github.com/flyingmutant/rapid/blob/v1.2.0/statemachine.go#L82
[280]: #StateMachineActions
[281]: #StateMachine
[282]: /builtin#string
[283]: #T
[284]: #T.Repeat
[285]: #StateMachine
[286]: #pkg-types
[287]: https://github.com/flyingmutant/rapid/blob/v1.2.0/generator.go#L21
[288]: #Generator
[289]: /builtin#any
[290]: https://github.com/flyingmutant/rapid/blob/v1.2.0/integers.go#L67
[291]: #Bool
[292]: #Generator
[293]: /builtin#bool
[294]: https://github.com/flyingmutant/rapid/blob/v1.2.0/integers.go#L71
[295]: #Byte
[296]: #Generator
[297]: /builtin#byte
[298]: https://github.com/flyingmutant/rapid/blob/v1.2.0/integers.go#L99
[299]: #ByteMax
[300]: /builtin#byte
[301]: #Generator
[302]: /builtin#byte
[303]: https://github.com/flyingmutant/rapid/blob/v1.2.0/integers.go#L84
[304]: #ByteMin
[305]: /builtin#byte
[306]: #Generator
[307]: /builtin#byte
[308]: https://github.com/flyingmutant/rapid/blob/v1.2.0/integers.go#L114
[309]: #ByteRange
[310]: /builtin#byte
[311]: /builtin#byte
[312]: #Generator
[313]: /builtin#byte
[314]: https://github.com/flyingmutant/rapid/blob/v1.2.0/combinators.go#L20
[315]: #Custom
[316]: /builtin#any
[317]: #T
[318]: #Generator
[319]: #example-Custom
[320]: https://github.com/flyingmutant/rapid/blob/v1.2.0/combinators.go#L56
[321]: #Deferred
[322]: /builtin#any
[323]: #Generator
[324]: #Generator
[325]: #example-Deferred
[326]: https://github.com/flyingmutant/rapid/blob/v1.2.0/floats.go#L27
[327]: #Float32
[328]: #Generator
[329]: /builtin#float32
[330]: #Float32Range
[331]: /math#MaxFloat32
[332]: /math#MaxFloat32
[333]: https://github.com/flyingmutant/rapid/blob/v1.2.0/floats.go#L37
[334]: #Float32Max
[335]: /builtin#float32
[336]: #Generator
[337]: /builtin#float32
[338]: #Float32Range
[339]: /math#MaxFloat32
[340]: https://github.com/flyingmutant/rapid/blob/v1.2.0/floats.go#L32
[341]: #Float32Min
[342]: /builtin#float32
[343]: #Generator
[344]: /builtin#float32
[345]: #Float32Range
[346]: /math#MaxFloat32
[347]: https://github.com/flyingmutant/rapid/blob/v1.2.0/floats.go#L43
[348]: #Float32Range
[349]: /builtin#float32
[350]: /builtin#float32
[351]: #Generator
[352]: /builtin#float32
[353]: https://github.com/flyingmutant/rapid/blob/v1.2.0/floats.go#L59
[354]: #Float64
[355]: #Generator
[356]: /builtin#float64
[357]: #Float64Range
[358]: /math#MaxFloat64
[359]: /math#MaxFloat64
[360]: https://github.com/flyingmutant/rapid/blob/v1.2.0/floats.go#L69
[361]: #Float64Max
[362]: /builtin#float64
[363]: #Generator
[364]: /builtin#float64
[365]: #Float64Range
[366]: /math#MaxFloat64
[367]: https://github.com/flyingmutant/rapid/blob/v1.2.0/floats.go#L64
[368]: #Float64Min
[369]: /builtin#float64
[370]: #Generator
[371]: /builtin#float64
[372]: #Float64Range
[373]: /math#MaxFloat64
[374]: https://github.com/flyingmutant/rapid/blob/v1.2.0/floats.go#L75
[375]: #Float64Range
[376]: /builtin#float64
[377]: /builtin#float64
[378]: #Generator
[379]: /builtin#float64
[380]: https://github.com/flyingmutant/rapid/blob/v1.2.0/integers.go#L72
[381]: #Int
[382]: #Generator
[383]: /builtin#int
[384]: https://github.com/flyingmutant/rapid/blob/v1.2.0/integers.go#L74
[385]: #Int16
[386]: #Generator
[387]: /builtin#int16
[388]: https://github.com/flyingmutant/rapid/blob/v1.2.0/integers.go#L102
[389]: #Int16Max
[390]: /builtin#int16
[391]: #Generator
[392]: /builtin#int16
[393]: https://github.com/flyingmutant/rapid/blob/v1.2.0/integers.go#L87
[394]: #Int16Min
[395]: /builtin#int16
[396]: #Generator
[397]: /builtin#int16
[398]: https://github.com/flyingmutant/rapid/blob/v1.2.0/integers.go#L123
[399]: #Int16Range
[400]: /builtin#int16
[401]: /builtin#int16
[402]: #Generator
[403]: /builtin#int16
[404]: https://github.com/flyingmutant/rapid/blob/v1.2.0/integers.go#L75
[405]: #Int32
[406]: #Generator
[407]: /builtin#int32
[408]: https://github.com/flyingmutant/rapid/blob/v1.2.0/integers.go#L103
[409]: #Int32Max
[410]: /builtin#int32
[411]: #Generator
[412]: /builtin#int32
[413]: https://github.com/flyingmutant/rapid/blob/v1.2.0/integers.go#L88
[414]: #Int32Min
[415]: /builtin#int32
[416]: #Generator
[417]: /builtin#int32
[418]: https://github.com/flyingmutant/rapid/blob/v1.2.0/integers.go#L126
[419]: #Int32Range
[420]: /builtin#int32
[421]: /builtin#int32
[422]: #Generator
[423]: /builtin#int32
[424]: https://github.com/flyingmutant/rapid/blob/v1.2.0/integers.go#L76
[425]: #Int64
[426]: #Generator
[427]: /builtin#int64
[428]: https://github.com/flyingmutant/rapid/blob/v1.2.0/integers.go#L104
[429]: #Int64Max
[430]: /builtin#int64
[431]: #Generator
[432]: /builtin#int64
[433]: https://github.com/flyingmutant/rapid/blob/v1.2.0/integers.go#L89
[434]: #Int64Min
[435]: /builtin#int64
[436]: #Generator
[437]: /builtin#int64
[438]: https://github.com/flyingmutant/rapid/blob/v1.2.0/integers.go#L129
[439]: #Int64Range
[440]: /builtin#int64
[441]: /builtin#int64
[442]: #Generator
[443]: /builtin#int64
[444]: https://github.com/flyingmutant/rapid/blob/v1.2.0/integers.go#L73
[445]: #Int8
[446]: #Generator
[447]: /builtin#int8
[448]: https://github.com/flyingmutant/rapid/blob/v1.2.0/integers.go#L101
[449]: #Int8Max
[450]: /builtin#int8
[451]: #Generator
[452]: /builtin#int8
[453]: https://github.com/flyingmutant/rapid/blob/v1.2.0/integers.go#L86
[454]: #Int8Min
[455]: /builtin#int8
[456]: #Generator
[457]: /builtin#int8
[458]: https://github.com/flyingmutant/rapid/blob/v1.2.0/integers.go#L120
[459]: #Int8Range
[460]: /builtin#int8
[461]: /builtin#int8
[462]: #Generator
[463]: /builtin#int8
[464]: https://github.com/flyingmutant/rapid/blob/v1.2.0/integers.go#L100
[465]: #IntMax
[466]: /builtin#int
[467]: #Generator
[468]: /builtin#int
[469]: https://github.com/flyingmutant/rapid/blob/v1.2.0/integers.go#L85
[470]: #IntMin
[471]: /builtin#int
[472]: #Generator
[473]: /builtin#int
[474]: https://github.com/flyingmutant/rapid/blob/v1.2.0/integers.go#L117
[475]: #IntRange
[476]: /builtin#int
[477]: /builtin#int
[478]: #Generator
[479]: /builtin#int
[480]: https://github.com/flyingmutant/rapid/blob/v1.2.0/combinators.go#L145
[481]: #Just
[482]: /builtin#any
[483]: #Generator
[484]: #SampledFrom
[485]: #example-Just
[486]: https://github.com/flyingmutant/rapid/blob/v1.2.0/make.go#L17
[487]: #Make
[488]: /builtin#any
[489]: #Generator
[490]: #example-Make
[491]: #example-Make-Tree
[492]: https://github.com/flyingmutant/rapid/blob/v1.2.0/combinators.go#L123
[493]: #Map
[494]: /builtin#any
[495]: /builtin#any
[496]: #Generator
[497]: #Generator
[498]: #example-Map
[499]: https://github.com/flyingmutant/rapid/blob/v1.2.0/collections.go#L105
[500]: #MapOf
[501]: /builtin#comparable
[502]: /builtin#any
[503]: #Generator
[504]: #Generator
[505]: #Generator
[506]: #MapOfN
[507]: #example-MapOf
[508]: https://github.com/flyingmutant/rapid/blob/v1.2.0/collections.go#L112
[509]: #MapOfN
[510]: /builtin#comparable
[511]: /builtin#any
[512]: #Generator
[513]: #Generator
[514]: /builtin#int
[515]: /builtin#int
[516]: #Generator
[517]: #example-MapOfN
[518]: https://github.com/flyingmutant/rapid/blob/v1.2.0/collections.go#L131
[519]: #MapOfNValues
[520]: /builtin#comparable
[521]: /builtin#any
[522]: #Generator
[523]: /builtin#int
[524]: /builtin#int
[525]: #Generator
[526]: #example-MapOfNValues
[527]: https://github.com/flyingmutant/rapid/blob/v1.2.0/collections.go#L124
[528]: #MapOfValues
[529]: /builtin#comparable
[530]: /builtin#any
[531]: #Generator
[532]: #Generator
[533]: #MapOfNValues
[534]: #example-MapOfValues
[535]: https://github.com/flyingmutant/rapid/blob/v1.2.0/combinators.go#L213
[536]: #OneOf
[537]: /builtin#any
[538]: #Generator
[539]: #Generator
[540]: #example-OneOf
[541]: https://github.com/flyingmutant/rapid/blob/v1.2.0/combinators.go#L178
[542]: #Permutation
[543]: /builtin#any
[544]: #Generator
[545]: #example-Permutation
[546]: https://github.com/flyingmutant/rapid/blob/v1.2.0/combinators.go#L241
[547]: #Ptr
[548]: /builtin#any
[549]: #Generator
[550]: /builtin#bool
[551]: #Generator
[552]: #example-Ptr
[553]: https://github.com/flyingmutant/rapid/blob/v1.2.0/strings.go#L74
[554]: #Rune
[555]: #Generator
[556]: /builtin#rune
[557]: #RuneFrom
[558]: #example-Rune
[559]: https://github.com/flyingmutant/rapid/blob/v1.2.0/strings.go#L80
[560]: #RuneFrom
[561]: /builtin#rune
[562]: /unicode
[563]: /unicode#RangeTable
[564]: #Generator
[565]: /builtin#rune
[566]: #example-RuneFrom
[567]: https://github.com/flyingmutant/rapid/blob/v1.2.0/combinators.go#L151
[568]: #SampledFrom
[569]: /builtin#any
[570]: #Generator
[571]: #example-SampledFrom
[572]: https://github.com/flyingmutant/rapid/blob/v1.2.0/collections.go#L17
[573]: #SliceOf
[574]: /builtin#any
[575]: #Generator
[576]: #Generator
[577]: #SliceOfN
[578]: #example-SliceOf
[579]: https://github.com/flyingmutant/rapid/blob/v1.2.0/strings.go#L238
[580]: #SliceOfBytesMatching
[581]: /builtin#string
[582]: #Generator
[583]: /builtin#byte
[584]: /regexp/syntax#Perl
[585]: #example-SliceOfBytesMatching
[586]: https://github.com/flyingmutant/rapid/blob/v1.2.0/collections.go#L35
[587]: #SliceOfDistinct
[588]: /builtin#any
[589]: /builtin#comparable
[590]: #Generator
[591]: #Generator
[592]: #SliceOfNDistinct
[593]: #example-SliceOfDistinct
[594]: https://github.com/flyingmutant/rapid/blob/v1.2.0/collections.go#L24
[595]: #SliceOfN
[596]: /builtin#any
[597]: #Generator
[598]: /builtin#int
[599]: /builtin#int
[600]: #Generator
[601]: #example-SliceOfN
[602]: https://github.com/flyingmutant/rapid/blob/v1.2.0/collections.go#L43
[603]: #SliceOfNDistinct
[604]: /builtin#any
[605]: /builtin#comparable
[606]: #Generator
[607]: /builtin#int
[608]: /builtin#int
[609]: #Generator
[610]: #ID
[611]: #example-SliceOfNDistinct
[612]: https://github.com/flyingmutant/rapid/blob/v1.2.0/strings.go#L143
[613]: #String
[614]: #Generator
[615]: /builtin#string
[616]: #StringOf
[617]: #Rune
[618]: #example-String
[619]: https://github.com/flyingmutant/rapid/blob/v1.2.0/strings.go#L224
[620]: #StringMatching
[621]: /builtin#string
[622]: #Generator
[623]: /builtin#string
[624]: /regexp/syntax#Perl
[625]: #example-StringMatching
[626]: https://github.com/flyingmutant/rapid/blob/v1.2.0/strings.go#L148
[627]: #StringN
[628]: /builtin#int
[629]: /builtin#int
[630]: /builtin#int
[631]: #Generator
[632]: /builtin#string
[633]: #StringOfN
[634]: #Rune
[635]: #example-StringN
[636]: https://github.com/flyingmutant/rapid/blob/v1.2.0/strings.go#L153
[637]: #StringOf
[638]: #Generator
[639]: /builtin#rune
[640]: #Generator
[641]: /builtin#string
[642]: #StringOfN
[643]: #example-StringOf
[644]: https://github.com/flyingmutant/rapid/blob/v1.2.0/strings.go#L163
[645]: #StringOfN
[646]: #Generator
[647]: /builtin#rune
[648]: /builtin#int
[649]: /builtin#int
[650]: /builtin#int
[651]: #Generator
[652]: /builtin#string
[653]: #example-StringOfN
[654]: https://github.com/flyingmutant/rapid/blob/v1.2.0/integers.go#L77
[655]: #Uint
[656]: #Generator
[657]: /builtin#uint
[658]: https://github.com/flyingmutant/rapid/blob/v1.2.0/integers.go#L79
[659]: #Uint16
[660]: #Generator
[661]: /builtin#uint16
[662]: https://github.com/flyingmutant/rapid/blob/v1.2.0/integers.go#L107
[663]: #Uint16Max
[664]: /builtin#uint16
[665]: #Generator
[666]: /builtin#uint16
[667]: https://github.com/flyingmutant/rapid/blob/v1.2.0/integers.go#L92
[668]: #Uint16Min
[669]: /builtin#uint16
[670]: #Generator
[671]: /builtin#uint16
[672]: https://github.com/flyingmutant/rapid/blob/v1.2.0/integers.go#L138
[673]: #Uint16Range
[674]: /builtin#uint16
[675]: /builtin#uint16
[676]: #Generator
[677]: /builtin#uint16
[678]: https://github.com/flyingmutant/rapid/blob/v1.2.0/integers.go#L80
[679]: #Uint32
[680]: #Generator
[681]: /builtin#uint32
[682]: https://github.com/flyingmutant/rapid/blob/v1.2.0/integers.go#L108
[683]: #Uint32Max
[684]: /builtin#uint32
[685]: #Generator
[686]: /builtin#uint32
[687]: https://github.com/flyingmutant/rapid/blob/v1.2.0/integers.go#L93
[688]: #Uint32Min
[689]: /builtin#uint32
[690]: #Generator
[691]: /builtin#uint32
[692]: https://github.com/flyingmutant/rapid/blob/v1.2.0/integers.go#L141
[693]: #Uint32Range
[694]: /builtin#uint32
[695]: /builtin#uint32
[696]: #Generator
[697]: /builtin#uint32
[698]: https://github.com/flyingmutant/rapid/blob/v1.2.0/integers.go#L81
[699]: #Uint64
[700]: #Generator
[701]: /builtin#uint64
[702]: https://github.com/flyingmutant/rapid/blob/v1.2.0/integers.go#L109
[703]: #Uint64Max
[704]: /builtin#uint64
[705]: #Generator
[706]: /builtin#uint64
[707]: https://github.com/flyingmutant/rapid/blob/v1.2.0/integers.go#L94
[708]: #Uint64Min
[709]: /builtin#uint64
[710]: #Generator
[711]: /builtin#uint64
[712]: https://github.com/flyingmutant/rapid/blob/v1.2.0/integers.go#L144
[713]: #Uint64Range
[714]: /builtin#uint64
[715]: /builtin#uint64
[716]: #Generator
[717]: /builtin#uint64
[718]: https://github.com/flyingmutant/rapid/blob/v1.2.0/integers.go#L78
[719]: #Uint8
[720]: #Generator
[721]: /builtin#uint8
[722]: https://github.com/flyingmutant/rapid/blob/v1.2.0/integers.go#L106
[723]: #Uint8Max
[724]: /builtin#uint8
[725]: #Generator
[726]: /builtin#uint8
[727]: https://github.com/flyingmutant/rapid/blob/v1.2.0/integers.go#L91
[728]: #Uint8Min
[729]: /builtin#uint8
[730]: #Generator
[731]: /builtin#uint8
[732]: https://github.com/flyingmutant/rapid/blob/v1.2.0/integers.go#L135
[733]: #Uint8Range
[734]: /builtin#uint8
[735]: /builtin#uint8
[736]: #Generator
[737]: /builtin#uint8
[738]: https://github.com/flyingmutant/rapid/blob/v1.2.0/integers.go#L105
[739]: #UintMax
[740]: /builtin#uint
[741]: #Generator
[742]: /builtin#uint
[743]: https://github.com/flyingmutant/rapid/blob/v1.2.0/integers.go#L90
[744]: #UintMin
[745]: /builtin#uint
[746]: #Generator
[747]: /builtin#uint
[748]: https://github.com/flyingmutant/rapid/blob/v1.2.0/integers.go#L132
[749]: #UintRange
[750]: /builtin#uint
[751]: /builtin#uint
[752]: #Generator
[753]: /builtin#uint
[754]: https://github.com/flyingmutant/rapid/blob/v1.2.0/integers.go#L82
[755]: #Uintptr
[756]: #Generator
[757]: /builtin#uintptr
[758]: https://github.com/flyingmutant/rapid/blob/v1.2.0/integers.go#L110
[759]: #UintptrMax
[760]: /builtin#uintptr
[761]: #Generator
[762]: /builtin#uintptr
[763]: https://github.com/flyingmutant/rapid/blob/v1.2.0/integers.go#L95
[764]: #UintptrMin
[765]: /builtin#uintptr
[766]: #Generator
[767]: /builtin#uintptr
[768]: https://github.com/flyingmutant/rapid/blob/v1.2.0/integers.go#L147
[769]: #UintptrRange
[770]: /builtin#uintptr
[771]: /builtin#uintptr
[772]: #Generator
[773]: /builtin#uintptr
[774]: https://github.com/flyingmutant/rapid/blob/v1.2.0/generator.go#L99
[775]: #Generator.AsAny
[776]: #Generator
[777]: #Generator
[778]: /builtin#any
[779]: https://github.com/flyingmutant/rapid/blob/v1.2.0/generator.go#L42
[780]: #Generator.Draw
[781]: #Generator
[782]: #T
[783]: /builtin#string
[784]: https://github.com/flyingmutant/rapid/blob/v1.2.0/generator.go#L81
[785]: #Generator.Example
[786]: #Generator
[787]: /builtin#int
[788]: https://github.com/flyingmutant/rapid/blob/v1.2.0/generator.go#L94
[789]: #Generator.Filter
[790]: #Generator
[791]: /builtin#bool
[792]: #Generator
[793]: https://github.com/flyingmutant/rapid/blob/v1.2.0/generator.go#L33
[794]: #Generator.String
[795]: #Generator
[796]: /builtin#string
[797]: https://github.com/flyingmutant/rapid/blob/v1.2.0/statemachine.go#L71
[798]: #StateMachine
[799]: #T
[800]: https://github.com/flyingmutant/rapid/blob/v1.2.0/engine.go#L505
[801]: #T
[802]: /testing#T
[803]: /testing#T.Helper
[804]: #T.Errorf
[805]: https://github.com/flyingmutant/rapid/blob/v1.2.0/engine.go#L621
[806]: #T.Cleanup
[807]: #T
[808]: #Check
[809]: #MakeFuzz
[810]: #Custom
[811]: #T.Context
[812]: https://github.com/flyingmutant/rapid/blob/v1.2.0/engine.go#L562
[813]: #T.Context
[814]: #T
[815]: /context
[816]: /context#Context
[817]: #Check
[818]: #MakeFuzz
[819]: #Custom
[820]: https://github.com/flyingmutant/rapid/blob/v1.2.0/engine.go#L730
[821]: #T.Error
[822]: #T
[823]: /builtin#any
[824]: #T.Log
[825]: #T.Fail
[826]: https://github.com/flyingmutant/rapid/blob/v1.2.0/engine.go#L721
[827]: #T.Errorf
[828]: #T
[829]: /builtin#string
[830]: /builtin#any
[831]: #T.Logf
[832]: #T.Fail
[833]: https://github.com/flyingmutant/rapid/blob/v1.2.0/engine.go#L760
[834]: #T.Fail
[835]: #T
[836]: https://github.com/flyingmutant/rapid/blob/v1.2.0/engine.go#L756
[837]: #T.FailNow
[838]: #T
[839]: https://github.com/flyingmutant/rapid/blob/v1.2.0/engine.go#L764
[840]: #T.Failed
[841]: #T
[842]: /builtin#bool
[843]: https://github.com/flyingmutant/rapid/blob/v1.2.0/engine.go#L748
[844]: #T.Fatal
[845]: #T
[846]: /builtin#any
[847]: #T.Log
[848]: #T.FailNow
[849]: https://github.com/flyingmutant/rapid/blob/v1.2.0/engine.go#L739
[850]: #T.Fatalf
[851]: #T
[852]: /builtin#string
[853]: /builtin#any
[854]: #T.Logf
[855]: #T.FailNow
[856]: https://github.com/flyingmutant/rapid/blob/v1.2.0/engine.go#L682
[857]: #T.Log
[858]: #T
[859]: /builtin#any
[860]: https://github.com/flyingmutant/rapid/blob/v1.2.0/engine.go#L673
[861]: #T.Logf
[862]: #T
[863]: /builtin#string
[864]: /builtin#any
[865]: https://github.com/flyingmutant/rapid/blob/v1.2.0/statemachine.go#L29
[866]: #T.Repeat
[867]: #T
[868]: /builtin#string
[869]: #T
[870]: #StateMachineActions
[871]: #example-T.Repeat-Queue
[872]: https://github.com/flyingmutant/rapid/blob/v1.2.0/engine.go#L701
[873]: #T.Skip
[874]: #T
[875]: /builtin#any
[876]: #T.Log
[877]: #T.SkipNow
[878]: https://github.com/flyingmutant/rapid/blob/v1.2.0/engine.go#L716
[879]: #T.SkipNow
[880]: #T
[881]: #T.Repeat
[882]: https://github.com/flyingmutant/rapid/blob/v1.2.0/engine.go#L692
[883]: #T.Skipf
[884]: #T
[885]: /builtin#string
[886]: /builtin#any
[887]: #T.Logf
[888]: #T.SkipNow
[889]: https://github.com/flyingmutant/rapid/blob/v1.2.0/engine.go#L464
[890]: #TB
[891]: /builtin#string
[892]: /builtin#string
[893]: /builtin#any
[894]: /builtin#any
[895]: /builtin#string
[896]: /builtin#any
[897]: /builtin#any
[898]: /builtin#string
[899]: /builtin#any
[900]: /builtin#any
[901]: /builtin#string
[902]: /builtin#any
[903]: /builtin#any
[904]: /builtin#bool
[905]: /testing#T
[906]: /testing#B
[907]: #T
[908]: #section-sourcefiles
[909]: https://github.com/flyingmutant/rapid/tree/v1.2.0
[910]: https://github.com/flyingmutant/rapid/blob/v1.2.0/collections.go
[911]: https://github.com/flyingmutant/rapid/blob/v1.2.0/combinators.go
[912]: https://github.com/flyingmutant/rapid/blob/v1.2.0/data.go
[913]: https://github.com/flyingmutant/rapid/blob/v1.2.0/doc.go
[914]: https://github.com/flyingmutant/rapid/blob/v1.2.0/engine.go
[915]: https://github.com/flyingmutant/rapid/blob/v1.2.0/floats.go
[916]: https://github.com/flyingmutant/rapid/blob/v1.2.0/generator.go
[917]: https://github.com/flyingmutant/rapid/blob/v1.2.0/integers.go
[918]: https://github.com/flyingmutant/rapid/blob/v1.2.0/make.go
[919]: https://github.com/flyingmutant/rapid/blob/v1.2.0/persist.go
[920]: https://github.com/flyingmutant/rapid/blob/v1.2.0/shrink.go
[921]: https://github.com/flyingmutant/rapid/blob/v1.2.0/statemachine.go
[922]: https://github.com/flyingmutant/rapid/blob/v1.2.0/strings.go
[923]: https://github.com/flyingmutant/rapid/blob/v1.2.0/utils.go
[924]: https://github.com/flyingmutant/rapid/blob/v1.2.0/vis.go
