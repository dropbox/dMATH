# Quickstart[¶][1]

This is a lightning introduction to the most important features of Hypothesis; enough to get you
started writing tests. The [tutorial][2] introduces these features (and more) in greater detail.

## Install Hypothesis[¶][3]

pip install hypothesis

## Write your first test[¶][4]

Create a new file called `example.py`, containing a simple test:

# contents of example.py
from hypothesis import [given][5], strategies as st

[@given][6]([st.integers][7]())
def test_integers(n):
    [print][8](f"called with {n}")
    assert [isinstance][9](n, [int][10])

test_integers()

[`@given`][11] is the standard entrypoint to Hypothesis. It takes a *strategy*, which describes the
type of inputs you want the decorated function to accept. When we call `test_integers`, Hypothesis
will generate random integers (because we used the [`integers()`][12] strategy) and pass them as
`n`. Let’s see that in action now by running `python example.py`:

called with 0
called with -18588
called with -672780074
called with 32616
...

We just called `test_integers()`, without passing a value for `n`, because Hypothesis generates
random values of `n` for us.

Note

By default, Hypothesis generates 100 random inputs. You can control this with the
[`max_examples`][13] setting.

## Running in a test suite[¶][14]

A Hypothesis test is still a regular python function, which means pytest or unittest will pick it up
and run it in all the normal ways.

# contents of example.py
from hypothesis import [given][15], strategies as st

[@given][16]([st.integers][17](0, 200))
def test_integers(n):
    assert n < 50

This test will clearly fail, which can be confirmed by running `pytest example.py`:

$ pytest example.py

    ...

    @given(st.integers())
    def test_integers(n):
>       assert n < 50
E       assert 50 < 50
E       Falsifying example: test_integers(
E           n=50,
E       )

## Arguments to [`@given`][18][¶][19]

You can pass multiple arguments to [`@given`][20]:

[@given][21]([st.integers][22](), [st.text][23]())
def test_integers(n, s):
    assert [isinstance][24](n, [int][25])
    assert [isinstance][26](s, [str][27])

Or use keyword arguments:

[@given][28](n=[st.integers][29](), s=[st.text][30]())
def test_integers(n, s):
    assert [isinstance][31](n, [int][32])
    assert [isinstance][33](s, [str][34])

Note

See [`@given`][35] for details about how [`@given`][36] handles different types of arguments.

## Filtering inside a test[¶][37]

Sometimes, you need to remove invalid cases from your test. The best way to do this is with
[`.filter()`][38]:

[@given][39]([st.integers][40]().filter(lambda n: n % 2 == 0))
def test_integers(n):
    assert n % 2 == 0

For more complicated conditions, you can use [`assume()`][41], which tells Hypothesis to discard any
test case with a false-y argument:

[@given][42]([st.integers][43](), [st.integers][44]())
def test_integers(n1, n2):
    [assume][45](n1 != n2)
    # n1 and n2 are guaranteed to be different here

Note

You can learn more about [`.filter()`][46] and [`assume()`][47] in the [Adapting strategies][48]
tutorial page.

## Dependent generation[¶][49]

You may want an input to depend on the value of another input. For instance, you might want to
generate two integers `n1` and `n2` where `n1 <= n2`.

You can do this using the [`@composite`][50] strategy. [`@composite`][51] lets you define a new
strategy which is itself built by drawing values from other strategies, using the
automatically-passed `draw` function.

[@st.composite][52]
def ordered_pairs(draw):
    n1 = draw([st.integers][53]())
    n2 = draw([st.integers][54](min_value=n1))
    return (n1, n2)

[@given][55](ordered_pairs())
def test_pairs_are_ordered(pair):
    n1, n2 = pair
    assert n1 <= n2

In more complex cases, you might need to interleave generation and test code. In this case, use
[`data()`][56].

[@given][57]([st.data][58](), [st.text][59](min_size=1))
def test_string_characters_are_substrings(data, string):
    assert [isinstance][60](string, [str][61])
    index = data.draw([st.integers][62](0, [len][63](string) - 1))
    assert string[index] in string

## Combining Hypothesis with pytest[¶][64]

Hypothesis works with pytest features, like [pytest.mark.parametrize][65]:

import [pytest][66]

from hypothesis import [given][67], strategies as st

@pytest.mark.parametrize("operation", [[reversed][68], [sorted][69]])
[@given][70]([st.lists][71]([st.integers][72]()))
def test_list_operation_preserves_length(operation, lst):
    assert [len][73](lst) == [len][74]([list][75](operation(lst)))

Hypothesis also works with pytest fixtures:

import [pytest][76]

[@pytest.fixture][77](scope="session")
def shared_mapping():
    return {n: 0 for n in [range][78](101)}

[@given][79]([st.integers][80](0, 100))
def test_shared_mapping_keys(shared_mapping, n):
    assert n in shared_mapping

[1]: #quickstart
[2]: tutorial/index.html
[3]: #install-hypothesis
[4]: #write-your-first-test
[5]: reference/api.html#hypothesis.given
[6]: reference/api.html#hypothesis.given
[7]: reference/strategies.html#hypothesis.strategies.integers
[8]: https://docs.python.org/3/library/functions.html#print
[9]: https://docs.python.org/3/library/functions.html#isinstance
[10]: https://docs.python.org/3/library/functions.html#int
[11]: reference/api.html#hypothesis.given
[12]: reference/strategies.html#hypothesis.strategies.integers
[13]: reference/api.html#hypothesis.settings.max_examples
[14]: #running-in-a-test-suite
[15]: reference/api.html#hypothesis.given
[16]: reference/api.html#hypothesis.given
[17]: reference/strategies.html#hypothesis.strategies.integers
[18]: reference/api.html#hypothesis.given
[19]: #arguments-to-given
[20]: reference/api.html#hypothesis.given
[21]: reference/api.html#hypothesis.given
[22]: reference/strategies.html#hypothesis.strategies.integers
[23]: reference/strategies.html#hypothesis.strategies.text
[24]: https://docs.python.org/3/library/functions.html#isinstance
[25]: https://docs.python.org/3/library/functions.html#int
[26]: https://docs.python.org/3/library/functions.html#isinstance
[27]: https://docs.python.org/3/library/stdtypes.html#str
[28]: reference/api.html#hypothesis.given
[29]: reference/strategies.html#hypothesis.strategies.integers
[30]: reference/strategies.html#hypothesis.strategies.text
[31]: https://docs.python.org/3/library/functions.html#isinstance
[32]: https://docs.python.org/3/library/functions.html#int
[33]: https://docs.python.org/3/library/functions.html#isinstance
[34]: https://docs.python.org/3/library/stdtypes.html#str
[35]: reference/api.html#hypothesis.given
[36]: reference/api.html#hypothesis.given
[37]: #filtering-inside-a-test
[38]: reference/strategies.html#hypothesis.strategies.SearchStrategy.filter
[39]: reference/api.html#hypothesis.given
[40]: reference/strategies.html#hypothesis.strategies.integers
[41]: reference/api.html#hypothesis.assume
[42]: reference/api.html#hypothesis.given
[43]: reference/strategies.html#hypothesis.strategies.integers
[44]: reference/strategies.html#hypothesis.strategies.integers
[45]: reference/api.html#hypothesis.assume
[46]: reference/strategies.html#hypothesis.strategies.SearchStrategy.filter
[47]: reference/api.html#hypothesis.assume
[48]: tutorial/adapting-strategies.html
[49]: #dependent-generation
[50]: reference/strategies.html#hypothesis.strategies.composite
[51]: reference/strategies.html#hypothesis.strategies.composite
[52]: reference/strategies.html#hypothesis.strategies.composite
[53]: reference/strategies.html#hypothesis.strategies.integers
[54]: reference/strategies.html#hypothesis.strategies.integers
[55]: reference/api.html#hypothesis.given
[56]: reference/strategies.html#hypothesis.strategies.data
[57]: reference/api.html#hypothesis.given
[58]: reference/strategies.html#hypothesis.strategies.data
[59]: reference/strategies.html#hypothesis.strategies.text
[60]: https://docs.python.org/3/library/functions.html#isinstance
[61]: https://docs.python.org/3/library/stdtypes.html#str
[62]: reference/strategies.html#hypothesis.strategies.integers
[63]: https://docs.python.org/3/library/functions.html#len
[64]: #combining-hypothesis-with-pytest
[65]: https://docs.pytest.org/en/stable/reference/reference.html#pytest-mark-parametrize-ref
[66]: https://docs.pytest.org/en/stable/index.html#module-pytest
[67]: reference/api.html#hypothesis.given
[68]: https://docs.python.org/3/library/functions.html#reversed
[69]: https://docs.python.org/3/library/functions.html#sorted
[70]: reference/api.html#hypothesis.given
[71]: reference/strategies.html#hypothesis.strategies.lists
[72]: reference/strategies.html#hypothesis.strategies.integers
[73]: https://docs.python.org/3/library/functions.html#len
[74]: https://docs.python.org/3/library/functions.html#len
[75]: https://docs.python.org/3/library/stdtypes.html#list
[76]: https://docs.pytest.org/en/stable/index.html#module-pytest
[77]: https://docs.pytest.org/en/stable/reference/reference.html#pytest.fixture
[78]: https://docs.python.org/3/library/stdtypes.html#range
[79]: reference/api.html#hypothesis.given
[80]: reference/strategies.html#hypothesis.strategies.integers
