# Welcome to Hypothesis![¶][1]

Hypothesis is the property-based testing library for Python. With Hypothesis, you write tests which
should pass for all inputs in whatever range you describe, and let Hypothesis randomly choose which
of those inputs to check - including edge cases you might not have thought about. For example:

from hypothesis import [given][2], strategies as st

[@given][3]([st.lists][4]([st.integers][5]() | [st.floats][6]()))
def test_sort_correctness_using_properties(lst):
    result = my_sort(lst)
    assert [set][7](lst) == [set][8](result)
    assert [all][9](a <= b for a, b in [zip][10](result, result[1:]))

You should start with the [tutorial][11], or alternatively the more condensed [quickstart][12].

## [Tutorial][13][¶][14]

An introduction to Hypothesis.

New users should start here, or with the more condensed [quickstart][15].

## [How-to guides][16][¶][17]

Practical guides for applying Hypothesis in specific scenarios.

## [Explanations][18][¶][19]

Commentary oriented towards deepening your understanding of Hypothesis.

## [API Reference][20][¶][21]

Technical API reference.

[1]: #welcome-to-hypothesis
[2]: reference/api.html#hypothesis.given
[3]: reference/api.html#hypothesis.given
[4]: reference/strategies.html#hypothesis.strategies.lists
[5]: reference/strategies.html#hypothesis.strategies.integers
[6]: reference/strategies.html#hypothesis.strategies.floats
[7]: https://docs.python.org/3/library/stdtypes.html#set
[8]: https://docs.python.org/3/library/stdtypes.html#set
[9]: https://docs.python.org/3/library/functions.html#all
[10]: https://docs.python.org/3/library/functions.html#zip
[11]: tutorial/index.html
[12]: quickstart.html
[13]: tutorial/index.html
[14]: #tutorial
[15]: quickstart.html
[16]: how-to/index.html
[17]: #how-to-guides
[18]: explanation/index.html
[19]: #explanations
[20]: reference/index.html
[21]: #api-reference
