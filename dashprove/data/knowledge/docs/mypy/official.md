[Home][1]
[Blog][2]
[Examples][3]
[Docs][4]
[GitHub][5]
[Wiki][6]
[Roadmap][7]
[About][8]
[Contact][9]
[Twitter][10]
[Newsletter][11]
[Atom][12]
Follow:
[Get the Code on GitHub!][13]
Mypy on Twitter is @mypyproject

[Follow @mypyproject on Twitter][14]

Why mypy?

*Compile-time type checking *
  Static typing makes it easier to find bugs with less debugging.
*Easier maintenance *
  Type declarations act as machine-checked documentation. Static typing makes your code easier to
  understand and easier to modify without introducing bugs.
*Grow your programs from dynamic to static typing *
  You can develop programs with dynamic typing and add static typing after your code has matured, or
  migrate existing Python code to static typing.

# mypy

Mypy is an optional static type checker for Python that aims to combine the benefits of dynamic (or
"duck") typing and static typing. Mypy combines the expressive power and convenience of Python with
a powerful type system and compile-time type checking. Mypy type checks standard Python programs;
run them using any Python VM with basically no runtime overhead.

# What's new

Mypy 1.18.1 released

11 Sept 2025: Mypy 1.18.1 was released. Read the [blog post][15] for the details. -Kevin Kannammalil

Mypy 1.17 released

14 Jul 2025: Mypy 1.17 was released. Read the [blog post][16] for the details. -Ethan Sarp

Mypy 1.16 released

29 May 2025: Mypy 1.16 was released. Read the [blog post][17] for the details. -Jukka Lehtosalo

Mypy 1.15 released

5 Feb 2025: Mypy 1.15 was released. Read the [blog post][18] for the details. -Wesley Collin Wright

[Older news][19]

# Seamless dynamic and static typing

From Python...
def fib(n):
    a, b = 0, 1
    while a < n:
        yield a
        a, b = b, a+b
...to statically typed Python
def fib(n: int) -> Iterator[int]:
    a, b = 0, 1
    while a < n:
        yield a
        a, b = b, a+b

Migrate existing code to static typing, a function at a time. You can freely mix static and dynamic
typing within a program, within a module or within an expression. No need to give up dynamic typing
— use static typing when it makes sense. Often just adding function signatures gives you statically
typed code. Mypy can infer the types of other variables.

# Python syntax

Mypy type checks programs that have type annotations conforming to [PEP 484][20]. Getting started is
easy if you know Python. The aim is to support almost all Python language constructs in mypy.

# Powerful type system

Mypy has a powerful, modern type system with features such as bidirectional type inference,
generics, callable types, abstract base classes, multiple inheritance and tuple types.

# Access to Python libs

Many commonly used libraries have stubs (statically typed interface definitions) that allow mypy to
check that your code uses the libraries correctly.

Learn more

* Follow the [mypy status blog][21].
* Browse [code examples][22].
* Read [the documentation][23].
* Read [the FAQ][24].
© 2014 the mypy project · Content available under a [Creative Commons license][25] · [Contact][26] ·
[Atom feed][27]

[1]: index.html
[2]: https://mypy-lang.blogspot.co.uk/
[3]: examples.html
[4]: https://mypy.readthedocs.org/en/stable/
[5]: https://github.com/python/mypy
[6]: https://github.com/python/mypy/wiki
[7]: https://github.com/python/mypy/blob/master/ROADMAP.md
[8]: about.html
[9]: contact.html
[10]: https://twitter.com/mypyproject
[11]: https://groups.google.com/d/forum/mypy-news?hl=en-GB
[12]: http://mypy-lang.blogspot.com/feeds/posts/default
[13]: https://github.com/python/mypy
[14]: https://twitter.com/mypyproject
[15]: https://mypy-lang.blogspot.com/2025/09/mypy-1181-released.html
[16]: https://mypy-lang.blogspot.com/2025/07/mypy-117-released.html
[17]: https://mypy-lang.blogspot.com/2025/05/mypy-116-released.html
[18]: https://mypy-lang.blogspot.com/2025/02/mypy-115-released.html
[19]: news.html
[20]: https://www.python.org/dev/peps/pep-0484/
[21]: http://mypy-lang.blogspot.co.uk/
[22]: examples.html
[23]: http://mypy.readthedocs.org/en/latest/
[24]: http://mypy.readthedocs.org/en/latest/faq.html
[25]: http://creativecommons.org/licenses/by-sa/3.0/
[26]: contact.html
[27]: http://mypy-lang.blogspot.com/feeds/posts/default
