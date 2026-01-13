# Crate rstest Copy item path

[Source][1]
Expand description

This crate will help you to write simpler tests by leveraging a software testing concept called
[test fixtures][2]. A fixture is something that you can use in your tests to encapsulate a test’s
dependencies.

The general idea is to have smaller tests that only describe the thing you’re testing while you hide
the auxiliary utilities your tests make use of somewhere else. For instance, if you have an
application that has many tests with users, shopping baskets, and products, you’d have to create a
user, a shopping basket, and product every single time in every test which becomes unwieldy quickly.
To cut down on that repetition, you can instead use fixtures to declare that you need those objects
for your function and the fixtures will take care of creating those by themselves. Focus on the
important stuff in your tests!

In `rstest` a fixture is a function that can return any kind of valid Rust type. This effectively
means that your fixtures are not limited by the kind of data they can return. A test can consume an
arbitrary number of fixtures at the same time.

### [§][3]What

The `rstest` crate defines the following procedural macros:

* [`[rstest]`][4]: Declare that a test or a group of tests that may take [fixtures][5], [input
  table][6], [list of values][7] or read data from [files][8].
* [`[fixture]`][9]: To mark a function as a fixture.

### [§][10]Why

Very often in Rust we write tests like this

`#[test]
fn should_process_two_users() {
    let mut repository = create_repository();
    repository.add("Bob", 21);
    repository.add("Alice", 22);

    let processor = string_processor();
    processor.send_all(&repository, "Good Morning");

    assert_eq!(2, processor.output.find("Good Morning").count());
    assert!(processor.output.contains("Bob"));
    assert!(processor.output.contains("Alice"));
}`

By making use of [`[rstest]`][11] we can isolate the dependencies `empty_repository` and
`string_processor` by passing them as fixtures:

`#[rstest]
fn should_process_two_users(mut empty_repository: impl Repository,
                            string_processor: FakeProcessor) {
    empty_repository.add("Bob", 21);
    empty_repository.add("Alice", 22);

    string_processor.send_all("Good Morning");

    assert_eq!(2, string_processor.output.find("Good Morning").count());
    assert!(string_processor.output.contains("Bob"));
    assert!(string_processor.output.contains("Alice"));
}`

… or if you use `"Alice"` and `"Bob"` in other tests, you can isolate `alice_and_bob` fixture and
use it directly:

`#[fixture]
fn alice_and_bob(mut empty_repository: impl Repository) -> impl Repository {
    empty_repository.add("Bob", 21);
    empty_repository.add("Alice", 22);
    empty_repository
}

#[rstest]
fn should_process_two_users(alice_and_bob: impl Repository,
                            string_processor: FakeProcessor) {
    string_processor.send_all("Good Morning");

    assert_eq!(2, string_processor.output.find("Good Morning").count());
    assert!(string_processor.output.contains("Bob"));
    assert!(string_processor.output.contains("Alice"));
}`

### [§][12]Features

* `async-timeout`: `timeout` for `async` tests (Default enabled)
* `crate-name`: Import `rstest` package with different name (Default enabled)

### [§][13]Injecting fixtures as function arguments

`rstest` functions can receive fixtures by using them as input arguments. A function decorated with
[`[rstest]`][14] will resolve each argument name by call the fixture function. Fixtures should be
annotated with the [`[fixture]`][15] attribute.

Fixtures will be resolved like function calls by following the standard resolution rules. Therefore,
an identically named fixture can be use in different context.

`mod empty_cases {
    use super::*;

    #[fixture]
    fn repository() -> impl Repository {
        DataSet::default()
    }

    #[rstest]
    fn should_do_nothing(repository: impl Repository) {
        //.. test impl ..
    }
}

mod non_trivial_case {
    use super::*;

    #[fixture]
    fn repository() -> impl Repository {
        let mut ds = DataSet::default();
        // Fill your dataset with interesting case
        ds
    }

    #[rstest]
    fn should_notify_all_entries(repository: impl Repository) {
        //.. test impl ..
    }
}
`

Last but not least, fixtures can be injected like we saw in `alice_and_bob` example.

### [§][16]Creating parametrized tests

You can use also [`[rstest]`][17] to create simple table-based tests. Let’s see the classic
Fibonacci example:

`use rstest::rstest;

#[rstest]
#[case(0, 0)]
#[case(1, 1)]
#[case(2, 1)]
#[case(3, 2)]
#[case(4, 3)]
#[case(5, 5)]
#[case(6, 8)]
fn fibonacci_test(#[case] input: u32,#[case] expected: u32) {
    assert_eq!(expected, fibonacci(input))
}

fn fibonacci(input: u32) -> u32 {
    match input {
        0 => 0,
        1 => 1,
        n => fibonacci(n - 2) + fibonacci(n - 1)
    }
}`

This will generate a bunch of tests, one for every `#[case(a, b)]`.

### [§][18]Creating a test for each combinations of given values

In some cases you need to test your code for each combinations of some input values. In this cases
[`[rstest]`][19] give you the ability to define a list of values (rust expressions) to use for an
arguments.

`
#[rstest]
fn should_terminate(
    #[values(State::Init, State::Start, State::Processing)]
    state: State,
    #[values(Event::Error, Event::Fatal)]
    event: Event
) {
    assert_eq!(State::Terminated, state.process(event))
}`

This will generate a test for each combination of `state` and `event`.

### [§][20]Magic Conversion

If you need a value where its type implement `FromStr()` trait you can use a literal string to build
it.

`#[rstest]
#[case("1.2.3.4:8080", 8080)]
#[case("127.0.0.1:9000", 9000)]
fn check_port(#[case] addr: SocketAddr, #[case] expected: u16) {
    assert_eq!(expected, addr.port());
}`

You can use this feature also in value list and in fixture default value.

## Structs[§][21]

*[Context][22]*
  A test context.

## Attribute Macros[§][23]

*[fixture][24]*
  Define a fixture that you can use in all `rstest`’s test arguments. You should just mark your
  function as `#[fixture]` and then use it as a test’s argument. Fixture functions can also use
  other fixtures.
*[rstest][25]*
  The attribute that you should use for your tests. Your annotated function’s arguments can be
  [injected][26] with [`fixture`][27]s, provided by [parametrized cases][28] or by [value
  lists][29].

[1]: ../src/rstest/lib.rs.html#1-1563
[2]: https://en.wikipedia.org/wiki/Test_fixture#Software
[3]: #what
[4]: attr.rstest.html
[5]: attr.rstest.html#injecting-fixtures
[6]: attr.rstest.html#test-parametrized-cases
[7]: attr.rstest.html#values-lists
[8]: attr.rstest.html#files-path-as-input-arguments
[9]: attr.fixture.html
[10]: #why
[11]: attr.rstest.html
[12]: #features
[13]: #injecting-fixtures-as-function-arguments
[14]: attr.rstest.html#injecting-fixtures
[15]: attr.fixture.html
[16]: #creating-parametrized-tests
[17]: attr.rstest.html#test-parametrized-cases
[18]: #creating-a-test-for-each-combinations-of-given-values
[19]: attr.rstest.html#values-lists
[20]: #magic-conversion
[21]: #structs
[22]: struct.Context.html
[23]: #attributes
[24]: attr.fixture.html
[25]: attr.rstest.html
[26]: attr.rstest.html#injecting-fixtures
[27]: attr.fixture.html
[28]: attr.rstest.html#test-parametrized-cases
[29]: attr.rstest.html#values-lists
