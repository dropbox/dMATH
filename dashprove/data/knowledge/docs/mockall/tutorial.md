# Crate mockall Copy item path

[Source][1]
Expand description

A powerful mock object library for Rust.

Mockall provides tools to create mock versions of almost any trait or struct. They can be used in
unit tests as a stand-in for the real object.

## [§][2]Usage

There are two ways to use Mockall. The easiest is to use [`#[automock]`][3]. It can mock most
traits, or structs that only have a single `impl` block. For things it can’t handle, there is
[`mock!`][4].

Whichever method is used, the basic idea is the same.

* Create a mock struct. It’s name will be the same as the original, with “Mock” prepended.
* In your test, instantiate the mock struct with its `new` or `default` method.
* Set expectations on the mock struct. Each expectation can have required argument matchers, a
  required call count, and a required position in a [`Sequence`][5]. Each expectation must also have
  a return value.
* Supply the mock object to the code that you’re testing. It will return the preprogrammed return
  values supplied in the previous step. Any accesses contrary to your expectations will cause a
  panic.

## [§][6]User Guide

* [`Getting started`][7]
* [`Static Return values`][8]
* [`Matching arguments`][9]
* [`Call counts`][10]
* [`Sequences`][11]
* [`Checkpoints`][12]
* [`Reference arguments`][13]
* [`Reference return values`][14]
* [`impl Trait`][15]
* [`Mocking structs`][16]
* [`Generic methods`][17]
* [`Generic traits and structs`][18]
* [`Associated types`][19]
* [`Multiple and inherited traits`][20]
* [`External traits`][21]
* [`Static methods`][22]
* [`Modules`][23]
* [`Foreign functions`][24]
* [`Debug`][25]
* [`Async Traits`][26]
* [`Crate features`][27]
* [`Examples`][28]

### [§][29]Getting Started

`use mockall::*;
use mockall::predicate::*;
#[automock]
trait MyTrait {
    fn foo(&self, x: u32) -> u32;
}

fn call_with_four(x: &dyn MyTrait) -> u32 {
    x.foo(4)
}

let mut mock = MockMyTrait::new();
mock.expect_foo()
    .with(predicate::eq(4))
    .times(1)
    .returning(|x| x + 1);
assert_eq!(5, call_with_four(&mock));`

### [§][30]Static Return values

Every expectation must have an associated return value (though when the **nightly** feature is
enabled expectations will automatically return the default values of their return types, if their
return types implement `Default`.). For methods that return a `static` value, the macros will
generate an `Expectation` struct like [`this`][31]. There are two ways to set such an expectation’s
return value: with a constant ([`return_const`][32]) or a closure ([`returning`][33]). A closure
will take the method’s arguments by value.

`#[automock]
trait MyTrait {
    fn foo(&self) -> u32;
    fn bar(&self, x: u32, y: u32) -> u32;
}

let mut mock = MockMyTrait::new();
mock.expect_foo()
    .return_const(42u32);
mock.expect_bar()
    .returning(|x, y| x + y);`

Additionally, constants that aren’t `Clone` can be returned with the [`return_once`][34] method.

`struct NonClone();
#[automock]
trait Foo {
    fn foo(&self) -> NonClone;
}

let mut mock = MockFoo::new();
let r = NonClone{};
mock.expect_foo()
    .return_once(move || r);`

`return_once` can also be used for computing the return value with an `FnOnce` closure. This is
useful for returning a non-`Clone` value and also triggering side effects at the same time.

`fn do_something() {}

struct NonClone();

#[automock]
trait Foo {
    fn foo(&self) -> NonClone;
}

let mut mock = MockFoo::new();
let r = NonClone{};
mock.expect_foo()
    .return_once(move || {
        do_something();
        r
    });`

Mock objects are always `Send`. If you need to use a return type that isn’t, you can use the
[`return_const_st`][35], [`returning_st`][36], or [`return_once_st`][37], methods. If you need to
match arguments that are not `Send`, you can use the [`withf_st`][38] These take a non-`Send` object
and add runtime access checks. The wrapped object will be `Send`, but accessing it from multiple
threads will cause a runtime panic.

`#[automock]
trait Foo {
    fn foo(&self, x: Rc<u32>) -> Rc<u32>;   // Rc<u32> isn't Send
}

let mut mock = MockFoo::new();
let x = Rc::new(5);
let argument = x.clone();
mock.expect_foo()
    .withf_st(move |x| *x == argument)
    .returning_st(move |_| Rc::new(42u32));
assert_eq!(42, *mock.foo(x));`

### [§][39]Matching arguments

Optionally, expectations may have argument matchers set. A matcher will verify that the expectation
was called with the expected arguments, or panic otherwise. A matcher is anything that implements
the [`Predicate`][40] trait. For example:

[ⓘ][41]
`#[automock]
trait Foo {
    fn foo(&self, x: u32);
}

let mut mock = MockFoo::new();
mock.expect_foo()
    .with(eq(42))
    .return_const(());

mock.foo(0);    // Panics!`

See [`predicate`][42] for a list of Mockall’s builtin predicate functions. For convenience,
[`withf`][43] is a shorthand for setting the commonly used [`function`][44] predicate. The arguments
to the predicate function are the method’s arguments, *by reference*. For example:

[ⓘ][45]
`#[automock]
trait Foo {
    fn foo(&self, x: u32, y: u32);
}

let mut mock = MockFoo::new();
mock.expect_foo()
    .withf(|x: &u32, y: &u32| x == y)
    .return_const(());

mock.foo(2 + 2, 5);    // Panics!`

#### [§][46]Matching multiple calls

Matchers can also be used to discriminate between different invocations of the same function. Used
that way, they can provide different return values for different arguments. The way this works is
that on a method call, all expectations set on a given method are evaluated in FIFO order. The first
matching expectation is used. Only if none of the expectations match does Mockall panic. For
example:

`#[automock]
trait Foo {
    fn foo(&self, x: u32) -> u32;
}

let mut mock = MockFoo::new();
mock.expect_foo()
    .with(eq(5))
    .return_const(50u32);
mock.expect_foo()
    .with(eq(6))
    .return_const(60u32);`

One common pattern is to use multiple expectations in order of decreasing specificity. The last
expectation can provide a default or fallback value, and earlier ones can be more specific. For
example:

`#[automock]
trait Foo {
    fn open(&self, path: String) -> Option<u32>;
}

let mut mock = MockFoo::new();
mock.expect_open()
    .with(eq(String::from("something.txt")))
    .returning(|_| Some(5));
mock.expect_open()
    .return_const(None);`

### [§][47]Call counts

By default, every expectation is allowed to be called an unlimited number of times. But Mockall can
optionally verify that an expectation was called a fixed number of times, or any number of times
within a given range.

[ⓘ][48]
`#[automock]
trait Foo {
    fn foo(&self, x: u32);
}

let mut mock = MockFoo::new();
mock.expect_foo()
    .times(1)
    .return_const(());

mock.foo(0);    // Ok
mock.foo(1);    // Panics!`

See also [`never`][49] and [`times`][50].

### [§][51]Sequences

By default expectations may be matched in any order. But it’s possible to specify the order by using
a [`Sequence`][52]. Any expectations may be added to the same sequence. They don’t even need to come
from the same object.

[ⓘ][53]
`#[automock]
trait Foo {
    fn foo(&self);
}

let mut seq = Sequence::new();

let mut mock1 = MockFoo::new();
mock1.expect_foo()
    .times(1)
    .in_sequence(&mut seq)
    .returning(|| ());

let mut mock2 = MockFoo::new();
mock2.expect_foo()
    .times(1)
    .in_sequence(&mut seq)
    .returning(|| ());

mock2.foo();    // Panics!  mock1.foo should've been called first.`

### [§][54]Checkpoints

Sometimes its useful to validate all expectations mid-test, throw them away, and add new ones.
That’s what checkpoints do. Every mock object has a `checkpoint` method. When called, it will
immediately validate all methods’ expectations. So any expectations that haven’t satisfied their
call count will panic. Afterwards, those expectations will be cleared so you can add new
expectations and keep testing.

[ⓘ][55]
`#[automock]
trait Foo {
    fn foo(&self);
}

let mut mock = MockFoo::new();
mock.expect_foo()
    .times(2)
    .returning(|| ());

mock.foo();
mock.checkpoint();  // Panics!  foo hasn't yet been called twice.`
[ⓘ][56]
`#[automock]
trait Foo {
    fn foo(&self);
}

let mut mock = MockFoo::new();
mock.expect_foo()
    .times(1)
    .returning(|| ());

mock.foo();
mock.checkpoint();
mock.foo();         // Panics!  The expectation has been cleared.`

### [§][57]Reference arguments

Mockall can mock methods with reference arguments, too. There’s one catch: the matcher
[`Predicate`][58] will take reference arguments by value, not by reference.

`#[automock]
trait Foo {
    fn foo(&self, x: &u32) -> u32;
}

let mut mock = MockFoo::new();
let e = mock.expect_foo()
    // Note that x is a &u32, not a &&u32
    .withf(|x: &u32| *x == 5)
    .returning(|x: &u32| *x + 1);

assert_eq!(6, mock.foo(&5));`

### [§][59]Reference return values

Mockall can also use reference return values. There is one restriction: the lifetime of the returned
reference must be either the same as the lifetime of the mock object, or `'static`.

Mockall creates different expectation types for methods that return references. Their API is the
same as the basic `Expectation`, except for setting return values.

Methods that return `'static` references work just like methods that return any other `'static`
value.

`struct Thing(u32);

#[automock]
trait Container {
    fn get(&self, i: u32) -> &'static Thing;
}

const THING: Thing = Thing(42);
let mut mock = MockContainer::new();
mock.expect_get()
    .return_const(&THING);

assert_eq!(42, mock.get(0).0);`

Methods that take a `&self` argument use an `Expectation` class like [this][60], which gets its
return value from the [`return_const`][61] method.

`struct Thing(u32);

#[automock]
trait Container {
    fn get(&self, i: u32) -> &Thing;
}

let thing = Thing(42);
let mut mock = MockContainer::new();
mock.expect_get()
    .return_const(thing);

assert_eq!(42, mock.get(0).0);`

Methods that take a `&mut self` argument use an `Expectation` class like [this][62], class,
regardless of whether the return value is actually mutable. They can take their return value either
from the [`return_var`][63] or [`returning`][64] methods.

`struct Thing(u32);

#[automock]
trait Container {
    fn get_mut(&mut self, i: u32) -> &mut Thing;
}

let thing = Thing(42);
let mut mock = MockContainer::new();
mock.expect_get_mut()
    .return_var(thing);

mock.get_mut(0).0 = 43;
assert_eq!(43, mock.get_mut(0).0);`

Unsized types that are common targets for [`Deref`][65] are special. Mockall will automatically use
the type’s owned form for the Expectation. Currently, the [`CStr`][66], [`OsStr`][67], [`Path`][68],
[`Slice`][69], and [`str`][70] types are supported. Using this feature is automatic:

`#[automock]
trait Foo {
    fn name(&self) -> &str;
}

let mut mock = MockFoo::new();
mock.expect_name().return_const("abcd".to_owned());
assert_eq!("abcd", mock.name());`

Similarly, Mockall will use a Boxed trait object for the Expectation of methods that return
references to trait objects.

`#[automock]
trait Foo {
    fn name(&self) -> &dyn Display;
}

let mut mock = MockFoo::new();
mock.expect_name().return_const(Box::new("abcd"));
assert_eq!("abcd", format!("{}", mock.name()));`

### [§][71]Impl Trait

Rust 1.26.0 introduced the `impl Trait` feature. It allows functions to return concrete but unnamed
types (and, less usefully, to take them as arguments). It’s *almost* the same as `Box<dyn Trait>`
but without the extra allocation. Mockall supports deriving mocks for methods that return `impl
Trait`, with limitations. When you derive the mock for such a method, Mockall internally transforms
the Expectation’s return type to `Box<dyn Trait>`, without changing the mock method’s signature. So
you can use it like this:

`struct Foo {}
#[automock]
impl Foo {
    fn foo(&self) -> impl Debug {
        // ...
    }
}

let mut mock = MockFoo::new();
mock.expect_foo()
    .returning(|| Box::new(String::from("Hello, World!")));
println!("{:?}", mock.foo());`

However, `impl Trait` isn’t *exactly* equivalent to `Box<dyn Trait>` but with fewer allocations.
There are some things the former can do but the latter can’t. For one thing, you can’t build a trait
object out of a `Sized` trait. So this won’t work:

[ⓘ][72]
`struct Foo {}
#[automock]
impl Foo {
    fn foo(&self) -> impl Clone {
        // ...
    }
}`

Nor can you create a trait object that implements two or more non-auto types. So this won’t work
either:

[ⓘ][73]
`struct Foo {}
#[automock]
impl Foo {
    fn foo(&self) -> impl Debug + Display {
        // ...
    }
}`

For such cases, there is no magic bullet. The best way to mock methods like those would be to
refactor them to return named (but possibly opaque) types instead.

See Also [`impl-trait-for-returning-complex-types-with-ease.html`][74]

#### [§][75]impl Future

Rust 1.36.0 added the `Future` trait. Unlike virtually every trait that preceeded it, `Box<dyn
Future>` is mostly useless. Instead, you usually need a `Pin<Box<dyn Future>>`. So that’s what
Mockall will do when you mock a method returning `impl Future` or the related `impl Stream`. Just
remember to use `pin` in your expectations, like this:

`struct Foo {}
#[automock]
impl Foo {
    fn foo(&self) -> impl Future<Output=i32> {
        // ...
    }
}

let mut mock = MockFoo::new();
mock.expect_foo()
    .returning(|| Box::pin(future::ready(42)));`

### [§][76]Mocking structs

Mockall mocks structs as well as traits. The problem here is a namespace problem: it’s hard to
supply the mock object to your code under test, because it has a different name. The solution is to
alter import paths during test. The easiest way to do that is with the [`mockall_double`][77] crate.

[`#[automock]`][78] works for structs that have a single `impl` block:

`use mockall_double::double;
mod thing {
    use mockall::automock;
    pub struct Thing{}
    #[automock]
    impl Thing {
        pub fn foo(&self) -> u32 {
            // ...
        }
    }
}

#[double]
use thing::Thing;

fn do_stuff(thing: &Thing) -> u32 {
    thing.foo()
}

#[cfg(test)]
mod t {
    use super::*;

    #[test]
    fn test_foo() {
        let mut mock = Thing::default();
        mock.expect_foo().returning(|| 42);
        do_stuff(&mock);
    }
}`

For structs with more than one `impl` block or that have unsupported `#[derive(X)]` attributes, e.g.
`Clone`, see [`mock!`][79] instead.

### [§][80]Generic methods

Mocking generic methods is possible, but the exact process depends on whether the parameters are
`'static`, non-`'static`, or lifetimes.

#### [§][81]With static parameters

With fully `'static` parameters, the mock method is generic and so is its expect_* method. The
expect_* method usually must be called with a turbofish. Expectations set with different generic
parameters operate completely independently of one another.

`#[automock]
trait Foo {
    fn foo<T: 'static>(&self, t: T) -> i32;
}

let mut mock = MockFoo::new();
mock.expect_foo::<i16>()
    .returning(|t| i32::from(t));
mock.expect_foo::<i8>()
    .returning(|t| -i32::from(t));

assert_eq!(5, mock.foo(5i16));
assert_eq!(-5, mock.foo(5i8));`

#### [§][82]With non-`static` type parameters

Mocking methods with non-`'static` type parameters is harder. The way Mockall does it is by turning
the generic parameters into trait objects before evaluating expectations. This makes the expect_*
method concrete, rather than generic. It also comes with many restrictions. See
[`#[concretize]`][83] for more details.

#### [§][84]With generic lifetimes

A method with a lifetime parameter is technically a generic method, but Mockall treats it like a
non-generic method that must work for all possible lifetimes. Mocking such a method is similar to
mocking a non-generic method, with a few additional restrictions. One restriction is that you can’t
match calls with `with`, you must use `withf` instead. Another is that the generic lifetime may not
appear as part of the return type. Finally, no method may have both generic lifetime parameters
*and* generic type parameters.

`struct X<'a>(&'a i32);

#[automock]
trait Foo {
    fn foo<'a>(&self, x: X<'a>) -> i32;
}

let mut mock = MockFoo::new();
mock.expect_foo()
    .withf(|f| *f.0 == 5)
    .return_const(42);
let x = X(&5);
assert_eq!(42, mock.foo(x));`

### [§][85]Generic traits and structs

Mocking generic structs and generic traits is not a problem. The mock struct will be generic, too.
As with generic methods, lifetime parameters are not allowed. However, as long as the generic
parameters are not used by any static methods, then the parameters need not be `'static'`.

`#[automock]
trait Foo<T> {
    fn foo(&self, t: T) -> i32;
}

let mut mock = MockFoo::<i16>::new();
mock.expect_foo()
    .returning(|t| i32::from(t));
assert_eq!(5, mock.foo(5i16));`

### [§][86]Associated types

Traits with associated types can be mocked too. Unlike generic traits, the mock struct will not be
generic. Instead, you must specify the associated types when defining the mock struct. They’re
specified as metaitems to the [`#[automock]`][87] attribute.

`#[automock(type Key=u16; type Value=i32;)]
pub trait A {
    type Key;
    type Value;
    fn foo(&self, k: Self::Key) -> Self::Value;
}

let mut mock = MockA::new();
mock.expect_foo()
    .returning(|x: u16| i32::from(x));
assert_eq!(4, mock.foo(4));`

### [§][88]Multiple and inherited traits

Creating a mock struct that implements multiple traits, whether inherited or not, requires using the
[`mock!`][89] macro. But once created, using it is just the same as using any other mock object:

`pub trait A {
    fn foo(&self);
}

pub trait B: A {
    fn bar(&self);
}

mock! {
    // Structure to mock
    C {}
    // First trait to implement on C
    impl A for C {
        fn foo(&self);
    }
    // Second trait to implement on C
    impl B for C {
        fn bar(&self);
    }
}
let mut mock = MockC::new();
mock.expect_foo().returning(|| ());
mock.expect_bar().returning(|| ());
mock.foo();
mock.bar();`

### [§][90]External traits

Mockall can mock traits and structs defined in external crates that are beyond your control, but you
must use [`mock!`][91] instead of [`#[automock]`][92]. Mock an external trait like this:

`mock! {
    MyStruct {}     // Name of the mock struct, less the "Mock" prefix
    impl Clone for MyStruct {   // specification of the trait to mock
        fn clone(&self) -> Self;
    }
}

let mut mock1 = MockMyStruct::new();
let mock2 = MockMyStruct::new();
mock1.expect_clone()
    .return_once(move || mock2);
let cloned = mock1.clone();`

### [§][93]Static methods

Mockall can also mock static methods. But be careful! The expectations are global. If you want to
use a static method in multiple tests, you must provide your own synchronization. See the
[`synchronization example`][94] for a basic implementation. For ordinary methods, expectations are
set on the mock object. But static methods don’t have any mock object. Instead, you must create a
`Context` object just to set their expectations.

`#[automock]
pub trait A {
    fn foo() -> u32;
}

let ctx = MockA::foo_context();
ctx.expect().returning(|| 99);
assert_eq!(99, MockA::foo());`

A common pattern is mocking a trait with a constructor method. In this case, you can easily set the
mock constructor method to return a mock object.

`struct Foo{}
#[automock]
impl Foo {
    fn from_i32(x: i32) -> Self {
        // ...
    }
    fn foo(&self) -> i32 {
        // ...
    }
}

let ctx = MockFoo::from_i32_context();
ctx.expect()
    .returning(|x| {
        let mut mock = MockFoo::default();
        mock.expect_foo()
            .return_const(x);
        mock
    });
let foo = MockFoo::from_i32(42);
assert_eq!(42, foo.foo());`

#### [§][95]Generic static methods

Mocking static methods of generic structs or traits, whether or not the methods themselves are
generic, should work seamlessly as long as the generic parameter is `'static`

`#[automock]
trait Foo<T: 'static> {
    fn new(t: T) -> MockFoo<T>;
}

let ctx = MockFoo::<u32>::new_context();
ctx.expect()
    .returning(|_| MockFoo::default());
let mock = MockFoo::<u32>::new(42u32);`

#### [§][96]Context checkpoints

The context object cleans up all expectations when it leaves scope. It also has a `checkpoint`
method that functions just like a mock object’s `checkpoint` method.

[ⓘ][97]
`#[automock]
pub trait A {
    fn foo() -> u32;
}

let ctx = MockA::foo_context();
ctx.expect()
    .times(1)
    .returning(|| 99);
ctx.checkpoint();   // Panics!`

A mock object’s checkpoint method does *not* checkpoint static methods. This behavior is useful when
using multiple mock objects at once. For example:

`#[automock]
pub trait A {
    fn build() -> Self;
    fn bar(&self) -> i32;
}

let ctx = MockA::build_context();
ctx.expect()
    .times(2)
    .returning(|| MockA::default());
let mut mock0 = MockA::build();
mock0.expect_bar().return_const(4);
mock0.bar();
mock0.checkpoint();     // Does not checkpoint the build method
let mock1 = MockA::build();`

One more thing: Mockall normally creates a zero-argument `new` method for every mock struct. But it
*won’t* do that when mocking a struct that already has a method named `new`. The `default` method
will still be present.

### [§][98]Modules

In addition to mocking types, Mockall can also derive mocks for entire modules of Rust functions.
Mockall will generate a new module named “mock_xxx”, if “xxx” is the original module’s name. You can
also use `#[double]` to selectively import the mock module.

Be careful! Module functions are static and so have the same caveats as [static methods][99]
described above.

`mod outer {
    use mockall::automock;
    #[automock()]
    pub(super) mod inner {
        pub fn bar(x: u32) -> i64 {
            // ...
        }
    }
}

#[double]
use outer::inner;

#[cfg(test)]
mod t {
    use super::*;

    #[test]
    fn test_foo_bar() {
        let ctx = inner::bar_context();
        ctx.expect()
            .returning(|x| i64::from(x + 1));
        assert_eq!(5, inner::bar(4));
    }
}`

#### [§][100]Foreign functions

One reason to mock modules is when working with foreign functions. Modules may contain foreign
functions, even though structs and traits may not. Like static methods, the expectations are global.

`mod outer {
    #[automock]
    pub mod ffi {
        extern "C" {
            pub fn foo(x: u32) -> i64;
        }
    }
}

#[double]
use outer::ffi;

fn do_stuff() -> i64 {
    unsafe{ ffi::foo(42) }
}

#[cfg(test)]
mod t {
    use super::*;

    #[test]
    fn test_foo() {
        let ctx = ffi::foo_context();
        ctx.expect()
            .returning(|x| i64::from(x + 1));
        assert_eq!(43, do_stuff());
    }
}`

### [§][101]Debug

`#[automock]` will automatically generate `Debug` impls when mocking traits and struct impls.
`mock!` will too, if you add a `#[derive(Debug)]`, like this:

`mock! {
    #[derive(Debug)]
    pub Foo {}
}`

### [§][102]Async Traits

Partial support for async traits was introduced in the Rust language since 1.75.0. Mockall is
compatible with them, as well as both [`async_trait`][103] and [`trait_variant`][104] crates, with
two important limitations:

* The `#[automock]` attribute must appear *before* the crate’s attribute.
* The `#[async_trait]` and `#[trait_variant::make]` macros must be imported with their canonical
  names.
`// async_trait works with both #[automock]
#[automock]
#[async_trait]
pub trait Foo {
   async fn foo(&self) -> u32;
}
// and mock!
mock! {
    pub Bar {}
    #[async_trait]
    impl Foo for Bar {
        async fn foo(&self) -> u32;
    }
}`

### [§][105]Crate features

Mockall has a **nightly** feature. Currently this feature has two effects:

* The compiler will produce better error messages.
* Expectations for methods whose return type implements `Default` needn’t have their return values
  explicitly set. Instead, they will automatically return the default value.

With **nightly** enabled, you can omit the return value like this:

`#[automock]
trait Foo {
    fn foo(&self) -> Vec<u32>;
}

let mut mock = MockFoo::new();
mock.expect_foo();
assert!(mock.foo().is_empty());`

### [§][106]Examples

For additional examples of Mockall in action, including detailed documentation on the autogenerated
methods, see [`examples`][107].

## Modules[§][108]

*[examples][109]*
  Examples of Mockall’s generated code
*[predicate][110]*
  Predicate factories

## Macros[§][111]

*[mock][112]*
  Manually mock a structure.

## Structs[§][113]

*[Sequence][114]*
  Used to enforce that mock calls must happen in the sequence specified.

## Traits[§][115]

*[Predicate][116]*
  Trait for generically evaluating a type against a dynamically created predicate function.
*[PredicateBooleanExt][117]*
  `Predicate` extension that adds boolean logic.
*[PredicateBoxExt][118]*
  `Predicate` extension for boxing a `Predicate`.
*[PredicateFileContentExt][119]*
  `Predicate` extension adapting a `slice` Predicate.
*[PredicateStrExt][120]*
  `Predicate` extension adapting a `str` Predicate.

## Attribute Macros[§][121]

*[automock][122]*
  Automatically generate mock types for structs and traits.
*[concretize][123]*
  Decorates a method or function to tell Mockall to treat its generic arguments as trait objects
  when creating expectations.

[1]: ../src/mockall/lib.rs.html#2-2094
[2]: #usage
[3]: attr.automock.html
[4]: macro.mock.html
[5]: struct.Sequence.html
[6]: #user-guide
[7]: #getting-started
[8]: #static-return-values
[9]: #matching-arguments
[10]: #call-counts
[11]: #sequences
[12]: #checkpoints
[13]: #reference-arguments
[14]: #reference-return-values
[15]: #impl-trait
[16]: #mocking-structs
[17]: #generic-methods
[18]: #generic-traits-and-structs
[19]: #associated-types
[20]: #multiple-and-inherited-traits
[21]: #external-traits
[22]: #static-methods
[23]: #modules
[24]: #foreign-functions
[25]: #debug
[26]: #async-traits
[27]: #crate-features
[28]: #examples
[29]: #getting-started
[30]: #static-return-values
[31]: examples/__mock_MockFoo_Foo/__foo/struct.Expectation.html
[32]: examples/__mock_MockFoo_Foo/__foo/struct.Expectation.html#method.return_const
[33]: examples/__mock_MockFoo_Foo/__foo/struct.Expectation.html#method.returning
[34]: examples/__mock_MockFoo_Foo/__foo/struct.Expectation.html#method.return_once
[35]: examples/__mock_MockFoo_Foo/__foo/struct.Expectation.html#method.return_const_st
[36]: examples/__mock_MockFoo_Foo/__foo/struct.Expectation.html#method.returning_st
[37]: examples/__mock_MockFoo_Foo/__foo/struct.Expectation.html#method.return_once_st
[38]: examples/__mock_MockFoo_Foo/__foo/struct.Expectation.html#method.withf_st
[39]: #matching-arguments
[40]: trait.Predicate.html
[41]: #
[42]: trait.Predicate.html
[43]: examples/__mock_MockFoo_Foo/__foo/struct.Expectation.html#method.withf
[44]: predicate/fn.function.html
[45]: #
[46]: #matching-multiple-calls
[47]: #call-counts
[48]: #
[49]: examples/__mock_MockFoo_Foo/__foo/struct.Expectation.html#method.never
[50]: examples/__mock_MockFoo_Foo/__foo/struct.Expectation.html#method.times
[51]: #sequences
[52]: struct.Sequence.html
[53]: #
[54]: #checkpoints
[55]: #
[56]: #
[57]: #reference-arguments
[58]: trait.Predicate.html
[59]: #reference-return-values
[60]: examples/__mock_MockFoo_Foo/__bar/struct.Expectation.html
[61]: examples/__mock_MockFoo_Foo/__bar/struct.Expectation.html#method.return_const
[62]: examples/__mock_MockFoo_Foo/__baz/struct.Expectation.html
[63]: examples/__mock_MockFoo_Foo/__baz/struct.Expectation.html#method.return_var
[64]: examples/__mock_MockFoo_Foo/__baz/struct.Expectation.html#method.returning
[65]: https://doc.rust-lang.org/nightly/core/ops/deref/trait.Deref.html
[66]: https://doc.rust-lang.org/nightly/core/ffi/c_str/struct.CStr.html
[67]: https://doc.rust-lang.org/nightly/std/ffi/os_str/struct.OsStr.html
[68]: https://doc.rust-lang.org/nightly/std/path/struct.Path.html
[69]: https://doc.rust-lang.org/nightly/alloc/slice/index.html
[70]: https://doc.rust-lang.org/nightly/alloc/str/index.html
[71]: #impl-trait
[72]: #
[73]: #
[74]: https://rust-lang-nursery.github.io/edition-guide/rust-2018/trait-system/impl-trait-for-return
ing-complex-types-with-ease
[75]: #impl-future
[76]: #mocking-structs
[77]: https://docs.rs/mockall_double/latest
[78]: attr.automock.html
[79]: macro.mock.html
[80]: #generic-methods
[81]: #with-static-parameters
[82]: #with-non-static-type-parameters
[83]: attr.concretize.html
[84]: #with-generic-lifetimes
[85]: #generic-traits-and-structs
[86]: #associated-types
[87]: attr.automock.html
[88]: #multiple-and-inherited-traits
[89]: macro.mock.html
[90]: #external-traits
[91]: macro.mock.html
[92]: attr.automock.html
[93]: #static-methods
[94]: https://github.com/asomers/mockall/blob/master/mockall/examples/synchronization.rs
[95]: #generic-static-methods
[96]: #context-checkpoints
[97]: #
[98]: #modules
[99]: #static-methods
[100]: #foreign-functions
[101]: #debug
[102]: #async-traits
[103]: https://docs.rs/async-trait/latest/async_trait/
[104]: https://docs.rs/trait-variant/latest/trait_variant/
[105]: #crate-features
[106]: #examples
[107]: examples/index.html
[108]: #modules-1
[109]: examples/index.html
[110]: predicate/index.html
[111]: #macros
[112]: macro.mock.html
[113]: #structs
[114]: struct.Sequence.html
[115]: #traits
[116]: trait.Predicate.html
[117]: trait.PredicateBooleanExt.html
[118]: trait.PredicateBoxExt.html
[119]: trait.PredicateFileContentExt.html
[120]: trait.PredicateStrExt.html
[121]: #attributes
[122]: attr.automock.html
[123]: attr.concretize.html
