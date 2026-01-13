# Rudra

Rudra is a static analyzer to detect common undefined behaviors in Rust programs. It is capable of
analyzing single Rust packages as well as all the packages on crates.io.

Rudra and its associated paper received the Distinguished Artifact Award at *the 28th ACM Symposium
on Operating Systems Principles 2021* (SOSP '21). ([PDF][1], [short talk][2], [long talk][3])

You can find the list of bugs found by Rudra at [Rudra-PoC][4] repository.

## Usage

The easiest way to use Rudra is to use [Docker][5].

1. First, make sure your system has Docker and Python 3 installed.
2. Add `rudra:latest` image on your system. There are two ways of doing this:
   
   * `docker pull ghcr.io/sslab-gatech/rudra:master && docker tag ghcr.io/sslab-gatech/rudra:master
     rudra:latest`
   * Alternatively, you can build your own image with `docker build . -t rudra:latest`
3. Run `./setup_rudra_runner_home.py <directory>` and set `RUDRA_RUNNER_HOME` to that directory.
   Example: `./setup_rudra_runner_home.py ~/rudra-home && export
   RUDRA_RUNNER_HOME=$HOME/rudra-home`.
   
   * There are two scripts, `./setup_rudra_runner_home.py` and `./setup_rudra_runner_home_fixed.py`.
     In general, `./setup_rudra_runner_home.py` should be used unless you want to reproduce the
     result of the paper with a fixed cargo index.
4. Add `docker-helper` in Rudra repository to `$PATH`. Now you are ready to test Rudra!

For development, you might want to install Rudra on your host system. See [DEV.md][6] for advanced
usage and development guide.

### Run Rudra on a single project

`docker-cargo-rudra <directory>
`

The log and report are printed to stderr by default.

### Run Rudra as GitHub Action

Rudra can be run as a GitHub Action allowing the static analyze to be used in an Action workflow.

# Run Rudra
- name: Rudra
  uses: sslab-gatech/Rudra@master

### Run Rudra with different compiler version

Rudra is tied to a specific Rust compiler version, and it can only analyze projects that compiles
with this version of the compiler. `master` branch uses `nightly-2021-10-21` version of Rust right
now. Check [the version page][7] for all supported versions.

### Known Issues

* Rudra does not support workspaces (#11). You can install Rudra on your host system (see
  [DEV.md][8]) and run analysis in the subdirectories to sidestep the problem for now.
* Rudra does not support suppressing warnings in specific locations. This could cause a usability
  issue when used in CI/CD due to false positives.

## Bug Types Detected by Rudra

Rudra currently detects the following bug types. For the full detail, please check our SOSP 2021
paper.

### Panic Safety (Unsafe code that can create memory-safety issues when panicked)

Detects when unsafe code may lead to memory safety issues if a user provided closure or trait
panics. For example, consider a function that dereferences a pointer with `ptr::read`, duplicating
its ownership and then calls a user provided function `f`. This can lead to a double-free if the
function `f` panics.

See [this section of the Rustonomicon][9] for more details.

while idx < len {
    let ch = unsafe { self.get_unchecked(idx..len).chars().next().unwrap() };
    let ch_len = ch.len_utf8();

    // Call to user provided predicate function f that can panic.
    if !f(ch) {
        del_bytes += ch_len;
    } else if del_bytes > 0 {
        unsafe {
            ptr::copy(
                self.vec.as_ptr().add(idx),
                self.vec.as_mut_ptr().add(idx - del_bytes),
                ch_len,
            );
        }
    }

    // Point idx to the next char
    idx += ch_len;
}

Example: [rust#78498][10]

### Higher Order Invariant (Assumed properties about traits)

When code assumes certain properties about trait methods that aren't enforced, such as expecting the
`Borrow` trait to return the same reference on multiple calls to `borrow`.

let mut g = Guard { len: buf.len(), buf }; 
// ...
  Ok(n) => g.len += n, 

Example: [rust#80894][11]

### Send Sync Variance (Unrestricted Send or Sync on generic types)

This occurs when a type generic over `T` implements Send or Sync without having correct bounds on
`T`.

unsafe impl<T: ?Sized + Send, U: ?Sized> Send for MappedMutexGuard<'_, T, U> {} 
unsafe impl<T: ?Sized + Sync, U: ?Sized> Sync for MappedMutexGuard<'_, T, U> {} 

Example: [futures#2239][12]

## Bugs Found by Rudra

Rudra was ran on the entirety of crates.io state as of July 4th, 2020 as well as the Rust standard
library from `nightly-2020-08-26`. It managed to find 264 new memory safety issues across the Rust
ecosystem which resulted in 76 CVEs.

The details of these bugs can be found in the [Rudra-PoC repo][13].

## License

Licensed under either of

* Apache License, Version 2.0, ([LICENSE-APACHE][14] or
  [http://www.apache.org/licenses/LICENSE-2.0][15])
* MIT license ([LICENSE-MIT][16] or [http://opensource.org/licenses/MIT][17])

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the
work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any
additional terms or conditions.

[1]: /sslab-gatech/Rudra/blob/master/rudra-sosp21.pdf
[2]: https://youtu.be/7pI9GfYEu-s
[3]: https://youtu.be/Hfl6EQquUU0
[4]: https://github.com/sslab-gatech/Rudra-PoC
[5]: https://www.docker.com/
[6]: /sslab-gatech/Rudra/blob/master/DEV.md
[7]: https://github.com/sslab-gatech/Rudra/pkgs/container/rudra/versions?filters%5Bversion_type%5D=t
agged
[8]: /sslab-gatech/Rudra/blob/master/DEV.md
[9]: https://doc.rust-lang.org/nomicon/exception-safety.html
[10]: https://github.com/rust-lang/rust/issues/78498
[11]: https://github.com/rust-lang/rust/issues/80894
[12]: https://github.com/rust-lang/futures-rs/issues/2239
[13]: https://github.com/sslab-gatech/Rudra-PoC
[14]: /sslab-gatech/Rudra/blob/master/LICENSE-APACHE
[15]: http://www.apache.org/licenses/LICENSE-2.0
[16]: /sslab-gatech/Rudra/blob/master/LICENSE-MIT
[17]: http://opensource.org/licenses/MIT
