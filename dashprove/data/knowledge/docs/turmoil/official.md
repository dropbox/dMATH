# Turmoil

**This is very experimental**

Add hardship to your tests.

Turmoil is a framework for testing distributed systems. It provides deterministic execution by
running multiple concurrent hosts within a single thread. It introduces "hardship" into the system
via changes in the simulated network and filesystem. Both can be controlled manually or with a
seeded rng.

[[Crates.io]][1] [[Documentation]][2] [[Build Status]][3] [[Discord chat]][4]

## Quickstart

Add this to your `Cargo.toml`.

[dev-dependencies]
turmoil = "0.7"

See crate documentation for simulation setup instructions.

### Examples

* [/tests][5] for TCP, UDP, and filesystem.
* [`gRPC`][6] using `tonic` and `hyper`.
* [`axum`][7]

### Filesystem Simulation (unstable)

*Requires the `unstable-fs` feature.*

[dev-dependencies]
turmoil = { version = "0.7", features = ["unstable-fs"] }

Turmoil provides simulated filesystem types for crash-consistency testing:

use turmoil::fs::shim::std::fs::OpenOptions;
use turmoil::fs::shim::std::os::unix::fs::FileExt;

let file = OpenOptions::new()
    .read(true)
    .write(true)
    .create(true)
    .open("/data/db")?;

file.write_all_at(b"data", 0)?;
file.sync_all()?;  // Data now durable, survives sim.crash()

Each host has isolated filesystem state. Use `Builder::fs_sync_probability()` to configure random
sync behavior for testing crash recovery.

## License

This project is licensed under the [MIT license][8].

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in
`turmoil` by you, shall be licensed as MIT, without any additional terms or conditions.

[1]: https://crates.io/crates/turmoil
[2]: https://docs.rs/turmoil
[3]: https://github.com/tokio-rs/turmoil/actions?query=workflow%3ACI+branch%3Amain
[4]: https://discord.com/channels/500028886025895936/628283075398467594
[5]: https://github.com/tokio-rs/turmoil/tree/main/tests
[6]: https://github.com/tokio-rs/turmoil/tree/main/examples/grpc
[7]: https://github.com/tokio-rs/turmoil/tree/main/examples/axum
[8]: /tokio-rs/turmoil/blob/main/LICENSE
