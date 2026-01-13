# γ-CROWN Makefile
# Build, test, benchmark, and profile commands

.PHONY: all build test bench bench-html profile flamegraph clean

# Default target
all: build test

# Build
build:
	cargo build --release

build-debug:
	cargo build

# Testing
test:
	cargo test --release

test-verbose:
	cargo test --release -- --nocapture

# Benchmarking with Criterion
bench:
	cargo bench

bench-propagate:
	cargo bench -p gamma-propagate

bench-gpu:
	cargo bench -p gamma-gpu

# Open HTML reports
bench-html:
	@echo "Opening benchmark reports..."
	@open target/criterion/report/index.html 2>/dev/null || \
		xdg-open target/criterion/report/index.html 2>/dev/null || \
		echo "Open target/criterion/report/index.html in your browser"

# Profiling with samply (Firefox Profiler compatible)
# Install: cargo install samply
profile-samply:
	@echo "Profiling with samply (opens Firefox Profiler)..."
	cargo build --release --example profile -p gamma-propagate
	samply record ./target/release/examples/profile

# Profiling with cargo-flamegraph
# Install: cargo install flamegraph
# macOS: Requires dtrace permissions (SIP may need adjustment)
# Linux: Requires perf
flamegraph:
	@echo "Generating flamegraph..."
	cargo flamegraph --release --example profile -p gamma-propagate -o flamegraph.svg
	@echo "Flamegraph saved to flamegraph.svg"

# Profile specific benchmark
flamegraph-bench:
	@echo "Generating flamegraph for benchmarks..."
	cargo flamegraph --release --bench propagation -p gamma-propagate -o flamegraph-bench.svg -- --bench
	@echo "Flamegraph saved to flamegraph-bench.svg"

# Memory profiling with DHAT (heap profiler)
# Add dhat = "0.3" to dev-dependencies and use #[global_allocator]
profile-dhat:
	@echo "DHAT profiling requires code instrumentation."
	@echo "See: https://docs.rs/dhat"

# Quick performance check
perf-check:
	cargo build --release --example profile -p gamma-propagate
	./target/release/examples/profile

# Clippy lints
lint:
	cargo clippy --all-targets --all-features -- -D warnings

# Format check
fmt:
	cargo fmt --all -- --check

fmt-fix:
	cargo fmt --all

# Clean
clean:
	cargo clean
	rm -f flamegraph*.svg
	rm -f perf.data*

# Install profiling tools
install-tools:
	@echo "Installing profiling tools..."
	cargo install samply
	cargo install flamegraph
	cargo install cargo-criterion
	@echo "Done. Note: flamegraph may require additional system setup."
	@echo "  macOS: May need to disable SIP or use dtrace permissions"
	@echo "  Linux: Ensure perf is installed (linux-tools-generic)"

# Help
help:
	@echo "γ-CROWN Build & Profile Commands"
	@echo ""
	@echo "Building:"
	@echo "  make build        - Release build"
	@echo "  make build-debug  - Debug build"
	@echo ""
	@echo "Testing:"
	@echo "  make test         - Run all tests"
	@echo "  make test-verbose - Run tests with output"
	@echo ""
	@echo "Benchmarking:"
	@echo "  make bench        - Run all Criterion benchmarks"
	@echo "  make bench-propagate - Benchmark gamma-propagate only"
	@echo "  make bench-gpu    - Benchmark gamma-gpu only"
	@echo "  make bench-html   - Open benchmark HTML reports"
	@echo ""
	@echo "Profiling:"
	@echo "  make perf-check   - Quick performance check"
	@echo "  make profile-samply - Profile with samply (Firefox Profiler)"
	@echo "  make flamegraph   - Generate flamegraph SVG"
	@echo ""
	@echo "Tools:"
	@echo "  make install-tools - Install samply, flamegraph, criterion"
	@echo "  make lint         - Run clippy"
	@echo "  make fmt          - Check formatting"
