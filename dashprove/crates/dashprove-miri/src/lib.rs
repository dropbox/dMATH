//! MIRI integration for undefined behavior detection in Rust code
//!
//! This crate provides integration with MIRI (Mid-level Intermediate Representation Interpreter)
//! for detecting undefined behavior in Rust code during DashProve verification.
//!
//! # Key Features
//!
//! - **MIRI test runner**: Execute tests under MIRI to detect undefined behavior
//! - **Output parser**: Parse MIRI output into structured results
//! - **Harness generator**: Generate MIRI test harnesses for specific functions
//! - **UB detection**: Detect memory safety issues, data races, and other UB
//!
//! # Architecture
//!
//! The crate is organized into several modules:
//!
//! - [`config`]: MIRI configuration options and detection status
//! - [`execution`]: MIRI test execution logic
//! - [`parser`]: MIRI output parsing into structured results
//! - [`harness`]: Test harness generation for MIRI verification
//! - [`error`]: Error types for MIRI operations
//!
//! # Example
//!
//! ```ignore
//! use dashprove_miri::{MiriRunner, MiriConfig, MiriDetection};
//!
//! // Detect MIRI availability
//! let detection = MiriRunner::detect(&MiriConfig::default()).await;
//!
//! // Run MIRI on a test project
//! let result = MiriRunner::run_tests(
//!     &config,
//!     &detection,
//!     project_path,
//!     Some("test_filter"),
//! ).await?;
//!
//! // Check for UB
//! for ub in &result.undefined_behaviors {
//!     println!("UB detected: {:?}", ub);
//! }
//! ```

// Crate-level lint configuration for pedantic clippy
#![allow(clippy::must_use_candidate)]
#![allow(clippy::missing_const_for_fn)]
#![allow(clippy::use_self)]
#![allow(clippy::unused_self)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::uninlined_format_args)]
#![allow(clippy::map_unwrap_or)]
#![allow(clippy::similar_names)]
#![allow(clippy::redundant_closure_for_method_calls)]
#![allow(clippy::option_if_let_else)]
#![allow(clippy::needless_pass_by_value)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::wildcard_imports)]
#![allow(clippy::trivially_copy_pass_by_ref)]
#![allow(clippy::match_same_arms)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::single_match_else)]
#![allow(clippy::struct_excessive_bools)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::match_wildcard_for_single_variants)]
#![allow(clippy::or_fun_call)]

pub mod config;
pub mod detection;
pub mod error;
pub mod execution;
pub mod harness;
pub mod parser;

pub use config::{MiriConfig, MiriFlags};
pub use detection::{detect_miri, MiriDetection, MiriVersion};
pub use error::{MiriError, MiriResult};
pub use execution::{run_miri, run_miri_on_file, setup_miri, MiriOutput};
pub use harness::{HarnessConfig, HarnessGenerator, MiriHarness};
pub use parser::{
    parse_miri_output, MiriDiagnostic, MiriDiagnosticLevel, MiriTestResult, MiriTestStatus,
    ParsedMiriOutput, UndefinedBehavior, UndefinedBehaviorKind,
};

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;
    use std::time::Duration;

    // ==================== Strategies ====================

    fn miri_flags_strategy() -> impl Strategy<Value = MiriFlags> {
        (
            any::<bool>(),
            any::<bool>(),
            any::<bool>(),
            any::<bool>(),
            any::<bool>(),
            proptest::option::of(any::<u64>()),
            proptest::option::of(0.0f64..1.0),
            prop::collection::vec("[a-zA-Z0-9_=-]{1,20}", 0..3),
        )
            .prop_map(
                |(
                    disable_isolation,
                    check_stacked_borrows,
                    check_data_races,
                    symbolic_alignment,
                    track_raw_pointers,
                    seed,
                    preemption_rate,
                    raw_flags,
                )| MiriFlags {
                    disable_isolation,
                    check_stacked_borrows,
                    check_data_races,
                    symbolic_alignment,
                    track_raw_pointers,
                    seed,
                    preemption_rate,
                    raw_flags,
                },
            )
    }

    #[allow(dead_code)]
    fn miri_config_strategy() -> impl Strategy<Value = MiriConfig> {
        (
            proptest::option::of("[a-zA-Z/]{1,20}"),
            1u64..600,
            miri_flags_strategy(),
            prop::collection::vec(("[A-Z]{1,10}", "[a-zA-Z0-9]{1,10}"), 0..3),
            proptest::option::of(1usize..16),
        )
            .prop_map(
                |(cargo_path, timeout_secs, flags, env_vars, jobs)| MiriConfig {
                    cargo_path: cargo_path.map(std::path::PathBuf::from),
                    timeout: Duration::from_secs(timeout_secs),
                    flags,
                    env_vars,
                    jobs,
                },
            )
    }

    fn harness_config_strategy() -> impl Strategy<Value = HarnessConfig> {
        (
            any::<bool>(),
            any::<bool>(),
            any::<bool>(),
            any::<bool>(),
            1usize..10,
            any::<bool>(),
        )
            .prop_map(
                |(
                    check_unsafe,
                    check_raw_pointers,
                    check_slices,
                    check_transmute,
                    max_recursion,
                    boundary_tests,
                )| HarnessConfig {
                    check_unsafe,
                    check_raw_pointers,
                    check_slices,
                    check_transmute,
                    max_recursion,
                    boundary_tests,
                },
            )
    }

    fn ub_kind_strategy() -> impl Strategy<Value = UndefinedBehaviorKind> {
        prop_oneof![
            Just(UndefinedBehaviorKind::UseAfterFree),
            Just(UndefinedBehaviorKind::DoubleFree),
            Just(UndefinedBehaviorKind::MemoryLeak),
            Just(UndefinedBehaviorKind::OutOfBounds),
            Just(UndefinedBehaviorKind::InvalidPointerOffset),
            Just(UndefinedBehaviorKind::NullPointerDereference),
            Just(UndefinedBehaviorKind::UninitializedRead),
            Just(UndefinedBehaviorKind::InvalidAlignment),
            Just(UndefinedBehaviorKind::StackedBorrowsViolation),
            Just(UndefinedBehaviorKind::DataRace),
            Just(UndefinedBehaviorKind::Deadlock),
            Just(UndefinedBehaviorKind::InvalidEnumDiscriminant),
            Just(UndefinedBehaviorKind::InvalidFunctionPointer),
            Just(UndefinedBehaviorKind::TypeValidation),
            "[a-zA-Z ]{1,20}".prop_map(UndefinedBehaviorKind::Other),
        ]
    }

    fn miri_test_status_strategy() -> impl Strategy<Value = MiriTestStatus> {
        prop_oneof![
            Just(MiriTestStatus::Passed),
            Just(MiriTestStatus::Failed),
            Just(MiriTestStatus::Ignored),
            Just(MiriTestStatus::UndefinedBehavior),
            Just(MiriTestStatus::TimedOut),
        ]
    }

    fn miri_diagnostic_level_strategy() -> impl Strategy<Value = MiriDiagnosticLevel> {
        prop_oneof![
            Just(MiriDiagnosticLevel::Error),
            Just(MiriDiagnosticLevel::Warning),
            Just(MiriDiagnosticLevel::Note),
            Just(MiriDiagnosticLevel::Help),
        ]
    }

    fn harness_input_strategy() -> impl Strategy<Value = harness::HarnessInput> {
        (
            "[a-z]{1,10}",
            "[a-zA-Z0-9_&]{1,15}",
            prop::collection::vec("[a-zA-Z0-9_.:]{1,15}", 1..5),
        )
            .prop_map(|(name, input_type, values)| harness::HarnessInput {
                name,
                input_type,
                values,
            })
    }

    fn miri_harness_strategy() -> impl Strategy<Value = MiriHarness> {
        (
            "[a-z_]{1,20}",
            "[a-zA-Z0-9_:]{1,30}",
            "[a-zA-Z0-9_(){};#\\[\\]\n ]{10,100}",
            "[a-zA-Z ]{1,50}",
            prop::collection::vec(harness_input_strategy(), 0..3),
        )
            .prop_map(|(name, target, code, description, inputs)| MiriHarness {
                name,
                target,
                code,
                description,
                inputs,
            })
    }

    proptest! {
        // ==================== MiriFlags Tests ====================

        #[test]
        fn miri_flags_default_values(_x in 0..1i32) {
            let flags = MiriFlags::default();
            prop_assert!(!flags.disable_isolation);
            prop_assert!(flags.check_stacked_borrows);
            prop_assert!(flags.check_data_races);
            prop_assert!(flags.symbolic_alignment);
            prop_assert!(!flags.track_raw_pointers);
            prop_assert!(flags.seed.is_none());
            prop_assert!(flags.preemption_rate.is_none());
        }

        #[test]
        fn miri_flags_strict_values(_x in 0..1i32) {
            let flags = MiriFlags::strict();
            prop_assert!(!flags.disable_isolation);
            prop_assert!(flags.check_stacked_borrows);
            prop_assert!(flags.check_data_races);
            prop_assert!(flags.symbolic_alignment);
            prop_assert!(flags.track_raw_pointers);
            prop_assert_eq!(flags.seed, Some(0));
            prop_assert!(flags.preemption_rate.is_some());
        }

        #[test]
        fn miri_flags_permissive_values(_x in 0..1i32) {
            let flags = MiriFlags::permissive();
            prop_assert!(flags.disable_isolation);
            prop_assert!(flags.check_stacked_borrows);
            prop_assert!(!flags.check_data_races);
            prop_assert!(!flags.symbolic_alignment);
            prop_assert!(!flags.track_raw_pointers);
        }

        #[test]
        fn miri_flags_to_miriflags_not_empty_when_symbolic(flags in miri_flags_strategy()) {
            let miriflags = flags.to_miriflags();
            if flags.symbolic_alignment {
                prop_assert!(miriflags.contains("-Zmiri-symbolic-alignment-check"));
            }
        }

        #[test]
        fn miri_flags_to_miriflags_contains_seed_when_set(seed in 0u64..1000) {
            let flags = MiriFlags {
                seed: Some(seed),
                ..Default::default()
            };
            let miriflags = flags.to_miriflags();
            let expected = format!("-Zmiri-seed={}", seed);
            prop_assert!(miriflags.contains(&expected), "expected '{}' in '{}'", expected, miriflags);
        }

        #[test]
        fn miri_flags_to_miriflags_contains_raw_flags(raw_flag in "[a-zA-Z0-9=-]{1,20}") {
            let flags = MiriFlags {
                raw_flags: vec![raw_flag.clone()],
                ..Default::default()
            };
            let miriflags = flags.to_miriflags();
            prop_assert!(miriflags.contains(&raw_flag));
        }

        // ==================== MiriConfig Tests ====================

        #[test]
        fn miri_config_default_values(_x in 0..1i32) {
            let config = MiriConfig::default();
            prop_assert!(config.cargo_path.is_none());
            prop_assert_eq!(config.timeout, Duration::from_secs(300));
            prop_assert!(config.env_vars.is_empty());
            prop_assert!(config.jobs.is_none());
        }

        #[test]
        fn miri_config_with_timeout_preserves(timeout_secs in 1u64..1000) {
            let config = MiriConfig::with_timeout(Duration::from_secs(timeout_secs));
            prop_assert_eq!(config.timeout, Duration::from_secs(timeout_secs));
        }

        #[test]
        fn miri_config_strict_has_strict_flags(_x in 0..1i32) {
            let config = MiriConfig::strict();
            prop_assert!(config.flags.track_raw_pointers);
            prop_assert_eq!(config.flags.seed, Some(0));
        }

        #[test]
        fn miri_config_permissive_has_permissive_flags(_x in 0..1i32) {
            let config = MiriConfig::permissive();
            prop_assert!(config.flags.disable_isolation);
            prop_assert!(!config.flags.check_data_races);
        }

        // ==================== HarnessConfig Tests ====================

        #[test]
        fn harness_config_default_values(_x in 0..1i32) {
            let config = HarnessConfig::default();
            prop_assert!(config.check_unsafe);
            prop_assert!(config.check_raw_pointers);
            prop_assert!(config.check_slices);
            prop_assert!(config.check_transmute);
            prop_assert!(config.max_recursion > 0);
            prop_assert!(config.boundary_tests);
        }

        #[test]
        fn harness_config_fields_preserved(config in harness_config_strategy()) {
            prop_assert!(config.max_recursion > 0);
        }

        // ==================== HarnessGenerator Tests ====================

        #[test]
        fn harness_generator_default_config_creates_instance(_x in 0..1i32) {
            let _gen = harness::HarnessGenerator::default_config();
            prop_assert!(true);
        }

        #[test]
        fn harness_generator_new_creates_instance(config in harness_config_strategy()) {
            let _gen = harness::HarnessGenerator::new(config);
            prop_assert!(true);
        }

        #[test]
        fn harness_generator_generate_function_harness(name in "[a-z]{1,10}", sig in "fn [a-z]{1,10}\\([a-z]: i32\\) -> i32") {
            let gen = harness::HarnessGenerator::default_config();
            let result = gen.generate_function_harness(&name, &sig, None);
            prop_assert!(result.is_ok());
            let harness = result.unwrap();
            prop_assert!(harness.code.contains("#[test]"));
            prop_assert!(harness.code.contains("fn miri_test_"));
        }

        #[test]
        fn harness_generator_generate_from_template(
            template in "[a-zA-Z0-9 ]{10,50}PLACEHOLDER[a-zA-Z0-9 ]{10,50}",
            replacement in "[a-zA-Z0-9]{1,10}"
        ) {
            let gen = harness::HarnessGenerator::default_config();
            let result = gen.generate_from_template(&template, &[("PLACEHOLDER", &replacement)]);
            prop_assert!(result.contains(&replacement));
            prop_assert!(!result.contains("PLACEHOLDER"));
        }

        // ==================== MiriHarness Tests ====================

        #[test]
        fn miri_harness_fields_preserved(harness in miri_harness_strategy()) {
            prop_assert!(!harness.name.is_empty());
            prop_assert!(!harness.target.is_empty());
            prop_assert!(!harness.code.is_empty());
            prop_assert!(!harness.description.is_empty());
        }

        // ==================== HarnessInput Tests ====================

        #[test]
        fn harness_input_fields_preserved(input in harness_input_strategy()) {
            prop_assert!(!input.name.is_empty());
            prop_assert!(!input.input_type.is_empty());
            prop_assert!(!input.values.is_empty());
        }

        // ==================== UndefinedBehaviorKind Tests ====================

        #[test]
        fn ub_kind_from_message_use_after_free(_x in 0..1i32) {
            let kind = UndefinedBehaviorKind::from_message("memory access to dangling pointer");
            prop_assert_eq!(kind, UndefinedBehaviorKind::UseAfterFree);
        }

        #[test]
        fn ub_kind_from_message_double_free(_x in 0..1i32) {
            let kind = UndefinedBehaviorKind::from_message("double free detected");
            prop_assert_eq!(kind, UndefinedBehaviorKind::DoubleFree);
        }

        #[test]
        fn ub_kind_from_message_out_of_bounds(_x in 0..1i32) {
            let kind = UndefinedBehaviorKind::from_message("out of bounds memory access");
            prop_assert_eq!(kind, UndefinedBehaviorKind::OutOfBounds);
        }

        #[test]
        fn ub_kind_from_message_null_pointer(_x in 0..1i32) {
            let kind = UndefinedBehaviorKind::from_message("null pointer dereference");
            prop_assert_eq!(kind, UndefinedBehaviorKind::NullPointerDereference);
        }

        #[test]
        fn ub_kind_from_message_uninitialized(_x in 0..1i32) {
            let kind = UndefinedBehaviorKind::from_message("reading uninitialized memory");
            prop_assert_eq!(kind, UndefinedBehaviorKind::UninitializedRead);
        }

        #[test]
        fn ub_kind_from_message_alignment(_x in 0..1i32) {
            let kind = UndefinedBehaviorKind::from_message("misaligned pointer access");
            prop_assert_eq!(kind, UndefinedBehaviorKind::InvalidAlignment);
        }

        #[test]
        fn ub_kind_from_message_stacked_borrows(_x in 0..1i32) {
            let kind = UndefinedBehaviorKind::from_message("stacked borrows violation");
            prop_assert_eq!(kind, UndefinedBehaviorKind::StackedBorrowsViolation);
        }

        #[test]
        fn ub_kind_from_message_data_race(_x in 0..1i32) {
            let kind = UndefinedBehaviorKind::from_message("data race detected");
            prop_assert_eq!(kind, UndefinedBehaviorKind::DataRace);
        }

        #[test]
        fn ub_kind_from_message_unknown(msg in "[xyz]{10,20}") {
            let kind = UndefinedBehaviorKind::from_message(&msg);
            prop_assert!(matches!(kind, UndefinedBehaviorKind::Other(_)));
        }

        // ==================== MiriTestStatus Tests ====================

        #[test]
        fn miri_test_status_eq_reflexive(status in miri_test_status_strategy()) {
            prop_assert_eq!(status, status);
        }

        // ==================== MiriDiagnosticLevel Tests ====================

        #[test]
        fn miri_diagnostic_level_eq_reflexive(level in miri_diagnostic_level_strategy()) {
            prop_assert_eq!(level, level);
        }

        // ==================== MiriTestResult Tests ====================

        #[test]
        fn miri_test_result_fields_preserved(
            name in "[a-zA-Z_:]{1,30}",
            status in miri_test_status_strategy(),
            duration in proptest::option::of(1u64..10000)
        ) {
            let result = MiriTestResult {
                name: name.clone(),
                status,
                duration_ms: duration,
                error: None,
            };
            prop_assert_eq!(result.name, name);
            prop_assert_eq!(result.status, status);
            prop_assert_eq!(result.duration_ms, duration);
        }

        // ==================== MiriDiagnostic Tests ====================

        #[test]
        fn miri_diagnostic_fields_preserved(
            level in miri_diagnostic_level_strategy(),
            message in "[a-zA-Z ]{1,50}",
            location in proptest::option::of("[a-zA-Z0-9/:]{1,30}")
        ) {
            let diag = MiriDiagnostic {
                level,
                message: message.clone(),
                location: location.clone(),
                code: None,
            };
            prop_assert_eq!(diag.level, level);
            prop_assert_eq!(diag.message, message);
            prop_assert_eq!(diag.location, location);
        }

        // ==================== UndefinedBehavior Tests ====================

        #[test]
        fn undefined_behavior_fields_preserved(
            kind in ub_kind_strategy(),
            message in "[a-zA-Z ]{1,30}",
            location in proptest::option::of("[a-zA-Z0-9/:]{1,30}")
        ) {
            let ub = UndefinedBehavior {
                kind: kind.clone(),
                message: message.clone(),
                location: location.clone(),
                backtrace: vec![],
                notes: vec![],
            };
            prop_assert_eq!(ub.message, message);
            prop_assert_eq!(ub.location, location);
        }

        // ==================== MiriSummary Tests ====================

        #[test]
        fn miri_summary_default_is_zero(_x in 0..1i32) {
            let summary = parser::MiriSummary::default();
            prop_assert_eq!(summary.total_tests, 0);
            prop_assert_eq!(summary.passed, 0);
            prop_assert_eq!(summary.failed, 0);
            prop_assert_eq!(summary.ignored, 0);
            prop_assert_eq!(summary.ub_count, 0);
            prop_assert!(summary.ub_by_kind.is_empty());
        }

        // ==================== ParsedMiriOutput Tests ====================

        #[test]
        fn parsed_miri_output_has_ub_when_non_empty(_x in 0..1i32) {
            let output = ParsedMiriOutput {
                undefined_behaviors: vec![UndefinedBehavior {
                    kind: UndefinedBehaviorKind::UseAfterFree,
                    message: "test".to_string(),
                    location: None,
                    backtrace: vec![],
                    notes: vec![],
                }],
                diagnostics: vec![],
                test_results: vec![],
                summary: parser::MiriSummary::default(),
                raw_stderr: String::new(),
            };
            prop_assert!(output.has_undefined_behavior());
        }

        #[test]
        fn parsed_miri_output_no_ub_when_empty(_x in 0..1i32) {
            let output = ParsedMiriOutput {
                undefined_behaviors: vec![],
                diagnostics: vec![],
                test_results: vec![],
                summary: parser::MiriSummary::default(),
                raw_stderr: String::new(),
            };
            prop_assert!(!output.has_undefined_behavior());
        }

        #[test]
        fn parsed_miri_output_all_passed_when_all_passed(_x in 0..1i32) {
            let output = ParsedMiriOutput {
                undefined_behaviors: vec![],
                diagnostics: vec![],
                test_results: vec![
                    MiriTestResult {
                        name: "test1".to_string(),
                        status: MiriTestStatus::Passed,
                        duration_ms: None,
                        error: None,
                    },
                    MiriTestResult {
                        name: "test2".to_string(),
                        status: MiriTestStatus::Ignored,
                        duration_ms: None,
                        error: None,
                    },
                ],
                summary: parser::MiriSummary::default(),
                raw_stderr: String::new(),
            };
            prop_assert!(output.all_tests_passed());
        }

        #[test]
        fn parsed_miri_output_not_all_passed_when_failed(_x in 0..1i32) {
            let output = ParsedMiriOutput {
                undefined_behaviors: vec![],
                diagnostics: vec![],
                test_results: vec![
                    MiriTestResult {
                        name: "test1".to_string(),
                        status: MiriTestStatus::Passed,
                        duration_ms: None,
                        error: None,
                    },
                    MiriTestResult {
                        name: "test2".to_string(),
                        status: MiriTestStatus::Failed,
                        duration_ms: None,
                        error: None,
                    },
                ],
                summary: parser::MiriSummary::default(),
                raw_stderr: String::new(),
            };
            prop_assert!(!output.all_tests_passed());
        }

        // ==================== MiriError Tests ====================

        #[test]
        fn miri_error_not_available_preserves_message(msg in "[a-zA-Z ]{1,30}") {
            let err = MiriError::NotAvailable(msg.clone());
            let display = format!("{}", err);
            prop_assert!(display.contains(&msg));
        }

        #[test]
        fn miri_error_execution_failed_preserves_message(msg in "[a-zA-Z ]{1,30}") {
            let err = MiriError::ExecutionFailed(msg.clone());
            let display = format!("{}", err);
            prop_assert!(display.contains(&msg));
        }

        #[test]
        fn miri_error_timeout_preserves_duration(secs in 1u64..1000) {
            let err = MiriError::Timeout(Duration::from_secs(secs));
            let display = format!("{}", err);
            prop_assert!(display.contains("timed out"));
        }

        #[test]
        fn miri_error_parse_error_preserves_message(msg in "[a-zA-Z ]{1,30}") {
            let err = MiriError::ParseError(msg.clone());
            let display = format!("{}", err);
            prop_assert!(display.contains(&msg));
        }

        #[test]
        fn miri_error_harness_generation_failed_preserves_message(msg in "[a-zA-Z ]{1,30}") {
            let err = MiriError::HarnessGenerationFailed(msg.clone());
            let display = format!("{}", err);
            prop_assert!(display.contains(&msg));
        }

        #[test]
        fn miri_error_io_error_preserves_message(msg in "[a-zA-Z ]{1,30}") {
            let err = MiriError::IoError(msg.clone());
            let display = format!("{}", err);
            prop_assert!(display.contains(&msg));
        }

        #[test]
        fn miri_error_invalid_config_preserves_message(msg in "[a-zA-Z ]{1,30}") {
            let err = MiriError::InvalidConfig(msg.clone());
            let display = format!("{}", err);
            prop_assert!(display.contains(&msg));
        }

        // ==================== MiriVersion Tests ====================

        #[test]
        fn miri_version_fields_preserved(
            version_string in "[a-zA-Z0-9. ]{1,30}",
            miri_version in proptest::option::of("[0-9.]{1,10}"),
            rust_version in proptest::option::of("[a-zA-Z0-9. ]{1,20}")
        ) {
            let version = MiriVersion {
                version_string: version_string.clone(),
                miri_version: miri_version.clone(),
                rust_version: rust_version.clone(),
            };
            prop_assert_eq!(version.version_string, version_string);
            prop_assert_eq!(version.miri_version, miri_version);
            prop_assert_eq!(version.rust_version, rust_version);
        }

        // ==================== MiriDetection Tests ====================

        #[test]
        fn miri_detection_not_found_not_available(reason in "[a-zA-Z ]{1,30}") {
            let detection = MiriDetection::NotFound(reason);
            prop_assert!(!detection.is_available());
            prop_assert!(detection.version().is_none());
            prop_assert!(detection.cargo_path().is_none());
        }

        #[test]
        fn miri_detection_available_is_available(_x in 0..1i32) {
            let detection = MiriDetection::Available {
                cargo_path: std::path::PathBuf::from("/usr/bin/cargo"),
                version: MiriVersion {
                    version_string: "miri 0.1.0".to_string(),
                    miri_version: Some("0.1.0".to_string()),
                    rust_version: None,
                },
            };
            prop_assert!(detection.is_available());
            prop_assert!(detection.version().is_some());
            prop_assert!(detection.cargo_path().is_some());
        }
    }
}

#[cfg(kani)]
mod kani_proofs {
    use super::*;
    use std::time::Duration;

    // MiriFlags invariants
    #[kani::proof]
    fn verify_miri_flags_default_values() {
        let flags = MiriFlags::default();
        kani::assert(
            !flags.disable_isolation,
            "Default disable_isolation is false",
        );
        kani::assert(
            flags.check_stacked_borrows,
            "Default check_stacked_borrows is true",
        );
        kani::assert(flags.check_data_races, "Default check_data_races is true");
        kani::assert(
            flags.symbolic_alignment,
            "Default symbolic_alignment is true",
        );
        kani::assert(
            !flags.track_raw_pointers,
            "Default track_raw_pointers is false",
        );
        kani::assert(flags.seed.is_none(), "Default seed is None");
        kani::assert(
            flags.preemption_rate.is_none(),
            "Default preemption_rate is None",
        );
        kani::assert(flags.raw_flags.is_empty(), "Default raw_flags is empty");
    }

    #[kani::proof]
    fn verify_miri_flags_strict_values() {
        let flags = MiriFlags::strict();
        kani::assert(
            !flags.disable_isolation,
            "Strict disable_isolation is false",
        );
        kani::assert(
            flags.check_stacked_borrows,
            "Strict check_stacked_borrows is true",
        );
        kani::assert(flags.check_data_races, "Strict check_data_races is true");
        kani::assert(
            flags.symbolic_alignment,
            "Strict symbolic_alignment is true",
        );
        kani::assert(
            flags.track_raw_pointers,
            "Strict track_raw_pointers is true",
        );
        kani::assert(flags.seed == Some(0), "Strict seed is Some(0)");
        kani::assert(
            flags.preemption_rate.is_some(),
            "Strict preemption_rate is Some",
        );
    }

    #[kani::proof]
    fn verify_miri_flags_permissive_values() {
        let flags = MiriFlags::permissive();
        kani::assert(
            flags.disable_isolation,
            "Permissive disable_isolation is true",
        );
        kani::assert(
            flags.check_stacked_borrows,
            "Permissive check_stacked_borrows is true",
        );
        kani::assert(
            !flags.check_data_races,
            "Permissive check_data_races is false",
        );
        kani::assert(
            !flags.symbolic_alignment,
            "Permissive symbolic_alignment is false",
        );
        kani::assert(
            !flags.track_raw_pointers,
            "Permissive track_raw_pointers is false",
        );
        kani::assert(flags.seed.is_none(), "Permissive seed is None");
        kani::assert(
            flags.preemption_rate.is_none(),
            "Permissive preemption_rate is None",
        );
    }

    // MiriConfig invariants
    #[kani::proof]
    fn verify_miri_config_default_values() {
        let config = MiriConfig::default();
        kani::assert(
            config.timeout == Duration::from_secs(300),
            "Default timeout is 300s",
        );
        kani::assert(config.cargo_path.is_none(), "Default cargo_path is None");
        kani::assert(config.env_vars.is_empty(), "Default env_vars is empty");
        kani::assert(config.jobs.is_none(), "Default jobs is None");
    }

    #[kani::proof]
    fn verify_miri_config_with_timeout() {
        let secs: u64 = kani::any();
        kani::assume(secs > 0 && secs < 100000);
        let timeout = Duration::from_secs(secs);
        let config = MiriConfig::with_timeout(timeout);
        kani::assert(
            config.timeout == timeout,
            "with_timeout must preserve value",
        );
    }

    #[kani::proof]
    fn verify_miri_config_strict_uses_strict_flags() {
        let config = MiriConfig::strict();
        let flags = config.flags;
        kani::assert(
            flags.track_raw_pointers,
            "Strict config must have track_raw_pointers",
        );
        kani::assert(flags.seed == Some(0), "Strict config must have seed=0");
    }

    #[kani::proof]
    fn verify_miri_config_permissive_uses_permissive_flags() {
        let config = MiriConfig::permissive();
        let flags = config.flags;
        kani::assert(
            flags.disable_isolation,
            "Permissive config must have disable_isolation",
        );
        kani::assert(
            !flags.check_data_races,
            "Permissive config must have check_data_races=false",
        );
    }

    // MiriTestStatus invariants
    #[kani::proof]
    fn verify_miri_test_status_equality_reflexive() {
        let status: MiriTestStatus = kani::any();
        kani::assert(
            status == status,
            "MiriTestStatus equality must be reflexive",
        );
    }

    // MiriDiagnosticLevel invariants
    #[kani::proof]
    fn verify_miri_diagnostic_level_equality_reflexive() {
        let level: MiriDiagnosticLevel = kani::any();
        kani::assert(
            level == level,
            "MiriDiagnosticLevel equality must be reflexive",
        );
    }

    // MiriSummary invariants
    #[kani::proof]
    fn verify_miri_summary_default_zeroes() {
        let summary = parser::MiriSummary::default();
        kani::assert(summary.total_tests == 0, "Default total_tests is 0");
        kani::assert(summary.passed == 0, "Default passed is 0");
        kani::assert(summary.failed == 0, "Default failed is 0");
        kani::assert(summary.ignored == 0, "Default ignored is 0");
        kani::assert(summary.ub_count == 0, "Default ub_count is 0");
        kani::assert(summary.ub_by_kind.is_empty(), "Default ub_by_kind is empty");
    }

    // HarnessConfig invariants
    #[kani::proof]
    fn verify_harness_config_default_values() {
        let config = HarnessConfig::default();
        kani::assert(config.check_unsafe, "Default check_unsafe is true");
        kani::assert(
            config.check_raw_pointers,
            "Default check_raw_pointers is true",
        );
        kani::assert(config.check_slices, "Default check_slices is true");
        kani::assert(config.check_transmute, "Default check_transmute is true");
        kani::assert(config.max_recursion == 3, "Default max_recursion is 3");
        kani::assert(config.boundary_tests, "Default boundary_tests is true");
    }

    // Additional MiriConfig invariants
    #[kani::proof]
    fn verify_miri_config_default_timeout() {
        let config = MiriConfig::default();
        kani::assert(
            config.timeout == std::time::Duration::from_secs(300),
            "Default timeout is 300 seconds",
        );
    }

    #[kani::proof]
    fn verify_miri_config_default_cargo_path_none() {
        let config = MiriConfig::default();
        kani::assert(config.cargo_path.is_none(), "Default cargo_path is None");
    }

    #[kani::proof]
    fn verify_miri_config_default_env_vars_empty() {
        let config = MiriConfig::default();
        kani::assert(config.env_vars.is_empty(), "Default env_vars is empty");
    }

    #[kani::proof]
    fn verify_miri_config_default_jobs_none() {
        let config = MiriConfig::default();
        kani::assert(config.jobs.is_none(), "Default jobs is None");
    }

    // MiriFlags invariants
    #[kani::proof]
    fn verify_miri_flags_default_disable_isolation() {
        let flags = MiriFlags::default();
        kani::assert(
            !flags.disable_isolation,
            "Default disable_isolation is false",
        );
    }

    #[kani::proof]
    fn verify_miri_flags_default_check_stacked_borrows() {
        let flags = MiriFlags::default();
        kani::assert(
            flags.check_stacked_borrows,
            "Default check_stacked_borrows is true",
        );
    }

    #[kani::proof]
    fn verify_miri_flags_default_check_data_races() {
        let flags = MiriFlags::default();
        kani::assert(flags.check_data_races, "Default check_data_races is true");
    }

    #[kani::proof]
    fn verify_miri_flags_default_symbolic_alignment() {
        let flags = MiriFlags::default();
        kani::assert(
            flags.symbolic_alignment,
            "Default symbolic_alignment is true",
        );
    }

    #[kani::proof]
    fn verify_miri_flags_default_track_raw_pointers() {
        let flags = MiriFlags::default();
        kani::assert(
            !flags.track_raw_pointers,
            "Default track_raw_pointers is false",
        );
    }

    #[kani::proof]
    fn verify_miri_flags_default_seed_none() {
        let flags = MiriFlags::default();
        kani::assert(flags.seed.is_none(), "Default seed is None");
    }

    #[kani::proof]
    fn verify_miri_flags_default_preemption_rate_none() {
        let flags = MiriFlags::default();
        kani::assert(
            flags.preemption_rate.is_none(),
            "Default preemption_rate is None",
        );
    }

    #[kani::proof]
    fn verify_miri_flags_default_raw_flags_empty() {
        let flags = MiriFlags::default();
        kani::assert(flags.raw_flags.is_empty(), "Default raw_flags is empty");
    }

    // Strict and permissive configs
    #[kani::proof]
    fn verify_miri_flags_strict_track_raw_pointers() {
        let flags = MiriFlags::strict();
        kani::assert(
            flags.track_raw_pointers,
            "Strict track_raw_pointers is true",
        );
    }

    #[kani::proof]
    fn verify_miri_flags_strict_seed() {
        let flags = MiriFlags::strict();
        kani::assert(flags.seed == Some(0), "Strict seed is Some(0)");
    }

    #[kani::proof]
    fn verify_miri_flags_permissive_disable_isolation() {
        let flags = MiriFlags::permissive();
        kani::assert(
            flags.disable_isolation,
            "Permissive disable_isolation is true",
        );
    }

    #[kani::proof]
    fn verify_miri_flags_permissive_check_data_races() {
        let flags = MiriFlags::permissive();
        kani::assert(
            !flags.check_data_races,
            "Permissive check_data_races is false",
        );
    }

    #[kani::proof]
    fn verify_miri_config_strict_uses_strict_flags() {
        let config = MiriConfig::strict();
        kani::assert(
            config.flags.track_raw_pointers,
            "Strict config uses strict flags",
        );
    }

    #[kani::proof]
    fn verify_miri_config_permissive_uses_permissive_flags() {
        let config = MiriConfig::permissive();
        kani::assert(
            config.flags.disable_isolation,
            "Permissive config uses permissive flags",
        );
    }
}
