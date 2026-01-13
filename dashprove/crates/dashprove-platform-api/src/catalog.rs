//! Built-in platform API catalogs
//!
//! Pre-defined API state machines for common platforms:
//! - Metal (Apple GPU)
//! - CUDA (NVIDIA GPU)
//! - POSIX (threading, file I/O)
//! - Vulkan (cross-platform GPU)

use crate::api::{ApiState, PlatformApi, StateTransition};
use crate::constraint::{ApiConstraint, Severity};

/// Trait for platform API catalogs
pub trait BuiltinCatalog {
    /// Get all APIs in this catalog
    fn apis() -> Vec<PlatformApi>;

    /// Get the platform name
    fn platform_name() -> &'static str;
}

/// Metal API catalog (Apple GPU)
pub struct MetalCatalog;

impl BuiltinCatalog for MetalCatalog {
    fn platform_name() -> &'static str {
        "Metal"
    }

    fn apis() -> Vec<PlatformApi> {
        vec![
            Self::mtl_command_buffer(),
            Self::mtl_command_queue(),
            Self::mtl_buffer(),
            Self::mtl_texture(),
            Self::mtl_render_command_encoder(),
            Self::mtl_compute_command_encoder(),
        ]
    }
}

impl MetalCatalog {
    /// MTLCommandBuffer state machine
    ///
    /// CRITICAL: addCompletedHandler must be called BEFORE commit.
    /// This constraint caused a SIGABRT in the MPS verification project (N=1305).
    pub fn mtl_command_buffer() -> PlatformApi {
        let mut api = PlatformApi::new("Metal", "MTLCommandBuffer")
            .with_description("Metal command buffer for GPU command submission");

        // States (simplified to essential states)
        api.add_state(
            ApiState::new("Created").with_description("Initial state after makeCommandBuffer()"),
        );
        api.add_state(
            ApiState::new("Encoding").with_description("Encoder is active, recording commands"),
        );
        api.add_state(
            ApiState::new("Committed")
                .with_description("Submitted to GPU, cannot modify")
                .as_terminal(),
        );

        api.set_initial_state("Created");

        // Transitions
        api.add_transition(
            StateTransition::new("makeRenderCommandEncoder", vec!["Created"], "Encoding")
                .with_description("Begin render encoding"),
        );
        api.add_transition(
            StateTransition::new("makeComputeCommandEncoder", vec!["Created"], "Encoding")
                .with_description("Begin compute encoding"),
        );
        api.add_transition(
            StateTransition::new("makeBlitCommandEncoder", vec!["Created"], "Encoding")
                .with_description("Begin blit encoding"),
        );
        api.add_transition(
            StateTransition::new("endEncoding", vec!["Encoding"], "Created")
                .with_description("Finish encoding, return to Created"),
        );

        // Handler registration (must happen before commit!)
        // Handlers keep the state unchanged
        api.add_transition(
            StateTransition::new("addCompletedHandler", vec!["Created"], "Created")
                .with_description("Register completion callback (from Created)"),
        );
        api.add_transition(
            StateTransition::new("addCompletedHandler", vec!["Encoding"], "Encoding")
                .with_description("Register completion callback (from Encoding)"),
        );
        api.add_transition(
            StateTransition::new("addScheduledHandler", vec!["Created"], "Created")
                .with_description("Register scheduled callback (from Created)"),
        );
        api.add_transition(
            StateTransition::new("addScheduledHandler", vec!["Encoding"], "Encoding")
                .with_description("Register scheduled callback (from Encoding)"),
        );

        // Commit
        api.add_transition(
            StateTransition::new("commit", vec!["Created"], "Committed")
                .with_description("Submit to GPU for execution"),
        );

        // CRITICAL CONSTRAINTS
        //
        // The state machine enforces that addCompletedHandler/addScheduledHandler
        // can only be called from Created or Encoding states (not after commit).
        // This is enforced via transitions - there's no transition from Committed
        // for these methods.
        //
        // Additional constraints below catch programming errors like trying to
        // create encoders after commit.

        api.never_call_from(
            "makeRenderCommandEncoder",
            &["Committed"],
            "Cannot create encoder after commit",
        );

        api.never_call_from(
            "makeComputeCommandEncoder",
            &["Committed"],
            "Cannot create encoder after commit",
        );

        api
    }

    /// MTLCommandQueue state machine
    pub fn mtl_command_queue() -> PlatformApi {
        let mut api = PlatformApi::new("Metal", "MTLCommandQueue")
            .with_description("Metal command queue for submitting command buffers");

        api.add_state(ApiState::new("Active").with_description("Queue is active"));
        api.add_state(
            ApiState::new("Released")
                .with_description("Queue has been released")
                .as_terminal(),
        );

        api.set_initial_state("Active");

        api.add_transition(
            StateTransition::new("makeCommandBuffer", vec!["Active"], "Active")
                .with_description("Create a new command buffer"),
        );
        api.add_transition(
            StateTransition::new(
                "makeCommandBufferWithUnretainedReferences",
                vec!["Active"],
                "Active",
            )
            .with_description("Create command buffer with unretained refs"),
        );

        api.add_constraint(
            ApiConstraint::forbidden_from_state(
                "makeCommandBuffer",
                "Released",
                "Cannot make command buffer on released queue",
            )
            .with_severity(Severity::Critical),
        );

        api
    }

    /// MTLBuffer state machine
    pub fn mtl_buffer() -> PlatformApi {
        let mut api =
            PlatformApi::new("Metal", "MTLBuffer").with_description("Metal buffer for GPU memory");

        api.add_state(ApiState::new("Valid").with_description("Buffer is valid for use"));
        api.add_state(ApiState::new("Mapped").with_description("Buffer is CPU-mapped"));
        api.add_state(
            ApiState::new("Released")
                .with_description("Buffer has been released")
                .as_terminal(),
        );

        api.set_initial_state("Valid");

        api.add_transition(
            StateTransition::new("contents", vec!["Valid", "Mapped"], "Mapped")
                .with_description("Get CPU pointer to buffer contents"),
        );
        api.add_transition(
            StateTransition::new("didModifyRange", vec!["Mapped"], "Valid")
                .with_description("Notify of CPU modifications"),
        );

        api.add_constraint(
            ApiConstraint::must_call_before(
                "contents",
                "didModifyRange",
                "Must access contents before calling didModifyRange",
            )
            .with_severity(Severity::Warning),
        );

        api
    }

    /// MTLTexture state machine
    pub fn mtl_texture() -> PlatformApi {
        let mut api = PlatformApi::new("Metal", "MTLTexture")
            .with_description("Metal texture for GPU image data");

        api.add_state(ApiState::new("Valid").with_description("Texture is valid"));
        api.add_state(
            ApiState::new("Released")
                .with_description("Texture has been released")
                .as_terminal(),
        );

        api.set_initial_state("Valid");

        api.add_transition(
            StateTransition::new("replaceRegion", vec!["Valid"], "Valid")
                .with_description("Replace texture region with CPU data"),
        );
        api.add_transition(
            StateTransition::new("getBytes", vec!["Valid"], "Valid")
                .with_description("Copy texture data to CPU memory"),
        );
        api.add_transition(
            StateTransition::new("makeTextureView", vec!["Valid"], "Valid")
                .with_description("Create a view of this texture"),
        );

        api
    }

    /// MTLRenderCommandEncoder state machine
    pub fn mtl_render_command_encoder() -> PlatformApi {
        let mut api = PlatformApi::new("Metal", "MTLRenderCommandEncoder")
            .with_description("Metal encoder for render commands");

        api.add_state(ApiState::new("Encoding").with_description("Recording render commands"));
        api.add_state(
            ApiState::new("Ended")
                .with_description("Encoding finished")
                .as_terminal(),
        );

        api.set_initial_state("Encoding");

        // State setting
        api.add_transition(
            StateTransition::new("setRenderPipelineState", vec!["Encoding"], "Encoding")
                .with_description("Set the render pipeline"),
        );
        api.add_transition(
            StateTransition::new("setVertexBuffer", vec!["Encoding"], "Encoding")
                .with_description("Bind vertex buffer"),
        );
        api.add_transition(
            StateTransition::new("setFragmentBuffer", vec!["Encoding"], "Encoding")
                .with_description("Bind fragment buffer"),
        );
        api.add_transition(
            StateTransition::new("setVertexTexture", vec!["Encoding"], "Encoding")
                .with_description("Bind vertex texture"),
        );
        api.add_transition(
            StateTransition::new("setFragmentTexture", vec!["Encoding"], "Encoding")
                .with_description("Bind fragment texture"),
        );

        // Draw calls
        api.add_transition(
            StateTransition::new("drawPrimitives", vec!["Encoding"], "Encoding")
                .with_description("Draw primitives"),
        );
        api.add_transition(
            StateTransition::new("drawIndexedPrimitives", vec!["Encoding"], "Encoding")
                .with_description("Draw indexed primitives"),
        );

        // End encoding
        api.add_transition(
            StateTransition::new("endEncoding", vec!["Encoding"], "Ended")
                .with_description("Finish encoding"),
        );

        // Constraints
        api.add_constraint(ApiConstraint::exactly_once(
            "endEncoding",
            "Encoder must be ended exactly once",
        ));

        api.never_call_from(
            "drawPrimitives",
            &["Ended"],
            "Cannot draw after endEncoding",
        );

        api
    }

    /// MTLComputeCommandEncoder state machine
    pub fn mtl_compute_command_encoder() -> PlatformApi {
        let mut api = PlatformApi::new("Metal", "MTLComputeCommandEncoder")
            .with_description("Metal encoder for compute commands");

        api.add_state(ApiState::new("Encoding").with_description("Recording compute commands"));
        api.add_state(
            ApiState::new("Ended")
                .with_description("Encoding finished")
                .as_terminal(),
        );

        api.set_initial_state("Encoding");

        api.add_transition(
            StateTransition::new("setComputePipelineState", vec!["Encoding"], "Encoding")
                .with_description("Set compute pipeline"),
        );
        api.add_transition(
            StateTransition::new("setBuffer", vec!["Encoding"], "Encoding")
                .with_description("Bind buffer"),
        );
        api.add_transition(
            StateTransition::new("setTexture", vec!["Encoding"], "Encoding")
                .with_description("Bind texture"),
        );
        api.add_transition(
            StateTransition::new("dispatchThreadgroups", vec!["Encoding"], "Encoding")
                .with_description("Dispatch compute work"),
        );
        api.add_transition(
            StateTransition::new("dispatchThreads", vec!["Encoding"], "Encoding")
                .with_description("Dispatch compute work by thread count"),
        );
        api.add_transition(
            StateTransition::new("endEncoding", vec!["Encoding"], "Ended")
                .with_description("Finish encoding"),
        );

        api.add_constraint(ApiConstraint::exactly_once(
            "endEncoding",
            "Encoder must be ended exactly once",
        ));

        api
    }
}

/// CUDA API catalog (NVIDIA GPU)
pub struct CudaCatalog;

impl BuiltinCatalog for CudaCatalog {
    fn platform_name() -> &'static str {
        "CUDA"
    }

    fn apis() -> Vec<PlatformApi> {
        vec![Self::cuda_stream(), Self::cuda_event(), Self::cuda_memory()]
    }
}

impl CudaCatalog {
    /// cudaStream_t state machine
    pub fn cuda_stream() -> PlatformApi {
        let mut api = PlatformApi::new("CUDA", "cudaStream_t")
            .with_description("CUDA stream for asynchronous operations");

        api.add_state(ApiState::new("Created").with_description("Stream created"));
        api.add_state(ApiState::new("Active").with_description("Stream has pending work"));
        api.add_state(ApiState::new("Synchronized").with_description("Stream synchronized"));
        api.add_state(
            ApiState::new("Destroyed")
                .with_description("Stream destroyed")
                .as_terminal(),
        );

        api.set_initial_state("Created");

        api.add_transition(
            StateTransition::new(
                "cudaMemcpyAsync",
                vec!["Created", "Active", "Synchronized"],
                "Active",
            )
            .with_description("Enqueue async memory copy"),
        );
        api.add_transition(
            StateTransition::new(
                "cudaLaunchKernel",
                vec!["Created", "Active", "Synchronized"],
                "Active",
            )
            .with_description("Launch kernel on stream"),
        );
        api.add_transition(
            StateTransition::new("cudaStreamSynchronize", vec!["Active"], "Synchronized")
                .with_description("Wait for stream to complete"),
        );
        api.add_transition(
            StateTransition::new(
                "cudaStreamDestroy",
                vec!["Created", "Synchronized"],
                "Destroyed",
            )
            .with_description("Destroy the stream"),
        );

        // Constraint: Must synchronize before destroy if active
        api.add_constraint(
            ApiConstraint::must_call_before(
                "cudaStreamSynchronize",
                "cudaStreamDestroy",
                "Stream should be synchronized before destruction to avoid losing work",
            )
            .with_severity(Severity::Warning),
        );

        api
    }

    /// cudaEvent_t state machine
    pub fn cuda_event() -> PlatformApi {
        let mut api = PlatformApi::new("CUDA", "cudaEvent_t")
            .with_description("CUDA event for synchronization");

        api.add_state(ApiState::new("Created").with_description("Event created"));
        api.add_state(ApiState::new("Recorded").with_description("Event recorded to stream"));
        api.add_state(ApiState::new("Completed").with_description("Event completed"));
        api.add_state(
            ApiState::new("Destroyed")
                .with_description("Event destroyed")
                .as_terminal(),
        );

        api.set_initial_state("Created");

        api.add_transition(
            StateTransition::new("cudaEventRecord", vec!["Created", "Completed"], "Recorded")
                .with_description("Record event on stream"),
        );
        api.add_transition(
            StateTransition::new("cudaEventSynchronize", vec!["Recorded"], "Completed")
                .with_description("Wait for event to complete"),
        );
        api.add_transition(
            StateTransition::new("cudaEventQuery", vec!["Recorded", "Completed"], "Recorded")
                .with_description("Query event status"),
        );
        api.add_transition(
            StateTransition::new(
                "cudaEventDestroy",
                vec!["Created", "Completed"],
                "Destroyed",
            )
            .with_description("Destroy the event"),
        );

        api
    }

    /// CUDA device memory state machine
    pub fn cuda_memory() -> PlatformApi {
        let mut api = PlatformApi::new("CUDA", "DeviceMemory")
            .with_description("CUDA device memory allocation");

        api.add_state(ApiState::new("Allocated").with_description("Memory allocated"));
        api.add_state(
            ApiState::new("Freed")
                .with_description("Memory freed")
                .as_terminal(),
        );

        api.set_initial_state("Allocated");

        api.add_transition(
            StateTransition::new("cudaMemcpy", vec!["Allocated"], "Allocated")
                .with_description("Copy data to/from device"),
        );
        api.add_transition(
            StateTransition::new("cudaMemset", vec!["Allocated"], "Allocated")
                .with_description("Set device memory"),
        );
        api.add_transition(
            StateTransition::new("cudaFree", vec!["Allocated"], "Freed")
                .with_description("Free device memory"),
        );

        // Double-free protection
        api.add_constraint(
            ApiConstraint::at_most_once("cudaFree", "Memory can only be freed once")
                .with_severity(Severity::Critical),
        );

        api.never_call_from("cudaMemcpy", &["Freed"], "Cannot access freed memory");

        api
    }
}

/// POSIX API catalog (threading, I/O)
pub struct PosixCatalog;

impl BuiltinCatalog for PosixCatalog {
    fn platform_name() -> &'static str {
        "POSIX"
    }

    fn apis() -> Vec<PlatformApi> {
        vec![
            Self::pthread_mutex(),
            Self::pthread_cond(),
            Self::file_descriptor(),
        ]
    }
}

impl PosixCatalog {
    /// pthread_mutex_t state machine
    pub fn pthread_mutex() -> PlatformApi {
        let mut api = PlatformApi::new("POSIX", "pthread_mutex_t")
            .with_description("POSIX mutex for thread synchronization");

        api.add_state(ApiState::new("Initialized").with_description("Mutex initialized"));
        api.add_state(ApiState::new("Locked").with_description("Mutex held by a thread"));
        api.add_state(ApiState::new("Unlocked").with_description("Mutex available"));
        api.add_state(
            ApiState::new("Destroyed")
                .with_description("Mutex destroyed")
                .as_terminal(),
        );

        api.set_initial_state("Initialized");

        // First lock from initialized state
        api.add_transition(
            StateTransition::new(
                "pthread_mutex_lock",
                vec!["Initialized", "Unlocked"],
                "Locked",
            )
            .with_description("Acquire the mutex"),
        );
        api.add_transition(
            StateTransition::new(
                "pthread_mutex_trylock",
                vec!["Initialized", "Unlocked"],
                "Locked",
            )
            .with_description("Try to acquire mutex (non-blocking)"),
        );
        api.add_transition(
            StateTransition::new("pthread_mutex_unlock", vec!["Locked"], "Unlocked")
                .with_description("Release the mutex"),
        );
        api.add_transition(
            StateTransition::new(
                "pthread_mutex_destroy",
                vec!["Initialized", "Unlocked"],
                "Destroyed",
            )
            .with_description("Destroy the mutex"),
        );

        // Constraints
        api.add_constraint(ApiConstraint::paired(
            "pthread_mutex_lock",
            "pthread_mutex_unlock",
            "Every lock must have a matching unlock",
        ));

        api.never_call_from(
            "pthread_mutex_destroy",
            &["Locked"],
            "Cannot destroy a locked mutex",
        );

        api
    }

    /// pthread_cond_t state machine
    pub fn pthread_cond() -> PlatformApi {
        let mut api = PlatformApi::new("POSIX", "pthread_cond_t")
            .with_description("POSIX condition variable");

        api.add_state(ApiState::new("Initialized").with_description("Condvar initialized"));
        api.add_state(
            ApiState::new("Destroyed")
                .with_description("Condvar destroyed")
                .as_terminal(),
        );

        api.set_initial_state("Initialized");

        api.add_transition(
            StateTransition::new("pthread_cond_wait", vec!["Initialized"], "Initialized")
                .with_description("Wait on condition"),
        );
        api.add_transition(
            StateTransition::new("pthread_cond_signal", vec!["Initialized"], "Initialized")
                .with_description("Signal one waiter"),
        );
        api.add_transition(
            StateTransition::new("pthread_cond_broadcast", vec!["Initialized"], "Initialized")
                .with_description("Signal all waiters"),
        );
        api.add_transition(
            StateTransition::new("pthread_cond_destroy", vec!["Initialized"], "Destroyed")
                .with_description("Destroy condition variable"),
        );

        api
    }

    /// File descriptor state machine
    pub fn file_descriptor() -> PlatformApi {
        let mut api =
            PlatformApi::new("POSIX", "FileDescriptor").with_description("POSIX file descriptor");

        api.add_state(ApiState::new("Open").with_description("File descriptor is open"));
        api.add_state(
            ApiState::new("Closed")
                .with_description("File descriptor is closed")
                .as_terminal(),
        );

        api.set_initial_state("Open");

        api.add_transition(
            StateTransition::new("read", vec!["Open"], "Open").with_description("Read from fd"),
        );
        api.add_transition(
            StateTransition::new("write", vec!["Open"], "Open").with_description("Write to fd"),
        );
        api.add_transition(
            StateTransition::new("close", vec!["Open"], "Closed").with_description("Close the fd"),
        );

        // Double-close protection
        api.add_constraint(
            ApiConstraint::at_most_once("close", "File descriptor can only be closed once")
                .with_severity(Severity::Critical),
        );

        api.never_call_from("read", &["Closed"], "Cannot read from closed fd");
        api.never_call_from("write", &["Closed"], "Cannot write to closed fd");

        api
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checker::ApiChecker;

    #[test]
    fn test_metal_catalog() {
        let apis = MetalCatalog::apis();
        assert!(!apis.is_empty());
        assert_eq!(MetalCatalog::platform_name(), "Metal");

        // Verify MTLCommandBuffer API
        let cmd_buf = apis.iter().find(|a| a.api_object == "MTLCommandBuffer");
        assert!(cmd_buf.is_some());
        let cmd_buf = cmd_buf.unwrap();

        // Check API has states and transitions
        assert!(
            cmd_buf.states.len() >= 3,
            "Should have Created, Encoding, Committed states"
        );
        assert!(!cmd_buf.transitions.is_empty(), "Should have transitions");
    }

    #[test]
    fn test_metal_command_buffer_constraint() {
        let api = MetalCatalog::mtl_command_buffer();
        let checker = ApiChecker::new(&api);

        // Valid: handler before commit
        let result = checker.check_sequence(&["addCompletedHandler", "commit"]);
        assert!(
            result.passed,
            "Handler before commit should pass: {:?}",
            result.violations
        );

        // Valid: commit without handler (handlers are optional)
        let result = checker.check_sequence(&["commit"]);
        assert!(
            result.passed,
            "Commit without handler is valid: {:?}",
            result.violations
        );

        // Invalid: handler after commit (state machine rejects - no transition from Committed)
        let result = checker.check_sequence(&["commit", "addCompletedHandler"]);
        assert!(!result.passed, "Handler after commit should fail");
    }

    #[test]
    fn test_cuda_catalog() {
        let apis = CudaCatalog::apis();
        assert!(!apis.is_empty());
        assert_eq!(CudaCatalog::platform_name(), "CUDA");
    }

    #[test]
    fn test_posix_mutex() {
        let api = PosixCatalog::pthread_mutex();
        let checker = ApiChecker::new(&api);

        // Valid: lock then unlock
        let result = checker.check_sequence(&["pthread_mutex_lock", "pthread_mutex_unlock"]);
        assert!(result.passed);

        // Invalid: unbalanced lock/unlock
        let result = checker.check_sequence(&[
            "pthread_mutex_lock",
            "pthread_mutex_unlock",
            "pthread_mutex_lock",
        ]);
        assert!(!result.passed);
    }

    #[test]
    fn test_posix_fd_double_close() {
        let api = PosixCatalog::file_descriptor();
        let checker = ApiChecker::new(&api);

        // Valid: read, write, close
        let result = checker.check_sequence(&["read", "write", "close"]);
        assert!(result.passed);

        // Invalid: double close (state machine prevents this)
        let result = checker.check_sequence(&["close", "close"]);
        assert!(!result.passed);
    }

    #[test]
    fn test_all_apis_validate() {
        // All built-in APIs should be valid
        for mut api in MetalCatalog::apis() {
            assert!(
                api.validate().is_ok(),
                "Metal API {} should be valid",
                api.api_object
            );
        }

        for mut api in CudaCatalog::apis() {
            assert!(
                api.validate().is_ok(),
                "CUDA API {} should be valid",
                api.api_object
            );
        }

        for mut api in PosixCatalog::apis() {
            assert!(
                api.validate().is_ok(),
                "POSIX API {} should be valid",
                api.api_object
            );
        }
    }

    #[test]
    fn test_posix_platform_name_is_posix() {
        let name = PosixCatalog::platform_name();
        assert_eq!(name, "POSIX", "Platform name must be exactly 'POSIX'");
        assert!(!name.is_empty(), "Platform name cannot be empty");
        assert_ne!(name, "xyzzy", "Platform name must not be placeholder");
    }

    #[test]
    fn test_posix_apis_returns_non_empty() {
        let apis = PosixCatalog::apis();
        assert!(
            !apis.is_empty(),
            "PosixCatalog::apis() must return non-empty vec"
        );
    }

    #[test]
    fn test_posix_apis_returns_expected_apis() {
        let apis = PosixCatalog::apis();
        let names: Vec<_> = apis.iter().map(|a| a.api_object.as_str()).collect();

        // Should have mutex, condvar, and file descriptor APIs
        assert!(
            names.contains(&"pthread_mutex_t"),
            "Should include pthread_mutex_t"
        );
        assert!(
            names.contains(&"pthread_cond_t"),
            "Should include pthread_cond_t"
        );
        assert!(
            names.contains(&"FileDescriptor"),
            "Should include FileDescriptor"
        );

        // Verify count matches expected
        assert_eq!(
            apis.len(),
            3,
            "Should return exactly 3 POSIX APIs (mutex, condvar, fd)"
        );
    }

    #[test]
    fn test_all_catalogs_have_non_empty_names() {
        assert!(
            !MetalCatalog::platform_name().is_empty(),
            "Metal platform name cannot be empty"
        );
        assert!(
            !CudaCatalog::platform_name().is_empty(),
            "CUDA platform name cannot be empty"
        );
        assert!(
            !PosixCatalog::platform_name().is_empty(),
            "POSIX platform name cannot be empty"
        );
    }

    #[test]
    fn test_all_catalogs_have_non_empty_apis() {
        assert!(
            !MetalCatalog::apis().is_empty(),
            "Metal APIs cannot be empty"
        );
        assert!(!CudaCatalog::apis().is_empty(), "CUDA APIs cannot be empty");
        assert!(
            !PosixCatalog::apis().is_empty(),
            "POSIX APIs cannot be empty"
        );
    }
}
