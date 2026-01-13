# Swift Testing

Swift Testing is a package with expressive and intuitive APIs that make testing your Swift code a
breeze.

[[CI status badge for main branch using main toolchain]][1] [[CI status badge for main branch using
6.2 toolchain]][2]

## Feature overview

### Clear, expressive API

Swift Testing has a clear and expressive API built using macros, so you can declare complex
behaviors with a small amount of code. The `#expect` API uses Swift expressions and operators, and
captures the evaluated values so you can quickly understand what went wrong when a test fails.

import Testing

@Test func helloWorld() {
  let greeting = "Hello, world!"
  #expect(greeting == "Hello") // Expectation failed: (greeting → "Hello, world!") == "Hello"
}

### Custom test behaviors

You can customize the behavior of tests or test suites using traits specified in your code. Traits
can describe the runtime conditions for a test, like which device a test should run on, or limit a
test to certain operating system versions. Traits can also help you use continuous integration
effectively by specifying execution time limits for your tests.

@Test(.enabled(if: AppFeatures.isCommentingEnabled))
func videoCommenting() async throws {
    let video = try #require(await videoLibrary.video(named: "A Beach"))
    #expect(video.comments.contains("So picturesque!"))
}

### Easy and flexible organization

Swift Testing provides many ways to keep your tests organized. Structure related tests using a
hierarchy of groups and subgroups. Apply tags to flexibly manage, edit, and run tests with common
characteristics across your test suite, like tests that target a specific device or use a specific
module. You can also give tests a descriptive name so you know what they’re doing at a glance.

@Test("Check video metadata",
      .tags(.metadata))
func videoMetadata() {
    let video = Video(fileName: "By the Lake.mov")
    let expectedMetadata = Metadata(duration: .seconds(90))
    #expect(video.metadata == expectedMetadata)
}

### Scalable coverage and execution

Parameterized tests help you run the same test over a sequence of values so you can write less code.
And all tests integrate seamlessly with Swift Concurrency and run in parallel by default.

@Test("Continents mentioned in videos", arguments: [
    "A Beach",
    "By the Lake",
    "Camping in the Woods"
])
func mentionedContinents(videoName: String) async throws {
    let videoLibrary = try await VideoLibrary()
    let video = try #require(await videoLibrary.video(named: videoName))
    #expect(video.mentionedContinents.count <= 3)
}

### Cross-platform support

Swift Testing is included in officially-supported Swift toolchains, including those for Apple
platforms, Linux, and Windows. To use the library, import the `Testing` module:

import Testing

You don't need to declare a package dependency to use Swift Testing. It's developed as open source
and discussed on the [Swift Forums][3] so the very best ideas, from anywhere, can help shape the
future of testing in Swift.

The table below describes the current level of support that Swift Testing has for various platforms:

────────────────┬──────────────┬──────────────────────
**Platform**    │**Support     │**Qualification^{[1][4
                │Status**      │]}**                  
────────────────┼──────────────┼──────────────────────
Apple platforms │Supported     │Automated             
────────────────┼──────────────┼──────────────────────
Linux           │Supported     │Automated             
────────────────┼──────────────┼──────────────────────
Windows         │Supported     │Automated             
────────────────┼──────────────┼──────────────────────
Wasm            │Experimental  │Automated (Build Only)
────────────────┼──────────────┼──────────────────────
Android         │Experimental  │Automated (Build Only)
────────────────┼──────────────┼──────────────────────
FreeBSD, OpenBSD│Experimental  │Manual                
────────────────┴──────────────┴──────────────────────

### Works with XCTest

If you already have tests written using XCTest, you can run them side-by-side with newer tests
written using Swift Testing. This helps you migrate tests incrementally, at your own pace.

## Documentation

Detailed documentation for Swift Testing can be found on the [Swift Package Index][5]. There, you
can delve into comprehensive guides, tutorials, and API references to make the most out of this
package. Swift Testing is included with the Swift 6 toolchain and Xcode 16. You do not need to add
it as a package dependency to your Swift package or Xcode project.

Important

Swift Testing depends on upcoming language and compiler features. If you are building Swift Testing
from source, be aware that the main branch of this repository requires a recent **main-branch
development snapshot** toolchain.

Other documentation resources for this project can be found in the [README][6] of the
`Documentation/` subdirectory.

## Footnotes

1. Most platforms have "Automated" qualification, where continuous integration automatically
   verifies that the project builds and passes all tests. This ensures that any changes meet our
   highest quality standards, so it is our goal for all supported platforms.
   
   Presently, some platforms rely on manual test ("Automated (Build Only)" qualification) or manual
   build and test ("Manual" qualification). [↩][7]

[1]: https://github.com/swiftlang/swift-testing/actions/workflows/main_using_main.yml
[2]: https://github.com/swiftlang/swift-testing/actions/workflows/main_using_release.yml
[3]: https://forums.swift.org/c/development/swift-testing/103
[4]: #user-content-fn-1-e1c6aa171b4876f7c42015f4068a75ca
[5]: https://swiftpackageindex.com/swiftlang/swift-testing/main/documentation/testing
[6]: https://github.com/swiftlang/swift-testing/blob/main/Documentation/README.md
[7]: #user-content-fnref-1-e1c6aa171b4876f7c42015f4068a75ca
