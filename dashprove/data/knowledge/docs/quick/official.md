[[Build Status]][1] [[CocoaPods]][2] [[Platforms]][3]

Quick is a behavior-driven development framework for Swift and Objective-C. Inspired by [RSpec][4],
[Specta][5], and [Ginkgo][6].

// Swift

import Quick
import Nimble

class TableOfContentsSpec: QuickSpec {
  override class func spec() {
    describe("the 'Documentation' directory") {
      it("has everything you need to get started") {
        let sections = Directory("Documentation").sections
        expect(sections).to(contain("Organized Tests with Quick Examples and Example Groups"))
        expect(sections).to(contain("Installing Quick"))
      }

      context("if it doesn't have what you're looking for") {
        it("needs to be updated") {
          let you = You(awesome: true)
          expect{you.submittedAnIssue}.toEventually(beTruthy())
        }
      }
    }
  }
}

#### Nimble

Quick comes together with [Nimble][7] — a matcher framework for your tests. You can learn why
`XCTAssert()` statements make your expectations unclear and how to fix that using Nimble assertions
[here][8].

## Swift Version

Certain versions of Quick and Nimble only support certain versions of Swift. Depending on which
version of Swift your project uses, you should use specific versions of Quick and Nimble. Use the
table below to determine which versions of Quick and Nimble are compatible with your project.

─────────────────────┬───────────────┬───────────────
Swift version        │Quick version  │Nimble version 
─────────────────────┼───────────────┼───────────────
Swift 5.2            │v3.0.0 or later│v9.0.0 or later
─────────────────────┼───────────────┼───────────────
Swift 4.2 / Swift 5  │v1.3.2 or later│v7.3.2 or later
─────────────────────┼───────────────┼───────────────
Swift 3 / Swift 4    │v1.0.0 or later│v5.0.0 or later
─────────────────────┼───────────────┼───────────────
Swift 2.2 / Swift 2.3│v0.9.3         │v4.1.0         
─────────────────────┴───────────────┴───────────────

## Documentation

All documentation can be found in the [Documentation folder][9], including [detailed installation
instructions][10] for CocoaPods, Git submodules, Swift Package Manager, and more. For example, you
can install Quick and [Nimble][11] using CocoaPods by adding the following to your `Podfile`:

# Podfile

use_frameworks!

target "MyApp" do
  # Normal libraries

  target 'MyApp_Tests' do
    inherit! :search_paths

    pod 'Quick'
    pod 'Nimble'
  end
end

You can also install Quick and Nimble using Swift Package Manager by adding the following to the
dependencies section your `Package.swift`:

dependencies: [
    .package(url: "https://github.com/Quick/Quick.git", from: "7.0.0"),
    .package(url: "https://github.com/Quick/Nimble.git", from: "12.0.0"),
],

## Projects using Quick

Over ten-thousand apps use either Quick and Nimble however, as they are not included in the app
binary, neither appear in “Top Used Libraries” blog posts. Therefore, it would be greatly
appreciated to remind contributors that their efforts are valued by compiling a list of
organizations and projects that use them.

Does your organization or project use Quick and Nimble? If yes, [please add your project to the
list][12].

## Who uses Quick

Similar to projects using Quick, it would be nice to hear why people use Quick and Nimble. Are there
features you love? Are there features that are just okay? Are there some features we have that no
one uses?

Have something positive to say about Quick (or Nimble)? If yes, [provide a testimonial here][13].

## Privacy Statement

Quick is a library that is only used for testing and should never be included in the binary
submitted to App Store Connect. Your app will be rejected if you do include Quick in the submitted
binary because Quick uses private APIs to better integrate with Xcode.

Despite not being shipped to Apple, Quick does not and will never collect any kind of analytics or
tracking.

## License

Apache 2.0 license. See the [`LICENSE`][14] file for details.

[1]: https://github.com/Quick/Quick/actions/workflows/ci-xcode.yml
[2]: https://cocoapods.org/pods/Quick
[3]: https://cocoapods.org/pods/Quick
[4]: https://github.com/rspec/rspec
[5]: https://github.com/specta/specta
[6]: https://github.com/onsi/ginkgo
[7]: https://github.com/Quick/Nimble
[8]: /Quick/Quick/blob/main/Documentation/en-us/NimbleAssertions.md
[9]: /Quick/Quick/blob/main/Documentation
[10]: /Quick/Quick/blob/main/Documentation/en-us/InstallingQuick.md
[11]: https://github.com/Quick/Nimble
[12]: https://github.com/Quick/Quick/wiki/Projects-using-Quick
[13]: https://github.com/Quick/Quick/wiki/Who-uses-Quick
[14]: /Quick/Quick/blob/main/LICENSE
