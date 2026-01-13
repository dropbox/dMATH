# Nimble

[[Build Status]][1] [[CocoaPods]][2] [[Carthage Compatible]][3] [[Platforms]][4]

Use Nimble to express the expected outcomes of Swift or Objective-C expressions. Inspired by
[Cedar][5].

// Swift
expect(1 + 1).to(equal(2))
expect(1.2).to(beCloseTo(1.1, within: 0.1))
expect(3) > 2
expect("seahorse").to(contain("sea"))
expect(["Atlantic", "Pacific"]).toNot(contain("Mississippi"))
expect(ocean.isClean).toEventually(beTruthy())

# Documentation

Nimble's documentation is now lives in [Sources/Nimble/Nimble.docc][6] as a Documentation Catalog.
You can easily browse it [quick.github.io/Nimble][7].

# Installing Nimble

> Nimble can be used on its own, or in conjunction with its sister project, [Quick][8]. To install
> both Quick and Nimble, follow [the installation instructions in the Quick Documentation][9].

Nimble can currently be installed in one of four ways: Swift Package Manager, CocoaPods, Carthage or
with git submodules.

## Swift Package Manager

### Xcode

To install Nimble via Xcode's Swift Package Manager Integration: Select your project configuration,
then the project tab, then the Package Dependencies tab. Click on the "plus" button at the bottom of
the list, then follow the wizard to add Quick to your project. Specify
`https://github.com/Quick/Nimble.git` as the url, and be sure to add Nimble as a dependency of your
unit test target, not your app target.

### Package.Swift

To use Nimble with Swift Package Manager to test your applications, add Nimble to your
`Package.Swift` and link it with your test target:

// swift-tools-version:5.7

import PackageDescription

let package = Package(
    name: "MyAwesomeLibrary",
    products: [
        // ...
    ],
    dependencies: [
        // ...
        .package(url:  "https://github.com/Quick/Nimble.git", from: "13.0.0"),
    ],
    targets: [
        // Targets are the basic building blocks of a package. A target can define a module or a tes
t suite.
        // Targets can depend on other targets in this package, and on products in packages this pac
kage depends on.
        .target(
            name: "MyAwesomeLibrary",
            dependencies: ...),
        .testTarget(
            name: "MyAwesomeLibraryTests",
            dependencies: ["MyAwesomeLibrary", "Nimble"]),
    ]
)

Please note that if you install Nimble using Swift Package Manager, then `raiseException` is not
available.

## CocoaPods

To use Nimble in CocoaPods to test your macOS, iOS, tvOS or watchOS applications, add Nimble to your
podfile and add the `use_frameworks!` line to enable Swift support for CocoaPods.

platform :ios, '13.0'

source 'https://github.com/CocoaPods/Specs.git'

# Whatever pods you need for your app go here

target 'YOUR_APP_NAME_HERE_Tests', :exclusive => true do
  use_frameworks!
  pod 'Nimble'
end

Finally run `pod install`.

## Carthage

To use Nimble in Carthage to test your macOS, iOS, tvOS or watchOS applications, add Nimble to your
`Cartfile.private`:

`github "Quick/Nimble" ~> 13.2
`

Then follow the rest of the [Carthage Quick Start][10] and link Nimble with your unit tests.

## Git Submodules

To use Nimble as a submodule to test your macOS, iOS or tvOS applications, follow these 4 easy
steps:

1. Clone the Nimble repository
2. Add Nimble.xcodeproj to the Xcode workspace for your project
3. Link Nimble.framework to your test target
4. Start writing expectations!

For more detailed instructions on each of these steps, read [How to Install Quick][11]. Ignore the
steps involving adding Quick to your project in order to install just Nimble.

# Privacy Statement

Nimble is a library that is only used for testing and should never be included in the binary
submitted to App Store Connect.

Despite not being shipped to Apple, Nimble does not and will never collect any kind of analytics or
tracking.

[1]: https://github.com/Quick/Nimble/actions/workflows/ci-xcode.yml
[2]: https://cocoapods.org/pods/Nimble
[3]: https://github.com/Carthage/Carthage
[4]: https://cocoapods.org/pods/Nimble
[5]: https://github.com/pivotal/cedar
[6]: /Quick/Nimble/blob/main/Sources/Nimble/Nimble.docc
[7]: https://quick.github.io/Nimble/documentation/nimble
[8]: https://github.com/Quick/Quick
[9]: https://github.com/Quick/Quick/blob/main/Documentation/en-us/InstallingQuick.md
[10]: https://github.com/carthage/carthage/?tab=readme-ov-file#quick-start
[11]: https://github.com/Quick/Quick#how-to-install-quick
