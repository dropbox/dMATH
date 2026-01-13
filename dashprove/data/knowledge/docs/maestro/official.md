[Edit][1]

1. [Getting Started][2]

# Maestro Documentation

Maestro is an open-source framework for mobile and web UI testing. Docs include setup guides,
examples, and steps to run automated tests.

🚀 **Running in the Cloud**

Ready to wire into CI or scale up your testing? Start running your flows on Maestro's
enterprise-grade cloud infrastructure: [Run Maestro tests in the Cloud][3] and follow the
[Quickstart Guide][4] to quickly experience the power of Cloud.

## Why Maestro?

Maestro is built on learnings from its predecessors (Appium, Espresso, UIAutomator, XCTest,
Selenium, Playwright) and allows you to easily define and test your Flows.

**What are Flows?** Think of Flows as parts of the user journey in your app. Login, Checkout and Add
to Cart are three examples of possible Flows that can be defined and tested using Maestro.

* Built-in tolerance to flakiness. UI elements will not always be where you expect them, screen tap
  will not always go through, etc. Maestro embraces the instability of mobile applications and
  devices and tries to counter it.
* Built-in tolerance to delays. No need to pepper your tests with `sleep()` calls. Maestro knows
  that it might take time to load the content (i.e. over the network) and automatically waits for it
  (but no longer than required).
* Blazingly fast iteration. Tests are interpreted, no need to compile anything. Maestro is able to
  continuously monitor your test files and rerun them as they change.
* Declarative yet powerful syntax. Define your tests in a `yaml` file.
* Simple setup. Maestro is a single binary that works anywhere.

## Examples

#### Twitter (Mobile)

#### Simple Examples

Android
iOS
Web

## Platform Support

Platform
Supported

[Android - Views][5]

✅

[Android - Jetpack Compose][6]

✅

[iOS - UIKit][7]

✅

[iOS - SwiftUI][8]

✅

[React Native][9]

✅

[Flutter][10]

✅

[Web Views][11]

✅

[Web (Desktop Browser)][12]

✅

.NET MAUI iOS

✅

.NET MAUI Android

✅

## Resources

* Blog Post: [**Introducing: Maestro — Painless Mobile UI Automation**][13]
* GitHub Repository: [**https://github.com/mobile-dev-inc/maestro**][14]
* Public Slack Channel: [**Join the workspace**][15]

## Get Started

Get started by installing the Maestro CLI:

[NextInstalling Maestro][16]

Last updated 2 months ago

Was this helpful?

[1]: https://github.com/mobile-dev-inc/maestro-docs/blob/main/README.md
[2]: /getting-started
[3]: https://maestro.dev/cloud
[4]: /cloud/cloud-quickstart
[5]: /platform-support/android-views
[6]: /platform-support/android-jetpack-compose
[7]: /platform-support/ios-uikit
[8]: /platform-support/ios-swiftui
[9]: /platform-support/react-native
[10]: /platform-support/flutter
[11]: /platform-support/web-views
[12]: /platform-support/web-desktop-browser
[13]: https://maestro.dev/blog/introducing-maestro-painless-mobile-ui-automation
[14]: https://github.com/mobile-dev-inc/maestro
[15]: https://slack.maestro.dev/
[16]: /getting-started/installing-maestro
