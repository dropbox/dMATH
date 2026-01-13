# Android testing samples

A collection of samples demonstrating different frameworks and techniques for automated testing.

### Espresso Samples

**[BasicSample][1]** - Basic Espresso sample

**[CustomMatcherSample][2]** - Shows how to extend Espresso to match the *hint* property of an
EditText

**[DataAdapterSample][3]** - Showcases the `onData()` entry point for Espresso, for lists and
AdapterViews

**[FragmentScenarioSample][4]** - Basic usage of `FragmentScenario` with Espresso.

**[IdlingResourceSample][5]** - Synchronization with background jobs

**[IntentsBasicSample][6]** - Basic usage of `intended()` and `intending()`

**[IntentsAdvancedSample][7]** - Simulates a user fetching a bitmap using the camera

**[MultiWindowSample][8]** - Shows how to point Espresso to different windows

**[RecyclerViewSample][9]** - RecyclerView actions for Espresso

**[ScreenshotSample][10]** - Screenshot capturing and saving using Espresso and androidx.test.core
APIs

**[WebBasicSample][11]** - Use Espresso-web to interact with WebViews

**[BasicSampleBundled][12]** - Basic sample for Eclipse and other IDEs

**[MultiProcessSample][13]** - Showcases how to use multiprocess Espresso.

### UiAutomator Sample

**[BasicSample][14]** - Basic UI Automator sample

### AndroidJUnitRunner Sample

**[AndroidJunitRunnerSample][15]** - Showcases test annotations, parameterized tests and testsuite
creation

### JUnit4 Rules Sample

**All previous samples use ActivityTestRule or IntentsTestRule but there's one specific to
ServiceTestRule:

**[BasicSample][16]** - Simple usage of `ActivityTestRule`

**[IntentsBasicSample][17]** - Simple usage of `IntentsTestRule`

**[ServiceTestRuleSample][18]** - Simple usage of `ServiceTestRule`

## Prerequisites

* Android SDK v28
* Android Build Tools v28.03

## Getting Started

These samples use the Gradle build system. To build a project, enter the project directory and use
the `./gradlew assemble` command or use "Import Project" in Android Studio.

* Use `./gradlew connectedAndroidTest` to run the tests on a connected emulator or device.
* Use `./gradlew test` to run the unit test on your local host.

There is a top-level `build.gradle` file if you want to build and test all samples from the root
directory. This is mostly helpful to build on a CI (Continuous Integration) server.

## AndroidX Test Library

Many of these samples use the AndroidX Test Library. Visit the [Testing site on
developer.android.com][19] for more information.

## Experimental Bazel Support

[[Build status]][20]

Some of these samples can be tested with [Bazel][21] on Linux. These samples contain a `BUILD.bazel`
file, which is similar to a `build.gradle` file. The external dependencies are defined in the top
level `WORKSPACE` file.

This is **experimental** feature. To run the tests, please install the latest version of Bazel
(0.12.0 or later) by following the [instructions on the Bazel website][22].

### Bazel commands

`# Clone the repository if you haven't.
$ git clone https://github.com/google/android-testing
$ cd android-testing

# Edit the path to your local SDK at the top of the WORKSPACE file
$ $EDITOR WORKSPACE

# Test everything in a headless mode (no graphical display)
$ bazel test //... --config=headless

# Test a single test, e.g. ui/espresso/BasicSample/BUILD.bazel
$ bazel test //ui/uiautomator/BasicSample:BasicSampleInstrumentationTest_21_x86 --config=headless

# Query for all android_instrumentation_test targets
$ bazel query 'kind(android_instrumentation_test, //...)'
//ui/uiautomator/BasicSample:BasicSampleInstrumentationTest_23_x86
//ui/uiautomator/BasicSample:BasicSampleInstrumentationTest_22_x86
//ui/uiautomator/BasicSample:BasicSampleInstrumentationTest_21_x86
//ui/uiautomator/BasicSample:BasicSampleInstrumentationTest_19_x86
//ui/espresso/RecyclerViewSample:RecyclerViewSampleInstrumentationTest_23_x86
//ui/espresso/RecyclerViewSample:RecyclerViewSampleInstrumentationTest_22_x86
//ui/espresso/RecyclerViewSample:RecyclerViewSampleInstrumentationTest_21_x86
//ui/espresso/RecyclerViewSample:RecyclerViewSampleInstrumentationTest_19_x86
//ui/espresso/MultiWindowSample:MultiWindowSampleInstrumentationTest_23_x86
//ui/espresso/MultiWindowSample:MultiWindowSampleInstrumentationTest_22_x86
...

# Test everything with GUI enabled
$ bazel test //... --config=gui

# Test with a local device or emulator. Ensure that `adb devices` lists the device.
$ bazel test //... --config=local_device

# If multiple devices are connected, add --device_serial_number=$identifier where $identifier is the
 name of the device in `adb devices`
$ bazel test //... --config=local_device --test_arg=--device_serial_number=$identifier
`

For more information, check out the documentation for [Android Instrumentation Tests in Bazel][23].
You may also want to check out [Building an Android App with Bazel][24], and the list of [Android
Rules][25] in the Bazel Build Encyclopedia.

Known issues:

* Building of APKs is supported on Linux, Mac and Windows, but testing is only supported on Linux.
* `android_instrumentation_test.target_device` attribute still needs to be specified even if
  `--config=local_device` is used.
* If using a local device or emulator, the APKs are not uninstalled automatically after the test.
  Use this command to remove the packages:
  
  * `adb shell pm list packages com.example.android.testing | cut -d ':' -f 2 | tr -d '\r' | xargs
    -L1 -t adb uninstall`

Please file Bazel related issues against the [Bazel][26] repository instead of this repository.

## Support

* Google+ Community: [https://plus.google.com/communities/105153134372062985968][27]
* Stack Overflow: [http://stackoverflow.com/questions/tagged/android-testing][28]

If you've found an error in this sample, please file an issue:
[https://github.com/googlesamples/android-testing][29]

Patches are encouraged, and may be submitted by forking this project and submitting a pull request
through GitHub. Please see CONTRIBUTING.md for more details.

## License

Copyright 2015 The Android Open Source Project, Inc.

Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements.
See the NOTICE file distributed with this work for additional information regarding copyright
ownership. The ASF licenses this file to you under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License. You may obtain a copy of the
License at

[http://www.apache.org/licenses/LICENSE-2.0][30]

Unless required by applicable law or agreed to in writing, software distributed under the License is
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
implied. See the License for the specific language governing permissions and limitations under the
License.

[1]: https://github.com/googlesamples/android-testing/blob/main/ui/espresso/BasicSample
[2]: https://github.com/googlesamples/android-testing/blob/main/ui/espresso/CustomMatcherSample
[3]: https://github.com/googlesamples/android-testing/blob/main/ui/espresso/DataAdapterSample
[4]: https://github.com/googlesamples/android-testing/blob/main/ui/espresso/FragmentScenarioSample
[5]: https://github.com/googlesamples/android-testing/blob/main/ui/espresso/IdlingResourceSample
[6]: https://github.com/googlesamples/android-testing/blob/main/ui/espresso/IntentsBasicSample
[7]: https://github.com/googlesamples/android-testing/blob/main/ui/espresso/IntentsAdvancedSample
[8]: https://github.com/googlesamples/android-testing/blob/main/ui/espresso/MultiWindowSample
[9]: https://github.com/googlesamples/android-testing/blob/main/ui/espresso/RecyclerViewSample
[10]: https://github.com/googlesamples/android-testing/blob/main/ui/espresso/ScreenshotSample
[11]: https://github.com/googlesamples/android-testing/blob/main/ui/espresso/WebBasicSample
[12]: https://github.com/googlesamples/android-testing/blob/main/ui/espresso/BasicSampleBundled
[13]: https://github.com/googlesamples/android-testing/blob/main/ui/espresso/MultiProcessSample
[14]: https://github.com/googlesamples/android-testing/tree/main/ui/uiautomator/BasicSample
[15]: https://github.com/googlesamples/android-testing/tree/main/runner/AndroidJunitRunnerSample
[16]: https://github.com/googlesamples/android-testing/blob/main/ui/espresso/BasicSample
[17]: https://github.com/googlesamples/android-testing/blob/main/ui/espresso/IntentsBasicSample
[18]: https://github.com/googlesamples/android-testing/tree/main/integration/ServiceTestRuleSample
[19]: https://developer.android.com/training/testing
[20]: https://buildkite.com/bazel/android-testing
[21]: https://bazel.build
[22]: https://docs.bazel.build/versions/master/install-ubuntu.html
[23]: https://docs.bazel.build/versions/master/android-instrumentation-test.html
[24]: https://docs.bazel.build/versions/master/tutorial/android-app.html
[25]: https://docs.bazel.build/versions/master/be/android.html
[26]: https://github.com/bazelbuild/bazel
[27]: https://plus.google.com/communities/105153134372062985968
[28]: http://stackoverflow.com/questions/tagged/android-testing
[29]: https://github.com/googlesamples/android-testing
[30]: http://www.apache.org/licenses/LICENSE-2.0
