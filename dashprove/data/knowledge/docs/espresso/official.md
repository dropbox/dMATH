* [ Android Developers ][1]
* [ Develop ][2]
* [ Test your app on Android ][3]

# Espresso Stay organized with collections Save and categorize content based on your preferences.

Use Espresso to write concise, beautiful, and reliable Android UI tests.

The following code snippet shows an example of an Espresso test:

### Kotlin

@Test
fun greeterSaysHello() {
    onView(withId(R.id.name_field)).perform(typeText("Steve"))
    onView(withId(R.id.greet_button)).perform(click())
    onView(withText("Hello Steve!")).check(matches(isDisplayed()))
}

### Java

@Test
public void greeterSaysHello() {
    onView(withId(R.id.name_field)).perform(typeText("Steve"));
    onView(withId(R.id.greet_button)).perform(click());
    onView(withText("Hello Steve!")).check(matches(isDisplayed()));
}

The core API is small, predictable, and easy to learn and yet remains open for customization.
Espresso tests state expectations, interactions, and assertions clearly without the distraction of
boilerplate content, custom infrastructure, or messy implementation details getting in the way.

Espresso tests run optimally fast! It lets you leave your waits, syncs, sleeps, and polls behind
while it manipulates and asserts on the application UI when it is at rest.

## Target audience

Espresso is targeted at developers, who believe that automated testing is an integral part of the
development lifecycle. While it can be used for black-box testing, Espresso’s full power is unlocked
by those who are familiar with the codebase under test.

## Synchronization capabilities

Each time your test invokes [`onView()`][4], Espresso waits to perform the corresponding UI action
or assertion until the following synchronization conditions are met:

* The message queue doesn't have any messages that Espresso needs to immediately process.
* There are no instances of `[AsyncTask][5]` currently executing a task.
* All developer-defined [idling resources][6] are idle.

By performing these checks, Espresso substantially increases the likelihood that only one UI action
or assertion can occur at any given time. This capability gives you more reliable and dependable
test results.

## Packages

* `espresso-core` - Contains core and basic `View` matchers, actions, and assertions. See
  [Basics][7] and [Recipes][8].
* [`espresso-web`][9] - Contains resources for `WebView` support.
* [`espresso-idling-resource`][10] - Espresso's mechanism for synchronization with background jobs.
* `espresso-contrib` - External contributions that contain `DatePicker`, `RecyclerView` and `Drawer`
  actions, accessibility checks, and `CountingIdlingResource`.
* [`espresso-intents`][11] - Extension to validate and stub intents for hermetic testing.
* `espresso-remote` - Location of Espresso's [multi-process][12] functionality.

You can learn more about the latest versions by reading the [release notes][13].

## Additional resources

For more information about using Espresso in Android tests, consult the following resources.

### Samples

* [Espresso Code Samples][14] includes a full selection of Espresso samples.
* [BasicSample][15]: Basic Espresso sample.
* [(more...)][16]

[1]: https://developer.android.com/
[2]: https://developer.android.com/develop
[3]: https://developer.android.com/training/testing
[4]: /reference/androidx/test/espresso/Espresso#onView(org.hamcrest.Matcher%3Candroid.view.View%3E)
[5]: /reference/android/os/AsyncTask
[6]: /training/testing/espresso/idling-resource
[7]: /training/testing/espresso/basics
[8]: /training/testing/espresso/recipes
[9]: /training/testing/espresso/web
[10]: /training/testing/espresso/idling-resource
[11]: /training/testing/espresso/intents
[12]: /training/testing/espresso/multiprocess
[13]: /topic/libraries/testing-support-library/release-notes
[14]: https://github.com/googlesamples/android-testing
[15]: https://github.com/android/testing-samples/tree/main/ui/espresso/BasicSample
[16]: /training/testing/espresso/additional-resources#samples
