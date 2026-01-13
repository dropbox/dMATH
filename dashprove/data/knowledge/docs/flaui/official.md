[[Alt text]][1]

### Badges

─────────────────┬─────────────────────────────────────────────────────
What             │Badge                                                
─────────────────┼─────────────────────────────────────────────────────
*Chat*           │[[Join the chat at https://gitter.im/FlaUI/Lobby]][2]
─────────────────┼─────────────────────────────────────────────────────
*Build*          │[[Build status]][3]                                  
─────────────────┼─────────────────────────────────────────────────────
*Tests*          │[[AppVeyor tests]][4]                                
─────────────────┼─────────────────────────────────────────────────────
*Libraries       │[[Nuget]][5] [[Nuget]][6] [[Nuget]][7]               
(NuGet)*         │                                                     
─────────────────┼─────────────────────────────────────────────────────
*CI Artefacts*   │[FlaUI CI][8]                                        
─────────────────┴─────────────────────────────────────────────────────

### Introduction

FlaUI is a .NET library which helps with automated UI testing of Windows applications (Win32,
WinForms, WPF, Store Apps, ...).
It is based on native UI Automation libraries from Microsoft and therefore kind of a wrapper around
them.
FlaUI wraps almost everything from the UI Automation libraries but also provides the native objects
in case someone has a special need which is not covered (yet) by FlaUI.
Some ideas are copied from the UIAComWrapper project or TestStack.White but rewritten from scratch
to have a clean codebase.

### Sponsoring

If you appreciate my work, feel free to support me by [sponsoring on github][9] or with a one-time
payment [over at PayPal][10].

### Why another library?

There are quite some automation solutions out there. Commercial ones like TestComplete, Ranorex,
CodedUI just to name a few. And also free ones which are mainly TestStack.White.
All of them are based on what Microsoft provides. These are the UI Automation libraries. There are
three versions of it:

* MSAA
  
  * MSAA is very obsolete and we'll skip this here (some like CodedUI still use it)
* UIA2: Managed Library for native UI Automation API
  
  * UIA2 is managed only, which would be good for C# but it does not support newer features (like
    touch) and it also does not work well with WPF or even worse with Windows Store Apps.
* UIA3: Com Library for native UI Automation API
  
  * UIA3 is the newest of them all and works great for WPF / Windows Store Apps but unfortunately,
    it can have some bugs with WinForms applications (see [FAQ][11]) which are not existent in UIA2.

So, the commercial solutions are mostly based on multiple of those and/or implement a lot of
workaround code to fix those issues. TestStack.White has two versions, one for UIA2 and one for UIA3
but because of the old codebase, it's fairly hard to bring UIA3 to work. For this, it also uses an
additional library, the UIAComWrapper which uses the same naming as the managed UIA2 and wraps the
UIA3 com interop with them (one more source for errors). FlaUI now tries to provide an interface for
UIA2 and UIA3 where the developer can choose, which version he wants to use. It should also provide
a very clean and modern codebase so that collaboration and further development is as easy as
possible.

### Usage

Installation

To use FlaUI, you need to reference the appropriate assemblies. So you should decide, if you want to
use UIA2 or UIA3 and install the appropriate library from NuGet. You can of course always download
the source and compile it yourself.

Usage in Code

The entry point is usually an application or the desktop so you get an automation element (like the
main window of the application). On this, you can then search sub-elements and interact with them.
There is a helper class to launch, attach or close applications. Since the application is not
related to any UIA library, you need to create the automation you want and use it to get your first
element, which then is your entry point.

using FlaUI.UIA3;

var app = FlaUI.Core.Application.Launch("notepad.exe");
using (var automation = new UIA3Automation())
{
        var window = app.GetMainWindow(automation);
        Console.WriteLine(window.Title);
        ...
}
using FlaUI.Core.AutomationElements;
using FlaUI.UIA3;

// Note: Works only pre-Windows 8 with the legacy calculator
var app = FlaUI.Core.Application.Launch("calc.exe");
using (var automation = new UIA3Automation())
{
        var window = app.GetMainWindow(automation);
        var button1 = window.FindFirstDescendant(cf => cf.ByText("1"))?.AsButton();
        button1?.Invoke();
        ...
}

### Further Resources

#### YouTube Tutorials

Have a look at [H Y R Tutorials][12]. This channel provides some videos to get you started with
FlaUI.

#### FlaUI UITests

FlaUI itself contains quite some UI tests itself. Browse to the code of them [here][13] and look how
they work.

#### Chat

Head over to the [chat][14] to ask your specific questions.

### Contribution

Feel free to fork FlaUI and send pull requests of your modifications.
You can also create issues if you find problems or have ideas on how to further improve FlaUI.

### Donors and Sponsors

* Thank you Gehtsoft USA LLC for the generous donation

### Acknowledgements

#### JetBrains

Thanks to [JetBrains][15] for providing a free license of [ReSharper][16].

#### AppVeyor

Thanks to [AppVeyor][17] for providing a free CI [build system for FlaUI][18].

#### TestStack.White

Thanks to the creators and maintainers (especially to [@JakeGinnivan][19] and [@petmongrels][20])
for their work and inspiration for this project.

#### Microsoft

Thanks to Microsoft for providing great tools which made developing this project possible.

[1]: /FlaUI/FlaUI/blob/master/FlaUI.png?raw=true
[2]: https://gitter.im/FlaUI/Lobby?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_conte
nt=badge
[3]: https://ci.appveyor.com/project/RomanBaeriswyl/flaui
[4]: https://camo.githubusercontent.com/5f08c063686acb36b29e50b2a787bb62aa5ec90dd6058cab21d0aba36d65
c418/68747470733a2f2f696d672e736869656c64732e696f2f6170707665796f722f74657374732f526f6d616e426165726
97377796c2f666c617569
[5]: https://www.nuget.org/packages/FlaUI.Core
[6]: https://www.nuget.org/packages/FlaUI.UIA3
[7]: https://www.nuget.org/packages/FlaUI.UIA2
[8]: https://ci.appveyor.com/project/RomanBaeriswyl/flaui/build/artifacts
[9]: https://github.com/sponsors/Roemer
[10]: https://paypal.me/rbaeriswyl
[11]: https://github.com/FlaUI/FlaUI/wiki/FAQ
[12]: https://www.youtube.com/playlist?list=PLacgMXFs7kl_fuSSe6lp6YRaeAp6vqra9
[13]: https://github.com/FlaUI/FlaUI/tree/master/src/FlaUI.Core.UITests
[14]: https://gitter.im/FlaUI/Lobby
[15]: https://www.jetbrains.com
[16]: https://www.jetbrains.com/resharper/
[17]: https://www.appveyor.com
[18]: https://ci.appveyor.com/project/RomanBaeriswyl/flaui
[19]: https://github.com/JakeGinnivan
[20]: https://github.com/petmongrels
