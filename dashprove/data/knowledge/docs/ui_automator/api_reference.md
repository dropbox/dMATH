* [ Android Developers ][1]
* [ Develop ][2]
* [ API reference ][3]
Stay organized with collections Save and categorize content based on your preferences.

# androidx.test.uiautomator

[Kotlin][4] |Java

## Interfaces

─────────────────────┬──────────────────────────────────────────────────────────────────────────────
`[Condition][5]`     │Represents a condition to be satisfied.                                       
─────────────────────┼──────────────────────────────────────────────────────────────────────────────
`[IAutomationSupport]│Provides auxiliary support for running test cases                             
[6]`                 │                                                                              
─────────────────────┼──────────────────────────────────────────────────────────────────────────────
`[Searchable][7]`    │The Searchable interface represents an object that can be searched for        
                     │matching UI elements.                                                         
─────────────────────┼──────────────────────────────────────────────────────────────────────────────
`[UiAccessibilityVali│A validator that runs during test actions to check the accessibility of the   
dator][8]`           │UI.                                                                           
─────────────────────┼──────────────────────────────────────────────────────────────────────────────
`[UiWatcher][9]`     │See `[registerWatcher][10]` on how to register a a condition watcher to be    
                     │called by the automation library.                                             
─────────────────────┴──────────────────────────────────────────────────────────────────────────────

## Classes

───────────────────┬────────────────────────────────────────────────────────────────────────────────
`[AccessibilityNode│                                                                                
InfoExt][11]`      │                                                                                
───────────────────┼────────────────────────────────────────────────────────────────────────────────
`[AccessibilityWind│                                                                                
owInfoExt][12]`    │                                                                                
───────────────────┼────────────────────────────────────────────────────────────────────────────────
`[By][13]`         │`[By][14]` is a utility class which enables the creation of `[BySelector][15]`s 
                   │in a concise manner.                                                            
───────────────────┼────────────────────────────────────────────────────────────────────────────────
`[By.Window][16]`  │This nested class is used to create a `[ByWindowSelector][17]` that matches a   
                   │window.                                                                         
───────────────────┼────────────────────────────────────────────────────────────────────────────────
`[BySelector][18]` │A `[BySelector][19]` specifies criteria for matching UI elements during a call  
                   │to `[findObject][20]`.                                                          
───────────────────┼────────────────────────────────────────────────────────────────────────────────
`[ByWindowSelector]│A `[ByWindowSelector][22]` specifies criteria for matching UI windows during a  
[21]`              │call to `[findWindow][23]`.                                                     
───────────────────┼────────────────────────────────────────────────────────────────────────────────
`[Configurator][24]│Allows you to set key parameters for running UiAutomator tests.                 
`                  │                                                                                
───────────────────┼────────────────────────────────────────────────────────────────────────────────
`[EventCondition][2│An `[EventCondition][26]` is a condition which depends on an event or series of 
5]`                │events having occurred.                                                         
───────────────────┼────────────────────────────────────────────────────────────────────────────────
`[ResultsReporter][│Allows to report test results to Android Studio.                                
27]`               │                                                                                
───────────────────┼────────────────────────────────────────────────────────────────────────────────
`[SearchCondition][│A `[SearchCondition][29]` is a condition that is satisfied by searching for UI  
28]`               │elements.                                                                       
───────────────────┼────────────────────────────────────────────────────────────────────────────────
`[StableResult][30]│Represents a node that is considered stable and it's returned by                
`                  │`[androidx.test.uiautomator.waitForStable][31]`.                                
───────────────────┼────────────────────────────────────────────────────────────────────────────────
`[UiAutomatorInstru│**This class is deprecated.**                                                   
mentationTestRunner│                                                                                
][32]`             │as it only handles deprecated `[UiAutomatorTestCase][33]`s.                     
───────────────────┼────────────────────────────────────────────────────────────────────────────────
`[UiAutomatorTestCa│**This class is deprecated.**                                                   
se][34]`           │                                                                                
                   │It is no longer necessary to extend UiAutomatorTestCase.                        
───────────────────┼────────────────────────────────────────────────────────────────────────────────
`[UiAutomatorTestSc│A UiAutomator scope that allows to easily access UiAutomator api and utils      
ope][35]`          │class.                                                                          
───────────────────┼────────────────────────────────────────────────────────────────────────────────
`[UiAutomatorTestSc│                                                                                
opeKt][36]`        │                                                                                
───────────────────┼────────────────────────────────────────────────────────────────────────────────
`[UiCollection][37]│Used to enumerate a container's UI elements for the purpose of counting, or     
`                  │targeting a sub elements by a child's text or description.                      
───────────────────┼────────────────────────────────────────────────────────────────────────────────
`[UiDevice][38]`   │UiDevice provides access to state information about the device.                 
───────────────────┼────────────────────────────────────────────────────────────────────────────────
`[UiDeviceExt][39]`│                                                                                
───────────────────┼────────────────────────────────────────────────────────────────────────────────
`[UiObject][40]`   │A UiObject is a representation of a view.                                       
───────────────────┼────────────────────────────────────────────────────────────────────────────────
`[UiObject2][41]`  │Represents a UI element, and exposes methods for performing gestures (clicks,   
                   │swipes) or searching through its children.                                      
───────────────────┼────────────────────────────────────────────────────────────────────────────────
`[UiObject2Conditio│A `[UiObject2Condition][43]` is a condition which is satisfied when a           
n][42]`            │`[UiObject2][44]` is in a particular state.                                     
───────────────────┼────────────────────────────────────────────────────────────────────────────────
`[UiObject2Ext][45]│                                                                                
`                  │                                                                                
───────────────────┼────────────────────────────────────────────────────────────────────────────────
`[UiScrollable][46]│UiScrollable is a `[UiCollection][47]` and provides support for searching for   
`                  │items in scrollable layout elements.                                            
───────────────────┼────────────────────────────────────────────────────────────────────────────────
`[UiSelector][48]` │Specifies the elements in the layout hierarchy for tests to target, filtered by 
                   │properties such as text value, content-description, class name, and state       
                   │information.                                                                    
───────────────────┼────────────────────────────────────────────────────────────────────────────────
`[UiWindow][49]`   │Represents a UI window on the screen and provides methods to access its         
                   │properties and perform actions.                                                 
───────────────────┼────────────────────────────────────────────────────────────────────────────────
`[Until][50]`      │The `[Until][51]` class provides factory methods for constructing common        
                   │conditions.                                                                     
───────────────────┼────────────────────────────────────────────────────────────────────────────────
`[UtilsKt][52]`    │                                                                                
───────────────────┴────────────────────────────────────────────────────────────────────────────────

## Enums

──────────────┬───────────────────────────────────────────────────────────────────────────
`[Direction][5│An enumeration used to specify the primary direction of certain gestures.  
3]`           │                                                                           
──────────────┴───────────────────────────────────────────────────────────────────────────

## Exceptions

─────────────────┬──────────────────────────────────────────────────────────────────────────────────
`[ElementNotFound│Thrown when an element is not found after invoking                                
Exception][54]`  │`[androidx.test.uiautomator.onElement][55]` or                                    
                 │`[androidx.test.uiautomator.onElements][56]`.                                     
─────────────────┼──────────────────────────────────────────────────────────────────────────────────
`[StaleObjectExce│A `[StaleObjectException][58]` exception is thrown when a `[UiObject2][59]` is    
ption][57]`      │used after the underlying `[android.view.View][60]` has been destroyed.           
─────────────────┼──────────────────────────────────────────────────────────────────────────────────
`[UiObjectNotFoun│Generated in test runs when a `[UiSelector][62]` selector could not be matched to 
dException][61]` │any UI element displayed.                                                         
─────────────────┴──────────────────────────────────────────────────────────────────────────────────

[1]: https://developer.android.com/
[2]: https://developer.android.com/develop
[3]: https://developer.android.com/reference
[4]: /reference/kotlin/androidx/test/uiautomator/package-summary
[5]: /reference/androidx/test/uiautomator/Condition
[6]: /reference/androidx/test/uiautomator/IAutomationSupport
[7]: /reference/androidx/test/uiautomator/Searchable
[8]: /reference/androidx/test/uiautomator/UiAccessibilityValidator
[9]: /reference/androidx/test/uiautomator/UiWatcher
[10]: /reference/androidx/test/uiautomator/UiDevice#registerWatcher(java.lang.String,androidx.test.u
iautomator.UiWatcher)
[11]: /reference/androidx/test/uiautomator/AccessibilityNodeInfoExt
[12]: /reference/androidx/test/uiautomator/AccessibilityWindowInfoExt
[13]: /reference/androidx/test/uiautomator/By
[14]: /reference/androidx/test/uiautomator/By
[15]: /reference/androidx/test/uiautomator/BySelector
[16]: /reference/androidx/test/uiautomator/By.Window
[17]: /reference/androidx/test/uiautomator/ByWindowSelector
[18]: /reference/androidx/test/uiautomator/BySelector
[19]: /reference/androidx/test/uiautomator/BySelector
[20]: /reference/androidx/test/uiautomator/UiDevice#findObject(androidx.test.uiautomator.BySelector)
[21]: /reference/androidx/test/uiautomator/ByWindowSelector
[22]: /reference/androidx/test/uiautomator/ByWindowSelector
[23]: /reference/androidx/test/uiautomator/UiDevice#findWindow(androidx.test.uiautomator.ByWindowSel
ector)
[24]: /reference/androidx/test/uiautomator/Configurator
[25]: /reference/androidx/test/uiautomator/EventCondition
[26]: /reference/androidx/test/uiautomator/EventCondition
[27]: /reference/androidx/test/uiautomator/ResultsReporter
[28]: /reference/androidx/test/uiautomator/SearchCondition
[29]: /reference/androidx/test/uiautomator/SearchCondition
[30]: /reference/androidx/test/uiautomator/StableResult
[31]: /reference/androidx/test/uiautomator/package-summary#(android.view.accessibility.Accessibility
NodeInfo).waitForStable(kotlin.Long,kotlin.Long,kotlin.Long,kotlin.Boolean)
[32]: /reference/androidx/test/uiautomator/UiAutomatorInstrumentationTestRunner
[33]: /reference/androidx/test/uiautomator/UiAutomatorTestCase
[34]: /reference/androidx/test/uiautomator/UiAutomatorTestCase
[35]: /reference/androidx/test/uiautomator/UiAutomatorTestScope
[36]: /reference/androidx/test/uiautomator/UiAutomatorTestScopeKt
[37]: /reference/androidx/test/uiautomator/UiCollection
[38]: /reference/androidx/test/uiautomator/UiDevice
[39]: /reference/androidx/test/uiautomator/UiDeviceExt
[40]: /reference/androidx/test/uiautomator/UiObject
[41]: /reference/androidx/test/uiautomator/UiObject2
[42]: /reference/androidx/test/uiautomator/UiObject2Condition
[43]: /reference/androidx/test/uiautomator/UiObject2Condition
[44]: /reference/androidx/test/uiautomator/UiObject2
[45]: /reference/androidx/test/uiautomator/UiObject2Ext
[46]: /reference/androidx/test/uiautomator/UiScrollable
[47]: /reference/androidx/test/uiautomator/UiCollection
[48]: /reference/androidx/test/uiautomator/UiSelector
[49]: /reference/androidx/test/uiautomator/UiWindow
[50]: /reference/androidx/test/uiautomator/Until
[51]: /reference/androidx/test/uiautomator/Until
[52]: /reference/androidx/test/uiautomator/UtilsKt
[53]: /reference/androidx/test/uiautomator/Direction
[54]: /reference/androidx/test/uiautomator/ElementNotFoundException
[55]: /reference/androidx/test/uiautomator/package-summary#(android.view.accessibility.Accessibility
NodeInfo).onElement(kotlin.Long,kotlin.Long,kotlin.Function1)
[56]: /reference/androidx/test/uiautomator/package-summary#(android.view.accessibility.Accessibility
NodeInfo).onElements(kotlin.Long,kotlin.Long,kotlin.Function1)
[57]: /reference/androidx/test/uiautomator/StaleObjectException
[58]: /reference/androidx/test/uiautomator/StaleObjectException
[59]: /reference/androidx/test/uiautomator/UiObject2
[60]: https://developer.android.com/reference/android/view/View.html
[61]: /reference/androidx/test/uiautomator/UiObjectNotFoundException
[62]: /reference/androidx/test/uiautomator/UiSelector
