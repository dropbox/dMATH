* [ Android Developers ][1]
* [ Develop ][2]
* [ Android Studio ][3]
* [ IDE guides ][4]

# UI/Application Exerciser Monkey Stay organized with collections Save and categorize content based
# on your preferences.

The Monkey is a program that runs on your [emulator][5] or device and generates pseudo-random
streams of user events such as clicks, touches, or gestures, as well as a number of system-level
events. You can use the Monkey to stress-test applications that you are developing, in a random yet
repeatable manner.

## Overview

The Monkey is a command-line tool that you can run on any emulator instance or on a device. It sends
a pseudo-random stream of user events into the system, which acts as a stress test on the
application software you are developing.

The Monkey includes a number of options, but they break down into four primary categories:

* Basic configuration options, such as setting the number of events to attempt.
* Operational constraints, such as restricting the test to a single package.
* Event types and frequencies.
* Debugging options.

When the Monkey runs, it generates events and sends them to the system. It also *watches* the system
under test and looks for three conditions, which it treats specially:

* If you have constrained the Monkey to run in one or more specific packages, it watches for
  attempts to navigate to any other packages, and blocks them.
* If your application crashes or receives any sort of unhandled exception, the Monkey will stop and
  report the error.
* If your application generates an *application not responding* error, the Monkey will stop and
  report the error.

Depending on the verbosity level you have selected, you will also see reports on the progress of the
Monkey and the events being generated.

## Basic use of the Monkey

You can launch the Monkey using a command line on your development machine or from a script. Because
the Monkey runs in the emulator/device environment, you must launch it from a shell in that
environment. You can do this by prefacing `adb shell` to each command, or by entering the shell and
entering Monkey commands directly.

The basic syntax is:

$ adb shell monkey [options] <event-count>

With no options specified, the Monkey will launch in a quiet (non-verbose) mode, and will send
events to any (and all) packages installed on your target. Here is a more typical command line,
which will launch your application and send 500 pseudo-random events to it:

$ adb shell monkey -p your.package.name -v 500

## Command options reference

The table below lists all options you can include on the Monkey command line.

───┬────────────────────────────────────────────┬───────────────────────────────────────────────────
Cat│Option                                      │Description                                        
ego│                                            │                                                   
ry │                                            │                                                   
───┼────────────────────────────────────────────┼───────────────────────────────────────────────────
Gen│`--help`                                    │Prints a simple usage guide.                       
era│                                            │                                                   
l  │                                            │                                                   
───┼────────────────────────────────────────────┴───────────────────────────────────────────────────
`-v│Each -v on the command line will increment  
`  │the verbosity level. Level 0 (the default)  
   │provides little information beyond startup  
   │notification, test completion, and final    
   │results. Level 1 provides more details about
   │the test as it runs, such as individual     
   │events being sent to your activities. Level 
   │2 provides more detailed setup information  
   │such as activities selected or not selected 
   │for testing.                                
───┼────────────────────────────────────────────┬
Eve│`-s <seed>`                                 │Seed value for pseudo-random number generator. If  
nts│                                            │you re-run the Monkey with the same seed value, it 
   │                                            │will generate the same sequence of events.         
───┼────────────────────────────────────────────┴───────────────────────────────────────────────────
`--│Inserts a fixed delay between events. You   
thr│can use this option to slow down the Monkey.
ott│If not specified, there is no delay and the 
le │events are generated as rapidly as possible.
<mi│                                            
lli│                                            
sec│                                            
ond│                                            
s>`│                                            
───┼────────────────────────────────────────────
`--│Adjust percentage of touch events. (Touch   
pct│events are a down-up event in a single place
-to│on the screen.)                             
uch│                                            
<pe│                                            
rce│                                            
nt>│                                            
`  │                                            
───┼────────────────────────────────────────────
`--│Adjust percentage of motion events. (Motion 
pct│events consist of a down event somewhere on 
-mo│the screen, a series of pseudo-random       
tio│movements, and an up event.)                
n  │                                            
<pe│                                            
rce│                                            
nt>│                                            
`  │                                            
───┼────────────────────────────────────────────
`--│Adjust percentage of trackball events.      
pct│(Trackball events consist of one or more    
-tr│random movements, sometimes followed by a   
ack│click.)                                     
bal│                                            
l  │                                            
<pe│                                            
rce│                                            
nt>│                                            
`  │                                            
───┼────────────────────────────────────────────
`--│Adjust percentage of "basic" navigation     
pct│events. (Navigation events consist of       
-na│up/down/left/right, as input from a         
v  │directional input device.)                  
<pe│                                            
rce│                                            
nt>│                                            
`  │                                            
───┼────────────────────────────────────────────
`--│Adjust percentage of "major" navigation     
pct│events. (These are navigation events that   
-ma│will typically cause actions within your UI,
jor│such as the center button in a 5-way pad,   
nav│the back key, or the menu key.)             
<pe│                                            
rce│                                            
nt>│                                            
`  │                                            
───┼────────────────────────────────────────────
`--│Adjust percentage of "system" key events.   
pct│(These are keys that are generally reserved 
-sy│for use by the system, such as Home, Back,  
ske│Start Call, End Call, or Volume controls.)  
ys │                                            
<pe│                                            
rce│                                            
nt>│                                            
`  │                                            
───┼────────────────────────────────────────────
`--│Adjust percentage of activity launches. At  
pct│random intervals, the Monkey will issue a   
-ap│startActivity() call, as a way of maximizing
psw│coverage of all activities within your      
itc│package.                                    
h  │                                            
<pe│                                            
rce│                                            
nt>│                                            
`  │                                            
───┼────────────────────────────────────────────
`--│Adjust percentage of other types of events. 
pct│This is a catch-all for all other types of  
-an│events such as keypresses, other less-used  
yev│buttons on the device, and so forth.        
ent│                                            
<pe│                                            
rce│                                            
nt>│                                            
`  │                                            
───┼────────────────────────────────────────────┬
Con│`-p <allowed-package-name>`                 │If you specify one or more packages this way, the  
str│                                            │Monkey will *only* allow the system to visit       
ain│                                            │activities within those packages. If your          
ts │                                            │application requires access to activities in other 
   │                                            │packages (e.g. to select a contact) you'll need to 
   │                                            │specify those packages as well. If you don't       
   │                                            │specify any packages, the Monkey will allow the    
   │                                            │system to launch activities in all packages. To    
   │                                            │specify multiple packages, use the -p option       
   │                                            │multiple times — one -p option per package.        
───┼────────────────────────────────────────────┴───────────────────────────────────────────────────
`-c│If you specify one or more categories this  
<ma│way, the Monkey will *only* allow the system
in-│to visit activities that are listed with one
cat│of the specified categories. If you don't   
ego│specify any categories, the Monkey will     
ry>│select activities listed with the category  
`  │Intent.CATEGORY_LAUNCHER or                 
   │Intent.CATEGORY_MONKEY. To specify multiple 
   │categories, use the -c option multiple times
   │— one -c option per category.               
───┼────────────────────────────────────────────┬
Deb│`--dbg-no-events`                           │When specified, the Monkey will perform the initial
ugg│                                            │launch into a test activity, but will not generate 
ing│                                            │any further events. For best results, combine with 
   │                                            │-v, one or more package constraints, and a non-zero
   │                                            │throttle to keep the Monkey running for 30 seconds 
   │                                            │or more. This provides an environment in which you 
   │                                            │can monitor package transitions invoked by your    
   │                                            │application.                                       
───┼────────────────────────────────────────────┴───────────────────────────────────────────────────
`--│If set, this option will generate profiling 
hpr│reports immediately before and after the    
of`│Monkey event sequence. This will generate   
   │large (~5Mb) files in data/misc, so use with
   │care. For information on analyzing profiling
   │reports, see [Profile your app              
   │performance][6].                            
───┼────────────────────────────────────────────
`--│Normally, the Monkey will stop when the     
ign│application crashes or experiences any type 
ore│of unhandled exception. If you specify this 
-cr│option, the Monkey will continue to send    
ash│events to the system, until the count is    
es`│completed.                                  
───┼────────────────────────────────────────────
`--│Normally, the Monkey will stop when the     
ign│application experiences any type of timeout 
ore│error such as a "Application Not Responding"
-ti│dialog. If you specify this option, the     
meo│Monkey will continue to send events to the  
uts│system, until the count is completed.       
`  │                                            
───┼────────────────────────────────────────────
`--│Normally, the Monkey will stop when the     
ign│application experiences any type of         
ore│permissions error, for example if it        
-se│attempts to launch an activity that requires
cur│certain permissions. If you specify this    
ity│option, the Monkey will continue to send    
-ex│events to the system, until the count is    
cep│completed.                                  
tio│                                            
ns`│                                            
───┼────────────────────────────────────────────
`--│Normally, when the Monkey stops due to an   
kil│error, the application that failed will be  
l-p│left running. When this option is set, it   
roc│will signal the system to stop the process  
ess│in which the error occurred. Note, under a  
-af│normal (successful) completion, the launched
ter│process(es) are not stopped, and the device 
-er│is simply left in the last state after the  
ror│final event.                                
`  │                                            
───┼────────────────────────────────────────────
`--│Watches for and reports crashes occurring in
mon│the Android system native code. If          
ito│--kill-process-after-error is set, the      
r-n│system will stop.                           
ati│                                            
ve-│                                            
cra│                                            
she│                                            
s` │                                            
───┼────────────────────────────────────────────
`--│Stops the Monkey from executing until a     
wai│debugger is attached to it.                 
t-d│                                            
bg`│                                            
───┴────────────────────────────────────────────

[1]: https://developer.android.com/
[2]: https://developer.android.com/develop
[3]: https://developer.android.com/studio
[4]: https://developer.android.com/studio/intro
[5]: /tools/help/emulator
[6]: /studio/profile
