Dec 02 2025

# PVS-Studio Messages

Dec 02 2025

* [Graph showing the number of diagnostic rules in PVS-Studio ][1]
* [What bugs can PVS-Studio detect?][2]
* [PVS-Studio diagnostic rule capabilities][3]
* [ Complete list of analyzer rules in XML][4]
* [General Analysis (C++)][5]
* [General Analysis (C#)][6]
* [General Analysis (Java)][7]
* [Micro-Optimizations (C++)][8]
* [Micro-Optimizations (C#)][9]
* [Diagnosis of 64-bit errors (Viva64, C++)][10]
* [Customer specific requests (C++)][11]
* [MISRA errors][12]
* [AUTOSAR errors][13]
* [OWASP errors (C++)][14]
* [OWASP errors (C#)][15]
* [OWASP errors (Java)][16]
* [Problems related to code analyzer][17]

## Graph showing the number of diagnostic rules in PVS-Studio

PVS-Studio is constantly evolving. Our team actively improves the tool's integration with various
CI/CD pipelines and IDEs, and supports new platforms and compilers. The number of diagnostic rules
in the analyzer is an effective way to showcase its enhancements.

[warnings/image1.png]

Figure 1. A graph showing the increasing number of diagnostics in PVS-Studio

We are continuously enhancing the analyzer's capability to detect new error patterns. Below, you can
learn more about new features in different analyzer versions. You can also explore the PVS-Studio
updates over the past year [in our blog][18].

## What bugs can PVS-Studio detect?

We organized most of the diagnostic rules into several groups, so that you can get a general idea of
what PVS-Studio is capable of.

Since the categorization is quite arbitrary, some diagnostic rules fall into multiple groups. For
example, the `if (abc == abc)` incorrect condition can be interpreted both as a simple typo and as a
security issue, because it may lead to a code vulnerability if the input data are incorrect.

Some of the errors, on the contrary, did not make it into the table because they were too specific.
Nevertheless, this table provides insight into the features of the static code analyzer.

## PVS-Studio diagnostic rule capabilities


──────┬─────────────────────────────────────────────────────────────────────────────────────────────
**Main│**Diagnostic rules**                                                                         
PVS-St│                                                                                             
udio  │                                                                                             
diagno│                                                                                             
stic  │                                                                                             
rule  │                                                                                             
capabi│                                                                                             
lities│                                                                                             
**    │                                                                                             
──────┼─────────────────────────────────────────────────────────────────────────────────────────────
64-bit│**C, C++: **V101-V128, V201-V207, V220, V221, V301-V303                                      
issues│                                                                                             
──────┼─────────────────────────────────────────────────────────────────────────────────────────────
Check │**C, C++: **V506, V507, V558, V723, V758, V1017, V1047                                       
that  │                                                                                             
addres│                                                                                             
ses to│                                                                                             
stack │                                                                                             
memory│                                                                                             
does  │                                                                                             
not   │                                                                                             
leave │                                                                                             
the   │                                                                                             
functi│                                                                                             
on    │                                                                                             
──────┼─────────────────────────────────────────────────────────────────────────────────────────────
Arithm│**C, C++: **V569, V636, V658, V784, V786, V1012, V1026, V1028, V1029, V1033, V1070, V1081,   
etic  │V1083, V1085, V1112                                                                          
over/u│                                                                                             
nderfl│                                                                                             
ow    │**C#: **V3041, V3200, V3204, V3217                                                           
      │                                                                                             
      │                                                                                             
      │**Java: **V6011, V6088, V6117                                                                
──────┼─────────────────────────────────────────────────────────────────────────────────────────────
Array │**C, C++: **V557, V582, V643, V781, V1038, V1111                                             
index │                                                                                             
out of│                                                                                             
bounds│**C#: **V3106, V3218                                                                         
      │                                                                                             
      │                                                                                             
      │**Java: **V6025, V6079                                                                       
──────┼─────────────────────────────────────────────────────────────────────────────────────────────
Double│**C, C++: **V586, V749, V1002, V1006                                                         
-free │                                                                                             
──────┼─────────────────────────────────────────────────────────────────────────────────────────────
Dead  │**C, C++: **V606, V607                                                                       
code  │                                                                                             
      │                                                                                             
      │**Java: **V6021                                                                              
──────┼─────────────────────────────────────────────────────────────────────────────────────────────
Microo│**C, C++: **V801, V802, V803, V804, V805, V806, V807, V808, V809, V810, V811, V812, V813,    
ptimiz│V814, V815, V816, V817, V818, V819, V820, V821, V822, V823, V824, V825, V826, V827, V828,    
ation │V829, V830, V831, V832, V833, V834, V835, V836, V837, V838, V839                             
      │                                                                                             
      │                                                                                             
      │**C#: **V4001, V4002, V4003, V4004, V4005, V4006, V4007, V4008                               
──────┼─────────────────────────────────────────────────────────────────────────────────────────────
Unreac│**C, C++: **V517, V551, V695, V734, V776, V779, V785                                         
hable │                                                                                             
code  │                                                                                             
      │**C#: **V3136, V3142, V3202                                                                  
      │                                                                                             
      │                                                                                             
      │**Java: **V6018, V6019                                                                       
──────┼─────────────────────────────────────────────────────────────────────────────────────────────
Uninit│**C, C++: **V573, V614, V679, V737, V788, V1007, V1050, V1077, V1086                         
ialize│                                                                                             
d     │                                                                                             
variab│**C#: **V3070, V3128                                                                         
les   │                                                                                             
      │                                                                                             
      │**Java: **V6036, V6050, V6052, V6090                                                         
──────┼─────────────────────────────────────────────────────────────────────────────────────────────
Unused│**C, C++: **V603, V751, V763, V1001, V1079                                                   
variab│                                                                                             
les   │                                                                                             
      │**C#: **V3061, V3065, V3077, V3117, V3137, V3143, V3196, V3203, V3220                        
      │                                                                                             
      │                                                                                             
      │**Java: **V6021, V6022, V6023                                                                
──────┼─────────────────────────────────────────────────────────────────────────────────────────────
Illega│**C, C++: **V610, V629, V673, V684, V770, V1093                                              
l     │                                                                                             
bitwis│                                                                                             
e/shif│**C#: **V3134                                                                                
t     │                                                                                             
operat│                                                                                             
ions  │**Java: **V6034, V6069                                                                       
──────┼─────────────────────────────────────────────────────────────────────────────────────────────
Undefi│**C, C++: **V567, V610, V611, V681, V694, V704, V708, V726, V736, V772, V1007, V1016, V1026, 
ned/un│V1032, V1061, V1066, V1069, V1082, V1091, V1094, V1097, V1099                                
specif│                                                                                             
ied   │                                                                                             
behavi│**Java: **V6128                                                                              
or    │                                                                                             
──────┼─────────────────────────────────────────────────────────────────────────────────────────────
Incorr│**C, C++: **V543, V544, V545, V556, V615, V636, V676, V716, V721, V724, V745, V750, V767,    
ect   │V768, V771, V772, V775, V1014, V1027, V1034, V1046, V1060, V1066, V1084                      
handli│                                                                                             
ng of │                                                                                             
the   │**C#: **V3041, V3059, V3076, V3111, V3121, V3148                                             
types │                                                                                             
(HRESU│                                                                                             
LT,   │**Java: **V6038, V6108                                                                       
BSTR, │                                                                                             
BOOL, │                                                                                             
VARIAN│                                                                                             
T_BOOL│                                                                                             
,     │                                                                                             
float,│                                                                                             
double│                                                                                             
)     │                                                                                             
──────┼─────────────────────────────────────────────────────────────────────────────────────────────
Improp│**C, C++: **V515, V518, V530, V540, V541, V554, V575, V597, V598, V618, V630, V632, V663,    
er    │V668, V698, V701, V702, V717, V718, V720, V723, V725, V727, V738, V742, V743, V748, V762,    
unders│V764, V780, V789, V797, V1014, V1024, V1031, V1035, V1045, V1052, V1053, V1054, V1057, V1060,
tandin│V1066, V1098, V1100, V1107, V1115                                                            
g of  │                                                                                             
functi│                                                                                             
on/cla│**C#: **V3010, V3057, V3068, V3072, V3073, V3074, V3078, V3082, V3084, V3094, V3096, V3097,  
ss    │V3102, V3103, V3104, V3108, V3114, V3115, V3118, V3123, V3126, V3145, V3178, V3186, V3192,   
operat│V3194, V3195, V3197                                                                          
ion   │                                                                                             
logic │                                                                                             
      │**Java: **V6009, V6010, V6016, V6026, V6029, V6049, V6055, V6058, V6064, V6068, V6081, V6110,
      │V6116, V6122, V6125                                                                          
──────┼─────────────────────────────────────────────────────────────────────────────────────────────
Mispri│**C, C++: **V501, V503, V504, V508, V511, V516, V519, V520, V521, V525, V527, V528, V529,    
nts   │V532, V533, V534, V535, V536, V537, V539, V546, V549, V552, V556, V559, V560, V561, V564,    
      │V568, V570, V571, V575, V577, V578, V584, V587, V588, V589, V590, V592, V602, V604, V606,    
      │V607, V616, V617, V620, V621, V622, V625, V626, V627, V633, V637, V638, V639, V644, V646,    
      │V650, V651, V653, V654, V655, V657, V660, V661, V662, V666, V669, V671, V672, V678, V682,    
      │V683, V693, V715, V722, V735, V741, V747, V753, V754, V756, V765, V767, V768, V770, V771,    
      │V787, V791, V792, V796, V1013, V1015, V1021, V1040, V1051, V1055, V1074, V1094, V1113        
      │                                                                                             
      │                                                                                             
      │**C#: **V3001, V3003, V3005, V3007, V3008, V3009, V3011, V3012, V3014, V3015, V3016, V3020,  
      │V3028, V3029, V3034, V3035, V3036, V3037, V3038, V3050, V3055, V3056, V3057, V3060, V3062,   
      │V3063, V3066, V3081, V3086, V3091, V3092, V3093, V3102, V3107, V3109, V3110, V3112, V3113,   
      │V3116, V3118, V3122, V3124, V3132, V3140, V3170, V3174, V3185, V3187, V3228                  
      │                                                                                             
      │                                                                                             
      │**Java: **V6001, V6005, V6009, V6012, V6014, V6015, V6016, V6017, V6021, V6026, V6028, V6029,
      │V6030, V6031, V6037, V6041, V6042, V6043, V6045, V6057, V6059, V6061, V6062, V6063, V6077,   
      │V6080, V6085, V6091, V6105, V6112                                                            
──────┼─────────────────────────────────────────────────────────────────────────────────────────────
Missin│**C, C++: **V599, V689                                                                       
g     │                                                                                             
Virtua│                                                                                             
l     │                                                                                             
destru│                                                                                             
ctor  │                                                                                             
──────┼─────────────────────────────────────────────────────────────────────────────────────────────
Coding│**C, C++: **V563, V612, V628, V640, V646, V705, V709, V715, V1044, V1073                     
style │                                                                                             
not   │                                                                                             
matchi│**C#: **V3007, V3018, V3033, V3043, V3067, V3069, V3138, V3150, V3172, V3183                 
ng the│                                                                                             
operat│                                                                                             
ion   │**Java: **V6040, V6047, V6063, V6086, V6089, V6132                                           
logic │                                                                                             
of the│                                                                                             
source│                                                                                             
code  │                                                                                             
──────┼─────────────────────────────────────────────────────────────────────────────────────────────
Copy-P│**C, C++: **V501, V517, V519, V523, V524, V571, V581, V649, V656, V666, V691, V760, V766,    
aste  │V778, V1037                                                                                  
      │                                                                                             
      │                                                                                             
      │**C#: **V3001, V3003, V3004, V3008, V3012, V3013, V3021, V3030, V3058, V3127, V3139, V3140,  
      │V3228                                                                                        
      │                                                                                             
      │                                                                                             
      │**Java: **V6003, V6004, V6012, V6021, V6027, V6032, V6033, V6039, V6067, V6072               
──────┼─────────────────────────────────────────────────────────────────────────────────────────────
Incorr│**C, C++: **V509, V565, V596, V667, V668, V740, V741, V746, V759, V1022, V1045, V1067, V1090 
ect   │                                                                                             
usage │                                                                                             
of    │**C#: **V3006, V3052, V3100, V3141, V3163, V3164, V5606, V5607                               
except│                                                                                             
ions  │                                                                                             
      │**Java: **V6006, V6051, V6103                                                                
──────┼─────────────────────────────────────────────────────────────────────────────────────────────
Buffer│**C, C++: **V512, V514, V594, V635, V641, V645, V752, V755                                   
overru│                                                                                             
n     │                                                                                             
──────┼─────────────────────────────────────────────────────────────────────────────────────────────
Securi│**C, C++: **V505, V510, V511, V512, V518, V531, V541, V547, V559, V560, V569, V570, V575,    
ty    │V576, V579, V583, V597, V598, V618, V623, V631, V642, V645, V675, V676, V724, V727, V729,    
issues│V733, V743, V745, V750, V771, V774, V782, V1001, V1003, V1005, V1010, V1017, V1055, V1072,   
      │V1076, V1113, V1118                                                                          
      │                                                                                             
      │                                                                                             
      │**C#: **V3022, V3023, V3025, V3027, V3039, V3053, V3063, V3225, V5601, V5608, V5609, V5610,  
      │V5611, V5612, V5613, V5614, V5615, V5616, V5617, V5618, V5619, V5620, V5621, V5622, V5623,   
      │V5624, V5625, V5626, V5627, V5628, V5629, V5630, V5631                                       
      │                                                                                             
      │                                                                                             
      │**Java: **V5305, V5309, V5312, V5313, V5314, V5315, V5316, V5317, V5318, V5319, V5320, V5321,
      │V5322, V5323, V5325, V5326, V5327, V5328, V5329, V5330, V5331, V5332, V5333, V5334, V5335,   
      │V5336, V5337, V5338, V6007, V6046, V6054, V6109                                              
──────┼─────────────────────────────────────────────────────────────────────────────────────────────
Operat│**C, C++: **V502, V562, V593, V634, V648, V727, V733, V1003, V1104                           
ion   │                                                                                             
priori│                                                                                             
ty    │**C#: **V3130, V3133, V3177, V3207                                                           
      │                                                                                             
      │                                                                                             
      │**Java: **V6044                                                                              
──────┼─────────────────────────────────────────────────────────────────────────────────────────────
Null  │**C, C++: **V522, V595, V664, V713, V757, V769                                               
pointe│                                                                                             
r /   │                                                                                             
null  │**C#: **V3019, V3042, V3080, V3095, V3105, V3125, V3141, V3145, V3146, V3148, V3149, V3153,  
refere│V3156, V3168, V3195                                                                          
nce   │                                                                                             
derefe│                                                                                             
rence │**Java: **V6008, V6060, V6093                                                                
──────┼─────────────────────────────────────────────────────────────────────────────────────────────
Unchec│**C, C++: **V595, V664, V783, V1004                                                          
ked   │                                                                                             
parame│                                                                                             
ter   │**C#: **V3095                                                                                
derefe│                                                                                             
rence │                                                                                             
      │**Java: **V6060                                                                              
──────┼─────────────────────────────────────────────────────────────────────────────────────────────
Synchr│**C, C++: **V712, V720, V744, V1011, V1018, V1025, V1036, V1088, V1089, V1114                
onizat│                                                                                             
ion   │                                                                                             
errors│**C#: **V3032, V3054, V3079, V3082, V3083, V3089, V3090, V3147, V3167, V3168, V3190, V3223   
      │                                                                                             
      │                                                                                             
      │**Java: **V6064, V6070, V6074, V6082, V6095, V6102, V6125, V6126, V6129                      
──────┼─────────────────────────────────────────────────────────────────────────────────────────────
Resour│**C, C++: **V599, V701, V773, V1020, V1023, V1100, V1106, V1110                              
ce    │                                                                                             
leaks │                                                                                             
      │**Java: **V6127                                                                              
──────┼─────────────────────────────────────────────────────────────────────────────────────────────
Check │**C, C++: **V609                                                                             
for   │                                                                                             
intege│                                                                                             
r     │**C#: **V3064, V3151, V3152                                                                  
divisi│                                                                                             
on by │                                                                                             
zero  │**Java: **V6020                                                                              
──────┼─────────────────────────────────────────────────────────────────────────────────────────────
Serial│**C, C++: **V513, V663, V739, V1024, V1095                                                   
izatio│                                                                                             
n /   │                                                                                             
deseri│**C#: **V3094, V3096, V3097, V3099, V3103, V3104, V3193, V5611                               
alizat│                                                                                             
ion   │                                                                                             
issues│**Java: **V6065, V6075, V6076, V6083, V6087                                                  
──────┼─────────────────────────────────────────────────────────────────────────────────────────────
Custom│**C, C++: **V2001, V2002, V2003, V2004, V2005, V2006, V2007, V2008, V2009, V2010, V2011,     
ized  │V2012, V2013, V2014, V2022                                                                   
user  │                                                                                             
rules │                                                                                             
──────┴─────────────────────────────────────────────────────────────────────────────────────────────

The table outlines the capabilities of PVS-Studio.

As you can see, the analyzer is best at detecting [security flaws][19], as well as errors caused by
typos and copy-paste operations.

To see these diagnostic rules in action, you can take a look at the [error base][20]. We collect all
the errors found by checking various open-source projects using PVS-Studio.

## Complete list of analyzer rules in XML

You can find a permanent link to a machine-readable map of all analyzer rules in XML format
[here][21].

## General Analysis (C++)

* [V501][22]. Identical sub-expressions to the left and to the right of 'foo' operator.
* [V502][23]. The '?:' operator may not work as expected. The '?:' operator has a lower priority
  than the 'foo' operator.
* [V503][24]. Nonsensical comparison: pointer < 0.
* [V504][25]. Semicolon ';' is probably missing after the 'return' keyword.
* [V505][26]. The 'alloca' function is used inside the loop. This can quickly overflow stack.
* [V506][27]. Pointer to local variable 'X' is stored outside the scope of this variable. Such a
  pointer will become invalid.
* [V507][28]. Pointer to local array 'X' is stored outside the scope of this array. Such a pointer
  will become invalid.
* [V508][29]. The 'new type(n)' pattern was detected. Probably meant: 'new type[n]'.
* [V509][30]. Exceptions raised inside noexcept functions must be wrapped in a try..catch block.
* [V510][31]. The 'Foo' function receives class-type variable as Nth actual argument. This is
  unexpected behavior.
* [V511][32]. The sizeof() operator returns pointer size instead of array size.
* [V512][33]. Call of the 'Foo' function will lead to buffer overflow.
* [V513][34]. Use _beginthreadex/_endthreadex functions instead of CreateThread/ExitThread
  functions.
* [V514][35]. Potential logical error. Size of a pointer is divided by another value.
* [V515][36]. The 'delete' operator is applied to non-pointer.
* [V516][37]. Non-null function pointer is compared to null. Consider inspecting the expression.
* [V517][38]. Potential logical error. The 'if (A) {...} else if (A) {...}' pattern was detected.
* [V518][39]. The 'malloc' function allocates suspicious amount of memory calculated by
  'strlen(expr)'. Perhaps the correct expression is strlen(expr) + 1.
* [V519][40]. The 'x' variable is assigned values twice successively. Perhaps this is a mistake.
* [V520][41]. Comma operator ',' in array index expression.
* [V521][42]. Expressions that use comma operator ',' are dangerous. Make sure the expression is
  correct.
* [V522][43]. Possible null pointer dereference.
* [V523][44]. The 'then' statement is equivalent to the 'else' statement.
* [V524][45]. It is suspicious that the body of 'Foo_1' function is fully equivalent to the body of
  'Foo_2' function.
* [V525][46]. Code contains collection of similar blocks. Check items X, Y, Z, ... in lines N1, N2,
  N3, ...
* [V526][47]. The 'strcmp' function returns 0 if corresponding strings are equal. Consider
  inspecting the condition for mistakes.
* [V527][48]. The 'zero' value is assigned to pointer. Probably meant: *ptr = zero.
* [V528][49]. Pointer is compared with 'zero' value. Probably meant: *ptr != zero.
* [V529][50]. Suspicious semicolon ';' after 'if/for/while' operator.
* [V530][51]. Return value of 'Foo' function is required to be used.
* [V531][52]. The sizeof() operator is multiplied by sizeof(). Consider inspecting the expression.
* [V532][53]. Consider inspecting the statement of '*pointer++' pattern. Probably meant:
  '(*pointer)++'.
* [V533][54]. It is possible that a wrong variable is incremented inside the 'for' operator.
  Consider inspecting 'X'.
* [V534][55]. It is possible that a wrong variable is compared inside the 'for' operator. Consider
  inspecting 'X'.
* [V535][56]. The 'X' variable is used for this loop and outer loops.
* [V536][57]. Constant value is represented by an octal form.
* [V537][58]. Potential incorrect use of item 'X'. Consider inspecting the expression.
* [V538][59]. The line contains control character 0x0B (vertical tabulation).
* [V539][60]. Iterators are passed as arguments to 'Foo' function. Consider inspecting the
  expression.
* [V540][61]. Member 'x' should point to string terminated by two 0 characters.
* [V541][62]. String is printed into itself. Consider inspecting the expression.
* [V542][63]. Suspicious type cast: 'Type1' to ' Type2'. Consider inspecting the expression.
* [V543][64]. It is suspicious that value 'X' is assigned to the variable 'Y' of HRESULT type.
* [V544][65]. It is suspicious that the value 'X' of HRESULT type is compared with 'Y'.
* [V545][66]. Conditional expression of 'if' statement is incorrect for the HRESULT type value
  'Foo'. The SUCCEEDED or FAILED macro should be used instead.
* [V546][67]. The 'Foo(Foo)' class member is initialized with itself.
* [V547][68]. Expression is always true/false.
* [V548][69]. TYPE X[][] is not equivalent to TYPE **X. Consider inspecting type casting.
* [V549][70]. The 'first' argument of 'Foo' function is equal to the 'second' argument.
* [V550][71]. Suspicious precise comparison. Consider using a comparison with defined precision:
  fabs(A - B) < Epsilon or fabs(A - B) > Epsilon.
* [V551][72]. Unreachable code under a 'case' label.
* [V552][73]. A bool type variable is incremented. Perhaps another variable should be incremented
  instead.
* [V553][74]. Length of function body or class declaration is more than 2000 lines. Consider
  refactoring the code.
* [V554][75]. Incorrect use of smart pointer.
* [V555][76]. Expression of the 'A - B > 0' kind will work as 'A != B'.
* [V556][77]. Values of different enum types are compared.
* [V557][78]. Possible array overrun.
* [V558][79]. Function returns pointer/reference to temporary local object.
* [V559][80]. Suspicious assignment inside the conditional expression of 'if/while/for' statement.
* [V560][81]. Part of conditional expression is always true/false.
* [V561][82]. Consider assigning value to 'foo' variable instead of declaring it anew.
* [V562][83]. Bool type value is compared with value of N. Consider inspecting the expression.
* [V563][84]. An 'else' branch may apply to the previous 'if' statement.
* [V564][85]. The '&' or '|' operator is applied to bool type value. Check for missing parentheses
  or use the '&&' or '||' operator.
* [V565][86]. Empty exception handler. Silent suppression of exceptions can hide errors in source
  code during testing.
* [V566][87]. Integer constant is converted to pointer. Check for an error or bad coding style.
* [V567][88]. Modification of variable is unsequenced relative to another operation on the same
  variable. This may lead to undefined behavior.
* [V568][89]. It is suspicious that the argument of sizeof() operator is the expression.
* [V569][90]. Truncation of constant value.
* [V570][91]. Variable is assigned to itself.
* [V571][92]. Recurring check. This condition was already verified in previous line.
* [V572][93]. Object created using 'new' operator is immediately cast to another type. Consider
  inspecting the expression.
* [V573][94]. Use of uninitialized variable 'Foo'. The variable was used to initialize itself.
* [V574][95]. Pointer is used both as an array and as a pointer to single object.
* [V575][96]. Function receives suspicious argument.
* [V576][97]. Incorrect format. Consider checking the Nth actual argument of the 'Foo' function.
* [V577][98]. Label is present inside switch(). Check for typos and consider using the 'default:'
  operator instead.
* [V578][99]. Suspicious bitwise operation was detected. Consider inspecting it.
* [V579][100]. The 'Foo' function receives the pointer and its size as arguments. This may be a
  potential error. Inspect the Nth argument.
* [V580][101]. Suspicious explicit type casting. Consider inspecting the expression.
* [V581][102]. Conditional expressions of 'if' statements located next to each other are identical.
* [V582][103]. Consider reviewing the source code that uses the container.
* [V583][104]. The '?:' operator, regardless of its conditional expression, always returns the same
  value.
* [V584][105]. Same value is present on both sides of the operator. The expression is incorrect or
  can be simplified.
* [V585][106]. Attempt to release memory that stores the 'Foo' local variable.
* [V586][107]. The 'Foo' function is called twice to deallocate the same resource.
* [V587][108]. Suspicious sequence of assignments: A = B; B = A;.
* [V588][109]. Expression of the 'A =+ B' kind is used. Possibly meant: 'A += B'. Consider
  inspecting the expression.
* [V589][110]. Expression of the 'A =- B' kind is used. Possibly meant: 'A -= B'. Consider
  inspecting the expression.
* [V590][111]. Possible excessive expression or typo. Consider inspecting the expression.
* [V591][112]. Non-void function must return value.
* [V592][113]. Expression is enclosed by parentheses twice: ((expression)). One pair of parentheses
  is unnecessary or typo is present.
* [V593][114]. Expression 'A = B == C' is calculated as 'A = (B == C)'. Consider inspecting the
  expression.
* [V594][115]. Pointer to array is out of array bounds.
* [V595][116]. Pointer was used before its check for nullptr. Check lines: N1, N2.
* [V596][117]. Object was created but is not used. Check for missing 'throw' keyword.
* [V597][118]. Compiler may delete 'memset' function call that is used to clear 'Foo' buffer. Use
  the RtlSecureZeroMemory() function to erase private data.
* [V598][119]. Memory manipulation function is used to work with a class object containing a virtual
  table pointer. The result of such an operation may be unexpected.
* [V599][120]. The virtual destructor is not present, although the 'Foo' class contains virtual
  functions.
* [V600][121]. The 'Foo' pointer is always not equal to NULL. Consider inspecting the condition.
* [V601][122]. Suspicious implicit type casting.
* [V602][123]. The '<' operator should probably be replaced with '<<'. Consider inspecting this
  expression.
* [V603][124]. Object was created but not used. If you wish to call constructor, use
  'this->Foo::Foo(....)'.
* [V604][125]. Number of iterations in loop equals size of a pointer. Consider inspecting the
  expression.
* [V605][126]. Unsigned value is compared to the NN number. Consider inspecting the expression.
* [V606][127]. Ownerless token 'Foo'.
* [V607][128]. Ownerless expression 'Foo'.
* [V608][129]. Recurring sequence of explicit type casts.
* [V609][130]. Possible division or mod by zero.
* [V610][131]. Undefined behavior. Check the shift operator.
* [V611][132]. Memory allocation and deallocation methods are incompatible.
* [V612][133]. Unconditional 'break/continue/return/goto' within a loop.
* [V613][134]. Suspicious pointer arithmetic with 'malloc/new'.
* [V614][135]. Use of 'Foo' uninitialized variable.
* [V615][136]. Suspicious explicit conversion from 'float *' type to 'double *' type.
* [V616][137]. Use of 'Foo' named constant with 0 value in bitwise operation.
* [V617][138]. Argument of the '|' bitwise operation always contains non-zero value. Consider
  inspecting the condition.
* [V618][139]. Dangerous call of 'Foo' function. The passed line may contain format specification.
  Example of safe code: printf("%s", str);
* [V619][140]. Array is used as pointer to single object.
* [V620][141]. Expression of sizeof(T)*N kind is summed up with pointer to T type. Consider
  inspecting the expression.
* [V621][142]. Loop may execute incorrectly or may not execute at all. Consider inspecting the 'for'
  operator.
* [V622][143]. First 'case' operator may be missing. Consider inspecting the 'switch' statement.
* [V623][144]. Temporary object is created and then destroyed. Consider inspecting the '?:'
  operator.
* [V624][145]. Use of constant NN. The resulting value may be inaccurate. Consider using the M_NN
  constant from <math.h>.
* [V625][146]. Initial and final values of the iterator are the same. Consider inspecting the 'for'
  operator.
* [V626][147]. It's possible that ',' should be replaced by ';'. Consider checking for typos.
* [V627][148]. Argument of sizeof() is a macro, which expands to a number. Consider inspecting the
  expression.
* [V628][149]. It is possible that a line was commented out improperly, thus altering the program's
  operation logic.
* [V629][150]. Bit shifting of the 32-bit value with a subsequent expansion to the 64-bit type.
  Consider inspecting the expression.
* [V630][151]. The 'malloc' function is used to allocate memory for an array of objects that are
  classes containing constructors/destructors.
* [V631][152]. Defining absolute path to file or directory is considered a poor coding style.
  Consider inspecting the 'Foo' function call.
* [V632][153]. Argument is of the 'T' type. Consider inspecting the NN argument of the 'Foo'
  function.
* [V633][154]. The '!=' operator should probably be used here. Consider inspecting the expression.
* [V634][155]. Priority of '+' operation is higher than priority of '<<' operation. Consider using
  parentheses in the expression.
* [V635][156]. Length should be probably multiplied by sizeof(wchar_t). Consider inspecting the
  expression.
* [V636][157]. Expression was implicitly cast from integer type to real type. Consider using an
  explicit type cast to avoid overflow or loss of a fractional part.
* [V637][158]. Use of two opposite conditions. The second condition is always false.
* [V638][159]. Terminal null is present inside a string. Use of '\0xNN' characters. Probably meant:
  '\xNN'.
* [V639][160]. One of closing ')' parentheses is probably positioned incorrectly. Consider
  inspecting the expression for function call.
* [V640][161]. Code's operational logic does not correspond with its formatting.
* [V641][162]. Buffer size is not a multiple of element size.
* [V642][163]. Function result is saved inside the 'byte' type variable. Significant bits may be
  lost. This may break the program's logic.
* [V643][164]. Suspicious pointer arithmetic. Value of 'char' type is added to a string pointer.
* [V644][165]. Suspicious function declaration. Consider creating a 'T' type object.
* [V645][166]. Function call may lead to buffer overflow. Bounds should not contain size of a
  buffer, but a number of characters it can hold.
* [V646][167]. The 'else' keyword may be missing. Consider inspecting the program's logic.
* [V647][168]. Value of 'A' type is assigned to a pointer of 'B' type.
* [V648][169]. Priority of '&&' operation is higher than priority of '||' operation.
* [V649][170]. Two 'if' statements with identical conditional expressions. The first 'if' statement
  contains function return. This means that the second 'if' statement is senseless.
* [V650][171]. Type casting is used 2 times in a row. The '+' operation is executed. Probably meant:
  (T1)((T2)a + b).
* [V651][172]. Suspicious operation of 'sizeof(X)/sizeof(T)' kind, where 'X' is of the 'class' type.
* [V652][173]. Operation is executed 3 or more times in a row.
* [V653][174]. Suspicious string consisting of two parts is used for initialization. Comma may be
  missing.
* [V654][175]. Condition of a loop is always true/false.
* [V655][176]. Strings were concatenated but not used. Consider inspecting the expression.
* [V656][177]. Variables are initialized through the call to the same function. It's probably an
  error or un-optimized code.
* [V657][178]. Function always returns the same value of NN. Consider inspecting the function.
* [V658][179]. Value is subtracted from unsigned variable. It can result in an overflow. In such a
  case, the comparison operation may behave unexpectedly.
* [V659][180]. Functions' declarations with 'Foo' name differ in 'const' keyword only, while these
  functions' bodies have different composition. It is suspicious and can possibly be an error.
* [V660][181]. Program contains an unused label and function call: 'CC:AA()'. Probably meant:
  'CC::AA()'.
* [V661][182]. Suspicious expression 'A[B < C]'. Probably meant 'A[B] < C'.
* [V662][183]. Different containers are used to set up initial and final values of iterator.
  Consider inspecting the loop expression.
* [V663][184]. Infinite loop is possible. The 'cin.eof()' condition is insufficient to break from
  the loop. Consider adding the 'cin.fail()' function call to the conditional expression.
* [V664][185]. Pointer is dereferenced in the member initializer list before it is checked for null
  in the body of a constructor.
* [V665][186]. Possible incorrect use of '#pragma warning(default: X)'. The '#pragma
  warning(push/pop)' should be used instead.
* [V666][187]. Value may not correspond with the length of a string passed with YY argument.
  Consider inspecting the NNth argument of the 'Foo' function.
* [V667][188]. The 'throw' operator does not have any arguments and is not located within the
  'catch' block.
* [V668][189]. Possible meaningless check for null, as memory was allocated using 'new' operator.
  Memory allocation will lead to an exception.
* [V669][190]. Argument is a non-constant reference. The analyzer is unable to determine the
  position where this argument is modified. Consider checking the function for an error.
* [V670][191]. Uninitialized class member is used to initialize another member. Remember that
  members are initialized in the order of their declarations inside a class.
* [V671][192]. The 'swap' function may interchange a variable with itself.
* [V672][193]. It is possible that creating a new variable is unnecessary. One of the function's
  arguments has the same name and this argument is a reference.
* [V673][194]. More than N bits are required to store the value, but the expression evaluates to the
  T type which can only hold K bits.
* [V674][195]. Expression contains a suspicious mix of integer and real types.
* [V675][196]. Writing into read-only memory.
* [V676][197]. Incorrect comparison of BOOL type variable with TRUE.
* [V677][198]. Custom declaration of standard type. Consider using the declaration from system
  header files instead.
* [V678][199]. Object is used as an argument to its own method. Consider checking the first actual
  argument of the 'Foo' function.
* [V679][200]. The 'X' variable was not initialized. This variable is passed by reference to the
  'Foo' function in which its value will be used.
* [V680][201]. The 'delete A, B' expression only destroys the 'A' object. Then the ',' operator
  returns a resulting value from the right side of the expression.
* [V681][202]. The language standard does not define order in which 'Foo' functions are called
  during evaluation of arguments.
* [V682][203]. Suspicious literal: '/r'. It is possible that a backslash should be used instead:
  '\r'.
* [V683][204]. The 'i' variable should probably be incremented instead of the 'n' variable. Consider
  inspecting the loop expression.
* [V684][205]. Value of variable is not modified. It is possible that '1' should be present instead
  of '0'. Consider inspecting the expression.
* [V685][206]. The expression contains a comma. Consider inspecting the return statement.
* [V686][207]. Pattern A || (A && ...) was detected. The expression is excessive or contains a
  logical error.
* [V687][208]. Size of array calculated by sizeof() operator was added to a pointer. It is possible
  that the number of elements should be calculated by sizeof(A)/sizeof(A[0]).
* [V688][209]. The 'foo' local variable has the same name as one of class members. This can result
  in confusion.
* [V689][210]. Destructor of 'Foo' class is not declared as virtual. A smart pointer may not destroy
  an object correctly.
* [V690][211]. The class implements a copy constructor/operator=, but lacks the operator=/copy
  constructor.
* [V691][212]. Empirical analysis. Possible typo inside the string literal. The 'foo' word is
  suspicious.
* [V692][213]. Inappropriate attempt to append a null character to a string. To determine the length
  of a string by 'strlen' function correctly, use a string ending with a null terminator in the
  first place.
* [V693][214]. It is possible that 'i < X.size()' should be used instead of 'X.size()'. Consider
  inspecting conditional expression of the loop.
* [V694][215]. The condition (ptr - const_value) is only false if the value of a pointer equals a
  magic constant.
* [V695][216]. Range intersections are possible within conditional expressions.
* [V696][217]. The 'continue' operator will terminate 'do { ... } while (FALSE)' loop because the
  condition is always false.
* [V697][218]. Number of elements in the allocated array equals the size of a pointer in bytes.
* [V698][219]. Functions of strcmp() kind can return any values, not only -1, 0, or 1.
* [V699][220]. It is possible that 'foo = bar == baz ? .... : ....' should be used here instead of
  'foo = bar = baz ? .... : ....'. Consider inspecting the expression.
* [V700][221]. It is suspicious that variable is initialized through itself. Consider inspecting the
  'T foo = foo = x;' expression.
* [V701][222]. Possible realloc() leak: when realloc() fails to allocate memory, original pointer is
  lost. Consider assigning realloc() to a temporary pointer.
* [V702][223]. Classes should always be derived from std::exception (and alike) as 'public'.
* [V703][224]. It is suspicious that the 'foo' field in derived class overwrites field in base
  class.
* [V704][225]. The expression is always false on newer compilers. Avoid using 'this == 0'
  comparison.
* [V705][226]. It is possible that 'else' block was forgotten or commented out, thus altering the
  program's operation logics.
* [V706][227]. Suspicious division: sizeof(X) / Value. Size of every element in X array is not equal
  to divisor.
* [V707][228]. Giving short names to global variables is considered to be bad practice.
* [V708][229]. Dangerous construction is used: 'm[x] = m.size()', where 'm' is of 'T' class. This
  may lead to undefined behavior.
* [V709][230]. Suspicious comparison found: 'a == b == c'. Remember that 'a == b == c' is not equal
  to 'a == b && b == c'.
* [V710][231]. Suspicious declaration. There is no point to declare constant reference to a number.
* [V711][232]. It is dangerous to create a local variable within a loop with a same name as a
  variable controlling this loop.
* [V712][233]. Compiler may optimize out this loop or make it infinite. Use volatile variable(s) or
  synchronization primitives to avoid this.
* [V713][234]. Pointer was used in the logical expression before its check for nullptr in the same
  logical expression.
* [V714][235]. Variable is not passed into foreach loop by reference, but its value is changed
  inside of the loop.
* [V715][236]. The 'while' operator has empty body. This pattern is suspicious.
* [V716][237]. Suspicious type conversion: HRESULT -> BOOL (BOOL -> HRESULT).
* [V717][238]. It is suspicious to cast object of base class V to derived class U.
* [V718][239]. The 'Foo' function should not be called from 'DllMain' function.
* [V719][240]. The switch statement does not cover all values of the enum.
* [V720][241]. The 'SuspendThread' function is usually used when developing a debugger. See
  documentation for details.
* [V721][242]. The VARIANT_BOOL type is used incorrectly. The true value (VARIANT_TRUE) is defined
  as -1.
* [V722][243]. Abnormality within similar comparisons. It is possible that a typo is present inside
  the expression.
* [V723][244]. Function returns a pointer to the internal string buffer of a local object, which
  will be destroyed.
* [V724][245]. Converting integers or pointers to BOOL can lead to a loss of high-order bits.
  Non-zero value can become 'FALSE'.
* [V725][246]. Dangerous cast of 'this' to 'void*' type in the 'Base' class, as it is followed by a
  subsequent cast to 'Class' type.
* [V726][247]. Attempt to free memory containing the 'int A[10]' array by using the 'free(A)'
  function.
* [V727][248]. Return value of 'wcslen' function is not multiplied by 'sizeof(wchar_t)'.
* [V728][249]. Excessive check can be simplified. The '||' operator is surrounded by opposite
  expressions 'x' and '!x'.
* [V729][250]. Function body contains the 'X' label that is not used by any 'goto' statements.
* [V730][251]. Not all members of a class are initialized inside the constructor.
* [V731][252]. The variable of char type is compared with pointer to string.
* [V732][253]. Unary minus operator does not modify a bool type value.
* [V733][254]. It is possible that macro expansion resulted in incorrect evaluation order.
* [V734][255]. Excessive expression. Examine the substrings "abc" and "abcd".
* [V735][256]. Possibly an incorrect HTML. The "</XX>" closing tag was encountered, while the
  "</YY>" tag was expected.
* [V736][257]. The behavior is undefined for arithmetic or comparisons with pointers that do not
  point to members of the same array.
* [V737][258]. It is possible that ',' comma is missing at the end of the string.
* [V738][259]. Temporary anonymous object is used.
* [V739][260]. EOF should not be compared with a value of the 'char' type. Consider using the 'int'
  type.
* [V740][261]. Exception is of the 'int' type because NULL is defined as 0. Keyword 'nullptr' can be
  used for 'pointer' type exception.
* [V741][262]. Use of the throw (a, b); pattern. It is possible that type name was omitted: throw
  MyException(a, b);.
* [V742][263]. Function receives an address of a 'char' type variable instead of pointer to a
  buffer.
* [V743][264]. The memory areas must not overlap. Use 'memmove' function.
* [V744][265]. Temporary object is immediately destroyed after being created. Consider naming the
  object.
* [V745][266]. A 'wchar_t *' type string is incorrectly converted to 'BSTR' type string.
* [V746][267]. Object slicing. An exception should be caught by reference rather than by value.
* [V747][268]. Suspicious expression inside parentheses. A function name may be missing.
* [V748][269]. Memory for 'getline' function should be allocated only by 'malloc' or 'realloc'
  functions. Consider inspecting the first parameter of 'getline' function.
* [V749][270]. Destructor of the object will be invoked a second time after leaving the object's
  scope.
* [V750][271]. BSTR string becomes invalid. Notice that BSTR strings store their length before start
  of the text.
* [V751][272]. Parameter is not used inside function's body.
* [V752][273]. Creating an object with placement new requires a buffer of large size.
* [V753][274]. The '&=' operation always sets a value of 'Foo' variable to zero.
* [V754][275]. The expression of 'foo(foo(x))' pattern is excessive or contains an error.
* [V755][276]. Copying from potentially tainted data source. Buffer overflow is possible.
* [V756][277]. The 'X' counter is not used inside a nested loop. Consider inspecting usage of 'Y'
  counter.
* [V757][278]. It is possible that an incorrect variable is compared with null after type conversion
  using 'dynamic_cast'.
* [V758][279]. Reference was invalidated because of destruction of the temporary object returned by
  the function.
* [V759][280]. Violated order of exception handlers. Exception caught by handler for base class.
* [V760][281]. Two identical text blocks were detected. The second block starts with NN string.
* [V761][282]. NN identical blocks were found.
* [V762][283]. Consider inspecting virtual function arguments. See NN argument of function 'Foo' in
  derived class and base class.
* [V763][284]. Parameter is always rewritten in function body before being used.
* [V764][285]. Possible incorrect order of arguments passed to function.
* [V765][286]. Compound assignment expression 'X += X + N' is suspicious. Consider inspecting it for
  a possible error.
* [V766][287]. An item with the same key has already been added.
* [V767][288]. Suspicious access to element by a constant index inside a loop.
* [V768][289]. Variable is of enum type. It is suspicious that it is used as a variable of a
  Boolean-type.
* [V769][290]. The pointer in the expression equals nullptr. The resulting value is meaningless and
  should not be used.
* [V770][291]. Possible use of left shift operator instead of comparison operator.
* [V771][292]. The '?:' operator uses constants from different enums.
* [V772][293]. Calling the 'delete' operator for a void pointer will cause undefined behavior.
* [V773][294]. Function exited without releasing the pointer/handle. A memory/resource leak is
  possible.
* [V774][295]. Pointer was used after the memory was released.
* [V775][296]. It is suspicious that the BSTR data type is compared using a relational operator.
* [V776][297]. Potentially infinite loop. The variable in the loop exit condition does not change
  its value between iterations.
* [V777][298]. Dangerous widening type conversion from an array of derived-class objects to a
  base-class pointer.
* [V778][299]. Two similar code fragments. Perhaps, it is a typo and 'X' variable should be used
  instead of 'Y'.
* [V779][300]. Unreachable code was detected. It is possible that an error is present.
* [V780][301]. The object of non-passive (non-PDS) type cannot be used with the function.
* [V781][302]. Value of a variable is checked after it is used. Possible error in program's logic.
  Check lines: N1, N2.
* [V782][303]. It is pointless to compute the distance between the elements of different arrays.
* [V783][304]. Possible dereference of invalid iterator 'X'.
* [V784][305]. The size of the bit mask is less than the size of the first operand. This will cause
  the loss of the higher bits.
* [V785][306]. Constant expression in switch statement.
* [V786][307]. Assigning the value C to the X variable looks suspicious. The value range of the
  variable: [A, B].
* [V787][308]. Wrong variable is probably used in the for operator as an index.
* [V788][309]. Review captured variable in lambda expression.
* [V789][310]. Iterators for the container, used in the range-based for loop, become invalid upon a
  function call.
* [V790][311]. It is suspicious that the assignment operator takes an object by a non-constant
  reference and returns this object.
* [V791][312]. The initial value of the index in the nested loop equals 'i'. Consider using 'i + 1'
  instead.
* [V792][313]. The function located to the right of the '|' and '&' operators will be called
  regardless of the value of the left operand. Consider using '||' and '&&' instead.
* [V793][314]. It is suspicious that the result of the statement is a part of the condition.
  Perhaps, this statement should have been compared with something else.
* [V794][315]. The assignment operator should be protected from the case of 'this == &src'.
* [V795][316]. Size of the 'time_t' type is not 64 bits. After the year 2038, the program will work
  incorrectly.
* [V796][317]. A 'break' statement is probably missing in a 'switch' statement.
* [V797][318]. The function is used as if it returned a bool type. The return value of the function
  should probably be compared with std::string::npos.
* [V798][319]. The size of the dynamic array can be less than the number of elements in the
  initializer.
* [V799][320]. Variable is not used after memory is allocated for it. Consider checking the use of
  this variable.
* [V1001][321]. Variable is assigned but not used by the end of the function.
* [V1002][322]. Class that contains pointers, constructor and destructor is copied by the
  automatically generated operator= or copy constructor.
* [V1003][323]. Macro expression is dangerous or suspicious.
* [V1004][324]. Pointer was used unsafely after its check for nullptr.
* [V1005][325]. The resource was acquired using 'X' function but was released using incompatible 'Y'
  function.
* [V1006][326]. Several shared_ptr objects are initialized by the same pointer. A double memory
  deallocation will occur.
* [V1007][327]. Value from the uninitialized optional is used. It may be an error.
* [V1008][328]. No more than one iteration of the loop will be performed. Consider inspecting the
  'for' operator.
* [V1009][329]. Check the array initialization. Only the first element is initialized explicitly.
* [V1010][330]. Unchecked tainted data is used in expression.
* [V1011][331]. Function execution could be deferred. Consider specifying execution policy
  explicitly.
* [V1012][332]. The expression is always false. Overflow check is incorrect.
* [V1013][333]. Suspicious subexpression in a sequence of similar comparisons.
* [V1014][334]. Structures with members of real type are compared byte-wise.
* [V1015][335]. Suspicious simultaneous use of bitwise and logical operators.
* [V1016][336]. The value is out of range of enum values. This causes unspecified or undefined
  behavior.
* [V1017][337]. Variable of the 'string_view' type references a temporary object, which will be
  removed after evaluation of an expression.
* [V1018][338]. Usage of a suspicious mutex wrapper. It is probably unused, uninitialized, or
  already locked.
* [V1019][339]. Compound assignment expression is used inside condition.
* [V1020][340]. Function exited without performing epilogue actions. It is possible that there is an
  error.
* [V1021][341]. The variable is assigned the same value on several loop iterations.
* [V1022][342]. Exception was thrown by pointer. Consider throwing it by value instead.
* [V1023][343]. A pointer without owner is added to the container by the 'emplace_back' method. A
  memory leak will occur in case of an exception.
* [V1024][344]. Potential use of invalid data. The stream is checked for EOF before reading from it
  but is not checked after reading.
* [V1025][345]. New variable with default value is created instead of 'std::unique_lock' that locks
  on the mutex.
* [V1026][346]. The variable is incremented in the loop. Undefined behavior will occur in case of
  signed integer overflow.
* [V1027][347]. Pointer to an object of the class is cast to unrelated class.
* [V1028][348]. Possible overflow. Consider casting operands, not the result.
* [V1029][349]. Numeric Truncation Error. Return value of function is written to N-bit variable.
* [V1030][350]. Variable is used after it is moved.
* [V1031][351]. Function is not declared. The passing of data to or from this function may be
  affected.
* [V1032][352]. Pointer is cast to a more strictly aligned pointer type.
* [V1033][353]. Variable is declared as auto in C. Its default type is int.
* [V1034][354]. Do not use real-type variables as loop counters.
* [V1035][355]. Only values returned from fgetpos() can be used as arguments to fsetpos().
* [V1036][356]. Potentially unsafe double-checked locking.
* [V1037][357]. Two or more case-branches perform the same actions.
* [V1038][358]. It is suspicious that a char or string literal is added to a pointer.
* [V1039][359]. Character escape is used in multicharacter literal. This causes
  implementation-defined behavior.
* [V1040][360]. Possible typo in the spelling of a pre-defined macro name.
* [V1041][361]. Class member is initialized with dangling reference.
* [V1042][362]. This file is marked with copyleft license, which requires you to open the derived
  source code.
* [V1043][363]. A global object variable is declared in the header. Multiple copies of it will be
  created in all translation units that include this header file.
* [V1044][364]. Loop break conditions do not depend on the number of iterations.
* [V1045][365]. The DllMain function throws an exception. Consider wrapping the throw operator in a
  try..catch block.
* [V1046][366]. Unsafe usage of the 'bool' and integer types together in the operation '&='.
* [V1047][367]. Lifetime of the lambda is greater than lifetime of the local variable captured by
  reference.
* [V1048][368]. Variable 'foo' was assigned the same value.
* [V1049][369]. The 'foo' include guard is already defined in the 'bar1.h' header. The 'bar2.h'
  header will be excluded from compilation.
* [V1050][370]. Uninitialized class member is used when initializing the base class.
* [V1051][371]. It is possible that an assigned variable should be checked in the next condition.
  Consider checking for typos.
* [V1052][372]. Declaring virtual methods in a class marked as 'final' is pointless.
* [V1053][373]. Calling the 'foo' virtual function in the constructor/destructor may lead to
  unexpected result at runtime.
* [V1054][374]. Object slicing. Derived class object was copied to the base class object.
* [V1055][375]. The 'sizeof' expression returns the size of the container type, not the number of
  elements. Consider using the 'size()' function.
* [V1056][376]. The predefined identifier '__func__' always contains the string 'operator()' inside
  function body of the overloaded 'operator()'.
* [V1057][377]. Pseudo random sequence is the same at every program run. Consider assigning the seed
  to a value not known at compile-time.
* [V1058][378]. Nonsensical comparison of two different functions' addresses.
* [V1059][379]. Macro name overrides a keyword/reserved name. This may lead to undefined behavior.
* [V1060][380]. Passing 'BSTR ' to the 'SysAllocString' function may lead to incorrect object
  creation.
* [V1061][381]. Extending 'std' or 'posix' namespace may result in undefined behavior.
* [V1062][382]. Class defines a custom new or delete operator. The opposite operator must also be
  defined.
* [V1063][383]. The modulo by 1 operation is meaningless. The result will always be zero.
* [V1064][384]. The left operand of integer division is less than the right one. The result will
  always be zero.
* [V1065][385]. Expression can be simplified: check similar operands.
* [V1066][386]. The 'SysFreeString' function should be called only for objects of the 'BSTR' type.
* [V1067][387]. Throwing from exception constructor may lead to unexpected behavior.
* [V1068][388]. Do not define an unnamed namespace in a header file.
* [V1069][389]. Do not concatenate string literals with different prefixes.
* [V1070][390]. Signed value is converted to an unsigned one with subsequent expansion to a larger
  type in ternary operator.
* [V1071][391]. Return value is not always used. Consider inspecting the 'foo' function.
* [V1072][392]. Buffer needs to be securely cleared on all execution paths.
* [V1073][393]. Check the following code block after the 'if' statement. Consider checking for
  typos.
* [V1074][394]. Boundary between numeric escape sequence and string is unclear. The escape sequence
  ends with a letter and the next character is also a letter. Check for typos.
* [V1075][395]. The function expects the file to be opened in one mode, but it was opened in
  different mode.
* [V1076][396]. Code contains invisible characters that may alter its logic. Consider enabling the
  display of invisible characters in the code editor.
* [V1077][397]. Conditional initialization inside the constructor may leave some members
  uninitialized.
* [V1078][398]. An empty container is iterated. The loop will not be executed.
* [V1079][399]. Parameter of 'std::stop_token' type is not used inside function's body.
* [V1080][400]. Call of 'std::is_constant_evaluated' function always returns the same value.
* [V1081][401]. Argument of abs() function is minimal negative value. Such absolute value can't be
  represented in two's complement. This leads to undefined behavior.
* [V1082][402]. Function marked as 'noreturn' may return control. This will result in undefined
  behavior.
* [V1083][403]. Signed integer overflow in arithmetic expression. This leads to undefined behavior.
* [V1084][404]. The expression is always true/false. The value is out of range of enum values.
* [V1085][405]. Negative value is implicitly converted to unsigned integer type in arithmetic
  expression.
* [V1086][406]. Call of the 'Foo' function will lead to buffer underflow.
* [V1087][407]. Upper bound of case range is less than its lower bound. This case may be
  unreachable.
* [V1088][408]. No objects are passed to the 'std::scoped_lock' constructor. No locking will be
  performed. This can cause concurrency issues.
* [V1089][409]. Waiting on condition variable without predicate. A thread can wait indefinitely or
  experience a spurious wake-up.
* [V1090][410]. The 'std::uncaught_exception' function is deprecated since C++17 and is removed in
  C++20. Consider replacing this function with 'std::uncaught_exceptions'.
* [V1091][411]. The pointer is cast to an integer type of a larger size. Casting pointer to a type
  of a larger size is an implementation-defined behavior.
* [V1092][412]. Recursive function call during the static/thread_local variable initialization might
  occur. This may lead to undefined behavior.
* [V1093][413]. The result of the right shift operation will always be 0. The right operand is
  greater than or equal to the number of bits in the left operand.
* [V1094][414]. Conditional escape sequence in literal. Its representation is
  implementation-defined.
* [V1095][415]. Usage of potentially invalid handle. The value should be non-negative.
* [V1096][416]. Variable with static storage duration is declared inside the inline function with
  external linkage. This may lead to ODR violation.
* [V1097][417]. Line splice results in a character sequence that matches the syntax of a
  universal-character-name. Using this sequence lead to undefined behavior.
* [V1098][418]. The 'emplace' / 'insert' function call contains potentially dangerous move
  operation. Moved object can be destroyed even if there is no insertion.
* [V1099][419]. Using the function of uninitialized derived class while initializing the base class
  will lead to undefined behavior.
* [V1100][420]. Unreal Engine. Declaring a pointer to a type derived from 'UObject' in a class that
  is not derived from 'UObject' is dangerous. The pointer may start pointing to an invalid object
  after garbage collection.
* [V1101][421]. Changing the default argument of a virtual function parameter in a derived class may
  result in unexpected behavior.
* [V1102][422]. Unreal Engine. Violation of naming conventions may cause Unreal Header Tool to work
  incorrectly.
* [V1103][423]. The values of padding bytes are unspecified. Comparing objects with padding using
  'memcmp' may lead to unexpected result.
* [V1104][424]. Priority of the 'M' operator is higher than that of the 'N' operator. Possible
  missing parentheses.
* [V1105][425]. Suspicious string modification using the 'operator+='. The right operand is
  implicitly converted to a character type.
* [V1106][426]. Qt. Class inherited from 'QObject' should contain at least one constructor that
  takes a pointer to 'QObject'.
* [V1107][427]. Function was declared as accepting unspecified number of parameters. Consider
  explicitly specifying the function parameters list.
* [V1108][428]. Constraint specified in a custom function annotation on the parameter is violated.
* [V1109][429]. Function is deprecated. Consider switching to an equivalent newer function.
* [V1110][430]. Constructor of a class inherited from 'QObject' does not use a pointer to a parent
  object.
* [V1111][431]. The index was used without check after it was checked in previous lines.
* [V1112][432]. Comparing expressions with different signedness can lead to unexpected results.
* [V1113][433]. Potential resource leak. Calling the 'memset' function will change the pointer
  itself, not the allocated resource. Check the first and third arguments.
* [V1114][434]. Suspicious use of type conversion operator when working with COM interfaces.
  Consider using the 'QueryInterface' member function.
* [V1115][435]. Function annotated with the 'pure' attribute has side effects.
* [V1116][436]. Creating an exception object without an explanatory message may result in
  insufficient logging.
* [V1117][437]. The declared function type is cv-qualified. The behavior when using this type is
  undefined.
* [V1118][438]. Excessive file permissions can lead to vulnerabilities. Consider restricting file
  permissions.

## General Analysis (C#)

* [V3001][439]. There are identical sub-expressions to the left and to the right of the 'foo'
  operator.
* [V3002][440]. The switch statement does not cover all values of the enum.
* [V3003][441]. The use of 'if (A) {...} else if (A) {...}' pattern was detected. There is a
  probability of logical error presence.
* [V3004][442]. The 'then' statement is equivalent to the 'else' statement.
* [V3005][443]. The 'x' variable is assigned to itself.
* [V3006][444]. The object was created but it is not being used. The 'throw' keyword could be
  missing.
* [V3007][445]. Odd semicolon ';' after 'if/for/while' operator.
* [V3008][446]. The 'x' variable is assigned values twice successively. Perhaps this is a mistake.
* [V3009][447]. It's odd that this method always returns one and the same value of NN.
* [V3010][448]. The return value of function 'Foo' is required to be utilized.
* [V3011][449]. Two opposite conditions were encountered. The second condition is always false.
* [V3012][450]. The '?:' operator, regardless of its conditional expression, always returns one and
  the same value.
* [V3013][451]. It is odd that the body of 'Foo_1' function is fully equivalent to the body of
  'Foo_2' function.
* [V3014][452]. It is likely that a wrong variable is being incremented inside the 'for' operator.
  Consider reviewing 'X'.
* [V3015][453]. It is likely that a wrong variable is being compared inside the 'for' operator.
  Consider reviewing 'X'.
* [V3016][454]. The variable 'X' is being used for this loop and for the outer loop.
* [V3017][455]. A pattern was detected: A || (A && ...). The expression is excessive or contains a
  logical error.
* [V3018][456]. Consider inspecting the application's logic. It's possible that 'else' keyword is
  missing.
* [V3019][457]. It is possible that an incorrect variable is compared with null after type
  conversion using 'as' keyword.
* [V3020][458]. An unconditional 'break/continue/return/goto' within a loop.
* [V3021][459]. There are two 'if' statements with identical conditional expressions. The first 'if'
  statement contains method return. This means that the second 'if' statement is senseless.
* [V3022][460]. Expression is always true/false.
* [V3023][461]. Consider inspecting this expression. The expression is excessive or contains a
  misprint.
* [V3024][462]. An odd precise comparison. Consider using a comparison with defined precision:
  Math.Abs(A - B) < Epsilon or Math.Abs(A - B) > Epsilon.
* [V3025][463]. Incorrect format. Consider checking the N format items of the 'Foo' function.
* [V3026][464]. The constant NN is being utilized. The resulting value could be inaccurate. Consider
  using the KK constant.
* [V3027][465]. The variable was utilized in the logical expression before it was verified against
  null in the same logical expression.
* [V3028][466]. Consider inspecting the 'for' operator. Initial and final values of the iterator are
  the same.
* [V3029][467]. The conditional expressions of the 'if' statements situated alongside each other are
  identical.
* [V3030][468]. Recurring check. This condition was already verified in previous line.
* [V3031][469]. An excessive check can be simplified. The operator '||' operator is surrounded by
  opposite expressions 'x' and '!x'.
* [V3032][470]. Waiting on this expression is unreliable, as compiler may optimize some of the
  variables. Use volatile variable(s) or synchronization primitives to avoid this.
* [V3033][471]. It is possible that this 'else' branch must apply to the previous 'if' statement.
* [V3034][472]. Consider inspecting the expression. Probably the '!=' should be used here.
* [V3035][473]. Consider inspecting the expression. Probably the '+=' should be used here.
* [V3036][474]. Consider inspecting the expression. Probably the '-=' should be used here.
* [V3037][475]. An odd sequence of assignments of this kind: A = B; B = A;
* [V3038][476]. The argument was passed to method several times. It is possible that another
  argument should be passed instead.
* [V3039][477]. Consider inspecting the 'Foo' function call. Defining an absolute path to the file
  or directory is considered a poor style.
* [V3040][478]. The expression contains a suspicious mix of integer and real types.
* [V3041][479]. The expression was implicitly cast from integer type to real type. Consider
  utilizing an explicit type cast to avoid the loss of a fractional part.
* [V3042][480]. Possible NullReferenceException. The '?.' and '.' operators are used for accessing
  members of the same object.
* [V3043][481]. The code's operational logic does not correspond with its formatting.
* [V3044][482]. WPF: writing and reading are performed on a different Dependency Properties.
* [V3045][483]. WPF: the names of the property registered for DependencyProperty, and of the
  property used to access it, do not correspond with each other.
* [V3046][484]. WPF: the type registered for DependencyProperty does not correspond with the type of
  the property used to access it.
* [V3047][485]. WPF: A class containing registered property does not correspond with a type that is
  passed as the ownerType.type.
* [V3048][486]. WPF: several Dependency Properties are registered with a same name within the owner
  type.
* [V3049][487]. WPF: readonly field of 'DependencyProperty' type is not initialized.
* [V3050][488]. Possibly an incorrect HTML. The </XX> closing tag was encountered, while the </YY>
  tag was expected.
* [V3051][489]. An excessive type cast or check. The object is already of the same type.
* [V3052][490]. The original exception object was swallowed. Stack of original exception could be
  lost.
* [V3053][491]. An excessive expression. Examine the substrings "abc" and "abcd".
* [V3054][492]. Potentially unsafe double-checked locking. Use volatile variable(s) or
  synchronization primitives to avoid this.
* [V3055][493]. Suspicious assignment inside the condition expression of 'if/while/for' operator.
* [V3056][494]. Consider reviewing the correctness of 'X' item's usage.
* [V3057][495]. Function receives an odd argument.
* [V3058][496]. An item with the same key has already been added.
* [V3059][497]. Consider adding '[Flags]' attribute to the enum.
* [V3060][498]. A value of variable is not modified. Consider inspecting the expression. It is
  possible that other value should be present instead of '0'.
* [V3061][499]. Parameter 'A' is always rewritten in method body before being used.
* [V3062][500]. An object is used as an argument to its own method. Consider checking the first
  actual argument of the 'Foo' method.
* [V3063][501]. A part of conditional expression is always true/false if it is evaluated.
* [V3064][502]. Division or mod division by zero.
* [V3065][503]. Parameter is not utilized inside method's body.
* [V3066][504]. Possible incorrect order of arguments passed to method.
* [V3067][505]. It is possible that 'else' block was forgotten or commented out, thus altering the
  program's operation logics.
* [V3068][506]. Calling overrideable class member from constructor is dangerous.
* [V3069][507]. It's possible that the line was commented out improperly, thus altering the
  program's operation logics.
* [V3070][508]. Uninitialized variables are used when initializing the 'A' variable.
* [V3071][509]. The object is returned from inside 'using' block. 'Dispose' will be invoked before
  exiting method.
* [V3072][510]. The 'A' class containing IDisposable members does not itself implement IDisposable.
* [V3073][511]. Not all IDisposable members are properly disposed. Call 'Dispose' when disposing 'A'
  class.
* [V3074][512]. The 'A' class contains 'Dispose' method. Consider making it implement 'IDisposable'
  interface.
* [V3075][513]. The operation is executed 2 or more times in succession.
* [V3076][514]. Comparison with 'double.NaN' is meaningless. Use 'double.IsNaN()' method instead.
* [V3077][515]. Property setter / event accessor does not utilize its 'value' parameter.
* [V3078][516]. Sorting keys priority will be reversed relative to the order of 'OrderBy' method
  calls. Perhaps, 'ThenBy' should be used instead.
* [V3079][517]. The 'ThreadStatic' attribute is applied to a non-static 'A' field and will be
  ignored.
* [V3080][518]. Possible null dereference.
* [V3081][519]. The 'X' counter is not used inside a nested loop. Consider inspecting usage of 'Y'
  counter.
* [V3082][520]. The 'Thread' object is created but is not started. It is possible that a call to
  'Start' method is missing.
* [V3083][521]. Unsafe invocation of event, NullReferenceException is possible. Consider assigning
  event to a local variable before invoking it.
* [V3084][522]. Anonymous function is used to unsubscribe from event. No handlers will be
  unsubscribed, as a separate delegate instance is created for each anonymous function declaration.
* [V3085][523]. The name of 'X' field/property in a nested type is ambiguous. The outer type
  contains static field/property with identical name.
* [V3086][524]. Variables are initialized through the call to the same function. It's probably an
  error or un-optimized code.
* [V3087][525]. Type of variable enumerated in 'foreach' is not guaranteed to be castable to the
  type of collection's elements.
* [V3088][526]. The expression was enclosed by parentheses twice: ((expression)). One pair of
  parentheses is unnecessary or misprint is present.
* [V3089][527]. Initializer of a field marked by [ThreadStatic] attribute will be called once on the
  first accessing thread. The field will have default value on different threads.
* [V3090][528]. Unsafe locking on an object.
* [V3091][529]. Empirical analysis. It is possible that a typo is present inside the string literal.
  The 'foo' word is suspicious.
* [V3092][530]. Range intersections are possible within conditional expressions.
* [V3093][531]. The operator evaluates both operands. Perhaps a short-circuit operator should be
  used instead.
* [V3094][532]. Possible exception when deserializing type. The Ctor(SerializationInfo,
  StreamingContext) constructor is missing.
* [V3095][533]. The object was used before it was verified against null. Check lines: N1, N2.
* [V3096][534]. Possible exception when serializing type. [Serializable] attribute is missing.
* [V3097][535]. Possible exception: type marked by [Serializable] contains non-serializable members
  not marked by [NonSerialized].
* [V3098][536]. The 'continue' operator will terminate 'do { ... } while (false)' loop because the
  condition is always false.
* [V3099][537]. Not all the members of type are serialized inside 'GetObjectData' method.
* [V3100][538]. NullReferenceException is possible. Unhandled exceptions in destructor lead to
  termination of runtime.
* [V3101][539]. Potential resurrection of 'this' object instance from destructor. Without
  re-registering for finalization, destructor will not be called a second time on resurrected
  object.
* [V3102][540]. Suspicious access to element by a constant index inside a loop.
* [V3103][541]. A private Ctor(SerializationInfo, StreamingContext) constructor in unsealed type
  will not be accessible when deserializing derived types.
* [V3104][542]. The 'GetObjectData' implementation in unsealed type is not virtual, incorrect
  serialization of derived type is possible.
* [V3105][543]. The 'a' variable was used after it was assigned through null-conditional operator.
  NullReferenceException is possible.
* [V3106][544]. Possibly index is out of bound.
* [V3107][545]. Identical expression to the left and to the right of compound assignment.
* [V3108][546]. It is not recommended to return null or throw exceptions from 'ToString()' method.
* [V3109][547]. The same sub-expression is present on both sides of the operator. The expression is
  incorrect or it can be simplified.
* [V3110][548]. Possible infinite recursion.
* [V3111][549]. Checking value for null will always return false when generic type is instantiated
  with a value type.
* [V3112][550]. An abnormality within similar comparisons. It is possible that a typo is present
  inside the expression.
* [V3113][551]. Consider inspecting the loop expression. It is possible that different variables are
  used inside initializer and iterator.
* [V3114][552]. IDisposable object is not disposed before method returns.
* [V3115][553]. It is not recommended to throw exceptions from 'Equals(object obj)' method.
* [V3116][554]. Consider inspecting the 'for' operator. It's possible that the loop will be executed
  incorrectly or won't be executed at all.
* [V3117][555]. Constructor parameter is not used.
* [V3118][556]. A component of TimeSpan is used, which does not represent full time interval.
  Possibly 'Total*' value was intended instead.
* [V3119][557]. Calling a virtual (overridden) event may lead to unpredictable behavior. Consider
  implementing event accessors explicitly or use 'sealed' keyword.
* [V3120][558]. Potentially infinite loop. The variable from the loop exit condition does not change
  its value between iterations.
* [V3121][559]. An enumeration was declared with 'Flags' attribute, but does not set any
  initializers to override default values.
* [V3122][560]. Uppercase (lowercase) string is compared with a different lowercase (uppercase)
  string.
* [V3123][561]. Perhaps the '??' operator works in a different way than it was expected. Its
  priority is lower than priority of other operators in its left part.
* [V3124][562]. Appending an element and checking for key uniqueness is performed on two different
  variables.
* [V3125][563]. The object was used after it was verified against null. Check lines: N1, N2.
* [V3126][564]. Type implementing IEquatable<T> interface does not override 'GetHashCode' method.
* [V3127][565]. Two similar code fragments were found. Perhaps, this is a typo and 'X' variable
  should be used instead of 'Y'.
* [V3128][566]. The field (property) is used before it is initialized in constructor.
* [V3129][567]. The value of the captured variable will be overwritten on the next iteration of the
  loop in each instance of anonymous function that captures it.
* [V3130][568]. Priority of the '&&' operator is higher than that of the '||' operator. Possible
  missing parentheses.
* [V3131][569]. The expression is checked for compatibility with the type 'A', but is casted to the
  'B' type.
* [V3132][570]. A terminal null is present inside a string. The '\0xNN' characters were encountered.
  Probably meant: '\xNN'.
* [V3133][571]. Postfix increment/decrement is senseless because this variable is overwritten.
* [V3134][572]. Shift by N bits is greater than the size of type.
* [V3135][573]. The initial value of the index in the nested loop equals 'i'. Consider using 'i + 1'
  instead.
* [V3136][574]. Constant expression in switch statement.
* [V3137][575]. The variable is assigned but is not used by the end of the function.
* [V3138][576]. String literal contains potential interpolated expression.
* [V3139][577]. Two or more case-branches perform the same actions.
* [V3140][578]. Property accessors use different backing fields.
* [V3141][579]. Expression under 'throw' is a potential null, which can lead to
  NullReferenceException.
* [V3142][580]. Unreachable code detected. It is possible that an error is present.
* [V3143][581]. The 'value' parameter is rewritten inside a property setter, and is not used after
  that.
* [V3144][582]. This file is marked with copyleft license, which requires you to open the derived
  source code.
* [V3145][583]. Unsafe dereference of a WeakReference target. The object could have been garbage
  collected before the 'Target' property was accessed.
* [V3146][584]. Possible null dereference. A method can return default null value.
* [V3147][585]. Non-atomic modification of volatile variable.
* [V3148][586]. Casting potential 'null' value to a value type can lead to NullReferenceException.
* [V3149][587]. Dereferencing the result of 'as' operator can lead to NullReferenceException.
* [V3150][588]. Loop break conditions do not depend on the number of iterations.
* [V3151][589]. Potential division by zero. Variable was used as a divisor before it was compared to
  zero. Check lines: N1, N2.
* [V3152][590]. Potential division by zero. Variable was compared to zero before it was used as a
  divisor. Check lines: N1, N2.
* [V3153][591]. Dereferencing the result of null-conditional access operator can lead to
  NullReferenceException.
* [V3154][592]. The 'a % b' expression always evaluates to 0.
* [V3155][593]. The expression is incorrect or it can be simplified.
* [V3156][594]. The argument of the method is not expected to be null.
* [V3157][595]. Suspicious division. Absolute value of the left operand is less than the right
  operand.
* [V3158][596]. Suspicious division. Absolute values of both operands are equal.
* [V3159][597]. Modified value of the operand is not used after the increment/decrement operation.
* [V3160][598]. Argument of incorrect type is passed to the 'Enum.HasFlag' method.
* [V3161][599]. Comparing value type variables with 'ReferenceEquals' is incorrect because compared
  values will be boxed.
* [V3162][600]. Suspicious return of an always empty collection.
* [V3163][601]. An exception handling block does not contain any code.
* [V3164][602]. Exception classes should be publicly accessible.
* [V3165][603]. The expression of the 'char' type is passed as an argument of the 'A' type whereas
  similar overload with the string parameter exists.
* [V3166][604]. Calling the 'SingleOrDefault' method may lead to 'InvalidOperationException'.
* [V3167][605]. Parameter of 'CancellationToken' type is not used inside function's body.
* [V3168][606]. Awaiting on expression with potential null value can lead to throwing of
  'NullReferenceException'.
* [V3169][607]. Suspicious return of a local reference variable which always equals null.
* [V3170][608]. Both operands of the '??' operator are identical.
* [V3171][609]. Potentially negative value is used as the size of an array.
* [V3172][610]. The 'if/if-else/for/while/foreach' statement and code block after it are not
  related. Inspect the program's logic.
* [V3173][611]. Possible incorrect initialization of variable. Consider verifying the initializer.
* [V3174][612]. Suspicious subexpression in a sequence of similar comparisons.
* [V3175][613]. Locking operations must be performed on the same thread. Using 'await' in a critical
  section may lead to a lock being released on a different thread.
* [V3176][614]. The '&=' or '|=' operator is redundant because the right operand is always
  true/false.
* [V3177][615]. Logical literal belongs to second operator with a higher priority. It is possible
  literal was intended to belong to '??' operator instead.
* [V3178][616]. Calling method or accessing property of potentially disposed object may result in
  exception.
* [V3179][617]. Calling element access method for potentially empty collection may result in
  exception.
* [V3180][618]. The 'HasFlag' method always returns 'true' because the value '0' is passed as its
  argument.
* [V3181][619]. The result of '&' operator is '0' because one of the operands is '0'.
* [V3182][620]. The result of '&' operator is always '0'.
* [V3183][621]. Code formatting implies that the statement should not be a part of the 'then' branch
  that belongs to the preceding 'if' statement.
* [V3184][622]. The argument's value is greater than the size of the collection. Passing the value
  into the 'Foo' method will result in an exception.
* [V3185][623]. An argument containing a file path could be mixed up with another argument. The
  other function parameter expects a file path instead.
* [V3186][624]. The arguments violate the bounds of collection. Passing these values into the method
  will result in an exception.
* [V3187][625]. Parts of an SQL query are not delimited by any separators or whitespaces. Executing
  this query may lead to an error.
* [V3188][626]. Unity Engine. The value of an expression is a potentially destroyed Unity object or
  null. Member invocation on this value may lead to an exception.
* [V3189][627]. The assignment to a member of the readonly field will have no effect when the field
  is of a value type. Consider restricting the type parameter to reference types.
* [V3190][628]. Concurrent modification of a variable may lead to errors.
* [V3191][629]. Iteration through collection makes no sense because it is always empty.
* [V3192][630]. Type member is used in the 'GetHashCode' method but is missing from the 'Equals'
  method.
* [V3193][631]. Data processing results are potentially used before asynchronous output reading is
  complete. Consider calling 'WaitForExit' overload with no arguments before using the data.
* [V3194][632]. Calling 'OfType' for collection will return an empty collection. It is not possible
  to cast collection elements to the type parameter.
* [V3195][633]. Collection initializer implicitly calls 'Add' method. Using it on member with
  default value of null will result in null dereference exception.
* [V3196][634]. Parameter is not utilized inside the method body, but an identifier with a similar
  name is used inside the same method.
* [V3197][635]. The compared value inside the 'Object.Equals' override is converted to a different
  type that does not contain the override.
* [V3198][636]. The variable is assigned the same value that it already holds.
* [V3199][637]. The index from end operator is used with the value that is less than or equal to
  zero. Collection index will be out of bounds.
* [V3200][638]. Possible overflow. The expression will be evaluated before casting. Consider casting
  one of the operands instead.
* [V3201][639]. Return value is not always used. Consider inspecting the 'foo' method.
* [V3202][640]. Unreachable code detected. The 'case' value is out of the range of the match
  expression.
* [V3203][641]. Method parameter is not used.
* [V3204][642]. The expression is always false due to implicit type conversion. Overflow check is
  incorrect.
* [V3205][643]. Unity Engine. Improper creation of 'MonoBehaviour' or 'ScriptableObject' object
  using the 'new' operator. Use the special object creation method instead.
* [V3206][644]. Unity Engine. A direct call to the coroutine-like method will not start it. Use the
  'StartCoroutine' method instead.
* [V3207][645]. The 'not A or B' logical pattern may not work as expected. The 'not' pattern is
  matched only to the first expression from the 'or' pattern.
* [V3208][646]. Unity Engine. Using 'WeakReference' with 'UnityEngine.Object' is not supported. GC
  will not reclaim the object's memory because it is linked to a native object.
* [V3209][647]. Unity Engine. Using await on 'Awaitable' object more than once can lead to exception
  or deadlock, as such objects are returned to the pool after being awaited.
* [V3210][648]. Unity Engine. Unity does not allow removing the 'Transform' component using
  'Destroy' or 'DestroyImmediate' methods. The method call will be ignored.
* [V3211][649]. Unity Engine. The operators '?.', '??' and '??=' do not correctly handle destroyed
  objects derived from 'UnityEngine.Object'.
* [V3212][650]. Unity Engine. Pattern matching does not correctly handle destroyed objects derived
  from 'UnityEngine.Object'.
* [V3213][651]. Unity Engine. The 'GetComponent' method must be instantiated with a type that
  inherits from 'UnityEngine.Component'.
* [V3214][652]. Unity Engine. Using Unity API in the background thread may result in an error.
* [V3215][653]. Unity Engine. Passing a method name as a string literal into the 'StartCoroutine' is
  unreliable.
* [V3216][654]. Unity Engine. Checking a field with a specific Unity Engine type for null may not
  work correctly due to implicit field initialization by the engine.
* [V3217][655]. Possible overflow as a result of an arithmetic operation.
* [V3218][656]. Loop condition may be incorrect due to an off-by-one error.
* [V3219][657]. The variable was changed after it was captured in a LINQ method with deferred
  execution. The original value will not be used when the method is executed.
* [V3220][658]. The result of the LINQ method with deferred execution is never used. The method will
  not be executed.
* [V3221][659]. Modifying a collection during its enumeration will lead to an exception.
* [V3222][660]. Potential resource leak. An inner IDisposable object might remain non-disposed if
  the constructor of the outer object throws an exception.
* [V3223][661]. Inconsistent use of a potentially shared variable with and without a lock can lead
  to a data race.
* [V3224][662]. Consider using an overload with 'IEqualityComparer', as it is present in similar
  cases for the same collection element type.
* [V3225][663]. A data reading method returns the number of bytes that were read and cannot return
  the value of -1.
* [V3226][664]. Potential resource leak. The disposing method will not be called if an exception
  occurs in the 'try' block. Consider calling it in the 'finally' block.
* [V3227][665]. The precedence of the arithmetic operator is higher than that of the shift operator.
  Consider using parentheses in the expression.
* [V3228][666]. It is possible that an assigned variable should be used in the next condition.
  Consider checking for misprints.
* [V3229][667]. The 'GetHashCode' method may return different hash codes for equal objects. It uses
  an object reference to generate a hash for a variable. Check the implementation of the 'Equals'
  method.

## General Analysis (Java)

* [V6001][668]. There are identical sub-expressions to the left and to the right of the 'foo'
  operator.
* [V6002][669]. The switch statement does not cover all values of the enum.
* [V6003][670]. The use of 'if (A) {...} else if (A) {...}' pattern was detected. There is a
  probability of logical error presence.
* [V6004][671]. The 'then' statement is equivalent to the 'else' statement.
* [V6005][672]. The 'x' variable is assigned to itself.
* [V6006][673]. The object was created but it is not being used. The 'throw' keyword could be
  missing.
* [V6007][674]. Expression is always true/false.
* [V6008][675]. Potential null dereference.
* [V6009][676]. Function receives an odd argument.
* [V6010][677]. The return value of function 'Foo' is required to be utilized.
* [V6011][678]. The expression contains a suspicious mix of integer and real types.
* [V6012][679]. The '?:' operator, regardless of its conditional expression, always returns one and
  the same value.
* [V6013][680]. Comparison of arrays, strings, collections by reference. Possibly an equality
  comparison was intended.
* [V6014][681]. It's odd that this method always returns one and the same value of NN.
* [V6015][682]. Consider inspecting the expression. Probably the '!='/'-='/'+=' should be used here.
* [V6016][683]. Suspicious access to element by a constant index inside a loop.
* [V6017][684]. The 'X' counter is not used inside a nested loop. Consider inspecting usage of 'Y'
  counter.
* [V6018][685]. Constant expression in switch statement.
* [V6019][686]. Unreachable code detected. It is possible that an error is present.
* [V6020][687]. Division or mod division by zero.
* [V6021][688]. The value is assigned to the 'x' variable but is not used.
* [V6022][689]. Parameter is not used inside method's body.
* [V6023][690]. Parameter 'A' is always rewritten in method body before being used.
* [V6024][691]. The 'continue' operator will terminate 'do { ... } while (false)' loop because the
  condition is always false.
* [V6025][692]. Possibly index is out of bound.
* [V6026][693]. This value is already assigned to the 'b' variable.
* [V6027][694]. Variables are initialized through the call to the same function. It's probably an
  error or un-optimized code.
* [V6028][695]. Identical expression to the left and to the right of compound assignment.
* [V6029][696]. Possible incorrect order of arguments passed to method.
* [V6030][697]. The function located to the right of the '|' and '&' operators will be called
  regardless of the value of the left operand. Consider using '||' and '&&' instead.
* [V6031][698]. The variable 'X' is being used for this loop and for the outer loop.
* [V6032][699]. It is odd that the body of 'Foo_1' function is fully equivalent to the body of
  'Foo_2' function.
* [V6033][700]. An item with the same key has already been added.
* [V6034][701]. Shift by N bits is inconsistent with the size of type.
* [V6035][702]. Double negation is present in the expression: !!x.
* [V6036][703]. The value from the uninitialized optional is used.
* [V6037][704]. An unconditional 'break/continue/return/goto' within a loop.
* [V6038][705]. Comparison with 'double.NaN' is meaningless. Use 'double.isNaN()' method instead.
* [V6039][706]. There are two 'if' statements with identical conditional expressions. The first 'if'
  statement contains method return. This means that the second 'if' statement is senseless.
* [V6040][707]. The code's operational logic does not correspond with its formatting.
* [V6041][708]. Suspicious assignment inside the conditional expression of 'if/while/do...while'
  statement.
* [V6042][709]. The expression is checked for compatibility with type 'A', but is cast to type 'B'.
* [V6043][710]. Consider inspecting the 'for' operator. Initial and final values of the iterator are
  the same.
* [V6044][711]. Postfix increment/decrement is senseless because this variable is overwritten.
* [V6045][712]. Suspicious subexpression in a sequence of similar comparisons.
* [V6046][713]. Incorrect format. Consider checking the N format items of the 'Foo' function.
* [V6047][714]. It is possible that this 'else' branch must apply to the previous 'if' statement.
* [V6048][715]. This expression can be simplified. One of the operands in the operation equals NN.
  Probably it is a mistake.
* [V6049][716]. Classes that define 'equals' method must also define 'hashCode' method.
* [V6050][717]. Class initialization cycle is present.
* [V6051][718]. Use of jump statements in 'finally' block can lead to the loss of unhandled
  exceptions.
* [V6052][719]. Calling an overridden method in parent-class constructor may lead to use of
  uninitialized data.
* [V6053][720]. Collection is modified while iteration is in progress.
  ConcurrentModificationException may occur.
* [V6054][721]. Classes should not be compared by their name.
* [V6055][722]. Expression inside assert statement can change object's state.
* [V6056][723]. Implementation of 'compareTo' overloads the method from a base class. Possibly, an
  override was intended.
* [V6057][724]. Consider inspecting this expression. The expression is excessive or contains a
  misprint.
* [V6058][725]. Comparing objects of incompatible types.
* [V6059][726]. Odd use of special character in regular expression. Possibly, it was intended to be
  escaped.
* [V6060][727]. The reference was used before it was verified against null.
* [V6061][728]. The used constant value is represented by an octal form.
* [V6062][729]. Possible infinite recursion.
* [V6063][730]. Odd semicolon ';' after 'if/for/while' operator.
* [V6064][731]. Suspicious invocation of Thread.run().
* [V6065][732]. A non-serializable class should not be serialized.
* [V6066][733]. Passing objects of incompatible types to the method of collection.
* [V6067][734]. Two or more case-branches perform the same actions.
* [V6068][735]. Suspicious use of BigDecimal class.
* [V6069][736]. Unsigned right shift assignment of negative 'byte' / 'short' value.
* [V6070][737]. Unsafe synchronization on an object.
* [V6071][738]. This file is marked with copyleft license, which requires you to open the derived
  source code.
* [V6072][739]. Two similar code fragments were found. Perhaps, this is a typo and 'X' variable
  should be used instead of 'Y'.
* [V6073][740]. It is not recommended to return null or throw exceptions from 'toString' / 'clone'
  methods.
* [V6074][741]. Non-atomic modification of volatile variable.
* [V6075][742]. The signature of method 'X' does not conform to serialization requirements.
* [V6076][743]. Recurrent serialization will use cached object state from first serialization.
* [V6077][744]. A suspicious label is present inside a switch(). It is possible that these are
  misprints and 'default:' label should be used instead.
* [V6078][745]. Potential Java SE API compatibility issue.
* [V6079][746]. Value of variable is checked after use. Potential logical error is present. Check
  lines: N1, N2.
* [V6080][747]. Consider checking for misprints. It's possible that an assigned variable should be
  checked in the next condition.
* [V6081][748]. Annotation that does not have 'RUNTIME' retention policy will not be accessible
  through Reflection API.
* [V6082][749]. Unsafe double-checked locking.
* [V6083][750]. Serialization order of fields should be preserved during deserialization.
* [V6084][751]. Suspicious return of an always empty collection.
* [V6085][752]. An abnormality within similar comparisons. It is possible that a typo is present
  inside the expression.
* [V6086][753]. Suspicious code formatting. 'else' keyword is probably missing.
* [V6087][754]. InvalidClassException may occur during deserialization.
* [V6088][755]. Result of this expression will be implicitly cast to 'Type'. Check if program logic
  handles it correctly.
* [V6089][756]. It's possible that the line was commented out improperly, thus altering the
  program's operation logics.
* [V6090][757]. Field 'A' is being used before it was initialized.
* [V6091][758]. Suspicious getter/setter implementation. The 'A' field should probably be
  returned/assigned instead.
* [V6092][759]. A resource is returned from try-with-resources statement. It will be closed before
  the method exits.
* [V6093][760]. Automatic unboxing of a variable may cause NullPointerException.
* [V6094][761]. The expression was implicitly cast from integer type to real type. Consider
  utilizing an explicit type cast to avoid the loss of a fractional part.
* [V6095][762]. Thread.sleep() inside synchronized block/method may cause decreased performance.
* [V6096][763]. An odd precise comparison. Consider using a comparison with defined precision:
  Math.abs(A - B) < Epsilon or Math.abs(A - B) > Epsilon.
* [V6097][764]. Lowercase 'L' at the end of a long literal can be mistaken for '1'.
* [V6098][765]. The method does not override another method from the base class.
* [V6099][766]. The initial value of the index in the nested loop equals 'i'. Consider using 'i + 1'
  instead.
* [V6100][767]. An object is used as an argument to its own method. Consider checking the first
  actual argument of the 'Foo' method.
* [V6101][768]. compareTo()-like methods can return not only the values -1, 0 and 1, but any values.
* [V6102][769]. Inconsistent synchronization of a field. Consider synchronizing the field on all
  usages.
* [V6103][770]. Ignored InterruptedException could lead to delayed thread shutdown.
* [V6104][771]. A pattern was detected: A || (A && ...). The expression is excessive or contains a
  logical error.
* [V6105][772]. Consider inspecting the loop expression. It is possible that different variables are
  used inside initializer and iterator.
* [V6106][773]. Casting expression to 'X' type before implicitly casting it to other type may be
  excessive or incorrect.
* [V6107][774]. The constant NN is being utilized. The resulting value could be inaccurate. Consider
  using the KK constant.
* [V6108][775]. Do not use real-type variables in 'for' loop counters.
* [V6109][776]. Potentially predictable seed is used in pseudo-random number generator.
* [V6110][777]. Using an environment variable could be unsafe or unreliable. Consider using trusted
  system property instead
* [V6111][778]. Potentially negative value is used as the size of an array.
* [V6112][779]. Calling the 'getClass' method repeatedly or on the value of the '.class' literal
  will always return the instance of the 'Class<Class>' type.
* [V6113][780]. Suspicious division. Absolute value of the left operand is less than the value of
  the right operand.
* [V6114][781]. The 'A' class containing Closeable members does not release the resources that the
  field is holding.
* [V6115][782]. Not all Closeable members are released inside the 'close' method.
* [V6116][783]. The class does not implement the Closeable interface, but it contains the 'close'
  method that releases resources.
* [V6117][784]. Possible overflow. The expression will be evaluated before casting. Consider casting
  one of the operands instead.
* [V6118][785]. The original exception object was swallowed. Cause of original exception could be
  lost.
* [V6119][786]. The result of '&' operator is always '0'.
* [V6120][787]. The result of the '&' operator is '0' because one of the operands is '0'.
* [V6121][788]. Return value is not always used. Consider inspecting the 'foo' method.
* [V6122][789]. The 'Y' (week year) pattern is used for date formatting. Check whether the 'y'
  (year) pattern was intended instead.
* [V6123][790]. Modified value of the operand is not used after the increment/decrement operation.
* [V6124][791]. Converting an integer literal to the type with a smaller value range will result in
  overflow.
* [V6125][792]. Calling the 'wait', 'notify', and 'notifyAll' methods outside of synchronized
  context will lead to 'IllegalMonitorStateException'.
* [V6126][793]. Native synchronization used on high-level concurrency class.
* [V6127][794]. Closeable object is not closed. This may lead to a resource leak.
* [V6128][795]. Using a Closeable object after it was closed can lead to an exception.
* [V6129][796]. Possible deadlock due to incorrect synchronization order between locks.
* [V6130][797]. Integer overflow in arithmetic expression.
* [V6131][798]. Casting to a type with a smaller range will result in an overflow.
* [V6132][799]. It is possible that 'else' block was forgotten or commented out, thus altering the
  program's operation logics.

## Micro-Optimizations (C++)

* [V801][800]. Decreased performance. It is better to redefine the N function argument as a
  reference. Consider replacing 'const T' with 'const .. &T' / 'const .. *T'.
* [V802][801]. On 32-bit/64-bit platform, structure size can be reduced from N to K bytes by
  rearranging the fields according to their sizes in decreasing order.
* [V803][802]. Decreased performance. It is more effective to use the prefix form of ++it. Replace
  iterator++ with ++iterator.
* [V804][803]. Decreased performance. The 'Foo' function is called twice in the specified expression
  to calculate length of the same string.
* [V805][804]. Decreased performance. It is inefficient to identify an empty string by using
  'strlen(str) > 0' construct. A more efficient way is to check: str[0] != '\0'.
* [V806][805]. Decreased performance. The expression of strlen(MyStr.c_str()) kind can be rewritten
  as MyStr.length().
* [V807][806]. Decreased performance. Consider creating a pointer/reference to avoid using the same
  expression repeatedly.
* [V808][807]. An array/object was declared but was not utilized.
* [V809][808]. Verifying that a pointer value is not NULL is not required. The 'if (ptr != NULL)'
  check can be removed.
* [V810][809]. Decreased performance. The 'A' function was called several times with identical
  arguments. The result should possibly be saved to a temporary variable, which then could be used
  while calling the 'B' function.
* [V811][810]. Decreased performance. Excessive type casting: string -> char * -> string.
* [V812][811]. Decreased performance. Ineffective use of the 'count' function. It can possibly be
  replaced by the call to the 'find' function.
* [V813][812]. Decreased performance. The argument should probably be rendered as a constant
  pointer/reference.
* [V814][813]. Decreased performance. The 'strlen' function was called multiple times inside the
  body of a loop.
* [V815][814]. Decreased performance. Consider replacing the expression 'AA' with 'BB'.
* [V816][815]. It is more efficient to catch exception by reference rather than by value.
* [V817][816]. It is more efficient to search for 'X' character rather than a string.
* [V818][817]. It is more efficient to use an initialization list rather than an assignment
  operator.
* [V819][818]. Decreased performance. Memory is allocated and released multiple times inside the
  loop body.
* [V820][819]. The variable is not used after copying. Copying can be replaced with move/swap for
  optimization.
* [V821][820]. The variable can be constructed in a lower level scope.
* [V822][821]. Decreased performance. A new object is created, while a reference to an object is
  expected.
* [V823][822]. Decreased performance. Object may be created in-place in a container. Consider
  replacing methods: 'insert' -> 'emplace', 'push_*' -> 'emplace_*'.
* [V824][823]. It is recommended to use the 'make_unique/make_shared' function to create smart
  pointers.
* [V825][824]. Expression is equivalent to moving one unique pointer to another. Consider using
  'std::move' instead.
* [V826][825]. Consider replacing standard container with a different one.
* [V827][826]. Maximum size of a vector is known at compile time. Consider pre-allocating it by
  calling reserve(N).
* [V828][827]. Decreased performance. Moving an object in a return statement prevents copy elision.
* [V829][828]. Lifetime of the heap-allocated variable is limited to the current function's scope.
  Consider allocating it on the stack instead.
* [V830][829]. Decreased performance. Consider replacing the use of 'std::optional::value()' with
  either the '*' or '->' operator.
* [V831][830]. Decreased performance. Consider replacing the call to the 'at()' method with the
  'operator[]'.
* [V832][831]. It's better to use '= default;' syntax instead of empty body.
* [V833][832]. Using 'std::move' function's with const object disables move semantics.
* [V834][833]. Incorrect type of a loop variable. This leads to the variable binding to a temporary
  object instead of a range element.
* [V835][834]. Passing cheap-to-copy argument by reference may lead to decreased performance.
* [V836][835]. Expression's value is copied at the variable declaration. The variable is never
  modified. Consider declaring it as a reference.
* [V837][836]. The 'emplace' / 'insert' function does not guarantee that arguments will not be
  copied or moved if there is no insertion. Consider using the 'try_emplace' function.
* [V838][837]. Temporary object is constructed during lookup in ordered associative container.
  Consider using a container with heterogeneous lookup to avoid construction of temporary objects.
* [V839][838]. Function returns a constant value. This may interfere with move semantics.

## Micro-Optimizations (C#)

* [V4001][839]. Unity Engine. Boxing inside a frequently called method may decrease performance.
* [V4002][840]. Unity Engine. Avoid storing consecutive concatenations inside a single string in
  performance-sensitive context. Consider using StringBuilder to improve performance.
* [V4003][841]. Unity Engine. Avoid capturing variable in performance-sensitive context. This can
  lead to decreased performance.
* [V4004][842]. Unity Engine. New array object is returned from method or property. Using such
  member in performance-sensitive context can lead to decreased performance.
* [V4005][843]. Unity Engine. The expensive operation is performed inside method or property. Using
  such member in performance-sensitive context can lead to decreased performance.
* [V4006][844]. Unity Engine. Multiple operations between complex and numeric values. Prioritizing
  operations between numeric values can optimize execution time.
* [V4007][845]. Unity Engine. Avoid creating and destroying UnityEngine objects in
  performance-sensitive context. Consider activating and deactivating them instead.
* [V4008][846]. Unity Engine. Avoid using memory allocation Physics APIs in performance-sensitive
  context.

## Diagnosis of 64-bit errors (Viva64, C++)

* [V101][847]. Implicit assignment type conversion to memsize type.
* [V102][848]. Usage of non memsize type for pointer arithmetic.
* [V103][849]. Implicit type conversion from memsize type to 32-bit type.
* [V104][850]. Implicit type conversion to memsize type in an arithmetic expression.
* [V105][851]. N operand of '?:' operation: implicit type conversion to memsize type.
* [V106][852]. Implicit type conversion N argument of function 'foo' to memsize type.
* [V107][853]. Implicit type conversion N argument of function 'foo' to 32-bit type.
* [V108][854]. Incorrect index type: 'foo[not a memsize-type]'. Use memsize type instead.
* [V109][855]. Implicit type conversion of return value to memsize type.
* [V110][856]. Implicit type conversion of return value from memsize type to 32-bit type.
* [V111][857]. Call of function 'foo' with variable number of arguments. N argument has memsize
  type.
* [V112][858]. Dangerous magic number N used.
* [V113][859]. Implicit type conversion from memsize to double type or vice versa.
* [V114][860]. Dangerous explicit type pointer conversion.
* [V115][861]. Memsize type is used for throw.
* [V116][862]. Memsize type is used for catch.
* [V117][863]. Memsize type is used in the union.
* [V118][864]. malloc() function accepts a dangerous expression in the capacity of an argument.
* [V119][865]. More than one sizeof() operator is used in one expression.
* [V120][866]. Member operator[] of object 'foo' is declared with 32-bit type argument, but is
  called with memsize type argument.
* [V121][867]. Implicit conversion of the type of 'new' operator's argument to size_t type.
* [V122][868]. Memsize type is used in the struct/class.
* [V123][869]. Allocation of memory by the pattern "(X*)malloc(sizeof(Y))" where the sizes of X and
  Y types are not equal.
* [V124][870]. Function 'Foo' writes/reads 'N' bytes. The alignment rules and type sizes have been
  changed. Consider reviewing this value.
* [V125][871]. It is not advised to declare type 'T' as 32-bit type.
* [V126][872]. Be advised that the size of the type 'long' varies between LLP64/LP64 data models.
* [V127][873]. An overflow of the 32-bit variable is possible inside a long cycle which utilizes a
  memsize-type loop counter.
* [V128][874]. A variable of the memsize type is read from a stream. Consider verifying the
  compatibility of 32 and 64 bit versions of the application in the context of a stored data.
* [V201][875]. Explicit conversion from 32-bit integer type to memsize type.
* [V202][876]. Explicit conversion from memsize type to 32-bit integer type.
* [V203][877]. Explicit type conversion from memsize to double type or vice versa.
* [V204][878]. Explicit conversion from 32-bit integer type to pointer type.
* [V205][879]. Explicit conversion of pointer type to 32-bit integer type.
* [V206][880]. Explicit conversion from 'void *' to 'int *'.
* [V207][881]. A 32-bit variable is utilized as a reference to a pointer. A write outside the bounds
  of this variable may occur.
* [V220][882]. Suspicious sequence of types castings: memsize -> 32-bit integer -> memsize.
* [V221][883]. Suspicious sequence of types castings: pointer -> memsize -> 32-bit integer.
* [V301][884]. Unexpected function overloading behavior. See N argument of function 'foo' in derived
  class 'derived' and base class 'base'.
* [V302][885]. Member operator[] of 'foo' class has a 32-bit type argument. Use memsize-type here.
* [V303][886]. The function is deprecated in the Win64 system. It is safer to use the 'foo'
  function.

## Customer specific requests (C++)

* [V2001][887]. Consider using the extended version of the 'foo' function here.
* [V2002][888]. Consider using the 'Ptr' version of the 'foo' function here.
* [V2003][889]. Explicit conversion from 'float/double' type to signed integer type.
* [V2004][890]. Explicit conversion from 'float/double' type to unsigned integer type.
* [V2005][891]. C-style explicit type casting is utilized. Consider using:
  static_cast/const_cast/reinterpret_cast.
* [V2006][892]. Implicit type conversion from enum type to integer type.
* [V2007][893]. This expression can be simplified. One of the operands in the operation equals NN.
  Probably it is a mistake.
* [V2008][894]. Cyclomatic complexity: NN. Consider refactoring the 'Foo' function.
* [V2009][895]. Consider passing the 'Foo' argument as a pointer/reference to const.
* [V2010][896]. Handling of two different exception types is identical.
* [V2011][897]. Consider inspecting signed and unsigned function arguments. See NN argument of
  function 'Foo' in derived class and base class.
* [V2012][898]. Possibility of decreased performance. It is advised to pass arguments to
  std::unary_function/std::binary_function template as references.
* [V2013][899]. Consider inspecting the correctness of handling the N argument in the 'Foo'
  function.
* [V2014][900]. Don't use terminating functions in library code.
* [V2015][901]. An identifier declared in an inner scope should not hide an identifier in an outer
  scope.
* [V2016][902]. Consider inspecting the function call. The function was annotated as dangerous.
* [V2017][903]. String literal is identical to variable name. It is possible that the variable
  should be used instead of the string literal.
* [V2018][904]. Cast should not remove 'const' qualifier from the type that is pointed to by a
  pointer or a reference.
* [V2019][905]. Cast should not remove 'volatile' qualifier from the type that is pointed to by a
  pointer or a reference.
* [V2020][906]. The loop body contains the 'break;' / 'continue;' statement. This may complicate the
  control flow.
* [V2021][907]. Using assertions may cause the abnormal program termination in undesirable contexts.
* [V2022][908]. Implicit type conversion from integer type to enum type.
* [V2023][909]. Absence of the 'override' specifier when overriding a virtual function may cause a
  mismatch of signatures.

## MISRA errors

* [V2501][910]. MISRA. Octal constants should not be used.
* [V2502][911]. MISRA. The 'goto' statement should not be used.
* [V2503][912]. MISRA. Implicitly specified enumeration constants should be unique – consider
  specifying non-unique constants explicitly.
* [V2504][913]. MISRA. Size of an array is not specified.
* [V2505][914]. MISRA. The 'goto' statement shouldn't jump to a label declared earlier.
* [V2506][915]. MISRA. A function should have a single point of exit at the end.
* [V2507][916]. MISRA. The body of a loop\conditional statement should be enclosed in braces.
* [V2508][917]. MISRA. The function with the 'atof/atoi/atol/atoll' name should not be used.
* [V2509][918]. MISRA. The function with the 'abort/exit/getenv/system' name should not be used.
* [V2510][919]. MISRA. The function with the 'qsort/bsearch' name should not be used.
* [V2511][920]. MISRA. Memory allocation and deallocation functions should not be used.
* [V2512][921]. MISRA. The macro with the 'setjmp' name and the function with the 'longjmp' name
  should not be used.
* [V2513][922]. MISRA. Unbounded functions performing string operations should not be used.
* [V2514][923]. MISRA. Unions should not be used.
* [V2515][924]. MISRA. Declaration should contain no more than two levels of pointer nesting.
* [V2516][925]. MISRA. The 'if' ... 'else if' construct should be terminated with an 'else'
  statement.
* [V2517][926]. MISRA. Literal suffixes should not contain lowercase characters.
* [V2518][927]. MISRA. The 'default' label should be either the first or the last label of a
  'switch' statement.
* [V2519][928]. MISRA. Every 'switch' statement should have a 'default' label, which, in addition to
  the terminating 'break' statement, should contain either a statement or a comment.
* [V2520][929]. MISRA. Every switch-clause should be terminated by an unconditional 'break' or
  'throw' statement.
* [V2521][930]. MISRA. Only the first member of enumerator list should be explicitly initialized,
  unless all members are explicitly initialized.
* [V2522][931]. MISRA. The 'switch' statement should have 'default' as the last label.
* [V2523][932]. MISRA. All integer constants of unsigned type should have 'u' or 'U' suffix.
* [V2524][933]. MISRA. A switch-label should only appear at the top level of the compound statement
  forming the body of a 'switch' statement.
* [V2525][934]. MISRA. Every 'switch' statement should contain non-empty switch-clauses.
* [V2526][935]. MISRA. The functions from time.h/ctime should not be used.
* [V2527][936]. MISRA. A switch-expression should not have Boolean type. Consider using of 'if-else'
  construct.
* [V2528][937]. MISRA. The comma operator should not be used.
* [V2529][938]. MISRA. Any label should be declared in the same block as 'goto' statement or in any
  block enclosing it.
* [V2530][939]. MISRA. Any loop should be terminated with no more than one 'break' or 'goto'
  statement.
* [V2531][940]. MISRA. Expression of essential type 'foo' should not be explicitly cast to essential
  type 'bar'.
* [V2532][941]. MISRA. String literal should not be assigned to object unless it has type of pointer
  to const-qualified char.
* [V2533][942]. MISRA. C-style and functional notation casts should not be performed.
* [V2534][943]. MISRA. The loop counter should not have floating-point type.
* [V2535][944]. MISRA. Unreachable code should not be present in the project.
* [V2536][945]. MISRA. Function should not contain labels not used by any 'goto' statements.
* [V2537][946]. MISRA. Functions should not have unused parameters.
* [V2538][947]. MISRA. The value of uninitialized variable should not be used.
* [V2539][948]. MISRA. Class destructor should not exit with an exception.
* [V2540][949]. MISRA. Arrays should not be partially initialized.
* [V2541][950]. MISRA. Function should not be declared implicitly.
* [V2542][951]. MISRA. Function with a non-void return type should return a value from all exit
  paths.
* [V2543][952]. MISRA. Value of the essential character type should be used appropriately in the
  addition/subtraction operations.
* [V2544][953]. MISRA. The values used in expressions should have appropriate essential types.
* [V2545][954]. MISRA. Conversion between pointers of different object types should not be
  performed.
* [V2546][955]. MISRA. Expression resulting from the macro expansion should be surrounded by
  parentheses.
* [V2547][956]. MISRA. The return value of non-void function should be used.
* [V2548][957]. MISRA. The address of an object with local scope should not be passed out of its
  scope.
* [V2549][958]. MISRA. Pointer to FILE should not be dereferenced.
* [V2550][959]. MISRA. Floating-point values should not be tested for equality or inequality.
* [V2551][960]. MISRA. Variable should be declared in a scope that minimizes its visibility.
* [V2552][961]. MISRA. Expressions with enum underlying type should have values corresponding to the
  enumerators of the enumeration.
* [V2553][962]. MISRA. Unary minus operator should not be applied to an expression of the unsigned
  type.
* [V2554][963]. MISRA. Expression containing increment (++) or decrement (--) should not have other
  side effects.
* [V2555][964]. MISRA. Incorrect shifting expression.
* [V2556][965]. MISRA. Use of a pointer to FILE when the associated stream has already been closed.
* [V2557][966]. MISRA. Operand of sizeof() operator should not have other side effects.
* [V2558][967]. MISRA. A pointer/reference parameter in a function should be declared as
  pointer/reference to const if the corresponding object was not modified.
* [V2559][968]. MISRA. Subtraction, >, >=, <, <= should be applied only to pointers that address
  elements of the same array.
* [V2560][969]. MISRA. There should be no user-defined variadic functions.
* [V2561][970]. MISRA. The result of an assignment expression should not be used.
* [V2562][971]. MISRA. Expressions with pointer type should not be used in the '+', '-', '+=' and
  '-=' operations.
* [V2563][972]. MISRA. Array indexing should be the only form of pointer arithmetic and it should be
  applied only to objects defined as an array type.
* [V2564][973]. MISRA. There should be no implicit integral-floating conversion.
* [V2565][974]. MISRA. A function should not call itself either directly or indirectly.
* [V2566][975]. MISRA. Constant expression evaluation should not result in an unsigned integer
  wrap-around.
* [V2567][976]. MISRA. Cast should not remove 'const' / 'volatile' qualification from the type that
  is pointed to by a pointer or a reference.
* [V2568][977]. MISRA. Both operands of an operator should be of the same type category.
* [V2569][978]. MISRA. The 'operator &&', 'operator ||', 'operator ,' and the unary 'operator &'
  should not be overloaded.
* [V2570][979]. MISRA. Operands of the logical '&&' or the '||' operators, the '!' operator should
  have 'bool' type.
* [V2571][980]. MISRA. Conversions between pointers to objects and integer types should not be
  performed.
* [V2572][981]. MISRA. Value of the expression should not be converted to the different essential
  type or the narrower essential type.
* [V2573][982]. MISRA. Identifiers that start with '__' or '_[A-Z]' are reserved.
* [V2574][983]. MISRA. Functions should not be declared at block scope.
* [V2575][984]. MISRA. The global namespace should only contain 'main', namespace declarations and
  'extern "C"' declarations.
* [V2576][985]. MISRA. The identifier 'main' should not be used for a function other than the global
  function 'main'.
* [V2577][986]. MISRA. The function argument corresponding to a parameter declared to have an array
  type should have an appropriate number of elements.
* [V2578][987]. MISRA. An identifier with array type passed as a function argument should not decay
  to a pointer.
* [V2579][988]. MISRA. Macro should not be defined with the same name as a keyword.
* [V2580][989]. MISRA. The 'restrict' specifier should not be used.
* [V2581][990]. MISRA. Single-line comments should not end with a continuation token.
* [V2582][991]. MISRA. Block of memory should only be freed if it was allocated by a Standard
  Library function.
* [V2583][992]. MISRA. Line whose first token is '#' should be a valid preprocessing directive.
* [V2584][993]. MISRA. Expression used in condition should have essential Boolean type.
* [V2585][994]. MISRA. Casts between a void pointer and an arithmetic type should not be performed.
* [V2586][995]. MISRA. Flexible array members should not be declared.
* [V2587][996]. MISRA. The '//' and '/*' character sequences should not appear within comments.
* [V2588][997]. MISRA. All memory or resources allocated dynamically should be explicitly released.
* [V2589][998]. MISRA. Casts between a pointer and a non-integer arithmetic type should not be
  performed.
* [V2590][999]. MISRA. Conversions should not be performed between pointer to function and any other
  type.
* [V2591][1000]. MISRA. Bit fields should only be declared with explicitly signed or unsigned
  integer type
* [V2592][1001]. MISRA. An identifier declared in an inner scope should not hide an identifier in an
  outer scope.
* [V2593][1002]. MISRA. Single-bit bit fields should not be declared as signed type.
* [V2594][1003]. MISRA. Controlling expressions should not be invariant.
* [V2595][1004]. MISRA. Array size should be specified explicitly when array declaration uses
  designated initialization.
* [V2596][1005]. MISRA. The value of a composite expression should not be assigned to an object with
  wider essential type.
* [V2597][1006]. MISRA. Cast should not convert pointer to function to any other pointer type.
* [V2598][1007]. MISRA. Variable length array types are not allowed.
* [V2599][1008]. MISRA. The standard signal handling functions should not be used.
* [V2600][1009]. MISRA. The standard input/output functions should not be used.
* [V2601][1010]. MISRA. Functions should be declared in prototype form with named parameters.
* [V2602][1011]. MISRA. Octal and hexadecimal escape sequences should be terminated.
* [V2603][1012]. MISRA. The 'static' keyword shall not be used between [] in the declaration of an
  array parameter.
* [V2604][1013]. MISRA. Features from <stdarg.h> should not be used.
* [V2605][1014]. MISRA. Features from <tgmath.h> should not be used.
* [V2606][1015]. MISRA. There should be no attempt to write to a stream that has been opened for
  reading.
* [V2607][1016]. MISRA. Inline functions should be declared with the static storage class.
* [V2608][1017]. MISRA. The 'static' storage class specifier should be used in all declarations of
  object and functions that have internal linkage.
* [V2609][1018]. MISRA. There should be no occurrence of undefined or critical unspecified
  behaviour.
* [V2610][1019]. MISRA. The ', " or \ characters and the /* or // character sequences should not
  occur in a header file name.
* [V2611][1020]. MISRA. Casts between a pointer to an incomplete type and any other type shouldn't
  be performed.
* [V2612][1021]. MISRA. Array element should not be initialized more than once.
* [V2613][1022]. MISRA. Operand that is a composite expression has more narrow essential type than
  the other operand.
* [V2614][1023]. MISRA. External identifiers should be distinct.
* [V2615][1024]. MISRA. A compatible declaration should be visible when an object or function with
  external linkage is defined.
* [V2616][1025]. MISRA. All conditional inclusion preprocessor directives should reside in the same
  file as the conditional inclusion directive to which they are related.
* [V2617][1026]. MISRA. Object should not be assigned or copied to an overlapping object.
* [V2618][1027]. MISRA. Identifiers declared in the same scope and name space should be distinct.
* [V2619][1028]. MISRA. Typedef names should be unique across all name spaces.
* [V2620][1029]. MISRA. Value of a composite expression should not be cast to a different essential
  type category or a wider essential type.
* [V2621][1030]. MISRA. Tag names should be unique across all name spaces.
* [V2622][1031]. MISRA. External object or function should be declared once in one and only one
  file.
* [V2623][1032]. MISRA. Macro identifiers should be distinct.
* [V2624][1033]. MISRA. The initializer for an aggregate or union should be enclosed in braces.
* [V2625][1034]. MISRA. Identifiers that define objects or functions with external linkage shall be
  unique.
* [V2626][1035]. MISRA. The 'sizeof' operator should not have an operand which is a function
  parameter declared as 'array of type'.
* [V2627][1036]. MISRA. Function type should not be type qualified.
* [V2628][1037]. MISRA. The pointer arguments to the Standard Library functions memcpy, memmove and
  memcmp should be pointers to qualified or unqualified versions of compatible types.
* [V2629][1038]. MISRA. Pointer arguments to the 'memcmp' function should point to an appropriate
  type.
* [V2630][1039]. MISRA. Bit field should not be declared as a member of a union.
* [V2631][1040]. MISRA. Pointers to variably-modified array types should not be used.
* [V2632][1041]. MISRA. Object with temporary lifetime should not undergo array-to-pointer
  conversion.
* [V2633][1042]. MISRA. Identifiers should be distinct from macro names.
* [V2634][1043]. MISRA. Features from <fenv.h> should not be used.
* [V2635][1044]. MISRA. The function with the 'system' name should not be used.
* [V2636][1045]. MISRA. The functions with the 'rand' and 'srand' name of <stdlib.h> should not be
  used.
* [V2637][1046]. MISRA. A 'noreturn' function should have 'void' return type.
* [V2638][1047]. MISRA. Generic association should list an appropriate type.
* [V2639][1048]. MISRA. Default association should appear as either the first or the last
  association of a generic selection.
* [V2640][1049]. MISRA. Thread objects, thread synchronization objects and thread-specific storage
  pointers should have appropriate storage duration.
* [V2641][1050]. MISRA. Types should be explicitly specified.
* [V2642][1051]. MISRA. The '_Atomic' specifier should not be applied to the incomplete type 'void'.
* [V2643][1052]. MISRA. All memory synchronization operation should be executed in sequentially
  consistent order.
* [V2644][1053]. MISRA. Controlling expression of generic selection must not have side effects.
* [V2645][1054]. MISRA. The language features specified in Annex K should not be used.
* [V2646][1055]. MISRA. All arguments of any multi-argument type-generic macros from <tgmath.h>
  should have the same type.
* [V2647][1056]. MISRA. Structure and union members of atomic objects should not be directly
  accessed.
* [V2648][1057]. MISRA. Null pointer constant must be derived by expansion of the NULL macro
  provided by the implementation.
* [V2649][1058]. MISRA. All arguments of any type-generic macros from <tgmath.h> should have an
  appropriate essential type.
* [V2650][1059]. MISRA. Controlling expression of generic selection should have essential type that
  matches its standard type
* [V2651][1060]. MISRA. Initializer using chained designators should not contain initializers
  without designators.
* [V2652][1061]. MISRA. Argument of an integer constant macro should have an appropriate form.
* [V2653][1062]. MISRA. The small integer variants of the minimum-width integer constant macros
  should not be used.
* [V2654][1063]. MISRA. Initializer list should not contain persistent side effects.
* [V2655][1064]. MISRA. The right operand of a logical '&&' or '||' operator should not contain
  persistent side effects.
* [V2656][1065]. MISRA. Standard Library function 'memcmp' should not be used to compare
  null-terminated strings.
* [V2657][1066]. MISRA. Obsolescent language features should not be used.
* [V2658][1067]. MISRA. Dead code should not be used in a project.
* [V2659][1068]. MISRA. Switch statements should be well-formed.
* [V2660][1069]. MISRA. A function declared with a _Noreturn specifier should not return to its
  caller.
* [V2661][1070]. MISRA. A 'for' loop should be well-formed.
* [V2662][1071]. MISRA. Any value passed to a function from <ctype.h> should be representable as an
  unsigned character or be the value EOF.
* [V2663][1072]. MISRA. The macro EOF should only be compared with the unmodified return value of
  any Standard Library function capable of returning EOF.
* [V2664][1073]. MISRA. Use of the string handling functions from <string.h> should not result in
  accesses beyond the bounds of the objects referenced by their pointer parameters.
* [V2665][1074]. MISRA. The size argument passed to function from <string.h> should have an
  appropriate value.
* [V2666][1075]. MISRA. All declarations of an object with an explicit alignment specification
  should specify the same alignment.

## AUTOSAR errors

* [V3501][1076]. AUTOSAR. Octal constants should not be used.
* [V3502][1077]. AUTOSAR. Size of an array is not specified.
* [V3503][1078]. AUTOSAR. The 'goto' statement shouldn't jump to a label declared earlier.
* [V3504][1079]. AUTOSAR. The body of a loop\conditional statement should be enclosed in braces.
* [V3505][1080]. AUTOSAR. The function with the 'atof/atoi/atol/atoll' name should not be used.
* [V3506][1081]. AUTOSAR. The function with the 'abort/exit/getenv/system' name should not be used.
* [V3507][1082]. AUTOSAR. The macro with the 'setjmp' name and the function with the 'longjmp' name
  should not be used.
* [V3508][1083]. AUTOSAR. Unbounded functions performing string operations should not be used.
* [V3509][1084]. AUTOSAR. Unions should not be used.
* [V3510][1085]. AUTOSAR. Declaration should contain no more than two levels of pointer nesting.
* [V3511][1086]. AUTOSAR. The 'if' ... 'else if' construct should be terminated with an 'else'
  statement.
* [V3512][1087]. AUTOSAR. Literal suffixes should not contain lowercase characters.
* [V3513][1088]. AUTOSAR. Every switch-clause should be terminated by an unconditional 'break' or
  'throw' statement.
* [V3514][1089]. AUTOSAR. The 'switch' statement should have 'default' as the last label.
* [V3515][1090]. AUTOSAR. All integer constants of unsigned type should have 'U' suffix.
* [V3516][1091]. AUTOSAR. A switch-label should only appear at the top level of the compound
  statement forming the body of a 'switch' statement.
* [V3517][1092]. AUTOSAR. The functions from time.h/ctime should not be used.
* [V3518][1093]. AUTOSAR. A switch-expression should not have Boolean type. Consider using of
  'if-else' construct.
* [V3519][1094]. AUTOSAR. The comma operator should not be used.
* [V3520][1095]. AUTOSAR. Any label should be declared in the same block as 'goto' statement or in
  any block enclosing it.
* [V3521][1096]. AUTOSAR. The loop counter should not have floating-point type.
* [V3522][1097]. AUTOSAR. Unreachable code should not be present in the project.
* [V3523][1098]. AUTOSAR. Functions should not have unused parameters.
* [V3524][1099]. AUTOSAR. The value of uninitialized variable should not be used.
* [V3525][1100]. AUTOSAR. Function with a non-void return type should return a value from all exit
  paths.
* [V3526][1101]. AUTOSAR. Expression resulting from the macro expansion should be surrounded by
  parentheses.
* [V3527][1102]. AUTOSAR. The return value of non-void function should be used.
* [V3528][1103]. AUTOSAR. The address of an object with local scope should not be passed out of its
  scope.
* [V3529][1104]. AUTOSAR. Floating-point values should not be tested for equality or inequality.
* [V3530][1105]. AUTOSAR. Variable should be declared in a scope that minimizes its visibility.
* [V3531][1106]. AUTOSAR. Expressions with enum underlying type should have values corresponding to
  the enumerators of the enumeration.
* [V3532][1107]. AUTOSAR. Unary minus operator should not be applied to an expression of the
  unsigned type.
* [V3533][1108]. AUTOSAR. Expression containing increment (++) or decrement (--) should not have
  other side effects.
* [V3534][1109]. AUTOSAR. Incorrect shifting expression.
* [V3535][1110]. AUTOSAR. Operand of sizeof() operator should not have other side effects.
* [V3536][1111]. AUTOSAR. A pointer/reference parameter in a function should be declared as
  pointer/reference to const if the corresponding object was not modified.
* [V3537][1112]. AUTOSAR. Subtraction, >, >=, <, <= should be applied only to pointers that address
  elements of the same array.
* [V3538][1113]. AUTOSAR. The result of an assignment expression should not be used.
* [V3539][1114]. AUTOSAR. Array indexing should be the only form of pointer arithmetic and it should
  be applied only to objects defined as an array type.
* [V3540][1115]. AUTOSAR. There should be no implicit integral-floating conversion.
* [V3541][1116]. AUTOSAR. A function should not call itself either directly or indirectly.
* [V3542][1117]. AUTOSAR. Constant expression evaluation should not result in an unsigned integer
  wrap-around.
* [V3543][1118]. AUTOSAR. Cast should not remove 'const' / 'volatile' qualification from the type
  that is pointed to by a pointer or a reference.
* [V3544][1119]. AUTOSAR. The 'operator &&', 'operator ||', 'operator ,' and the unary 'operator &'
  should not be overloaded.
* [V3545][1120]. AUTOSAR. Operands of the logical '&&' or the '||' operators, the '!' operator
  should have 'bool' type.
* [V3546][1121]. AUTOSAR. Conversions between pointers to objects and integer types should not be
  performed.
* [V3547][1122]. AUTOSAR. Identifiers that start with '__' or '_[A-Z]' are reserved.
* [V3548][1123]. AUTOSAR. Functions should not be declared at block scope.
* [V3549][1124]. AUTOSAR. The global namespace should only contain 'main', namespace declarations
  and 'extern "C"' declarations.
* [V3550][1125]. AUTOSAR. The identifier 'main' should not be used for a function other than the
  global function 'main'.
* [V3551][1126]. AUTOSAR. An identifier with array type passed as a function argument should not
  decay to a pointer.
* [V3552][1127]. AUTOSAR. Cast should not convert a pointer to a function to any other pointer type,
  including a pointer to function type.
* [V3553][1128]. AUTOSAR. The standard signal handling functions should not be used.
* [V3554][1129]. AUTOSAR. The standard input/output functions should not be used.
* [V3555][1130]. AUTOSAR. The 'static' storage class specifier should be used in all declarations of
  functions that have internal linkage.

## OWASP errors (C++)

* [V5001][1131]. OWASP. It is highly probable that the semicolon ';' is missing after 'return'
  keyword.
* [V5002][1132]. OWASP. An empty exception handler. Silent suppression of exceptions can hide the
  presence of bugs in source code during testing.
* [V5003][1133]. OWASP. The object was created but it is not being used. The 'throw' keyword could
  be missing.
* [V5004][1134]. OWASP. Consider inspecting the expression. Bit shifting of the 32-bit value with a
  subsequent expansion to the 64-bit type.
* [V5005][1135]. OWASP. A value is being subtracted from the unsigned variable. This can result in
  an overflow. In such a case, the comparison operation can potentially behave unexpectedly.
* [V5006][1136]. OWASP. More than N bits are required to store the value, but the expression
  evaluates to the T type which can only hold K bits.
* [V5007][1137]. OWASP. Consider inspecting the loop expression. It is possible that the 'i'
  variable should be incremented instead of the 'n' variable.
* [V5008][1138]. OWASP. Classes should always be derived from std::exception (and alike) as
  'public'.
* [V5009][1139]. OWASP. Unchecked tainted data is used in expression.
* [V5010][1140]. OWASP. The variable is incremented in the loop. Undefined behavior will occur in
  case of signed integer overflow.
* [V5011][1141]. OWASP. Possible overflow. Consider casting operands, not the result.
* [V5012][1142]. OWASP. Potentially unsafe double-checked locking.
* [V5013][1143]. OWASP. Storing credentials inside source code can lead to security issues.
* [V5014][1144]. OWASP. Cryptographic function is deprecated. Its use can lead to security issues.
  Consider switching to an equivalent newer function.

## OWASP errors (C#)

* [V5601][1145]. OWASP. Storing credentials inside source code can lead to security issues.
* [V5602][1146]. OWASP. The object was created but it is not being used. The 'throw' keyword could
  be missing.
* [V5603][1147]. OWASP. The original exception object was swallowed. Stack of original exception
  could be lost.
* [V5604][1148]. OWASP. Potentially unsafe double-checked locking. Use volatile variable(s) or
  synchronization primitives to avoid this.
* [V5605][1149]. OWASP. Unsafe invocation of event, NullReferenceException is possible. Consider
  assigning event to a local variable before invoking it.
* [V5606][1150]. OWASP. An exception handling block does not contain any code.
* [V5607][1151]. OWASP. Exception classes should be publicly accessible.
* [V5608][1152]. OWASP. Possible SQL injection. Potentially tainted data is used to create SQL
  command.
* [V5609][1153]. OWASP. Possible path traversal vulnerability. Potentially tainted data is used as a
  path.
* [V5610][1154]. OWASP. Possible XSS vulnerability. Potentially tainted data might be used to
  execute a malicious script.
* [V5611][1155]. OWASP. Potential insecure deserialization vulnerability. Potentially tainted data
  is used to create an object using deserialization.
* [V5612][1156]. OWASP. Do not use old versions of SSL/TLS protocols as it may cause security
  issues.
* [V5613][1157]. OWASP. Use of outdated cryptographic algorithm is not recommended.
* [V5614][1158]. OWASP. Potential XXE vulnerability. Insecure XML parser is used to process
  potentially tainted data.
* [V5615][1159]. OWASP. Potential XEE vulnerability. Insecure XML parser is used to process
  potentially tainted data.
* [V5616][1160]. OWASP. Possible command injection. Potentially tainted data is used to create OS
  command.
* [V5617][1161]. OWASP. Assigning potentially negative or large value as timeout of HTTP session can
  lead to excessive session expiration time.
* [V5618][1162]. OWASP. Possible server-side request forgery. Potentially tainted data is used in
  the URL.
* [V5619][1163]. OWASP. Possible log injection. Potentially tainted data is written into logs.
* [V5620][1164]. OWASP. Possible LDAP injection. Potentially tainted data is used in a search
  filter.
* [V5621][1165]. OWASP. Error message contains potentially sensitive data that may be exposed.
* [V5622][1166]. OWASP. Possible XPath injection. Potentially tainted data is used in the XPath
  expression.
* [V5623][1167]. OWASP. Possible open redirect vulnerability. Potentially tainted data is used in
  the URL.
* [V5624][1168]. OWASP. Use of potentially tainted data in configuration may lead to security
  issues.
* [V5625][1169]. OWASP. Referenced package contains vulnerability.
* [V5626][1170]. OWASP. Possible ReDoS vulnerability. Potentially tainted data is processed by
  regular expression that contains an unsafe pattern.
* [V5627][1171]. OWASP. Possible NoSQL injection. Potentially tainted data is used to create query.
* [V5628][1172]. OWASP. Possible Zip Slip vulnerability. Potentially tainted data is used in the
  path to extract the file.
* [V5629][1173]. OWASP. Code contains invisible characters that may alter its logic. Consider
  enabling the display of invisible characters in the code editor.
* [V5630][1174]. OWASP. Possible cookie injection. Potentially tainted data is used to create a
  cookie.
* [V5631][1175]. OWASP. Use of externally-controlled format string. Potentially tainted data is used
  as a format string.

## OWASP errors (Java)

* [V5301][1176]. OWASP. An exception handling block does not contain any code.
* [V5302][1177]. OWASP. Exception classes should be publicly accessible.
* [V5303][1178]. OWASP. The object was created but it is not being used. The 'throw' keyword could
  be missing.
* [V5304][1179]. OWASP. Unsafe double-checked locking.
* [V5305][1180]. OWASP. Storing credentials inside source code can lead to security issues.
* [V5306][1181]. OWASP. The original exception object was swallowed. Cause of original exception
  could be lost.
* [V5307][1182]. OWASP. Potentially predictable seed is used in pseudo-random number generator.
* [V5308][1183]. OWASP. Possible overflow. The expression will be evaluated before casting. Consider
  casting one of the operands instead.
* [V5309][1184]. OWASP. Possible SQL injection. Potentially tainted data is used to create SQL
  command.
* [V5310][1185]. OWASP. Possible command injection. Potentially tainted data is used to create OS
  command.
* [V5311][1186]. OWASP. Possible argument injection. Potentially tainted data is used to create OS
  command.
* [V5312][1187]. OWASP. Possible XPath injection. Potentially tainted data is used in the XPath
  expression.
* [V5313][1188]. OWASP. Do not use old versions of SSL/TLS protocols as it may cause security
  issues.
* [V5314][1189]. OWASP. Use of an outdated hash algorithm is not recommended.
* [V5315][1190]. OWASP. Use of an outdated cryptographic algorithm is not recommended.
* [V5316][1191]. Do not use the 'HttpServletRequest.getRequestedSessionId' method because it uses a
  session ID provided by a client.
* [V5317][1192]. OWASP. Implementing a cryptographic algorithm is not advised because an attacker
  might break it.
* [V5318][1193]. OWASP. Setting POSIX file permissions to 'all' or 'others' groups can lead to
  unintended access to files or directories.
* [V5319][1194]. OWASP. Possible log injection. Potentially tainted data is written into logs.
* [V5320][1195]. OWASP. Use of potentially tainted data in configuration may lead to security
  issues.
* [V5321][1196]. OWASP. Possible LDAP injection. Potentially tainted data is used in a search
  filter.
* [V5322][1197]. OWASP. Possible reflection injection. Potentially tainted data is used to select
  class or method.
* [V5323][1198]. OWASP. Potentially tainted data is used to define 'Access-Control-Allow-Origin'
  header.
* [V5324][1199]. OWASP. Possible open redirect vulnerability. Potentially tainted data is used in
  the URL.
* [V5325][1200]. OWASP. Setting the value of the 'Access-Control-Allow-Origin' header to '*' is
  potentially insecure.
* [V5326][1201]. OWASP. A password for a database connection should not be empty
* [V5327][1202]. OWASP. Possible regex injection. Potentially tainted data is used to create regular
  expression
* [V5328][1203]. OWASP. Using non-restrictive authorization checks could lead to security
  violations.
* [V5329][1204]. OWASP. Using non-atomic file creation is not recommended because an attacker might
  intercept file ownership.
* [V5330][1205]. OWASP. Possible XSS injection. Potentially tainted data might be used to execute a
  malicious script.
* [V5331][1206]. OWASP. Hardcoded IP addresses are not secure.
* [V5332][1207]. OWASP. Possible path traversal vulnerability. Potentially tainted data might be
  used to access files or folders outside a target directory.
* [V5333][1208]. OWASP. Possible insecure deserialization vulnerability. Potentially tainted data is
  used to create an object during deserialization.
* [V5334][1209]. OWASP. Possible server-side request forgery. Potentially tainted data in the URL is
  used to access a remote resource.
* [V5335][1210]. OWASP. Potential XXE vulnerability. Insecure XML parser is used to process
  potentially tainted data.
* [V5336][1211]. OWASP. Potential XEE vulnerability. Insecure XML parser is used to process
  potentially tainted data.
* [V5337][1212]. OWASP. Possible NoSQL injection. Potentially tainted data is used to create query.
* [V5338][1213]. OWASP. Possible Zip Slip vulnerability. Potentially tainted data might be used to
  extract the file.

## Problems related to code analyzer

* [V001][1214]. A code fragment from 'file' cannot be analyzed.
* [V002][1215]. Some diagnostic messages may contain incorrect line number.
* [V003][1216]. Unrecognized error found...
* [V004][1217]. Diagnostics from the 64-bit rule set are not entirely accurate without the
  appropriate 64-bit compiler. Consider utilizing 64-bit compiler if possible.
* [V005][1218]. Cannot determine active configuration for project. Please check projects and
  solution configurations.
* [V006][1219]. File cannot be processed. Analysis aborted by timeout.
* [V007][1220]. Deprecated CLR switch was detected. Incorrect diagnostics are possible.
* [V008][1221]. Unable to start the analysis on this file.
* [V010][1222]. Analysis for the project type or platform toolsets is not supported in this tool.
  Use direct analyzer integration or compiler monitoring instead.
* [V011][1223]. Presence of #line directives may cause some diagnostic messages to have incorrect
  file name and line number.
* [V012][1224]. Some warnings could have been disabled.
* [V013][1225]. Intermodular analysis may be incomplete, as it is not run on all source files.
* [V014][1226]. The version of your suppress file is outdated. Appending new suppressed messages to
  it is not possible. Consider re-generating your suppress file to continue updating it.
* [V015][1227]. All analyzer messages were filtered out or marked as false positive. Use filter
  buttons or 'Don't Check Files' settings to enable message display.
* [V016][1228]. User annotation was not applied to a virtual function. To force the annotation, use
  the 'enable_on_virtual' flag.
* [V017][1229]. The analyzer terminated abnormally due to lack of memory.
* [V018][1230]. False Alarm marks without hash codes were ignored because the 'V_HASH_ONLY' option
  is enabled.
* [V019][1231]. Error occurred while working with the user annotation mechanism.
* [V020][1232]. Error occurred while working with rules configuration files.
* [V051][1233]. Some of the references in project are missing or incorrect. The analysis results
  could be incomplete. Consider making the project fully compilable and building it before analysis.
* [V052][1234]. A critical error had occurred.
* [V061][1235]. An error has occurred.
* [V062][1236]. Failed to run analyzer core. Make sure the correct 64-bit Java 11 or higher
  executable is used, or specify it manually.
* [V063][1237]. Analysis aborted by timeout.
* [V064][1238]. Unable to run a rule to analyze a code fragment.
* [V065][1239]. Unable to start analysis on a module.

Was this page helpful?

**Message submitted. ** [check circle]

Your message has been sent. We will email you at

If you do not see the email in your inbox, please check if it is filtered to one of the following
folders:

* Promotion
* Updates
* Spam

If you can’t find an answer to your question, fill in the form below and our developers will contact
you

By clicking this button you agree to our [Privacy Policy][1240] statement

[1]: #ID456ABC175A
[2]: #WhatBugsCanPVSStudioDetect
[3]: #IDE49231CD27
[4]: #ListOfAllAnalyzerRulesInXML
[5]: #GeneralAnalysisCPP
[6]: #GeneralAnalysisCS
[7]: #GeneralAnalysisJAVA
[8]: #MicroOptimizationsCPP
[9]: #MicroOptimizationsCS
[10]: #64CPP
[11]: #CustomersSpecificRequestsCPP
[12]: #MISRA
[13]: #AUTOSAR
[14]: #OWASPCPP
[15]: #OWASPCS
[16]: #OWASPJava
[17]: #ProblemsRelatedToCodeAnalyzer
[18]: /en/blog/posts/1096/
[19]: /en/pvs-studio/sast/
[20]: /en/blog/examples/
[21]: http://files.pvs-studio.com/rules/RulesMap.xml
[22]: /en/docs/warnings/v501/
[23]: /en/docs/warnings/v502/
[24]: /en/docs/warnings/v503/
[25]: /en/docs/warnings/v504/
[26]: /en/docs/warnings/v505/
[27]: /en/docs/warnings/v506/
[28]: /en/docs/warnings/v507/
[29]: /en/docs/warnings/v508/
[30]: /en/docs/warnings/v509/
[31]: /en/docs/warnings/v510/
[32]: /en/docs/warnings/v511/
[33]: /en/docs/warnings/v512/
[34]: /en/docs/warnings/v513/
[35]: /en/docs/warnings/v514/
[36]: /en/docs/warnings/v515/
[37]: /en/docs/warnings/v516/
[38]: /en/docs/warnings/v517/
[39]: /en/docs/warnings/v518/
[40]: /en/docs/warnings/v519/
[41]: /en/docs/warnings/v520/
[42]: /en/docs/warnings/v521/
[43]: /en/docs/warnings/v522/
[44]: /en/docs/warnings/v523/
[45]: /en/docs/warnings/v524/
[46]: /en/docs/warnings/v525/
[47]: /en/docs/warnings/v526/
[48]: /en/docs/warnings/v527/
[49]: /en/docs/warnings/v528/
[50]: /en/docs/warnings/v529/
[51]: /en/docs/warnings/v530/
[52]: /en/docs/warnings/v531/
[53]: /en/docs/warnings/v532/
[54]: /en/docs/warnings/v533/
[55]: /en/docs/warnings/v534/
[56]: /en/docs/warnings/v535/
[57]: /en/docs/warnings/v536/
[58]: /en/docs/warnings/v537/
[59]: /en/docs/warnings/v538/
[60]: /en/docs/warnings/v539/
[61]: /en/docs/warnings/v540/
[62]: /en/docs/warnings/v541/
[63]: /en/docs/warnings/v542/
[64]: /en/docs/warnings/v543/
[65]: /en/docs/warnings/v544/
[66]: /en/docs/warnings/v545/
[67]: /en/docs/warnings/v546/
[68]: /en/docs/warnings/v547/
[69]: /en/docs/warnings/v548/
[70]: /en/docs/warnings/v549/
[71]: /en/docs/warnings/v550/
[72]: /en/docs/warnings/v551/
[73]: /en/docs/warnings/v552/
[74]: /en/docs/warnings/v553/
[75]: /en/docs/warnings/v554/
[76]: /en/docs/warnings/v555/
[77]: /en/docs/warnings/v556/
[78]: /en/docs/warnings/v557/
[79]: /en/docs/warnings/v558/
[80]: /en/docs/warnings/v559/
[81]: /en/docs/warnings/v560/
[82]: /en/docs/warnings/v561/
[83]: /en/docs/warnings/v562/
[84]: /en/docs/warnings/v563/
[85]: /en/docs/warnings/v564/
[86]: /en/docs/warnings/v565/
[87]: /en/docs/warnings/v566/
[88]: /en/docs/warnings/v567/
[89]: /en/docs/warnings/v568/
[90]: /en/docs/warnings/v569/
[91]: /en/docs/warnings/v570/
[92]: /en/docs/warnings/v571/
[93]: /en/docs/warnings/v572/
[94]: /en/docs/warnings/v573/
[95]: /en/docs/warnings/v574/
[96]: /en/docs/warnings/v575/
[97]: /en/docs/warnings/v576/
[98]: /en/docs/warnings/v577/
[99]: /en/docs/warnings/v578/
[100]: /en/docs/warnings/v579/
[101]: /en/docs/warnings/v580/
[102]: /en/docs/warnings/v581/
[103]: /en/docs/warnings/v582/
[104]: /en/docs/warnings/v583/
[105]: /en/docs/warnings/v584/
[106]: /en/docs/warnings/v585/
[107]: /en/docs/warnings/v586/
[108]: /en/docs/warnings/v587/
[109]: /en/docs/warnings/v588/
[110]: /en/docs/warnings/v589/
[111]: /en/docs/warnings/v590/
[112]: /en/docs/warnings/v591/
[113]: /en/docs/warnings/v592/
[114]: /en/docs/warnings/v593/
[115]: /en/docs/warnings/v594/
[116]: /en/docs/warnings/v595/
[117]: /en/docs/warnings/v596/
[118]: /en/docs/warnings/v597/
[119]: /en/docs/warnings/v598/
[120]: /en/docs/warnings/v599/
[121]: /en/docs/warnings/v600/
[122]: /en/docs/warnings/v601/
[123]: /en/docs/warnings/v602/
[124]: /en/docs/warnings/v603/
[125]: /en/docs/warnings/v604/
[126]: /en/docs/warnings/v605/
[127]: /en/docs/warnings/v606/
[128]: /en/docs/warnings/v607/
[129]: /en/docs/warnings/v608/
[130]: /en/docs/warnings/v609/
[131]: /en/docs/warnings/v610/
[132]: /en/docs/warnings/v611/
[133]: /en/docs/warnings/v612/
[134]: /en/docs/warnings/v613/
[135]: /en/docs/warnings/v614/
[136]: /en/docs/warnings/v615/
[137]: /en/docs/warnings/v616/
[138]: /en/docs/warnings/v617/
[139]: /en/docs/warnings/v618/
[140]: /en/docs/warnings/v619/
[141]: /en/docs/warnings/v620/
[142]: /en/docs/warnings/v621/
[143]: /en/docs/warnings/v622/
[144]: /en/docs/warnings/v623/
[145]: /en/docs/warnings/v624/
[146]: /en/docs/warnings/v625/
[147]: /en/docs/warnings/v626/
[148]: /en/docs/warnings/v627/
[149]: /en/docs/warnings/v628/
[150]: /en/docs/warnings/v629/
[151]: /en/docs/warnings/v630/
[152]: /en/docs/warnings/v631/
[153]: /en/docs/warnings/v632/
[154]: /en/docs/warnings/v633/
[155]: /en/docs/warnings/v634/
[156]: /en/docs/warnings/v635/
[157]: /en/docs/warnings/v636/
[158]: /en/docs/warnings/v637/
[159]: /en/docs/warnings/v638/
[160]: /en/docs/warnings/v639/
[161]: /en/docs/warnings/v640/
[162]: /en/docs/warnings/v641/
[163]: /en/docs/warnings/v642/
[164]: /en/docs/warnings/v643/
[165]: /en/docs/warnings/v644/
[166]: /en/docs/warnings/v645/
[167]: /en/docs/warnings/v646/
[168]: /en/docs/warnings/v647/
[169]: /en/docs/warnings/v648/
[170]: /en/docs/warnings/v649/
[171]: /en/docs/warnings/v650/
[172]: /en/docs/warnings/v651/
[173]: /en/docs/warnings/v652/
[174]: /en/docs/warnings/v653/
[175]: /en/docs/warnings/v654/
[176]: /en/docs/warnings/v655/
[177]: /en/docs/warnings/v656/
[178]: /en/docs/warnings/v657/
[179]: /en/docs/warnings/v658/
[180]: /en/docs/warnings/v659/
[181]: /en/docs/warnings/v660/
[182]: /en/docs/warnings/v661/
[183]: /en/docs/warnings/v662/
[184]: /en/docs/warnings/v663/
[185]: /en/docs/warnings/v664/
[186]: /en/docs/warnings/v665/
[187]: /en/docs/warnings/v666/
[188]: /en/docs/warnings/v667/
[189]: /en/docs/warnings/v668/
[190]: /en/docs/warnings/v669/
[191]: /en/docs/warnings/v670/
[192]: /en/docs/warnings/v671/
[193]: /en/docs/warnings/v672/
[194]: /en/docs/warnings/v673/
[195]: /en/docs/warnings/v674/
[196]: /en/docs/warnings/v675/
[197]: /en/docs/warnings/v676/
[198]: /en/docs/warnings/v677/
[199]: /en/docs/warnings/v678/
[200]: /en/docs/warnings/v679/
[201]: /en/docs/warnings/v680/
[202]: /en/docs/warnings/v681/
[203]: /en/docs/warnings/v682/
[204]: /en/docs/warnings/v683/
[205]: /en/docs/warnings/v684/
[206]: /en/docs/warnings/v685/
[207]: /en/docs/warnings/v686/
[208]: /en/docs/warnings/v687/
[209]: /en/docs/warnings/v688/
[210]: /en/docs/warnings/v689/
[211]: /en/docs/warnings/v690/
[212]: /en/docs/warnings/v691/
[213]: /en/docs/warnings/v692/
[214]: /en/docs/warnings/v693/
[215]: /en/docs/warnings/v694/
[216]: /en/docs/warnings/v695/
[217]: /en/docs/warnings/v696/
[218]: /en/docs/warnings/v697/
[219]: /en/docs/warnings/v698/
[220]: /en/docs/warnings/v699/
[221]: /en/docs/warnings/v700/
[222]: /en/docs/warnings/v701/
[223]: /en/docs/warnings/v702/
[224]: /en/docs/warnings/v703/
[225]: /en/docs/warnings/v704/
[226]: /en/docs/warnings/v705/
[227]: /en/docs/warnings/v706/
[228]: /en/docs/warnings/v707/
[229]: /en/docs/warnings/v708/
[230]: /en/docs/warnings/v709/
[231]: /en/docs/warnings/v710/
[232]: /en/docs/warnings/v711/
[233]: /en/docs/warnings/v712/
[234]: /en/docs/warnings/v713/
[235]: /en/docs/warnings/v714/
[236]: /en/docs/warnings/v715/
[237]: /en/docs/warnings/v716/
[238]: /en/docs/warnings/v717/
[239]: /en/docs/warnings/v718/
[240]: /en/docs/warnings/v719/
[241]: /en/docs/warnings/v720/
[242]: /en/docs/warnings/v721/
[243]: /en/docs/warnings/v722/
[244]: /en/docs/warnings/v723/
[245]: /en/docs/warnings/v724/
[246]: /en/docs/warnings/v725/
[247]: /en/docs/warnings/v726/
[248]: /en/docs/warnings/v727/
[249]: /en/docs/warnings/v728/
[250]: /en/docs/warnings/v729/
[251]: /en/docs/warnings/v730/
[252]: /en/docs/warnings/v731/
[253]: /en/docs/warnings/v732/
[254]: /en/docs/warnings/v733/
[255]: /en/docs/warnings/v734/
[256]: /en/docs/warnings/v735/
[257]: /en/docs/warnings/v736/
[258]: /en/docs/warnings/v737/
[259]: /en/docs/warnings/v738/
[260]: /en/docs/warnings/v739/
[261]: /en/docs/warnings/v740/
[262]: /en/docs/warnings/v741/
[263]: /en/docs/warnings/v742/
[264]: /en/docs/warnings/v743/
[265]: /en/docs/warnings/v744/
[266]: /en/docs/warnings/v745/
[267]: /en/docs/warnings/v746/
[268]: /en/docs/warnings/v747/
[269]: /en/docs/warnings/v748/
[270]: /en/docs/warnings/v749/
[271]: /en/docs/warnings/v750/
[272]: /en/docs/warnings/v751/
[273]: /en/docs/warnings/v752/
[274]: /en/docs/warnings/v753/
[275]: /en/docs/warnings/v754/
[276]: /en/docs/warnings/v755/
[277]: /en/docs/warnings/v756/
[278]: /en/docs/warnings/v757/
[279]: /en/docs/warnings/v758/
[280]: /en/docs/warnings/v759/
[281]: /en/docs/warnings/v760/
[282]: /en/docs/warnings/v761/
[283]: /en/docs/warnings/v762/
[284]: /en/docs/warnings/v763/
[285]: /en/docs/warnings/v764/
[286]: /en/docs/warnings/v765/
[287]: /en/docs/warnings/v766/
[288]: /en/docs/warnings/v767/
[289]: /en/docs/warnings/v768/
[290]: /en/docs/warnings/v769/
[291]: /en/docs/warnings/v770/
[292]: /en/docs/warnings/v771/
[293]: /en/docs/warnings/v772/
[294]: /en/docs/warnings/v773/
[295]: /en/docs/warnings/v774/
[296]: /en/docs/warnings/v775/
[297]: /en/docs/warnings/v776/
[298]: /en/docs/warnings/v777/
[299]: /en/docs/warnings/v778/
[300]: /en/docs/warnings/v779/
[301]: /en/docs/warnings/v780/
[302]: /en/docs/warnings/v781/
[303]: /en/docs/warnings/v782/
[304]: /en/docs/warnings/v783/
[305]: /en/docs/warnings/v784/
[306]: /en/docs/warnings/v785/
[307]: /en/docs/warnings/v786/
[308]: /en/docs/warnings/v787/
[309]: /en/docs/warnings/v788/
[310]: /en/docs/warnings/v789/
[311]: /en/docs/warnings/v790/
[312]: /en/docs/warnings/v791/
[313]: /en/docs/warnings/v792/
[314]: /en/docs/warnings/v793/
[315]: /en/docs/warnings/v794/
[316]: /en/docs/warnings/v795/
[317]: /en/docs/warnings/v796/
[318]: /en/docs/warnings/v797/
[319]: /en/docs/warnings/v798/
[320]: /en/docs/warnings/v799/
[321]: /en/docs/warnings/v1001/
[322]: /en/docs/warnings/v1002/
[323]: /en/docs/warnings/v1003/
[324]: /en/docs/warnings/v1004/
[325]: /en/docs/warnings/v1005/
[326]: /en/docs/warnings/v1006/
[327]: /en/docs/warnings/v1007/
[328]: /en/docs/warnings/v1008/
[329]: /en/docs/warnings/v1009/
[330]: /en/docs/warnings/v1010/
[331]: /en/docs/warnings/v1011/
[332]: /en/docs/warnings/v1012/
[333]: /en/docs/warnings/v1013/
[334]: /en/docs/warnings/v1014/
[335]: /en/docs/warnings/v1015/
[336]: /en/docs/warnings/v1016/
[337]: /en/docs/warnings/v1017/
[338]: /en/docs/warnings/v1018/
[339]: /en/docs/warnings/v1019/
[340]: /en/docs/warnings/v1020/
[341]: /en/docs/warnings/v1021/
[342]: /en/docs/warnings/v1022/
[343]: /en/docs/warnings/v1023/
[344]: /en/docs/warnings/v1024/
[345]: /en/docs/warnings/v1025/
[346]: /en/docs/warnings/v1026/
[347]: /en/docs/warnings/v1027/
[348]: /en/docs/warnings/v1028/
[349]: /en/docs/warnings/v1029/
[350]: /en/docs/warnings/v1030/
[351]: /en/docs/warnings/v1031/
[352]: /en/docs/warnings/v1032/
[353]: /en/docs/warnings/v1033/
[354]: /en/docs/warnings/v1034/
[355]: /en/docs/warnings/v1035/
[356]: /en/docs/warnings/v1036/
[357]: /en/docs/warnings/v1037/
[358]: /en/docs/warnings/v1038/
[359]: /en/docs/warnings/v1039/
[360]: /en/docs/warnings/v1040/
[361]: /en/docs/warnings/v1041/
[362]: /en/docs/warnings/v1042/
[363]: /en/docs/warnings/v1043/
[364]: /en/docs/warnings/v1044/
[365]: /en/docs/warnings/v1045/
[366]: /en/docs/warnings/v1046/
[367]: /en/docs/warnings/v1047/
[368]: /en/docs/warnings/v1048/
[369]: /en/docs/warnings/v1049/
[370]: /en/docs/warnings/v1050/
[371]: /en/docs/warnings/v1051/
[372]: /en/docs/warnings/v1052/
[373]: /en/docs/warnings/v1053/
[374]: /en/docs/warnings/v1054/
[375]: /en/docs/warnings/v1055/
[376]: /en/docs/warnings/v1056/
[377]: /en/docs/warnings/v1057/
[378]: /en/docs/warnings/v1058/
[379]: /en/docs/warnings/v1059/
[380]: /en/docs/warnings/v1060/
[381]: /en/docs/warnings/v1061/
[382]: /en/docs/warnings/v1062/
[383]: /en/docs/warnings/v1063/
[384]: /en/docs/warnings/v1064/
[385]: /en/docs/warnings/v1065/
[386]: /en/docs/warnings/v1066/
[387]: /en/docs/warnings/v1067/
[388]: /en/docs/warnings/v1068/
[389]: /en/docs/warnings/v1069/
[390]: /en/docs/warnings/v1070/
[391]: /en/docs/warnings/v1071/
[392]: /en/docs/warnings/v1072/
[393]: /en/docs/warnings/v1073/
[394]: /en/docs/warnings/v1074/
[395]: /en/docs/warnings/v1075/
[396]: /en/docs/warnings/v1076/
[397]: /en/docs/warnings/v1077/
[398]: /en/docs/warnings/v1078/
[399]: /en/docs/warnings/v1079/
[400]: /en/docs/warnings/v1080/
[401]: /en/docs/warnings/v1081/
[402]: /en/docs/warnings/v1082/
[403]: /en/docs/warnings/v1083/
[404]: /en/docs/warnings/v1084/
[405]: /en/docs/warnings/v1085/
[406]: /en/docs/warnings/v1086/
[407]: /en/docs/warnings/v1087/
[408]: /en/docs/warnings/v1088/
[409]: /en/docs/warnings/v1089/
[410]: /en/docs/warnings/v1090/
[411]: /en/docs/warnings/v1091/
[412]: /en/docs/warnings/v1092/
[413]: /en/docs/warnings/v1093/
[414]: /en/docs/warnings/v1094/
[415]: /en/docs/warnings/v1095/
[416]: /en/docs/warnings/v1096/
[417]: /en/docs/warnings/v1097/
[418]: /en/docs/warnings/v1098/
[419]: /en/docs/warnings/v1099/
[420]: /en/docs/warnings/v1100/
[421]: /en/docs/warnings/v1101/
[422]: /en/docs/warnings/v1102/
[423]: /en/docs/warnings/v1103/
[424]: /en/docs/warnings/v1104/
[425]: /en/docs/warnings/v1105/
[426]: /en/docs/warnings/v1106/
[427]: /en/docs/warnings/v1107/
[428]: /en/docs/warnings/v1108/
[429]: /en/docs/warnings/v1109/
[430]: /en/docs/warnings/v1110/
[431]: /en/docs/warnings/v1111/
[432]: /en/docs/warnings/v1112/
[433]: /en/docs/warnings/v1113/
[434]: /en/docs/warnings/v1114/
[435]: /en/docs/warnings/v1115/
[436]: /en/docs/warnings/v1116/
[437]: /en/docs/warnings/v1117/
[438]: /en/docs/warnings/v1118/
[439]: /en/docs/warnings/v3001/
[440]: /en/docs/warnings/v3002/
[441]: /en/docs/warnings/v3003/
[442]: /en/docs/warnings/v3004/
[443]: /en/docs/warnings/v3005/
[444]: /en/docs/warnings/v3006/
[445]: /en/docs/warnings/v3007/
[446]: /en/docs/warnings/v3008/
[447]: /en/docs/warnings/v3009/
[448]: /en/docs/warnings/v3010/
[449]: /en/docs/warnings/v3011/
[450]: /en/docs/warnings/v3012/
[451]: /en/docs/warnings/v3013/
[452]: /en/docs/warnings/v3014/
[453]: /en/docs/warnings/v3015/
[454]: /en/docs/warnings/v3016/
[455]: /en/docs/warnings/v3017/
[456]: /en/docs/warnings/v3018/
[457]: /en/docs/warnings/v3019/
[458]: /en/docs/warnings/v3020/
[459]: /en/docs/warnings/v3021/
[460]: /en/docs/warnings/v3022/
[461]: /en/docs/warnings/v3023/
[462]: /en/docs/warnings/v3024/
[463]: /en/docs/warnings/v3025/
[464]: /en/docs/warnings/v3026/
[465]: /en/docs/warnings/v3027/
[466]: /en/docs/warnings/v3028/
[467]: /en/docs/warnings/v3029/
[468]: /en/docs/warnings/v3030/
[469]: /en/docs/warnings/v3031/
[470]: /en/docs/warnings/v3032/
[471]: /en/docs/warnings/v3033/
[472]: /en/docs/warnings/v3034/
[473]: /en/docs/warnings/v3035/
[474]: /en/docs/warnings/v3036/
[475]: /en/docs/warnings/v3037/
[476]: /en/docs/warnings/v3038/
[477]: /en/docs/warnings/v3039/
[478]: /en/docs/warnings/v3040/
[479]: /en/docs/warnings/v3041/
[480]: /en/docs/warnings/v3042/
[481]: /en/docs/warnings/v3043/
[482]: /en/docs/warnings/v3044/
[483]: /en/docs/warnings/v3045/
[484]: /en/docs/warnings/v3046/
[485]: /en/docs/warnings/v3047/
[486]: /en/docs/warnings/v3048/
[487]: /en/docs/warnings/v3049/
[488]: /en/docs/warnings/v3050/
[489]: /en/docs/warnings/v3051/
[490]: /en/docs/warnings/v3052/
[491]: /en/docs/warnings/v3053/
[492]: /en/docs/warnings/v3054/
[493]: /en/docs/warnings/v3055/
[494]: /en/docs/warnings/v3056/
[495]: /en/docs/warnings/v3057/
[496]: /en/docs/warnings/v3058/
[497]: /en/docs/warnings/v3059/
[498]: /en/docs/warnings/v3060/
[499]: /en/docs/warnings/v3061/
[500]: /en/docs/warnings/v3062/
[501]: /en/docs/warnings/v3063/
[502]: /en/docs/warnings/v3064/
[503]: /en/docs/warnings/v3065/
[504]: /en/docs/warnings/v3066/
[505]: /en/docs/warnings/v3067/
[506]: /en/docs/warnings/v3068/
[507]: /en/docs/warnings/v3069/
[508]: /en/docs/warnings/v3070/
[509]: /en/docs/warnings/v3071/
[510]: /en/docs/warnings/v3072/
[511]: /en/docs/warnings/v3073/
[512]: /en/docs/warnings/v3074/
[513]: /en/docs/warnings/v3075/
[514]: /en/docs/warnings/v3076/
[515]: /en/docs/warnings/v3077/
[516]: /en/docs/warnings/v3078/
[517]: /en/docs/warnings/v3079/
[518]: /en/docs/warnings/v3080/
[519]: /en/docs/warnings/v3081/
[520]: /en/docs/warnings/v3082/
[521]: /en/docs/warnings/v3083/
[522]: /en/docs/warnings/v3084/
[523]: /en/docs/warnings/v3085/
[524]: /en/docs/warnings/v3086/
[525]: /en/docs/warnings/v3087/
[526]: /en/docs/warnings/v3088/
[527]: /en/docs/warnings/v3089/
[528]: /en/docs/warnings/v3090/
[529]: /en/docs/warnings/v3091/
[530]: /en/docs/warnings/v3092/
[531]: /en/docs/warnings/v3093/
[532]: /en/docs/warnings/v3094/
[533]: /en/docs/warnings/v3095/
[534]: /en/docs/warnings/v3096/
[535]: /en/docs/warnings/v3097/
[536]: /en/docs/warnings/v3098/
[537]: /en/docs/warnings/v3099/
[538]: /en/docs/warnings/v3100/
[539]: /en/docs/warnings/v3101/
[540]: /en/docs/warnings/v3102/
[541]: /en/docs/warnings/v3103/
[542]: /en/docs/warnings/v3104/
[543]: /en/docs/warnings/v3105/
[544]: /en/docs/warnings/v3106/
[545]: /en/docs/warnings/v3107/
[546]: /en/docs/warnings/v3108/
[547]: /en/docs/warnings/v3109/
[548]: /en/docs/warnings/v3110/
[549]: /en/docs/warnings/v3111/
[550]: /en/docs/warnings/v3112/
[551]: /en/docs/warnings/v3113/
[552]: /en/docs/warnings/v3114/
[553]: /en/docs/warnings/v3115/
[554]: /en/docs/warnings/v3116/
[555]: /en/docs/warnings/v3117/
[556]: /en/docs/warnings/v3118/
[557]: /en/docs/warnings/v3119/
[558]: /en/docs/warnings/v3120/
[559]: /en/docs/warnings/v3121/
[560]: /en/docs/warnings/v3122/
[561]: /en/docs/warnings/v3123/
[562]: /en/docs/warnings/v3124/
[563]: /en/docs/warnings/v3125/
[564]: /en/docs/warnings/v3126/
[565]: /en/docs/warnings/v3127/
[566]: /en/docs/warnings/v3128/
[567]: /en/docs/warnings/v3129/
[568]: /en/docs/warnings/v3130/
[569]: /en/docs/warnings/v3131/
[570]: /en/docs/warnings/v3132/
[571]: /en/docs/warnings/v3133/
[572]: /en/docs/warnings/v3134/
[573]: /en/docs/warnings/v3135/
[574]: /en/docs/warnings/v3136/
[575]: /en/docs/warnings/v3137/
[576]: /en/docs/warnings/v3138/
[577]: /en/docs/warnings/v3139/
[578]: /en/docs/warnings/v3140/
[579]: /en/docs/warnings/v3141/
[580]: /en/docs/warnings/v3142/
[581]: /en/docs/warnings/v3143/
[582]: /en/docs/warnings/v3144/
[583]: /en/docs/warnings/v3145/
[584]: /en/docs/warnings/v3146/
[585]: /en/docs/warnings/v3147/
[586]: /en/docs/warnings/v3148/
[587]: /en/docs/warnings/v3149/
[588]: /en/docs/warnings/v3150/
[589]: /en/docs/warnings/v3151/
[590]: /en/docs/warnings/v3152/
[591]: /en/docs/warnings/v3153/
[592]: /en/docs/warnings/v3154/
[593]: /en/docs/warnings/v3155/
[594]: /en/docs/warnings/v3156/
[595]: /en/docs/warnings/v3157/
[596]: /en/docs/warnings/v3158/
[597]: /en/docs/warnings/v3159/
[598]: /en/docs/warnings/v3160/
[599]: /en/docs/warnings/v3161/
[600]: /en/docs/warnings/v3162/
[601]: /en/docs/warnings/v3163/
[602]: /en/docs/warnings/v3164/
[603]: /en/docs/warnings/v3165/
[604]: /en/docs/warnings/v3166/
[605]: /en/docs/warnings/v3167/
[606]: /en/docs/warnings/v3168/
[607]: /en/docs/warnings/v3169/
[608]: /en/docs/warnings/v3170/
[609]: /en/docs/warnings/v3171/
[610]: /en/docs/warnings/v3172/
[611]: /en/docs/warnings/v3173/
[612]: /en/docs/warnings/v3174/
[613]: /en/docs/warnings/v3175/
[614]: /en/docs/warnings/v3176/
[615]: /en/docs/warnings/v3177/
[616]: /en/docs/warnings/v3178/
[617]: /en/docs/warnings/v3179/
[618]: /en/docs/warnings/v3180/
[619]: /en/docs/warnings/v3181/
[620]: /en/docs/warnings/v3182/
[621]: /en/docs/warnings/v3183/
[622]: /en/docs/warnings/v3184/
[623]: /en/docs/warnings/v3185/
[624]: /en/docs/warnings/v3186/
[625]: /en/docs/warnings/v3187/
[626]: /en/docs/warnings/v3188/
[627]: /en/docs/warnings/v3189/
[628]: /en/docs/warnings/v3190/
[629]: /en/docs/warnings/v3191/
[630]: /en/docs/warnings/v3192/
[631]: /en/docs/warnings/v3193/
[632]: /en/docs/warnings/v3194/
[633]: /en/docs/warnings/v3195/
[634]: /en/docs/warnings/v3196/
[635]: /en/docs/warnings/v3197/
[636]: /en/docs/warnings/v3198/
[637]: /en/docs/warnings/v3199/
[638]: /en/docs/warnings/v3200/
[639]: /en/docs/warnings/v3201/
[640]: /en/docs/warnings/v3202/
[641]: /en/docs/warnings/v3203/
[642]: /en/docs/warnings/v3204/
[643]: /en/docs/warnings/v3205/
[644]: /en/docs/warnings/v3206/
[645]: /en/docs/warnings/v3207/
[646]: /en/docs/warnings/v3208/
[647]: /en/docs/warnings/v3209/
[648]: /en/docs/warnings/v3210/
[649]: /en/docs/warnings/v3211/
[650]: /en/docs/warnings/v3212/
[651]: /en/docs/warnings/v3213/
[652]: /en/docs/warnings/v3214/
[653]: /en/docs/warnings/v3215/
[654]: /en/docs/warnings/v3216/
[655]: /en/docs/warnings/v3217/
[656]: /en/docs/warnings/v3218/
[657]: /en/docs/warnings/v3219/
[658]: /en/docs/warnings/v3220/
[659]: /en/docs/warnings/v3221/
[660]: /en/docs/warnings/v3222/
[661]: /en/docs/warnings/v3223/
[662]: /en/docs/warnings/v3224/
[663]: /en/docs/warnings/v3225/
[664]: /en/docs/warnings/v3226/
[665]: /en/docs/warnings/v3227/
[666]: /en/docs/warnings/v3228/
[667]: /en/docs/warnings/v3229/
[668]: /en/docs/warnings/v6001/
[669]: /en/docs/warnings/v6002/
[670]: /en/docs/warnings/v6003/
[671]: /en/docs/warnings/v6004/
[672]: /en/docs/warnings/v6005/
[673]: /en/docs/warnings/v6006/
[674]: /en/docs/warnings/v6007/
[675]: /en/docs/warnings/v6008/
[676]: /en/docs/warnings/v6009/
[677]: /en/docs/warnings/v6010/
[678]: /en/docs/warnings/v6011/
[679]: /en/docs/warnings/v6012/
[680]: /en/docs/warnings/v6013/
[681]: /en/docs/warnings/v6014/
[682]: /en/docs/warnings/v6015/
[683]: /en/docs/warnings/v6016/
[684]: /en/docs/warnings/v6017/
[685]: /en/docs/warnings/v6018/
[686]: /en/docs/warnings/v6019/
[687]: /en/docs/warnings/v6020/
[688]: /en/docs/warnings/v6021/
[689]: /en/docs/warnings/v6022/
[690]: /en/docs/warnings/v6023/
[691]: /en/docs/warnings/v6024/
[692]: /en/docs/warnings/v6025/
[693]: /en/docs/warnings/v6026/
[694]: /en/docs/warnings/v6027/
[695]: /en/docs/warnings/v6028/
[696]: /en/docs/warnings/v6029/
[697]: /en/docs/warnings/v6030/
[698]: /en/docs/warnings/v6031/
[699]: /en/docs/warnings/v6032/
[700]: /en/docs/warnings/v6033/
[701]: /en/docs/warnings/v6034/
[702]: /en/docs/warnings/v6035/
[703]: /en/docs/warnings/v6036/
[704]: /en/docs/warnings/v6037/
[705]: /en/docs/warnings/v6038/
[706]: /en/docs/warnings/v6039/
[707]: /en/docs/warnings/v6040/
[708]: /en/docs/warnings/v6041/
[709]: /en/docs/warnings/v6042/
[710]: /en/docs/warnings/v6043/
[711]: /en/docs/warnings/v6044/
[712]: /en/docs/warnings/v6045/
[713]: /en/docs/warnings/v6046/
[714]: /en/docs/warnings/v6047/
[715]: /en/docs/warnings/v6048/
[716]: /en/docs/warnings/v6049/
[717]: /en/docs/warnings/v6050/
[718]: /en/docs/warnings/v6051/
[719]: /en/docs/warnings/v6052/
[720]: /en/docs/warnings/v6053/
[721]: /en/docs/warnings/v6054/
[722]: /en/docs/warnings/v6055/
[723]: /en/docs/warnings/v6056/
[724]: /en/docs/warnings/v6057/
[725]: /en/docs/warnings/v6058/
[726]: /en/docs/warnings/v6059/
[727]: /en/docs/warnings/v6060/
[728]: /en/docs/warnings/v6061/
[729]: /en/docs/warnings/v6062/
[730]: /en/docs/warnings/v6063/
[731]: /en/docs/warnings/v6064/
[732]: /en/docs/warnings/v6065/
[733]: /en/docs/warnings/v6066/
[734]: /en/docs/warnings/v6067/
[735]: /en/docs/warnings/v6068/
[736]: /en/docs/warnings/v6069/
[737]: /en/docs/warnings/v6070/
[738]: /en/docs/warnings/v6071/
[739]: /en/docs/warnings/v6072/
[740]: /en/docs/warnings/v6073/
[741]: /en/docs/warnings/v6074/
[742]: /en/docs/warnings/v6075/
[743]: /en/docs/warnings/v6076/
[744]: /en/docs/warnings/v6077/
[745]: /en/docs/warnings/v6078/
[746]: /en/docs/warnings/v6079/
[747]: /en/docs/warnings/v6080/
[748]: /en/docs/warnings/v6081/
[749]: /en/docs/warnings/v6082/
[750]: /en/docs/warnings/v6083/
[751]: /en/docs/warnings/v6084/
[752]: /en/docs/warnings/v6085/
[753]: /en/docs/warnings/v6086/
[754]: /en/docs/warnings/v6087/
[755]: /en/docs/warnings/v6088/
[756]: /en/docs/warnings/v6089/
[757]: /en/docs/warnings/v6090/
[758]: /en/docs/warnings/v6091/
[759]: /en/docs/warnings/v6092/
[760]: /en/docs/warnings/v6093/
[761]: /en/docs/warnings/v6094/
[762]: /en/docs/warnings/v6095/
[763]: /en/docs/warnings/v6096/
[764]: /en/docs/warnings/v6097/
[765]: /en/docs/warnings/v6098/
[766]: /en/docs/warnings/v6099/
[767]: /en/docs/warnings/v6100/
[768]: /en/docs/warnings/v6101/
[769]: /en/docs/warnings/v6102/
[770]: /en/docs/warnings/v6103/
[771]: /en/docs/warnings/v6104/
[772]: /en/docs/warnings/v6105/
[773]: /en/docs/warnings/v6106/
[774]: /en/docs/warnings/v6107/
[775]: /en/docs/warnings/v6108/
[776]: /en/docs/warnings/v6109/
[777]: /en/docs/warnings/v6110/
[778]: /en/docs/warnings/v6111/
[779]: /en/docs/warnings/v6112/
[780]: /en/docs/warnings/v6113/
[781]: /en/docs/warnings/v6114/
[782]: /en/docs/warnings/v6115/
[783]: /en/docs/warnings/v6116/
[784]: /en/docs/warnings/v6117/
[785]: /en/docs/warnings/v6118/
[786]: /en/docs/warnings/v6119/
[787]: /en/docs/warnings/v6120/
[788]: /en/docs/warnings/v6121/
[789]: /en/docs/warnings/v6122/
[790]: /en/docs/warnings/v6123/
[791]: /en/docs/warnings/v6124/
[792]: /en/docs/warnings/v6125/
[793]: /en/docs/warnings/v6126/
[794]: /en/docs/warnings/v6127/
[795]: /en/docs/warnings/v6128/
[796]: /en/docs/warnings/v6129/
[797]: /en/docs/warnings/v6130/
[798]: /en/docs/warnings/v6131/
[799]: /en/docs/warnings/v6132/
[800]: /en/docs/warnings/v801/
[801]: /en/docs/warnings/v802/
[802]: /en/docs/warnings/v803/
[803]: /en/docs/warnings/v804/
[804]: /en/docs/warnings/v805/
[805]: /en/docs/warnings/v806/
[806]: /en/docs/warnings/v807/
[807]: /en/docs/warnings/v808/
[808]: /en/docs/warnings/v809/
[809]: /en/docs/warnings/v810/
[810]: /en/docs/warnings/v811/
[811]: /en/docs/warnings/v812/
[812]: /en/docs/warnings/v813/
[813]: /en/docs/warnings/v814/
[814]: /en/docs/warnings/v815/
[815]: /en/docs/warnings/v816/
[816]: /en/docs/warnings/v817/
[817]: /en/docs/warnings/v818/
[818]: /en/docs/warnings/v819/
[819]: /en/docs/warnings/v820/
[820]: /en/docs/warnings/v821/
[821]: /en/docs/warnings/v822/
[822]: /en/docs/warnings/v823/
[823]: /en/docs/warnings/v824/
[824]: /en/docs/warnings/v825/
[825]: /en/docs/warnings/v826/
[826]: /en/docs/warnings/v827/
[827]: /en/docs/warnings/v828/
[828]: /en/docs/warnings/v829/
[829]: /en/docs/warnings/v830/
[830]: /en/docs/warnings/v831/
[831]: /en/docs/warnings/v832/
[832]: /en/docs/warnings/v833/
[833]: /en/docs/warnings/v834/
[834]: /en/docs/warnings/v835/
[835]: /en/docs/warnings/v836/
[836]: /en/docs/warnings/v837/
[837]: /en/docs/warnings/v838/
[838]: /en/docs/warnings/v839/
[839]: /en/docs/warnings/v4001/
[840]: /en/docs/warnings/v4002/
[841]: /en/docs/warnings/v4003/
[842]: /en/docs/warnings/v4004/
[843]: /en/docs/warnings/v4005/
[844]: /en/docs/warnings/v4006/
[845]: /en/docs/warnings/v4007/
[846]: /en/docs/warnings/v4008/
[847]: /en/docs/warnings/v101/
[848]: /en/docs/warnings/v102/
[849]: /en/docs/warnings/v103/
[850]: /en/docs/warnings/v104/
[851]: /en/docs/warnings/v105/
[852]: /en/docs/warnings/v106/
[853]: /en/docs/warnings/v107/
[854]: /en/docs/warnings/v108/
[855]: /en/docs/warnings/v109/
[856]: /en/docs/warnings/v110/
[857]: /en/docs/warnings/v111/
[858]: /en/docs/warnings/v112/
[859]: /en/docs/warnings/v113/
[860]: /en/docs/warnings/v114/
[861]: /en/docs/warnings/v115/
[862]: /en/docs/warnings/v116/
[863]: /en/docs/warnings/v117/
[864]: /en/docs/warnings/v118/
[865]: /en/docs/warnings/v119/
[866]: /en/docs/warnings/v120/
[867]: /en/docs/warnings/v121/
[868]: /en/docs/warnings/v122/
[869]: /en/docs/warnings/v123/
[870]: /en/docs/warnings/v124/
[871]: /en/docs/warnings/v125/
[872]: /en/docs/warnings/v126/
[873]: /en/docs/warnings/v127/
[874]: /en/docs/warnings/v128/
[875]: /en/docs/warnings/v201/
[876]: /en/docs/warnings/v202/
[877]: /en/docs/warnings/v203/
[878]: /en/docs/warnings/v204/
[879]: /en/docs/warnings/v205/
[880]: /en/docs/warnings/v206/
[881]: /en/docs/warnings/v207/
[882]: /en/docs/warnings/v220/
[883]: /en/docs/warnings/v221/
[884]: /en/docs/warnings/v301/
[885]: /en/docs/warnings/v302/
[886]: /en/docs/warnings/v303/
[887]: /en/docs/warnings/v2001/
[888]: /en/docs/warnings/v2002/
[889]: /en/docs/warnings/v2003/
[890]: /en/docs/warnings/v2004/
[891]: /en/docs/warnings/v2005/
[892]: /en/docs/warnings/v2006/
[893]: /en/docs/warnings/v2007/
[894]: /en/docs/warnings/v2008/
[895]: /en/docs/warnings/v2009/
[896]: /en/docs/warnings/v2010/
[897]: /en/docs/warnings/v2011/
[898]: /en/docs/warnings/v2012/
[899]: /en/docs/warnings/v2013/
[900]: /en/docs/warnings/v2014/
[901]: /en/docs/warnings/v2015/
[902]: /en/docs/warnings/v2016/
[903]: /en/docs/warnings/v2017/
[904]: /en/docs/warnings/v2018/
[905]: /en/docs/warnings/v2019/
[906]: /en/docs/warnings/v2020/
[907]: /en/docs/warnings/v2021/
[908]: /en/docs/warnings/v2022/
[909]: /en/docs/warnings/v2023/
[910]: /en/docs/warnings/v2501/
[911]: /en/docs/warnings/v2502/
[912]: /en/docs/warnings/v2503/
[913]: /en/docs/warnings/v2504/
[914]: /en/docs/warnings/v2505/
[915]: /en/docs/warnings/v2506/
[916]: /en/docs/warnings/v2507/
[917]: /en/docs/warnings/v2508/
[918]: /en/docs/warnings/v2509/
[919]: /en/docs/warnings/v2510/
[920]: /en/docs/warnings/v2511/
[921]: /en/docs/warnings/v2512/
[922]: /en/docs/warnings/v2513/
[923]: /en/docs/warnings/v2514/
[924]: /en/docs/warnings/v2515/
[925]: /en/docs/warnings/v2516/
[926]: /en/docs/warnings/v2517/
[927]: /en/docs/warnings/v2518/
[928]: /en/docs/warnings/v2519/
[929]: /en/docs/warnings/v2520/
[930]: /en/docs/warnings/v2521/
[931]: /en/docs/warnings/v2522/
[932]: /en/docs/warnings/v2523/
[933]: /en/docs/warnings/v2524/
[934]: /en/docs/warnings/v2525/
[935]: /en/docs/warnings/v2526/
[936]: /en/docs/warnings/v2527/
[937]: /en/docs/warnings/v2528/
[938]: /en/docs/warnings/v2529/
[939]: /en/docs/warnings/v2530/
[940]: /en/docs/warnings/v2531/
[941]: /en/docs/warnings/v2532/
[942]: /en/docs/warnings/v2533/
[943]: /en/docs/warnings/v2534/
[944]: /en/docs/warnings/v2535/
[945]: /en/docs/warnings/v2536/
[946]: /en/docs/warnings/v2537/
[947]: /en/docs/warnings/v2538/
[948]: /en/docs/warnings/v2539/
[949]: /en/docs/warnings/v2540/
[950]: /en/docs/warnings/v2541/
[951]: /en/docs/warnings/v2542/
[952]: /en/docs/warnings/v2543/
[953]: /en/docs/warnings/v2544/
[954]: /en/docs/warnings/v2545/
[955]: /en/docs/warnings/v2546/
[956]: /en/docs/warnings/v2547/
[957]: /en/docs/warnings/v2548/
[958]: /en/docs/warnings/v2549/
[959]: /en/docs/warnings/v2550/
[960]: /en/docs/warnings/v2551/
[961]: /en/docs/warnings/v2552/
[962]: /en/docs/warnings/v2553/
[963]: /en/docs/warnings/v2554/
[964]: /en/docs/warnings/v2555/
[965]: /en/docs/warnings/v2556/
[966]: /en/docs/warnings/v2557/
[967]: /en/docs/warnings/v2558/
[968]: /en/docs/warnings/v2559/
[969]: /en/docs/warnings/v2560/
[970]: /en/docs/warnings/v2561/
[971]: /en/docs/warnings/v2562/
[972]: /en/docs/warnings/v2563/
[973]: /en/docs/warnings/v2564/
[974]: /en/docs/warnings/v2565/
[975]: /en/docs/warnings/v2566/
[976]: /en/docs/warnings/v2567/
[977]: /en/docs/warnings/v2568/
[978]: /en/docs/warnings/v2569/
[979]: /en/docs/warnings/v2570/
[980]: /en/docs/warnings/v2571/
[981]: /en/docs/warnings/v2572/
[982]: /en/docs/warnings/v2573/
[983]: /en/docs/warnings/v2574/
[984]: /en/docs/warnings/v2575/
[985]: /en/docs/warnings/v2576/
[986]: /en/docs/warnings/v2577/
[987]: /en/docs/warnings/v2578/
[988]: /en/docs/warnings/v2579/
[989]: /en/docs/warnings/v2580/
[990]: /en/docs/warnings/v2581/
[991]: /en/docs/warnings/v2582/
[992]: /en/docs/warnings/v2583/
[993]: /en/docs/warnings/v2584/
[994]: /en/docs/warnings/v2585/
[995]: /en/docs/warnings/v2586/
[996]: /en/docs/warnings/v2587/
[997]: /en/docs/warnings/v2588/
[998]: /en/docs/warnings/v2589/
[999]: /en/docs/warnings/v2590/
[1000]: /en/docs/warnings/v2591/
[1001]: /en/docs/warnings/v2592/
[1002]: /en/docs/warnings/v2593/
[1003]: /en/docs/warnings/v2594/
[1004]: /en/docs/warnings/v2595/
[1005]: /en/docs/warnings/v2596/
[1006]: /en/docs/warnings/v2597/
[1007]: /en/docs/warnings/v2598/
[1008]: /en/docs/warnings/v2599/
[1009]: /en/docs/warnings/v2600/
[1010]: /en/docs/warnings/v2601/
[1011]: /en/docs/warnings/v2602/
[1012]: /en/docs/warnings/v2603/
[1013]: /en/docs/warnings/v2604/
[1014]: /en/docs/warnings/v2605/
[1015]: /en/docs/warnings/v2606/
[1016]: /en/docs/warnings/v2607/
[1017]: /en/docs/warnings/v2608/
[1018]: /en/docs/warnings/v2609/
[1019]: /en/docs/warnings/v2610/
[1020]: /en/docs/warnings/v2611/
[1021]: /en/docs/warnings/v2612/
[1022]: /en/docs/warnings/v2613/
[1023]: /en/docs/warnings/v2614/
[1024]: /en/docs/warnings/v2615/
[1025]: /en/docs/warnings/v2616/
[1026]: /en/docs/warnings/v2617/
[1027]: /en/docs/warnings/v2618/
[1028]: /en/docs/warnings/v2619/
[1029]: /en/docs/warnings/v2620/
[1030]: /en/docs/warnings/v2621/
[1031]: /en/docs/warnings/v2622/
[1032]: /en/docs/warnings/v2623/
[1033]: /en/docs/warnings/v2624/
[1034]: /en/docs/warnings/v2625/
[1035]: /en/docs/warnings/v2626/
[1036]: /en/docs/warnings/v2627/
[1037]: /en/docs/warnings/v2628/
[1038]: /en/docs/warnings/v2629/
[1039]: /en/docs/warnings/v2630/
[1040]: /en/docs/warnings/v2631/
[1041]: /en/docs/warnings/v2632/
[1042]: /en/docs/warnings/v2633/
[1043]: /en/docs/warnings/v2634/
[1044]: /en/docs/warnings/v2635/
[1045]: /en/docs/warnings/v2636/
[1046]: /en/docs/warnings/v2637/
[1047]: /en/docs/warnings/v2638/
[1048]: /en/docs/warnings/v2639/
[1049]: /en/docs/warnings/v2640/
[1050]: /en/docs/warnings/v2641/
[1051]: /en/docs/warnings/v2642/
[1052]: /en/docs/warnings/v2643/
[1053]: /en/docs/warnings/v2644/
[1054]: /en/docs/warnings/v2645/
[1055]: /en/docs/warnings/v2646/
[1056]: /en/docs/warnings/v2647/
[1057]: /en/docs/warnings/v2648/
[1058]: /en/docs/warnings/v2649/
[1059]: /en/docs/warnings/v2650/
[1060]: /en/docs/warnings/v2651/
[1061]: /en/docs/warnings/v2652/
[1062]: /en/docs/warnings/v2653/
[1063]: /en/docs/warnings/v2654/
[1064]: /en/docs/warnings/v2655/
[1065]: /en/docs/warnings/v2656/
[1066]: /en/docs/warnings/v2657/
[1067]: /en/docs/warnings/v2658/
[1068]: /en/docs/warnings/v2659/
[1069]: /en/docs/warnings/v2660/
[1070]: /en/docs/warnings/v2661/
[1071]: /en/docs/warnings/v2662/
[1072]: /en/docs/warnings/v2663/
[1073]: /en/docs/warnings/v2664/
[1074]: /en/docs/warnings/v2665/
[1075]: /en/docs/warnings/v2666/
[1076]: /en/docs/warnings/v3501/
[1077]: /en/docs/warnings/v3502/
[1078]: /en/docs/warnings/v3503/
[1079]: /en/docs/warnings/v3504/
[1080]: /en/docs/warnings/v3505/
[1081]: /en/docs/warnings/v3506/
[1082]: /en/docs/warnings/v3507/
[1083]: /en/docs/warnings/v3508/
[1084]: /en/docs/warnings/v3509/
[1085]: /en/docs/warnings/v3510/
[1086]: /en/docs/warnings/v3511/
[1087]: /en/docs/warnings/v3512/
[1088]: /en/docs/warnings/v3513/
[1089]: /en/docs/warnings/v3514/
[1090]: /en/docs/warnings/v3515/
[1091]: /en/docs/warnings/v3516/
[1092]: /en/docs/warnings/v3517/
[1093]: /en/docs/warnings/v3518/
[1094]: /en/docs/warnings/v3519/
[1095]: /en/docs/warnings/v3520/
[1096]: /en/docs/warnings/v3521/
[1097]: /en/docs/warnings/v3522/
[1098]: /en/docs/warnings/v3523/
[1099]: /en/docs/warnings/v3524/
[1100]: /en/docs/warnings/v3525/
[1101]: /en/docs/warnings/v3526/
[1102]: /en/docs/warnings/v3527/
[1103]: /en/docs/warnings/v3528/
[1104]: /en/docs/warnings/v3529/
[1105]: /en/docs/warnings/v3530/
[1106]: /en/docs/warnings/v3531/
[1107]: /en/docs/warnings/v3532/
[1108]: /en/docs/warnings/v3533/
[1109]: /en/docs/warnings/v3534/
[1110]: /en/docs/warnings/v3535/
[1111]: /en/docs/warnings/v3536/
[1112]: /en/docs/warnings/v3537/
[1113]: /en/docs/warnings/v3538/
[1114]: /en/docs/warnings/v3539/
[1115]: /en/docs/warnings/v3540/
[1116]: /en/docs/warnings/v3541/
[1117]: /en/docs/warnings/v3542/
[1118]: /en/docs/warnings/v3543/
[1119]: /en/docs/warnings/v3544/
[1120]: /en/docs/warnings/v3545/
[1121]: /en/docs/warnings/v3546/
[1122]: /en/docs/warnings/v3547/
[1123]: /en/docs/warnings/v3548/
[1124]: /en/docs/warnings/v3549/
[1125]: /en/docs/warnings/v3550/
[1126]: /en/docs/warnings/v3551/
[1127]: /en/docs/warnings/v3552/
[1128]: /en/docs/warnings/v3553/
[1129]: /en/docs/warnings/v3554/
[1130]: /en/docs/warnings/v3555/
[1131]: /en/docs/warnings/v5001/
[1132]: /en/docs/warnings/v5002/
[1133]: /en/docs/warnings/v5003/
[1134]: /en/docs/warnings/v5004/
[1135]: /en/docs/warnings/v5005/
[1136]: /en/docs/warnings/v5006/
[1137]: /en/docs/warnings/v5007/
[1138]: /en/docs/warnings/v5008/
[1139]: /en/docs/warnings/v5009/
[1140]: /en/docs/warnings/v5010/
[1141]: /en/docs/warnings/v5011/
[1142]: /en/docs/warnings/v5012/
[1143]: /en/docs/warnings/v5013/
[1144]: /en/docs/warnings/v5014/
[1145]: /en/docs/warnings/v5601/
[1146]: /en/docs/warnings/v5602/
[1147]: /en/docs/warnings/v5603/
[1148]: /en/docs/warnings/v5604/
[1149]: /en/docs/warnings/v5605/
[1150]: /en/docs/warnings/v5606/
[1151]: /en/docs/warnings/v5607/
[1152]: /en/docs/warnings/v5608/
[1153]: /en/docs/warnings/v5609/
[1154]: /en/docs/warnings/v5610/
[1155]: /en/docs/warnings/v5611/
[1156]: /en/docs/warnings/v5612/
[1157]: /en/docs/warnings/v5613/
[1158]: /en/docs/warnings/v5614/
[1159]: /en/docs/warnings/v5615/
[1160]: /en/docs/warnings/v5616/
[1161]: /en/docs/warnings/v5617/
[1162]: /en/docs/warnings/v5618/
[1163]: /en/docs/warnings/v5619/
[1164]: /en/docs/warnings/v5620/
[1165]: /en/docs/warnings/v5621/
[1166]: /en/docs/warnings/v5622/
[1167]: /en/docs/warnings/v5623/
[1168]: /en/docs/warnings/v5624/
[1169]: /en/docs/warnings/v5625/
[1170]: /en/docs/warnings/v5626/
[1171]: /en/docs/warnings/v5627/
[1172]: /en/docs/warnings/v5628/
[1173]: /en/docs/warnings/v5629/
[1174]: /en/docs/warnings/v5630/
[1175]: /en/docs/warnings/v5631/
[1176]: /en/docs/warnings/v5301/
[1177]: /en/docs/warnings/v5302/
[1178]: /en/docs/warnings/v5303/
[1179]: /en/docs/warnings/v5304/
[1180]: /en/docs/warnings/v5305/
[1181]: /en/docs/warnings/v5306/
[1182]: /en/docs/warnings/v5307/
[1183]: /en/docs/warnings/v5308/
[1184]: /en/docs/warnings/v5309/
[1185]: /en/docs/warnings/v5310/
[1186]: /en/docs/warnings/v5311/
[1187]: /en/docs/warnings/v5312/
[1188]: /en/docs/warnings/v5313/
[1189]: /en/docs/warnings/v5314/
[1190]: /en/docs/warnings/v5315/
[1191]: /en/docs/warnings/v5316/
[1192]: /en/docs/warnings/v5317/
[1193]: /en/docs/warnings/v5318/
[1194]: /en/docs/warnings/v5319/
[1195]: /en/docs/warnings/v5320/
[1196]: /en/docs/warnings/v5321/
[1197]: /en/docs/warnings/v5322/
[1198]: /en/docs/warnings/v5323/
[1199]: /en/docs/warnings/v5324/
[1200]: /en/docs/warnings/v5325/
[1201]: /en/docs/warnings/v5326/
[1202]: /en/docs/warnings/v5327/
[1203]: /en/docs/warnings/v5328/
[1204]: /en/docs/warnings/v5329/
[1205]: /en/docs/warnings/v5330/
[1206]: /en/docs/warnings/v5331/
[1207]: /en/docs/warnings/v5332/
[1208]: /en/docs/warnings/v5333/
[1209]: /en/docs/warnings/v5334/
[1210]: /en/docs/warnings/v5335/
[1211]: /en/docs/warnings/v5336/
[1212]: /en/docs/warnings/v5337/
[1213]: /en/docs/warnings/v5338/
[1214]: /en/docs/warnings/v001/
[1215]: /en/docs/warnings/v002/
[1216]: /en/docs/warnings/v003/
[1217]: /en/docs/warnings/v004/
[1218]: /en/docs/warnings/v005/
[1219]: /en/docs/warnings/v006/
[1220]: /en/docs/warnings/v007/
[1221]: /en/docs/warnings/v008/
[1222]: /en/docs/warnings/v010/
[1223]: /en/docs/warnings/v011/
[1224]: /en/docs/warnings/v012/
[1225]: /en/docs/warnings/v013/
[1226]: /en/docs/warnings/v014/
[1227]: /en/docs/warnings/v015/
[1228]: /en/docs/warnings/v016/
[1229]: /en/docs/warnings/v017/
[1230]: /en/docs/warnings/v018/
[1231]: /en/docs/warnings/v019/
[1232]: /en/docs/warnings/v020/
[1233]: /en/docs/warnings/v051/
[1234]: /en/docs/warnings/v052/
[1235]: /en/docs/warnings/v061/
[1236]: /en/docs/warnings/v062/
[1237]: /en/docs/warnings/v063/
[1238]: /en/docs/warnings/v064/
[1239]: /en/docs/warnings/v065/
[1240]: https://pvs-studio.com/en/privacy-policy/
