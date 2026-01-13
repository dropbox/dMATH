* triton.language
* [ View page source][1]

# triton.language[¶][2]

## Programming Model[¶][3]

──────────────────────┬─────────────────────────────────────────────────────────────────────
[`tensor`][4]         │Represents an N-dimensional array of values or pointers.             
──────────────────────┼─────────────────────────────────────────────────────────────────────
[`tensor_descriptor`][│A descriptor representing a tensor in global memory.                 
5]                    │                                                                     
──────────────────────┼─────────────────────────────────────────────────────────────────────
[`program_id`][6]     │Returns the id of the current program instance along the given       
                      │`axis`.                                                              
──────────────────────┼─────────────────────────────────────────────────────────────────────
[`num_programs`][7]   │Returns the number of program instances launched along the given     
                      │`axis`.                                                              
──────────────────────┴─────────────────────────────────────────────────────────────────────

## Creation Ops[¶][8]

───────────────┬────────────────────────────────────────────────────────────────────────────
[`arange`][9]  │Returns contiguous values within the half-open interval `[start, end)`.     
───────────────┼────────────────────────────────────────────────────────────────────────────
[`cat`][10]    │Concatenate the given blocks                                                
───────────────┼────────────────────────────────────────────────────────────────────────────
[`full`][11]   │Returns a tensor filled with the scalar value for the given `shape` and     
               │`dtype`.                                                                    
───────────────┼────────────────────────────────────────────────────────────────────────────
[`zeros`][12]  │Returns a tensor filled with the scalar value 0 for the given `shape` and   
               │`dtype`.                                                                    
───────────────┼────────────────────────────────────────────────────────────────────────────
[`zeros_like`][│Returns a tensor of zeros with the same shape and type as a given tensor.   
13]            │                                                                            
───────────────┼────────────────────────────────────────────────────────────────────────────
[`cast`][14]   │Casts a tensor to the given `dtype`.                                        
───────────────┴────────────────────────────────────────────────────────────────────────────

## Shape Manipulation Ops[¶][15]

────────────────┬───────────────────────────────────────────────────────────────────────────────────
[`broadcast`][16│Tries to broadcast the two given blocks to a common compatible shape.              
]               │                                                                                   
────────────────┼───────────────────────────────────────────────────────────────────────────────────
[`broadcast_to`]│Tries to broadcast the given tensor to a new `shape`.                              
[17]            │                                                                                   
────────────────┼───────────────────────────────────────────────────────────────────────────────────
[`expand_dims`][│Expand the shape of a tensor, by inserting new length-1 dimensions.                
18]             │                                                                                   
────────────────┼───────────────────────────────────────────────────────────────────────────────────
[`interleave`][1│Interleaves the values of two tensors along their last dimension.                  
9]              │                                                                                   
────────────────┼───────────────────────────────────────────────────────────────────────────────────
[`join`][20]    │Join the given tensors in a new, minor dimension.                                  
────────────────┼───────────────────────────────────────────────────────────────────────────────────
[`permute`][21] │Permutes the dimensions of a tensor.                                               
────────────────┼───────────────────────────────────────────────────────────────────────────────────
[`ravel`][22]   │Returns a contiguous flattened view of `x`.                                        
────────────────┼───────────────────────────────────────────────────────────────────────────────────
[`reshape`][23] │Returns a tensor with the same number of elements as input but with the provided   
                │shape.                                                                             
────────────────┼───────────────────────────────────────────────────────────────────────────────────
[`split`][24]   │Split a tensor in two along its last dim, which must have size 2.                  
────────────────┼───────────────────────────────────────────────────────────────────────────────────
[`trans`][25]   │Permutes the dimensions of a tensor.                                               
────────────────┼───────────────────────────────────────────────────────────────────────────────────
[`view`][26]    │Returns a tensor with the same elements as input but a different shape.            
────────────────┴───────────────────────────────────────────────────────────────────────────────────

## Linear Algebra Ops[¶][27]

───────────────┬────────────────────────────────────────────────────────────────
[`dot`][28]    │Returns the matrix product of two blocks.                       
───────────────┼────────────────────────────────────────────────────────────────
[`dot_scaled`][│Returns the matrix product of two blocks in microscaling format.
29]            │                                                                
───────────────┴────────────────────────────────────────────────────────────────

## Memory/Pointer Ops[¶][30]

───────────────────────┬────────────────────────────────────────────────────────────────────────────
[`load`][31]           │Return a tensor of data whose values are loaded from memory at location     
                       │defined by pointer:                                                         
───────────────────────┼────────────────────────────────────────────────────────────────────────────
[`store`][32]          │Store a tensor of data into memory locations defined by pointer.            
───────────────────────┼────────────────────────────────────────────────────────────────────────────
[`make_tensor_descripto│Make a tensor descriptor object                                             
r`][33]                │                                                                            
───────────────────────┼────────────────────────────────────────────────────────────────────────────
[`load_tensor_descripto│Load a block of data from a tensor descriptor.                              
r`][34]                │                                                                            
───────────────────────┼────────────────────────────────────────────────────────────────────────────
[`store_tensor_descript│Store a block of data to a tensor descriptor.                               
or`][35]               │                                                                            
───────────────────────┼────────────────────────────────────────────────────────────────────────────
[`make_block_ptr`][36] │Returns a pointer to a block in a parent tensor                             
───────────────────────┼────────────────────────────────────────────────────────────────────────────
[`advance`][37]        │Advance a block pointer                                                     
───────────────────────┴────────────────────────────────────────────────────────────────────────────

## Indexing Ops[¶][38]

─────────┬──────────────────────────────────────────────────────────────────────────────────────────
[`flip`][│Flips a tensor x along the dimension dim.                                                 
39]      │                                                                                          
─────────┼──────────────────────────────────────────────────────────────────────────────────────────
[`where`]│Returns a tensor of elements from either `x` or `y`, depending on `condition`.            
[40]     │                                                                                          
─────────┼──────────────────────────────────────────────────────────────────────────────────────────
[`swizzle│Transforms the indices of a row-major size_i * size_j matrix into the indices of a        
2d`][41] │column-major matrix for each group of size_g rows.                                        
─────────┴──────────────────────────────────────────────────────────────────────────────────────────

## Math Ops[¶][42]

───────────┬────────────────────────────────────────────────────────────────────────────────────────
[`abs`][43]│Computes the element-wise absolute value of `x`.                                        
───────────┼────────────────────────────────────────────────────────────────────────────────────────
[`cdiv`][44│Computes the ceiling division of `x` by `div`                                           
]          │                                                                                        
───────────┼────────────────────────────────────────────────────────────────────────────────────────
[`ceil`][45│Computes the element-wise ceil of `x`.                                                  
]          │                                                                                        
───────────┼────────────────────────────────────────────────────────────────────────────────────────
[`clamp`][4│Clamps the input tensor `x` within the range [min, max].                                
6]         │                                                                                        
───────────┼────────────────────────────────────────────────────────────────────────────────────────
[`cos`][47]│Computes the element-wise cosine of `x`.                                                
───────────┼────────────────────────────────────────────────────────────────────────────────────────
[`div_rn`][│Computes the element-wise precise division (rounding to nearest wrt the IEEE standard)  
48]        │of `x` and `y`.                                                                         
───────────┼────────────────────────────────────────────────────────────────────────────────────────
[`erf`][49]│Computes the element-wise error function of `x`.                                        
───────────┼────────────────────────────────────────────────────────────────────────────────────────
[`exp`][50]│Computes the element-wise exponential of `x`.                                           
───────────┼────────────────────────────────────────────────────────────────────────────────────────
[`exp2`][51│Computes the element-wise exponential (base 2) of `x`.                                  
]          │                                                                                        
───────────┼────────────────────────────────────────────────────────────────────────────────────────
[`fdiv`][52│Computes the element-wise fast division of `x` and `y`.                                 
]          │                                                                                        
───────────┼────────────────────────────────────────────────────────────────────────────────────────
[`floor`][5│Computes the element-wise floor of `x`.                                                 
3]         │                                                                                        
───────────┼────────────────────────────────────────────────────────────────────────────────────────
[`fma`][54]│Computes the element-wise fused multiply-add of `x`, `y`, and `z`.                      
───────────┼────────────────────────────────────────────────────────────────────────────────────────
[`log`][55]│Computes the element-wise natural logarithm of `x`.                                     
───────────┼────────────────────────────────────────────────────────────────────────────────────────
[`log2`][56│Computes the element-wise logarithm (base 2) of `x`.                                    
]          │                                                                                        
───────────┼────────────────────────────────────────────────────────────────────────────────────────
[`maximum`]│Computes the element-wise maximum of `x` and `y`.                                       
[57]       │                                                                                        
───────────┼────────────────────────────────────────────────────────────────────────────────────────
[`minimum`]│Computes the element-wise minimum of `x` and `y`.                                       
[58]       │                                                                                        
───────────┼────────────────────────────────────────────────────────────────────────────────────────
[`rsqrt`][5│Computes the element-wise inverse square root of `x`.                                   
9]         │                                                                                        
───────────┼────────────────────────────────────────────────────────────────────────────────────────
[`sigmoid`]│Computes the element-wise sigmoid of `x`.                                               
[60]       │                                                                                        
───────────┼────────────────────────────────────────────────────────────────────────────────────────
[`sin`][61]│Computes the element-wise sine of `x`.                                                  
───────────┼────────────────────────────────────────────────────────────────────────────────────────
[`softmax`]│Computes the element-wise softmax of `x`.                                               
[62]       │                                                                                        
───────────┼────────────────────────────────────────────────────────────────────────────────────────
[`sqrt`][63│Computes the element-wise fast square root of `x`.                                      
]          │                                                                                        
───────────┼────────────────────────────────────────────────────────────────────────────────────────
[`sqrt_rn`]│Computes the element-wise precise square root (rounding to nearest wrt the IEEE         
[64]       │standard) of `x`.                                                                       
───────────┼────────────────────────────────────────────────────────────────────────────────────────
[`umulhi`][│Computes the element-wise most significant N bits of the 2N-bit product of `x` and `y`. 
65]        │                                                                                        
───────────┴────────────────────────────────────────────────────────────────────────────────────────

## Reduction Ops[¶][66]

────────────┬───────────────────────────────────────────────────────────────────────────────────
[`argmax`][6│Returns the maximum index of all elements in the `input` tensor along the provided 
7]          │`axis`                                                                             
────────────┼───────────────────────────────────────────────────────────────────────────────────
[`argmin`][6│Returns the minimum index of all elements in the `input` tensor along the provided 
8]          │`axis`                                                                             
────────────┼───────────────────────────────────────────────────────────────────────────────────
[`max`][69] │Returns the maximum of all elements in the `input` tensor along the provided `axis`
────────────┼───────────────────────────────────────────────────────────────────────────────────
[`min`][70] │Returns the minimum of all elements in the `input` tensor along the provided `axis`
────────────┼───────────────────────────────────────────────────────────────────────────────────
[`reduce`][7│Applies the combine_fn to all elements in `input` tensors along the provided `axis`
1]          │                                                                                   
────────────┼───────────────────────────────────────────────────────────────────────────────────
[`sum`][72] │Returns the sum of all elements in the `input` tensor along the provided `axis`    
────────────┼───────────────────────────────────────────────────────────────────────────────────
[`xor_sum`][│Returns the xor sum of all elements in the `input` tensor along the provided `axis`
73]         │                                                                                   
────────────┴───────────────────────────────────────────────────────────────────────────────────

## Scan/Sort Ops[¶][74]

───────────────┬────────────────────────────────────────────────────────────────────────────────────
[`associative_s│Applies the combine_fn to each elements with a carry in `input` tensors along the   
can`][75]      │provided `axis` and update the carry                                                
───────────────┼────────────────────────────────────────────────────────────────────────────────────
[`cumprod`][76]│Returns the cumprod of all elements in the `input` tensor along the provided `axis` 
───────────────┼────────────────────────────────────────────────────────────────────────────────────
[`cumsum`][77] │Returns the cumsum of all elements in the `input` tensor along the provided `axis`  
───────────────┼────────────────────────────────────────────────────────────────────────────────────
[`histogram`][7│computes an histogram based on input tensor with num_bins bins, the bins have a     
8]             │width of 1 and start at 0.                                                          
───────────────┼────────────────────────────────────────────────────────────────────────────────────
[`sort`][79]   │                                                                                    
───────────────┼────────────────────────────────────────────────────────────────────────────────────
[`gather`][80] │Gather from a tensor along a given dimension.                                       
───────────────┴────────────────────────────────────────────────────────────────────────────────────

## Atomic Ops[¶][81]

────────────────┬───────────────────────────────────────────────────────────────────────────────
[`atomic_add`][8│Performs an atomic add at the memory location specified by `pointer`.          
2]              │                                                                               
────────────────┼───────────────────────────────────────────────────────────────────────────────
[`atomic_and`][8│Performs an atomic logical and at the memory location specified by `pointer`.  
3]              │                                                                               
────────────────┼───────────────────────────────────────────────────────────────────────────────
[`atomic_cas`][8│Performs an atomic compare-and-swap at the memory location specified by        
4]              │`pointer`.                                                                     
────────────────┼───────────────────────────────────────────────────────────────────────────────
[`atomic_max`][8│Performs an atomic max at the memory location specified by `pointer`.          
5]              │                                                                               
────────────────┼───────────────────────────────────────────────────────────────────────────────
[`atomic_min`][8│Performs an atomic min at the memory location specified by `pointer`.          
6]              │                                                                               
────────────────┼───────────────────────────────────────────────────────────────────────────────
[`atomic_or`][87│Performs an atomic logical or at the memory location specified by `pointer`.   
]               │                                                                               
────────────────┼───────────────────────────────────────────────────────────────────────────────
[`atomic_xchg`][│Performs an atomic exchange at the memory location specified by `pointer`.     
88]             │                                                                               
────────────────┼───────────────────────────────────────────────────────────────────────────────
[`atomic_xor`][8│Performs an atomic logical xor at the memory location specified by `pointer`.  
9]              │                                                                               
────────────────┴───────────────────────────────────────────────────────────────────────────────

## Random Number Generation[¶][90]

────────────┬───────────────────────────────────────────────────────────────────────────────────────
[`randint4x`│Given a `seed` scalar and an `offset` block, returns four blocks of random `int32`.    
][91]       │                                                                                       
────────────┼───────────────────────────────────────────────────────────────────────────────────────
[`randint`][│Given a `seed` scalar and an `offset` block, returns a single block of random `int32`. 
92]         │                                                                                       
────────────┼───────────────────────────────────────────────────────────────────────────────────────
[`rand`][93]│Given a `seed` scalar and an `offset` block, returns a block of random `float32` in    
            │\(U(0, 1)\).                                                                           
────────────┼───────────────────────────────────────────────────────────────────────────────────────
[`randn`][94│Given a `seed` scalar and an `offset` block, returns a block of random `float32` in    
]           │\(\mathcal{N}(0, 1)\).                                                                 
────────────┴───────────────────────────────────────────────────────────────────────────────────────

## Iterators[¶][95]

─────────────────┬────────────────────────────────────
[`range`][96]    │Iterator that counts upward forever.
─────────────────┼────────────────────────────────────
[`static_range`][│Iterator that counts upward forever.
97]              │                                    
─────────────────┴────────────────────────────────────

## Inline Assembly[¶][98]

───────────────────────────┬──────────────────────────────────────
[`inline_asm_elementwise`][│Execute inline assembly over a tensor.
99]                        │                                      
───────────────────────────┴──────────────────────────────────────

## Compiler Hint Ops[¶][100]

───────────────────┬────────────────────────────────────────────────────────────────────────
[`assume`][101]    │Allow compiler to assume the `cond` is True.                            
───────────────────┼────────────────────────────────────────────────────────────────────────
[`debug_barrier`][1│Insert a barrier to synchronize all threads in a block.                 
02]                │                                                                        
───────────────────┼────────────────────────────────────────────────────────────────────────
[`max_constancy`][1│Let the compiler know that the value first values in `input` are        
03]                │constant.                                                               
───────────────────┼────────────────────────────────────────────────────────────────────────
[`max_contiguous`][│Let the compiler know that the value first values in `input` are        
104]               │contiguous.                                                             
───────────────────┼────────────────────────────────────────────────────────────────────────
[`multiple_of`][105│Let the compiler know that the values in `input` are all multiples of   
]                  │`value`.                                                                
───────────────────┴────────────────────────────────────────────────────────────────────────

## Debug Ops[¶][106]

──────────────────┬────────────────────────────────────────────────
[`static_print`][1│Print the values at compile time.               
07]               │                                                
──────────────────┼────────────────────────────────────────────────
[`static_assert`][│Assert the condition at compile time.           
108]              │                                                
──────────────────┼────────────────────────────────────────────────
[`device_print`][1│Print the values at runtime from the device.    
09]               │                                                
──────────────────┼────────────────────────────────────────────────
[`device_assert`][│Assert the condition at runtime from the device.
110]              │                                                
──────────────────┴────────────────────────────────────────────────
[ Previous][111] [Next ][112]

© Copyright 2020, Philippe Tillet.

Built with [Sphinx][113] using a [theme][114] provided by [Read the Docs][115].

[1]: ../_sources/python-api/triton.language.rst.txt
[2]: #triton-language
[3]: #programming-model
[4]: generated/triton.language.tensor.html#triton.language.tensor
[5]: generated/triton.language.tensor_descriptor.html#triton.language.tensor_descriptor
[6]: generated/triton.language.program_id.html#triton.language.program_id
[7]: generated/triton.language.num_programs.html#triton.language.num_programs
[8]: #creation-ops
[9]: generated/triton.language.arange.html#triton.language.arange
[10]: generated/triton.language.cat.html#triton.language.cat
[11]: generated/triton.language.full.html#triton.language.full
[12]: generated/triton.language.zeros.html#triton.language.zeros
[13]: generated/triton.language.zeros_like.html#triton.language.zeros_like
[14]: generated/triton.language.cast.html#triton.language.cast
[15]: #shape-manipulation-ops
[16]: generated/triton.language.broadcast.html#triton.language.broadcast
[17]: generated/triton.language.broadcast_to.html#triton.language.broadcast_to
[18]: generated/triton.language.expand_dims.html#triton.language.expand_dims
[19]: generated/triton.language.interleave.html#triton.language.interleave
[20]: generated/triton.language.join.html#triton.language.join
[21]: generated/triton.language.permute.html#triton.language.permute
[22]: generated/triton.language.ravel.html#triton.language.ravel
[23]: generated/triton.language.reshape.html#triton.language.reshape
[24]: generated/triton.language.split.html#triton.language.split
[25]: generated/triton.language.trans.html#triton.language.trans
[26]: generated/triton.language.view.html#triton.language.view
[27]: #linear-algebra-ops
[28]: generated/triton.language.dot.html#triton.language.dot
[29]: generated/triton.language.dot_scaled.html#triton.language.dot_scaled
[30]: #memory-pointer-ops
[31]: generated/triton.language.load.html#triton.language.load
[32]: generated/triton.language.store.html#triton.language.store
[33]: generated/triton.language.make_tensor_descriptor.html#triton.language.make_tensor_descriptor
[34]: generated/triton.language.load_tensor_descriptor.html#triton.language.load_tensor_descriptor
[35]: generated/triton.language.store_tensor_descriptor.html#triton.language.store_tensor_descriptor
[36]: generated/triton.language.make_block_ptr.html#triton.language.make_block_ptr
[37]: generated/triton.language.advance.html#triton.language.advance
[38]: #indexing-ops
[39]: generated/triton.language.flip.html#triton.language.flip
[40]: generated/triton.language.where.html#triton.language.where
[41]: generated/triton.language.swizzle2d.html#triton.language.swizzle2d
[42]: #math-ops
[43]: generated/triton.language.abs.html#triton.language.abs
[44]: generated/triton.language.cdiv.html#triton.language.cdiv
[45]: generated/triton.language.ceil.html#triton.language.ceil
[46]: generated/triton.language.clamp.html#triton.language.clamp
[47]: generated/triton.language.cos.html#triton.language.cos
[48]: generated/triton.language.div_rn.html#triton.language.div_rn
[49]: generated/triton.language.erf.html#triton.language.erf
[50]: generated/triton.language.exp.html#triton.language.exp
[51]: generated/triton.language.exp2.html#triton.language.exp2
[52]: generated/triton.language.fdiv.html#triton.language.fdiv
[53]: generated/triton.language.floor.html#triton.language.floor
[54]: generated/triton.language.fma.html#triton.language.fma
[55]: generated/triton.language.log.html#triton.language.log
[56]: generated/triton.language.log2.html#triton.language.log2
[57]: generated/triton.language.maximum.html#triton.language.maximum
[58]: generated/triton.language.minimum.html#triton.language.minimum
[59]: generated/triton.language.rsqrt.html#triton.language.rsqrt
[60]: generated/triton.language.sigmoid.html#triton.language.sigmoid
[61]: generated/triton.language.sin.html#triton.language.sin
[62]: generated/triton.language.softmax.html#triton.language.softmax
[63]: generated/triton.language.sqrt.html#triton.language.sqrt
[64]: generated/triton.language.sqrt_rn.html#triton.language.sqrt_rn
[65]: generated/triton.language.umulhi.html#triton.language.umulhi
[66]: #reduction-ops
[67]: generated/triton.language.argmax.html#triton.language.argmax
[68]: generated/triton.language.argmin.html#triton.language.argmin
[69]: generated/triton.language.max.html#triton.language.max
[70]: generated/triton.language.min.html#triton.language.min
[71]: generated/triton.language.reduce.html#triton.language.reduce
[72]: generated/triton.language.sum.html#triton.language.sum
[73]: generated/triton.language.xor_sum.html#triton.language.xor_sum
[74]: #scan-sort-ops
[75]: generated/triton.language.associative_scan.html#triton.language.associative_scan
[76]: generated/triton.language.cumprod.html#triton.language.cumprod
[77]: generated/triton.language.cumsum.html#triton.language.cumsum
[78]: generated/triton.language.histogram.html#triton.language.histogram
[79]: generated/triton.language.sort.html#triton.language.sort
[80]: generated/triton.language.gather.html#triton.language.gather
[81]: #atomic-ops
[82]: generated/triton.language.atomic_add.html#triton.language.atomic_add
[83]: generated/triton.language.atomic_and.html#triton.language.atomic_and
[84]: generated/triton.language.atomic_cas.html#triton.language.atomic_cas
[85]: generated/triton.language.atomic_max.html#triton.language.atomic_max
[86]: generated/triton.language.atomic_min.html#triton.language.atomic_min
[87]: generated/triton.language.atomic_or.html#triton.language.atomic_or
[88]: generated/triton.language.atomic_xchg.html#triton.language.atomic_xchg
[89]: generated/triton.language.atomic_xor.html#triton.language.atomic_xor
[90]: #random-number-generation
[91]: generated/triton.language.randint4x.html#triton.language.randint4x
[92]: generated/triton.language.randint.html#triton.language.randint
[93]: generated/triton.language.rand.html#triton.language.rand
[94]: generated/triton.language.randn.html#triton.language.randn
[95]: #iterators
[96]: generated/triton.language.range.html#triton.language.range
[97]: generated/triton.language.static_range.html#triton.language.static_range
[98]: #inline-assembly
[99]: generated/triton.language.inline_asm_elementwise.html#triton.language.inline_asm_elementwise
[100]: #compiler-hint-ops
[101]: generated/triton.language.assume.html#triton.language.assume
[102]: generated/triton.language.debug_barrier.html#triton.language.debug_barrier
[103]: generated/triton.language.max_constancy.html#triton.language.max_constancy
[104]: generated/triton.language.max_contiguous.html#triton.language.max_contiguous
[105]: generated/triton.language.multiple_of.html#triton.language.multiple_of
[106]: #debug-ops
[107]: generated/triton.language.static_print.html#triton.language.static_print
[108]: generated/triton.language.static_assert.html#triton.language.static_assert
[109]: generated/triton.language.device_print.html#triton.language.device_print
[110]: generated/triton.language.device_assert.html#triton.language.device_assert
[111]: generated/triton.Config.html
[112]: generated/triton.language.tensor.html
[113]: https://www.sphinx-doc.org/
[114]: https://github.com/readthedocs/sphinx_rtd_theme
[115]: https://readthedocs.org
