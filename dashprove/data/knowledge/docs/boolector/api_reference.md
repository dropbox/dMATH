* »
* Boolector C API documentation

# Boolector C API documentation[¶][1]

## Interface[¶][2]

* [C Interface][3]
  
  * [Macros][4]
  * [Typedefs][5]
  * [Functions][6]
  * [Deprecated][7]

## Quickstart[¶][8]

> First, we create a Boolector instance:
> 
>   Btor *btor = boolector_new ();
> 
> We can configure this instance via [`boolector_set_opt()`][9] For example, if we want to enable
> model generation:
> 
>   boolector_set_opt (btor, BTOR_OPT_MODEL_GEN, 1);
> 
> For a detailed description of all configurable options, see [`boolector_set_opt()`][10].
> 
> Next, we can create expressions and assert formulas via [`boolector_assert()`][11].
> 
> Note
> 
> Boolector’s internal design is motivated by hardware design. Hence we do not distinguish between
> type *Boolean* and type *bit vector of length 1*.
> 
> If incremental usage is enabled, formulas can optionally be assumed via
> [`boolector_assume()`][12].
> 
> Note
> 
> Assumptions are invalidated after a call to [`boolector_sat()`][13].
> 
> Alternatively, we can parse an input file prior to creating and asserting expressions. For
> example, to parse an input file example.btor, we can use [`boolector_parse()`][14] (auto detects
> the input format) or [`boolector_parse_btor()`][15] (for parsing input files in [BTOR][16]
> format).
> 
> char *error_msg;
> int status;
> int result;
> FILE *fd = fopen ("example.btor", "r");
> result = boolector_parse_btor (btor, fd, "example.btor", &error_msg, &status);
> 
> In case the input issues a call to check sat (in case of SMT-LIB v2 or incremental SMT-LIB v1),
> this function either returns [`BOOLECTOR_SAT`][17], [`BOOLECTOR_UNSAT`][18] or
> [`BOOLECTOR_UNKNOWN`][19]. In any other non-error case it returns [`BOOLECTOR_PARSE_UNKNOWN`][20].
> For a more detailed description of the parsers return values, see [`boolector_parse()`][21],
> [`boolector_parse_btor()`][22]. [`boolector_parse_btor2()`][23], [`boolector_parse_smt1()`][24]
> and [`boolector_parse_smt2()`][25].
> 
> If the parser encounters an error, it returns [`BOOLECTOR_PARSE_ERROR`][26] and an explanation of
> that error is stored in `error_msg`. If the input file specifies a (known) status of the input
> formula (either satisfiable or unsatisfiable), that status is stored in `status`.
> 
> As an example for generating and asserting expressions via [`boolector_assert()`][27], consider
> the following example:
> 
> 0 < x <= 100 && 0 < y <= 100 && x * y < 100
> 
> Assume that this example is given with x and y as natural numbers. We encode it with bit-vectors
> of size 8, and to preserve semantics, we have to ensure that the multiplication does not overflow.
> 
> We first create a bit-vector sort of size 8.
> 
>   BoolectorSort bvsort8 = boolector_bitvec_sort (btor, 8);
> 
> Then, we create and assert the following expressions:
> 
>   BoolectorNode *x       = boolector_var (btor, bvsort8, "x");
>   BoolectorNode *y       = boolector_var (btor, bvsort8, "y");
>   BoolectorNode *zero    = boolector_zero (btor, bvsort8);
>   BoolectorNode *hundred = boolector_int (btor, 100, bvsort8);
> 
>   // 0 < x
>   BoolectorNode *ult_x = boolector_ult (btor, zero, x);
>   boolector_assert (btor, ult_x);
> 
>   // x <= 100
>   BoolectorNode *ulte_x = boolector_ulte (btor, x, hundred);
>   boolector_assert (btor, ulte_x);
> 
>   // 0 < y
>   BoolectorNode *ult_y = boolector_ult (btor, zero, y);
>   boolector_assert (btor, ult_y);
> 
>   // y <= 100
>   BoolectorNode *ulte_y = boolector_ulte (btor, y, hundred);
>   boolector_assert (btor, ulte_y);
> 
>   // x * y
>   BoolectorNode *mul = boolector_mul (btor, x, y);
> 
>   // x * y < 100
>   BoolectorNode *ult = boolector_ult (btor, mul, hundred);
>   boolector_assert (btor, ult);
>   BoolectorNode *umulo  = boolector_umulo (btor, x, y);
>   BoolectorNode *numulo = boolector_not (btor, umulo);  // prevent overflow
>   boolector_assert (btor, numulo);
> 
> The satisfiability of the resulting formula can be determined via [`boolector_sat()`][28].
> 
>   int result = boolector_sat (btor);
> 
> If the resulting formula is satisfiable and model generation has been enabled via
> [`boolector_set_opt()`][29], we can either print the resulting model via
> [`boolector_print_model()`][30], or query assignments of bit vector and array variables or
> uninterpreted functions via [`boolector_bv_assignment()`][31],
> [`boolector_array_assignment()`][32] and [`boolector_uf_assignment()`][33].
> 
> Note
> 
> Querying assignments is not limited to variables. You can query the assignment of any arbitrary
> expression.
> 
> The example above is satisfiable, and we can now either query the assignments of variables `x` and
> `y` or print the resulting model via [`boolector_print_model()`][34].
> 
>   const char *xstr = boolector_bv_assignment (btor, x);  // returns "00000100"
>   const char *ystr = boolector_bv_assignment (btor, y);  // returns "00010101"
> 
> Boolector supports printing models in its own format (“btor”) or in [SMT-LIB v2][35] format
> (“smt2”). We print the resulting model in [BTOR][36] format:
> 
>   boolector_print_model (btor, "btor", stdout);
> 
> A possible model is shown below and gives the assignments of bit vector variables `x` and `y`. The
> first column indicates the id of the input, the second column its assignment, and the third column
> its name (or symbol) if any.
> 
> 2 00000001 x
> 3 01011111 y
> 
> In the case that the formula includes arrays as inputs, their values at a certain index are
> indicated as follows:
> 
> 4[00] 01 A
> 
> Here, array `A` has id 4 with index and element bit width of 2, and its value at index 0 is 1.
> 
> We now print the model of the example above in [SMT-LIB v2][37] format.
> 
>   boolector_print_model (btor, "smt2", stdout);
> 
> A possible model is shown below:
> 
> (
>   (define-fun x () (_ BitVec 8) #b00000001)
>   (define-fun y () (_ BitVec 8) #b01011111)
> )
> 
> Note
> 
> Boolector internally represents arrays as uninterpreted functions and prints array models as
> models for UF.
> 
> Finally, we have to clean up all created expressions (see [Internals][38] and
> [`boolector_release()`][39]) and delete Boolector instance `btor` via [`boolector_delete()`][40].
> Queried assignment strings have to be freed via [`boolector_free_bv_assignment()`][41],
> [`boolector_free_array_assignment()`][42] and [`boolector_free_uf_assignment()`][43].
> 
>   // Release expressions
>   boolector_release (btor, x);
>   boolector_release (btor, y);
>   boolector_release (btor, zero);
>   boolector_release (btor, hundred);
>   boolector_release (btor, ult_x);
>   boolector_release (btor, ulte_x);
>   boolector_release (btor, ult_y);
>   boolector_release (btor, ulte_y);
>   boolector_release (btor, mul);
>   boolector_release (btor, ult);
>   boolector_release (btor, numulo);
>   boolector_release (btor, umulo);
> 
>   // Release assigments
>   boolector_free_bv_assignment (btor, xstr);
>   boolector_free_bv_assignment (btor, ystr);
> 
>   // Release sorts
>   boolector_release_sort (btor, bvsort8);
> 
>   // Delete Boolector instance
>   boolector_delete (btor);
> 
> The source code of the example above can be found at [examples/api/c/quickstart.c][44].

## Options[¶][45]

> Boolector can be configured either via [`boolector_set_opt()`][46], or via environment variables
> of the form:
> 
> BTOR<capitalized option name without '_' and ':'>=<value>
> 
> E.g., given a Boolector instance `btor`, model generation is enabled either via
> 
> boolector_set_opt (btor, BTOR_OPT_MODEL_GEN, 1);
> 
> or via setting the environment variable:
> 
> BTORMODELGEN=1
> 
> For a list and detailed descriptions of all available options, see [`boolector_set_opt()`][47].

### API Tracing[¶][48]

> API tracing allows to record every call to Boolector’s public API. The resulting trace can be
> replayed and the replayed sequence behaves exactly like the original Boolector run. This is
> particularly useful for debugging purposes, as it enables replaying erroneous behaviour. API
> tracing can be enabled either via [`boolector_set_trapi()`][49] or by setting the environment
> variable `BTORAPITRACE=<filename>`.
> 
> For example, given a Boolector instance `btor`, API tracing is enabled as follows:
> 
> FILE *fd = fopen ("error.trace", "r");
> boolector_set_trapi (btor, fd);
> 
> or
> 
> BTORAPITRACE="error.trace"

## Internals[¶][50]

> Boolector internally maintains a **directed acyclic graph (DAG)** of expressions. As a
> consequence, each expression maintains a reference counter, which is initially set to 1. Each time
> an expression is shared, i.e., for each API call that returns an expression (a BoolectorNode), its
> reference counter is incremented by 1. Not considering API calls that created expressions, this
> mainly applies to [`boolector_copy()`][51], which simply increments the reference counter of an
> expression, and [`boolector_match_node()`][52] and [`boolector_match_node_by_id()`][53], which
> retrieve nodes of a given Boolector instance by id and a given node’s id. Expressions are released
> via [`boolector_release()`][54], and if its reference counter is decremented to zero, it is
> deleted from memory.
> 
> Note that by **asserting** an expression, it will be **permanently added** to the formula. This
> means that Boolector internally holds its reference until it is either eliminated via rewriting,
> or the Boolector instance is deleted. Following from that, it is safe to release an expression as
> soon as you asserted it, as long as you don’t need it for further querying.

### Operators[¶][55]

> Boolector internally describes expressions by means of a set of base operators. Boolector’s API,
> however, provides a richer set of operators for convenience, where non-base operators are
> internally rewritten to use base operators only. For example, two’s complement
> ([`boolector_neg()`][56]) is expressed by means of one’s complement.
> 
> Note
> 
> This behaviour is not influenced by the configured rewrite level.

### Rewriting and Preprocessing[¶][57]

> Boolector simplifies expressions and the expression DAG by means of rewriting. It supports three
> so-called **rewrite levels**. Increasing rewrite levels increase the extent of rewriting and
> preprocessing performed. Rewrite level of 0 is equivalent to disabling rewriting and preprocessing
> at all.
> 
> Note
> 
> Rewriting expressions by means of base operators can not be disabled, not even at rewrite level 0.
> 
> Boolector not only simplifies expressions during construction of the expression DAG but also
> performs preprocessing on the DAG. For each call to [`boolector_sat()`][58], various
> simplification techniques and preprocessing phases are initiated. You can force Boolector to
> initiate simplifying the expression DAG via [`boolector_simplify()`][59]. The rewrite level can be
> configured via [`boolector_set_opt()`][60].

## Examples[¶][61]

### Quickstart Example[¶][62]

> #include "boolector.h"
> 
> int
> main ()
> {
>   // Create Boolector instance
>   Btor *btor = boolector_new ();
>   // Enable model generation
>   boolector_set_opt (btor, BTOR_OPT_MODEL_GEN, 1);
> 
>   // Create bit-vector sort of size 8
>   BoolectorSort bvsort8 = boolector_bitvec_sort (btor, 8);
> 
>   // Create expressions
>   BoolectorNode *x       = boolector_var (btor, bvsort8, "x");
>   BoolectorNode *y       = boolector_var (btor, bvsort8, "y");
>   BoolectorNode *zero    = boolector_zero (btor, bvsort8);
>   BoolectorNode *hundred = boolector_int (btor, 100, bvsort8);
> 
>   // 0 < x
>   BoolectorNode *ult_x = boolector_ult (btor, zero, x);
>   boolector_assert (btor, ult_x);
> 
>   // x <= 100
>   BoolectorNode *ulte_x = boolector_ulte (btor, x, hundred);
>   boolector_assert (btor, ulte_x);
> 
>   // 0 < y
>   BoolectorNode *ult_y = boolector_ult (btor, zero, y);
>   boolector_assert (btor, ult_y);
> 
>   // y <= 100
>   BoolectorNode *ulte_y = boolector_ulte (btor, y, hundred);
>   boolector_assert (btor, ulte_y);
> 
>   // x * y
>   BoolectorNode *mul = boolector_mul (btor, x, y);
> 
>   // x * y < 100
>   BoolectorNode *ult = boolector_ult (btor, mul, hundred);
>   boolector_assert (btor, ult);
>   BoolectorNode *umulo  = boolector_umulo (btor, x, y);
>   BoolectorNode *numulo = boolector_not (btor, umulo);  // prevent overflow
>   boolector_assert (btor, numulo);
> 
>   int result = boolector_sat (btor);
>   printf ("Expect: sat\n");
>   printf ("Boolector: ");
>   if (result == BOOLECTOR_SAT)
>   {
>     printf ("sat\n");
>   }
>   else if (result == BOOLECTOR_UNSAT)
>   {
>     printf ("unsat\n");
>   }
>   else
>   {
>     printf ("unknown\n");
>   }
>   printf ("\n");
> 
>   const char *xstr = boolector_bv_assignment (btor, x);  // returns "00000100"
>   const char *ystr = boolector_bv_assignment (btor, y);  // returns "00010101"
>   printf ("assignment of x: %s\n", xstr);
>   printf ("assignment of y: %s\n", ystr);
>   printf ("\n");
> 
>   printf ("Print model in BTOR format:\n");
>   boolector_print_model (btor, "btor", stdout);
>   printf ("\n");
>   printf ("Print model in SMT-LIBv2 format:\n");
>   boolector_print_model (btor, "smt2", stdout);
>   printf ("\n");
> 
>   // Release expressions
>   boolector_release (btor, x);
>   boolector_release (btor, y);
>   boolector_release (btor, zero);
>   boolector_release (btor, hundred);
>   boolector_release (btor, ult_x);
>   boolector_release (btor, ulte_x);
>   boolector_release (btor, ult_y);
>   boolector_release (btor, ulte_y);
>   boolector_release (btor, mul);
>   boolector_release (btor, ult);
>   boolector_release (btor, numulo);
>   boolector_release (btor, umulo);
> 
>   // Release assigments
>   boolector_free_bv_assignment (btor, xstr);
>   boolector_free_bv_assignment (btor, ystr);
> 
>   // Release sorts
>   boolector_release_sort (btor, bvsort8);
> 
>   // Delete Boolector instance
>   boolector_delete (btor);
> }

### Bit-Vector Examples[¶][63]

> #include <assert.h>
> #include <stdio.h>
> #include <stdlib.h>
> #include "boolector.h"
> 
> #define BV1_EXAMPLE_NUM_BITS 8
> 
> /* We verify the XOR swap algorithm. The XOR bitwise operation can
>  * be used to swap variables without using a temporary variable:
>  * int x, y;
>  * ...
>  * x = x ^ y
>  * y = x ^ y
>  * x = x ^ y
>  */
> 
> int
> main (void)
> {
>   Btor *btor;
>   BoolectorNode *x, *y, *temp, *old_x, *old_y, *eq1, *eq2, *and, *formula;
>   BoolectorSort s;
>   int result;
> 
>   btor = boolector_new ();
>   s    = boolector_bitvec_sort (btor, BV1_EXAMPLE_NUM_BITS);
>   x    = boolector_var (btor, s, NULL);
>   y    = boolector_var (btor, s, NULL);
> 
>   /* remember initial values of x and y */
>   old_x = boolector_copy (btor, x);
>   old_y = boolector_copy (btor, y);
> 
>   /* x = x ^ y */
>   temp = boolector_xor (btor, x, y);
>   boolector_release (btor, x);
>   x = temp;
> 
>   /* y = x ^ y */
>   temp = boolector_xor (btor, x, y);
>   boolector_release (btor, y);
>   y = temp;
> 
>   /* x = x ^ y */
>   temp = boolector_xor (btor, x, y);
>   boolector_release (btor, x);
>   x = temp;
> 
>   /* Now, we have to show that old_x = y and old_y = x */
>   eq1 = boolector_eq (btor, old_x, y);
>   eq2 = boolector_eq (btor, old_y, x);
>   and = boolector_and (btor, eq1, eq2);
> 
>   /* In order to prove that this is a theorem, we negate the whole
>    * formula and show that the negation is unsatisfiable */
>   formula = boolector_not (btor, and);
> 
>   /* We assert the formula and call Boolector */
>   boolector_assert (btor, formula);
>   result = boolector_sat (btor);
>   printf ("Expect: unsat\n");
>   printf ("Boolector: %s\n",
>           result == BOOLECTOR_SAT
>               ? "sat"
>               : (result == BOOLECTOR_UNSAT ? "unsat" : "unknown"));
>   if (result != BOOLECTOR_UNSAT) abort ();
> 
>   /* cleanup */
>   boolector_release (btor, x);
>   boolector_release (btor, old_x);
>   boolector_release (btor, y);
>   boolector_release (btor, old_y);
>   boolector_release (btor, eq1);
>   boolector_release (btor, eq2);
>   boolector_release (btor, and);
>   boolector_release (btor, formula);
>   boolector_release_sort (btor, s);
>   assert (boolector_get_refs (btor) == 0);
>   boolector_delete (btor);
>   return 0;
> }
> #include <assert.h>
> #include <stdio.h>
> #include <stdlib.h>
> #include "boolector.h"
> 
> #define BV2_EXAMPLE_NUM_BITS 8
> 
> /* We try to show the following theorem:
>  * v1 > 0 & v2 > 0  =>  v1 + v2 > 0
>  *
>  * The theorem is valid if v1 and v2 are naturals, but not if they
>  * are two's complement bit-vectors as addition can overflow.
>  */
> 
> int
> main (void)
> {
>   Btor *btor;
>   BoolectorNode *v1, *v2, *add, *zero, *vars_sgt_zero, *impl;
>   BoolectorNode *v1_sgt_zero, *v2_sgt_zero, *add_sgt_zero, *formula;
>   BoolectorSort s;
>   const char *assignments[10];
>   int result, i;
> 
>   btor = boolector_new ();
>   boolector_set_opt (btor, BTOR_OPT_MODEL_GEN, 1);
> 
>   s    = boolector_bitvec_sort (btor, BV2_EXAMPLE_NUM_BITS);
>   v1   = boolector_var (btor, s, NULL);
>   v2   = boolector_var (btor, s, NULL);
>   zero = boolector_zero (btor, s);
> 
>   v1_sgt_zero   = boolector_sgt (btor, v1, zero);
>   v2_sgt_zero   = boolector_sgt (btor, v2, zero);
>   vars_sgt_zero = boolector_and (btor, v1_sgt_zero, v2_sgt_zero);
> 
>   add          = boolector_add (btor, v1, v2);
>   add_sgt_zero = boolector_sgt (btor, add, zero);
> 
>   impl = boolector_implies (btor, vars_sgt_zero, add_sgt_zero);
> 
>   /* We negate the formula and try to show that the negation is unsatisfiable */
>   formula = boolector_not (btor, impl);
> 
>   /* We assert the formula and call Boolector */
>   boolector_assert (btor, formula);
>   result = boolector_sat (btor);
>   printf ("Expect: sat\n");
>   printf ("Boolector: %s\n",
>           result == BOOLECTOR_SAT
>               ? "sat"
>               : (result == BOOLECTOR_UNSAT ? "unsat" : "unknown"));
>   if (result != BOOLECTOR_SAT) abort ();
> 
>   /* The formula is not valid, we have found a counter-example.
>    * Now, we are able to obtain assignments to arbitrary expressions */
>   i                = 0;
>   assignments[i++] = boolector_bv_assignment (btor, zero);
>   assignments[i++] = boolector_bv_assignment (btor, v1);
>   assignments[i++] = boolector_bv_assignment (btor, v2);
>   assignments[i++] = boolector_bv_assignment (btor, add);
>   assignments[i++] = boolector_bv_assignment (btor, v1_sgt_zero);
>   assignments[i++] = boolector_bv_assignment (btor, v2_sgt_zero);
>   assignments[i++] = boolector_bv_assignment (btor, vars_sgt_zero);
>   assignments[i++] = boolector_bv_assignment (btor, add_sgt_zero);
>   assignments[i++] = boolector_bv_assignment (btor, impl);
>   assignments[i++] = boolector_bv_assignment (btor, formula);
> 
>   i = 0;
>   printf ("Assignment to 0: %s\n", assignments[i++]);
>   printf ("Assignment to v1: %s\n", assignments[i++]);
>   printf ("Assignment to v2: %s\n", assignments[i++]);
>   printf ("Assignment to v1 + v2: %s\n", assignments[i++]);
>   printf ("Assignment to v1 > 0: %s\n", assignments[i++]);
>   printf ("Assignment to v2 > 0: %s\n", assignments[i++]);
>   printf ("Assignment to v1 > 0 & v2 > 0: %s\n", assignments[i++]);
>   printf ("Assignment to v1 + v2 > 0: %s\n", assignments[i++]);
>   printf ("Assignment to v1 > 0 & v2 > 0  => v1 + v2 > 0: %s\n",
>           assignments[i++]);
>   printf ("Assignment to !(v1 > 0 & v2 > 0  => v1 + v2 > 0): %s\n",
>           assignments[i++]);
>   for (i = 0; i < 10; i++) boolector_free_bv_assignment (btor, assignments[i]);
> 
>   /* cleanup */
>   boolector_release (btor, zero);
>   boolector_release (btor, v1);
>   boolector_release (btor, v2);
>   boolector_release (btor, add);
>   boolector_release (btor, impl);
>   boolector_release (btor, formula);
>   boolector_release (btor, v1_sgt_zero);
>   boolector_release (btor, v2_sgt_zero);
>   boolector_release (btor, vars_sgt_zero);
>   boolector_release (btor, add_sgt_zero);
>   boolector_release_sort (btor, s);
>   assert (boolector_get_refs (btor) == 0);
>   boolector_delete (btor);
>   return 0;
> }

### Array Examples[¶][64]

> #include <assert.h>
> #include <stdio.h>
> #include <stdlib.h>
> #include "boolector.h"
> 
> #define ARRAY1_EXAMPLE_ELEM_BW 8
> #define ARRAY1_EXAMPLE_INDEX_BW 3
> #define ARRAY1_EXAMPLE_ARRAY_SIZE (1 << ARRAY1_EXAMPLE_INDEX_BW)
> 
> /* We verify the following linear search algorithm. We iterate over an array
>  * and compute a maximum value as the following pseudo code shows:
>  *
>  * unsigned int array[ARRAY_SIZE];
>  * unsigned int max;
>  * int i;
>  * ...
>  * max = array[0];
>  * for (i = 1; i < ARRAY_SIZE; i++)
>  *   if (array[i] > max)
>  *     max = array[i]
>  *
>  * Finally, we prove that it is not possible to find an array position
>  * such that the value stored at this position is greater than 'max'.
>  * If we can show this, we have proved that this algorithm indeed finds
>  * a maximum value. Note that we prove that the algorithm finds an
>  * arbitrary maximum (multiple maxima are possible), not necessarily
>  * the first maximum.
>  */
> 
> int
> main (void)
> {
>   Btor *btor;
>   BoolectorNode *array, *read, *max, *temp, *ugt, *formula, *index;
>   BoolectorNode *indices[ARRAY1_EXAMPLE_ARRAY_SIZE];
>   BoolectorSort sort_elem, sort_index, sort_array;
>   int i, result;
> 
>   btor       = boolector_new ();
>   sort_index = boolector_bitvec_sort (btor, ARRAY1_EXAMPLE_INDEX_BW);
>   sort_elem  = boolector_bitvec_sort (btor, ARRAY1_EXAMPLE_ELEM_BW);
>   sort_array = boolector_array_sort (btor, sort_index, sort_elem);
> 
>   /* We create all possible constants that are used as read indices */
>   for (i = 0; i < ARRAY1_EXAMPLE_ARRAY_SIZE; i++)
>     indices[i] = boolector_int (btor, i, sort_index);
> 
>   array = boolector_array (btor, sort_array, 0);
>   /* Current maximum is first element of array */
>   max = boolector_read (btor, array, indices[0]);
>   /* Symbolic loop unrolling */
>   for (i = 1; i < ARRAY1_EXAMPLE_ARRAY_SIZE; i++)
>   {
>     read = boolector_read (btor, array, indices[i]);
>     ugt  = boolector_ugt (btor, read, max);
>     /* found a new maximum? */
>     temp = boolector_cond (btor, ugt, read, max);
>     boolector_release (btor, max);
>     max = temp;
>     boolector_release (btor, read);
>     boolector_release (btor, ugt);
>   }
> 
>   /* Now we show that 'max' is indeed a maximum */
>   /* We read at an arbitrary position */
>   index = boolector_var (btor, sort_index, NULL);
>   read  = boolector_read (btor, array, index);
> 
>   /* We assume that it is possible that the read value is greater than 'max' */
>   formula = boolector_ugt (btor, read, max);
> 
>   /* We assert the formula and call Boolector */
>   boolector_assert (btor, formula);
>   result = boolector_sat (btor);
>   printf ("Expect: unsat\n");
>   printf ("Boolector: %s\n",
>           result == BOOLECTOR_SAT
>               ? "sat"
>               : (result == BOOLECTOR_UNSAT ? "unsat" : "unknown"));
>   if (result != BOOLECTOR_UNSAT) abort ();
> 
>   /* clean up */
>   for (i = 0; i < ARRAY1_EXAMPLE_ARRAY_SIZE; i++)
>     boolector_release (btor, indices[i]);
>   boolector_release (btor, formula);
>   boolector_release (btor, read);
>   boolector_release (btor, index);
>   boolector_release (btor, max);
>   boolector_release (btor, array);
>   boolector_release_sort (btor, sort_array);
>   boolector_release_sort (btor, sort_index);
>   boolector_release_sort (btor, sort_elem);
>   assert (boolector_get_refs (btor) == 0);
>   boolector_delete (btor);
>   return 0;
> }
> #include <assert.h>
> #include <stdio.h>
> #include <stdlib.h>
> #include "boolector.h"
> 
> #define ARRAY2_EXAMPLE_ELEM_BW 8
> #define ARRAY2_EXAMPLE_INDEX_BW 1
> 
> /* We demonstrate Boolector's ability to obtain Array models.
>  * We check the following formula for satisfiability:
>  * write (array1, 0, 3) = write (array2, 1, 5)
>  */
> 
> int
> main (void)
> {
>   Btor *btor;
>   BoolectorNode *array1, *array2, *zero, *one, *val1, *val2;
>   BoolectorNode *write1, *write2, *formula;
>   BoolectorSort sort_index, sort_elem, sort_array;
>   char **indices, **values;
>   int32_t result;
>   uint32_t i, size;
> 
>   btor       = boolector_new ();
>   boolector_set_opt (btor, BTOR_OPT_OUTPUT_NUMBER_FORMAT, BTOR_OUTPUT_BASE_HEX);
> 
>   sort_index = boolector_bitvec_sort (btor, ARRAY2_EXAMPLE_INDEX_BW);
>   sort_elem  = boolector_bitvec_sort (btor, ARRAY2_EXAMPLE_ELEM_BW);
>   sort_array = boolector_array_sort (btor, sort_index, sort_elem);
>   boolector_set_opt (btor, BTOR_OPT_MODEL_GEN, 1);
> 
>   zero   = boolector_zero (btor, sort_index);
>   one    = boolector_one (btor, sort_index);
>   val1   = boolector_int (btor, 3, sort_elem);
>   val2   = boolector_int (btor, 5, sort_elem);
>   array1 = boolector_array (btor, sort_array, 0);
>   array2 = boolector_array (btor, sort_array, 0);
>   write1 = boolector_write (btor, array1, zero, val1);
>   write2 = boolector_write (btor, array2, one, val2);
>   /* Note: we compare two arrays for equality ---> needs extensional theory */
>   formula = boolector_eq (btor, write1, write2);
>   boolector_assert (btor, formula);
>   result = boolector_sat (btor);
>   printf ("Expect: sat\n");
>   printf ("Boolector: %s\n",
>           result == BOOLECTOR_SAT
>               ? "sat"
>               : (result == BOOLECTOR_UNSAT ? "unsat" : "unknown"));
>   if (result != BOOLECTOR_SAT) abort ();
> 
>   printf ("\nModel:\n");
>   /* Formula is satisfiable, we can obtain array models: */
>   boolector_array_assignment (btor, array1, &indices, &values, &size);
>   if (size > 0)
>   {
>     printf ("Array1:\n");
>     for (i = 0; i < size; i++)
>       printf ("Array1[#x%s] = #x%s\n", indices[i], values[i]);
>     boolector_free_array_assignment (btor, indices, values, size);
>   }
> 
>   boolector_array_assignment (btor, array2, &indices, &values, &size);
>   if (size > 0)
>   {
>     printf ("\nArray2:\n");
>     for (i = 0; i < size; i++)
>       printf ("Array2[#x%s] = #x%s\n", indices[i], values[i]);
>     boolector_free_array_assignment (btor, indices, values, size);
>   }
> 
>   boolector_array_assignment (btor, write1, &indices, &values, &size);
>   if (size > 0)
>   {
>     printf ("\nWrite1:\n");
>     for (i = 0; i < size; i++)
>       printf ("Write1[#x%s] = #x%s\n", indices[i], values[i]);
>     boolector_free_array_assignment (btor, indices, values, size);
>   }
> 
>   boolector_array_assignment (btor, write2, &indices, &values, &size);
>   if (size > 0)
>   {
>     printf ("\nWrite2:\n");
>     for (i = 0; i < size; i++)
>       printf ("Write2[#x%s] = #x%s\n", indices[i], values[i]);
>     boolector_free_array_assignment (btor, indices, values, size);
>   }
> 
>   /* clean up */
>   boolector_release (btor, formula);
>   boolector_release (btor, write1);
>   boolector_release (btor, write2);
>   boolector_release (btor, array1);
>   boolector_release (btor, array2);
>   boolector_release (btor, val1);
>   boolector_release (btor, val2);
>   boolector_release (btor, zero);
>   boolector_release (btor, one);
>   boolector_release_sort (btor, sort_array);
>   boolector_release_sort (btor, sort_index);
>   boolector_release_sort (btor, sort_elem);
>   assert (boolector_get_refs (btor) == 0);
>   boolector_delete (btor);
>   return 0;
> }
> #include <assert.h>
> #include <limits.h>
> #include <stdlib.h>
> 
> #include "boolector.h"
> 
> #define ARRAY3_EXAMPLE_ELEM_BW 8
> #define ARRAY3_EXAMPLE_INDEX_BW 1
> 
> int
> main ()
> {
>   int result;
>   BoolectorNode *array, *index1, *index2, *read1, *read2, *eq, *ne;
>   BoolectorSort sort_index, sort_elem, sort_array;
>   Btor *btor;
> 
>   btor       = boolector_new ();
>   sort_index = boolector_bitvec_sort (btor, ARRAY3_EXAMPLE_INDEX_BW);
>   sort_elem  = boolector_bitvec_sort (btor, ARRAY3_EXAMPLE_ELEM_BW);
>   sort_array = boolector_array_sort (btor, sort_index, sort_elem);
>   boolector_set_opt (btor, BTOR_OPT_INCREMENTAL, 1);
> 
>   array  = boolector_array (btor, sort_array, 0);
>   index1 = boolector_var (btor, sort_index, 0);
>   index2 = boolector_var (btor, sort_index, 0);
>   read1  = boolector_read (btor, array, index1);
>   read2  = boolector_read (btor, array, index2);
>   eq     = boolector_eq (btor, index1, index2);
>   ne     = boolector_ne (btor, read1, read2);
> 
>   /* we enforce that index1 is equal to index 2 */
>   boolector_assert (btor, eq);
>   result = boolector_sat (btor);
>   printf ("Expect: sat\n");
>   printf ("Boolector: %s\n",
>           result == BOOLECTOR_SAT
>               ? "sat"
>               : (result == BOOLECTOR_UNSAT ? "unsat" : "unknown"));
>   if (result != BOOLECTOR_SAT) abort ();
>   /* now we additionally assume that the read values differ
>    * the instance is now unsatasfiable as read congruence is violated */
>   boolector_assume (btor, ne);
>   result = boolector_sat (btor);
>   assert (result == BOOLECTOR_UNSAT);
>   /* after the SAT call the assumptions are gone
>    * the instance is now satisfiable again */
>   result = boolector_sat (btor);
>   printf ("Expect: sat\n");
>   printf ("Boolector: %s\n",
>           result == BOOLECTOR_SAT
>               ? "sat"
>               : (result == BOOLECTOR_UNSAT ? "unsat" : "unknown"));
>   if (result != BOOLECTOR_SAT) abort ();
>   boolector_release (btor, array);
>   boolector_release (btor, index1);
>   boolector_release (btor, index2);
>   boolector_release (btor, read1);
>   boolector_release (btor, read2);
>   boolector_release (btor, eq);
>   boolector_release (btor, ne);
>   boolector_release_sort (btor, sort_array);
>   boolector_release_sort (btor, sort_index);
>   boolector_release_sort (btor, sort_elem);
>   boolector_delete (btor);
>   return 0;
> }

[Next ][65] [ Previous][66]

© Copyright 2021, the authors of Boolector.

Built with [Sphinx][67] using a [theme][68] provided by [Read the Docs][69].

[1]: #boolector-c-api-documentation
[2]: #interface
[3]: cboolector_index.html
[4]: cboolector_index.html#macros
[5]: cboolector_index.html#typedefs
[6]: cboolector_index.html#functions
[7]: cboolector_index.html#deprecated
[8]: #quickstart
[9]: cboolector_index.html#c.boolector_set_opt
[10]: cboolector_index.html#c.boolector_set_opt
[11]: cboolector_index.html#c.boolector_assert
[12]: cboolector_index.html#c.boolector_assume
[13]: cboolector_index.html#c.boolector_sat
[14]: cboolector_index.html#c.boolector_parse
[15]: cboolector_index.html#c.boolector_parse_btor
[16]: http://fmv.jku.at/papers/BrummayerBiereLonsing-BPR08.pdf
[17]: cboolector_index.html#c.BOOLECTOR_SAT
[18]: cboolector_index.html#c.BOOLECTOR_UNSAT
[19]: cboolector_index.html#c.BOOLECTOR_UNKNOWN
[20]: cboolector_index.html#c.BOOLECTOR_PARSE_UNKNOWN
[21]: cboolector_index.html#c.boolector_parse
[22]: cboolector_index.html#c.boolector_parse_btor
[23]: cboolector_index.html#c.boolector_parse_btor2
[24]: cboolector_index.html#c.boolector_parse_smt1
[25]: cboolector_index.html#c.boolector_parse_smt2
[26]: cboolector_index.html#c.BOOLECTOR_PARSE_ERROR
[27]: cboolector_index.html#c.boolector_assert
[28]: cboolector_index.html#c.boolector_sat
[29]: cboolector_index.html#c.boolector_set_opt
[30]: cboolector_index.html#c.boolector_print_model
[31]: cboolector_index.html#c.boolector_bv_assignment
[32]: cboolector_index.html#c.boolector_array_assignment
[33]: cboolector_index.html#c.boolector_uf_assignment
[34]: cboolector_index.html#c.boolector_print_model
[35]: http://smtlib.cs.uiowa.edu/papers/smt-lib-reference-v2.0-r12.09.09.pdf
[36]: http://fmv.jku.at/papers/BrummayerBiereLonsing-BPR08.pdf
[37]: http://smtlib.cs.uiowa.edu/papers/smt-lib-reference-v2.0-r12.09.09.pdf
[38]: #c-internals
[39]: cboolector_index.html#c.boolector_release
[40]: cboolector_index.html#c.boolector_delete
[41]: cboolector_index.html#c.boolector_free_bv_assignment
[42]: cboolector_index.html#c.boolector_free_array_assignment
[43]: cboolector_index.html#c.boolector_free_uf_assignment
[44]: https://github.com/boolector/boolector/tree/master/examples/api/c/quickstart.c
[45]: #options
[46]: cboolector_index.html#c.boolector_set_opt
[47]: cboolector_index.html#c.boolector_set_opt
[48]: #api-tracing
[49]: cboolector_index.html#c.boolector_set_trapi
[50]: #internals
[51]: cboolector_index.html#c.boolector_copy
[52]: cboolector_index.html#c.boolector_match_node
[53]: cboolector_index.html#c.boolector_match_node_by_id
[54]: cboolector_index.html#c.boolector_release
[55]: #operators
[56]: cboolector_index.html#c.boolector_neg
[57]: #rewriting-and-preprocessing
[58]: cboolector_index.html#c.boolector_sat
[59]: cboolector_index.html#c.boolector_simplify
[60]: cboolector_index.html#c.boolector_set_opt
[61]: #examples
[62]: #quickstart-example
[63]: #bit-vector-examples
[64]: #array-examples
[65]: cboolector_index.html
[66]: index.html
[67]: https://www.sphinx-doc.org/
[68]: https://github.com/readthedocs/sphinx_rtd_theme
[69]: https://readthedocs.org
