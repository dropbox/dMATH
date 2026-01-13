# Crate cargo_spellcheck Copy item path

[Source][1]
Expand description

cargo-spellcheck

A syntax tree based doc comment and common mark spell checker.

## Re-exports[§][2]

*`pub use [doc_chunks][3] as documentation;`*
*`pub use self::[action][4]::*;`*

## Modules[§][5]

*[action][6]*
  Covers all user triggered actions (except for signals).
*[errors][7]*
  Global error usage without cluttering each file.
*[util][8]*

## Structs[§][9]

*[Args][10]*
*[CheckableChunk][11]*
  A chunk of documentation that is supposed to be checked.
*[Clusters][12]*
  Cluster comments together, such they appear as continuous text blocks.
*[Common][13]*
*[Config][14]*
*[Documentation][15]*
  Collection of all the documentation entries across the project
*[HunspellConfig][16]*
*[LanguageToolConfig][17]*
*[LineColumn][18]*
  A line-column pair representing the start or end of a `Span`.
*[ManifestMetadata][19]*
*[ManifestMetadataSpellcheck][20]*
*[MultipleCheckerTypes][21]*
*[PlainOverlay][22]*
  A plain representation of cmark riddled chunk.
*[Span][23]*
  Relative span in relation to the beginning of a doc comment.
*[Suggestion][24]*
  A suggestion for certain offending span.
*[SuggestionSet][25]*
  A set of suggestions across multiple files, clustered per file
*[TinHat][26]*
  Blocks (UNIX) signals.
*[UnknownCheckerTypeVariant][27]*

## Enums[§][28]

*[CheckerType][29]*
  Checker types to be derived from the stringly typed arguments.
*[CommentVariant][30]*
  Track what kind of comment the literal is
*[CommentVariantCategory][31]*
  Determine if a `CommentVariant` is a documentation comment or not.
*[ConfigWriteDestination][32]*
*[ContentOrigin][33]*
  Definition of the source of a checkable chunk
*[Detector][34]*
  Bitflag of available checkers by compilation / configuration.
*[ExitCode][35]*
  A simple exit code representation.
*[Sub][36]*
*[UnifiedArgs][37]*
  Unified arguments with configuration fallbacks.

## Functions[§][38]

*[byte_range_to_char_range][39]*
  Convert a given byte range of a string, that is known to be at valid char bounds, to a character
  range.
*[byte_range_to_char_range_many][40]*
  Convert many byte ranges to character ranges.
*[condition_display_content][41]*
  For long lines, literal will be trimmed to display in one terminal line. Misspelled words that are
  too long shall also be ellipsized.
*[derive_job_count][42]*
  Set the worker pool job/thread count.
*[extract_delimiter][43]*
  Extract line delimiter of a string.
*[generate_completions][44]*
*[get_terminal_size][45]*
  Terminal size in characters.
*[iter_with_line_column][46]*
  Iterate over annotated chars starting from line 1 and column 0 assuming `s` starts there.
*[iter_with_line_column_from][47]*
  Iterate over a str and annotate with line and column.
*[load_span_from][48]*
  Extract `span` from a `Read`-able source as `String`.
*[run][49]*
  The inner main.
*[signal_handler][50]*
  Handle incoming signals.
*[sub_char_range][51]*
  Extract a subset of chars by iterating. Range must be in characters.
*[sub_chars][52]*
  Extract a subset of chars by iterating. Range must be in characters.

## Type Aliases[§][53]

*[Range][54]*
  Range based on `usize`, simplification.

[1]: ../src/cargo_spellcheck/lib.rs.html#1-170
[2]: #reexports
[3]: https://docs.rs/doc-chunks/0.2.1/x86_64-unknown-linux-gnu/doc_chunks/index.html
[4]: action/index.html
[5]: #modules
[6]: action/index.html
[7]: errors/index.html
[8]: util/index.html
[9]: #structs
[10]: struct.Args.html
[11]: struct.CheckableChunk.html
[12]: struct.Clusters.html
[13]: struct.Common.html
[14]: struct.Config.html
[15]: struct.Documentation.html
[16]: struct.HunspellConfig.html
[17]: struct.LanguageToolConfig.html
[18]: struct.LineColumn.html
[19]: struct.ManifestMetadata.html
[20]: struct.ManifestMetadataSpellcheck.html
[21]: struct.MultipleCheckerTypes.html
[22]: struct.PlainOverlay.html
[23]: struct.Span.html
[24]: struct.Suggestion.html
[25]: struct.SuggestionSet.html
[26]: struct.TinHat.html
[27]: struct.UnknownCheckerTypeVariant.html
[28]: #enums
[29]: enum.CheckerType.html
[30]: enum.CommentVariant.html
[31]: enum.CommentVariantCategory.html
[32]: enum.ConfigWriteDestination.html
[33]: enum.ContentOrigin.html
[34]: enum.Detector.html
[35]: enum.ExitCode.html
[36]: enum.Sub.html
[37]: enum.UnifiedArgs.html
[38]: #functions
[39]: fn.byte_range_to_char_range.html
[40]: fn.byte_range_to_char_range_many.html
[41]: fn.condition_display_content.html
[42]: fn.derive_job_count.html
[43]: fn.extract_delimiter.html
[44]: fn.generate_completions.html
[45]: fn.get_terminal_size.html
[46]: fn.iter_with_line_column.html
[47]: fn.iter_with_line_column_from.html
[48]: fn.load_span_from.html
[49]: fn.run.html
[50]: fn.signal_handler.html
[51]: fn.sub_char_range.html
[52]: fn.sub_chars.html
[53]: #types
[54]: type.Range.html
