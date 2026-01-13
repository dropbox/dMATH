# Crate cargo_semver_checks Copy item path

[Source][1]

## Structs[§][2]

*[Check][3]*
  Test a release for semver violations.
*[CrateReport][4]*
  Report of semver check of one crate.
*[FeatureFlag][5]*
  A feature flag for gating unstable `cargo-semver-checks` features.
*[GlobalConfig][6]*
*[OverrideStack][7]*
  A stack of [`OverrideMap`][8] values capturing our precedence rules.
*[PackageSelection][9]*
*[QueryOverride][10]*
  Configured values for a [`SemverQuery`][11] that differ from the lint’s defaults.
*[Report][12]*
  Report of the whole analysis. Contains a report for each crate checked.
*[Rustdoc][13]*
*[SemverQuery][14]*
  A query that can be executed on a pair of rustdoc output files, returning instances of a
  particular kind of semver violation.
*[Witness][15]*
  Data for generating a **witness** from the results of a [`SemverQuery`][16].
*[WitnessGeneration][17]*
  Options for generating **witness code**. A witness is a minimal buildable example of how
  downstream code could break for a specific breaking change.

## Enums[§][18]

*[ActualSemverUpdate][19]*
  Kind of semver update.
*[LintLevel][20]*
  The level of intensity of the error when a lint occurs.
*[ReleaseType][21]*
  The kind of release we’re making.
*[RequiredSemverUpdate][22]*
*[ScopeSelection][23]*

## Type Aliases[§][24]

*[OverrideMap][25]*
  A mapping of lint ids to configured values that override that lint’s defaults.

[1]: ../src/cargo_semver_checks/lib.rs.html#1-934
[2]: #structs
[3]: struct.Check.html
[4]: struct.CrateReport.html
[5]: struct.FeatureFlag.html
[6]: struct.GlobalConfig.html
[7]: struct.OverrideStack.html
[8]: type.OverrideMap.html
[9]: struct.PackageSelection.html
[10]: struct.QueryOverride.html
[11]: struct.SemverQuery.html
[12]: struct.Report.html
[13]: struct.Rustdoc.html
[14]: struct.SemverQuery.html
[15]: struct.Witness.html
[16]: struct.SemverQuery.html
[17]: struct.WitnessGeneration.html
[18]: #enums
[19]: enum.ActualSemverUpdate.html
[20]: enum.LintLevel.html
[21]: enum.ReleaseType.html
[22]: enum.RequiredSemverUpdate.html
[23]: enum.ScopeSelection.html
[24]: #types
[25]: type.OverrideMap.html
