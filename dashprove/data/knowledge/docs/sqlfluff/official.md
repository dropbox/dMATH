# 📜 The SQL Linter for Humans[¶][1]

Bored of not having a good SQL linter that works with whichever dialect you’re working with? Fluff
is an extensible and modular linter designed to help you write good SQL and catch errors and bad SQL
before it hits your database.

Notable releases:

* **1.0.x**: First *stable* release, no major changes to take advantage of a point of relative
  stability.
* **2.0.x**: Recode of rules, whitespace fixing consolidation, `sqlfluff format` and removal of
  support for dbt versions pre 1.1. Note, that this release brings with it some breaking changes to
  rule coding and configuration, see [Upgrading from 1.x to 2.0][2].
* **3.0.x**: `sqlfluff fix` now defaults to *not* asking for confirmation and the –force option was
  removed. Richer information returned by the `sqlfluff lint` command (although in a different
  structure to previous versions). See [Upgrading to 3.x][3].

For more detail on other releases, see our [Release Notes][4].

Want to see where and how people are using SQLFluff in their projects? Head over to [SQLFluff in the
Wild][5] for inspiration.

## Getting Started[¶][6]

To get started just install the package, make a sql file and then run SQLFluff and point it at the
file. For more details or if you don’t have python or pip already installed see [Getting
Started][7].

$ pip install sqlfluff
$ echo "  SELECT a  +  b FROM tbl;  " > test.sql
$ sqlfluff lint test.sql --dialect ansi
== [test.sql] FAIL
L:   1 | P:   1 | LT01 | Expected only single space before 'SELECT' keyword.
                       | Found '  '. [layout.spacing]
L:   1 | P:   1 | LT02 | First line should not be indented.
                       | [layout.indent]
L:   1 | P:   1 | LT13 | Files must not begin with newlines or whitespace.
                       | [layout.start_of_file]
L:   1 | P:  11 | LT01 | Expected only single space before binary operator '+'.
                       | Found '  '. [layout.spacing]
L:   1 | P:  14 | LT01 | Expected only single space before naked identifier.
                       | Found '  '. [layout.spacing]
L:   1 | P:  27 | LT01 | Unnecessary trailing whitespace at end of file.
                       | [layout.spacing]
L:   1 | P:  27 | LT12 | Files must end with a single trailing newline.
                       | [layout.end_of_file]
All Finished 📜 🎉!

## Contents[¶][8]

Documentation for SQLFluff:

* [Getting Started][9]
  
  * [Installing Python][10]
  * [Installing SQLFluff][11]
  * [Basic Usage][12]
  * [Custom Usage][13]
  * [Going further][14]
* [Why SQLFluff?][15]
  
  * [Quality assurance][16]
  * [Modularity][17]
  * [Vision for SQLFluff][18]
* [Guides & How-tos][19]
  
  * [Setting up SQLFluff][20]
    
    * [Rolling out SQLFluff with a new team][21]
    * [Developing Custom Rules][22]
  * [Troubleshooting SQLFluff][23]
    
    * [How to Troubleshoot SQLFluff][24]
  * [Contributing to SQLFluff][25]
    
    * [Architecture][26]
    * [How to use Git][27]
    * [Contributing dialect changes][28]
    * [Developing Rules][29]
    * [Developing Plugins][30]
    * [Documentation Contributions][31]
* [Configuration][32]
  
  * [Setting Configuration][33]
    
    * [Configuration Files][34]
    * [In-File Configuration Directives][35]
  * [Rule Configuration][36]
    
    * [Enabling and Disabling Rules][37]
    * [Downgrading rules to warnings][38]
    * [Layout & Spacing Configuration][39]
  * [Layout & Whitespace Configuration][40]
    
    * [Spacing][41]
    * [Templating and alignment coordinate space][42]
    * [Advanced: coordinate space override][43]
    * [Line Breaks][44]
    * [Indentation][45]
    * [Configuring Layout][46]
  * [Templating Configuration][47]
    
    * [Jinja templater][48]
    * [Placeholder templater][49]
    * [Python templater][50]
    * [`dbt` templater][51]
    * [Generic Variable Templating][52]
  * [Ignoring Errors & Files][53]
    
    * [Ignoring individual lines][54]
    * [Ignoring line ranges][55]
    * [`.sqlfluffignore`][56]
    * [Ignoring types of errors][57]
  * [Default Configuration][58]
* [Production Usage & Security][59]
  
  * [Security Considerations][60]
  * [Using SQLFluff directly as a CLI application][61]
  * [Using SQLFluff on changes using `diff-quality`][62]
    
    * [Adding `diff-quality` to your builds][63]
  * [Using `pre-commit`][64]
    
    * [Ignoring files while using `pre-commit`][65]
  * [Using GitHub Actions to Annotate PRs][66]
* [Reference][67]
  
  * [Dialects Reference][68]
    
    * [ANSI][69]
    * [AWS Athena][70]
    * [Google BigQuery][71]
    * [ClickHouse][72]
    * [Databricks][73]
    * [IBM Db2][74]
    * [Apache Doris][75]
    * [DuckDB][76]
    * [Exasol][77]
    * [Apache Flink SQL][78]
    * [Greenplum][79]
    * [Apache Hive][80]
    * [Apache Impala][81]
    * [MariaDB][82]
    * [Materialize][83]
    * [MySQL][84]
    * [Oracle][85]
    * [PostgreSQL][86]
    * [AWS Redshift][87]
    * [Snowflake][88]
    * [Salesforce Object Query Language (SOQL)][89]
    * [Apache Spark SQL][90]
    * [SQLite][91]
    * [StarRocks][92]
    * [Teradata][93]
    * [Trino][94]
    * [Microsoft T-SQL][95]
    * [Vertica][96]
  * [Rules Reference][97]
    
    * [Core Rules][98]
    * [Rule Index][99]
    * [Aliasing bundle][100]
    * [Ambiguous bundle][101]
    * [Capitalisation bundle][102]
    * [Convention bundle][103]
    * [Jinja bundle][104]
    * [Layout bundle][105]
    * [References bundle][106]
    * [Structure bundle][107]
    * [TSQL bundle][108]
  * [CLI Reference][109]
    
    * [sqlfluff][110]
  * [Python API][111]
    
    * [Simple API commands][112]
    * [Advanced API usage][113]
  * [Internal API][114]
    
    * [`sqlfluff.core.config`: Configuration & `FluffConfig`][115]
    * [`sqlfluff.core.rules.base`: Base Rule Classes][116]
    * [`sqlfluff.utils.functional`: Functional Traversal API][117]
    * [`sqlfluff.utils.reflow`: Whitespace Reflow API][118]
  * [Release Notes][119]
    
    * [Upgrading to 3.x][120]
    * [Upgrading to 2.3][121]
    * [Upgrading to 2.2][122]
    * [Upgrading from 1.x to 2.0][123]
    * [Upgrading to 1.4][124]
    * [Upgrading to 1.3][125]
    * [Upgrading to 1.2][126]
    * [Upgrades pre 1.0][127]
* [SQLFluff in the Wild][128]
* [SQLFluff Slack][129]
* [SQLFluff on Twitter][130]

# Indices and tables[¶][131]

* [Index][132]
* [Module Index][133]
* [Search Page][134]

[ [Logo] ][135]


### Navigation

Documentation for SQLFluff:

* [Getting Started][136]
* [Why SQLFluff?][137]
* [Guides & How-tos][138]
* [Configuration][139]
* [Production Usage & Security][140]
* [Reference][141]
* [SQLFluff in the Wild][142]
* [SQLFluff Slack][143]
* [SQLFluff on Twitter][144]

### Related Topics

* [Documentation overview][145]
  
  * Next: [Getting Started][146]
©2024, Alan Cruickshank. | Powered by [Sphinx 8.2.3][147] & [Alabaster 1.0.0][148] | [Page
source][149]

[1]: #the-sql-linter-for-humans
[2]: reference/releasenotes.html#upgrading-2-0
[3]: reference/releasenotes.html#upgrading-3-0
[4]: reference/releasenotes.html#releasenotes
[5]: inthewild.html#inthewildref
[6]: #getting-started
[7]: gettingstarted.html#gettingstartedref
[8]: #contents
[9]: gettingstarted.html
[10]: gettingstarted.html#installing-python
[11]: gettingstarted.html#installing-sqlfluff
[12]: gettingstarted.html#basic-usage
[13]: gettingstarted.html#custom-usage
[14]: gettingstarted.html#going-further
[15]: why_sqlfluff.html
[16]: why_sqlfluff.html#quality-assurance
[17]: why_sqlfluff.html#modularity
[18]: why_sqlfluff.html#vision-for-sqlfluff
[19]: guides/index.html
[20]: guides/index.html#setting-up-sqlfluff
[21]: guides/setup/teamrollout.html
[22]: guides/setup/developing_custom_rules.html
[23]: guides/index.html#troubleshooting-sqlfluff
[24]: guides/troubleshooting/how_to.html
[25]: guides/index.html#contributing-to-sqlfluff
[26]: guides/contributing/architecture.html
[27]: guides/contributing/git.html
[28]: guides/contributing/dialect.html
[29]: guides/contributing/rules.html
[30]: guides/contributing/plugins.html
[31]: guides/contributing/docs.html
[32]: configuration/index.html
[33]: configuration/setting_configuration.html
[34]: configuration/setting_configuration.html#configuration-files
[35]: configuration/setting_configuration.html#in-file-configuration-directives
[36]: configuration/rule_configuration.html
[37]: configuration/rule_configuration.html#enabling-and-disabling-rules
[38]: configuration/rule_configuration.html#downgrading-rules-to-warnings
[39]: configuration/rule_configuration.html#layout-spacing-configuration
[40]: configuration/layout.html
[41]: configuration/layout.html#spacing
[42]: configuration/layout.html#templating-and-alignment-coordinate-space
[43]: configuration/layout.html#advanced-coordinate-space-override
[44]: configuration/layout.html#line-breaks
[45]: configuration/layout.html#indentation
[46]: configuration/layout.html#configuring-layout
[47]: configuration/templating/index.html
[48]: configuration/templating/jinja.html
[49]: configuration/templating/placeholder.html
[50]: configuration/templating/python.html
[51]: configuration/templating/dbt.html
[52]: configuration/templating/index.html#generic-variable-templating
[53]: configuration/ignoring_configuration.html
[54]: configuration/ignoring_configuration.html#ignoring-individual-lines
[55]: configuration/ignoring_configuration.html#ignoring-line-ranges
[56]: configuration/ignoring_configuration.html#sqlfluffignore
[57]: configuration/ignoring_configuration.html#ignoring-types-of-errors
[58]: configuration/default_configuration.html
[59]: production/index.html
[60]: production/security.html
[61]: production/cli_use.html
[62]: production/diff_quality.html
[63]: production/diff_quality.html#adding-diff-quality-to-your-builds
[64]: production/pre_commit.html
[65]: production/pre_commit.html#ignoring-files-while-using-pre-commit
[66]: production/github_actions.html
[67]: reference/index.html
[68]: reference/dialects.html
[69]: reference/dialects.html#ansi
[70]: reference/dialects.html#aws-athena
[71]: reference/dialects.html#google-bigquery
[72]: reference/dialects.html#clickhouse
[73]: reference/dialects.html#databricks
[74]: reference/dialects.html#ibm-db2
[75]: reference/dialects.html#apache-doris
[76]: reference/dialects.html#duckdb
[77]: reference/dialects.html#exasol
[78]: reference/dialects.html#apache-flink-sql
[79]: reference/dialects.html#greenplum
[80]: reference/dialects.html#apache-hive
[81]: reference/dialects.html#apache-impala
[82]: reference/dialects.html#mariadb
[83]: reference/dialects.html#materialize
[84]: reference/dialects.html#mysql
[85]: reference/dialects.html#oracle
[86]: reference/dialects.html#postgresql
[87]: reference/dialects.html#aws-redshift
[88]: reference/dialects.html#snowflake
[89]: reference/dialects.html#salesforce-object-query-language-soql
[90]: reference/dialects.html#apache-spark-sql
[91]: reference/dialects.html#sqlite
[92]: reference/dialects.html#starrocks
[93]: reference/dialects.html#teradata
[94]: reference/dialects.html#trino
[95]: reference/dialects.html#microsoft-t-sql
[96]: reference/dialects.html#vertica
[97]: reference/rules.html
[98]: reference/rules.html#core-rules
[99]: reference/rules.html#rule-index
[100]: reference/rules.html#aliasing-bundle
[101]: reference/rules.html#ambiguous-bundle
[102]: reference/rules.html#capitalisation-bundle
[103]: reference/rules.html#convention-bundle
[104]: reference/rules.html#jinja-bundle
[105]: reference/rules.html#layout-bundle
[106]: reference/rules.html#references-bundle
[107]: reference/rules.html#structure-bundle
[108]: reference/rules.html#tsql-bundle
[109]: reference/cli.html
[110]: reference/cli.html#sqlfluff
[111]: reference/api.html
[112]: reference/api.html#module-sqlfluff
[113]: reference/api.html#advanced-api-usage
[114]: reference/internals/index.html
[115]: reference/internals/config.html
[116]: reference/internals/rules.html
[117]: reference/internals/functional.html
[118]: reference/internals/reflow.html
[119]: reference/releasenotes.html
[120]: reference/releasenotes.html#upgrading-to-3-x
[121]: reference/releasenotes.html#upgrading-to-2-3
[122]: reference/releasenotes.html#upgrading-to-2-2
[123]: reference/releasenotes.html#upgrading-from-1-x-to-2-0
[124]: reference/releasenotes.html#upgrading-to-1-4
[125]: reference/releasenotes.html#upgrading-to-1-3
[126]: reference/releasenotes.html#upgrading-to-1-2
[127]: reference/releasenotes.html#upgrades-pre-1-0
[128]: inthewild.html
[129]: jointhecommunity.html
[130]: jointhecommunity.html#sqlfluff-on-twitter
[131]: #indices-and-tables
[132]: genindex.html
[133]: py-modindex.html
[134]: search.html
[135]: #
[136]: gettingstarted.html
[137]: why_sqlfluff.html
[138]: guides/index.html
[139]: configuration/index.html
[140]: production/index.html
[141]: reference/index.html
[142]: inthewild.html
[143]: jointhecommunity.html
[144]: jointhecommunity.html#sqlfluff-on-twitter
[145]: #
[146]: gettingstarted.html
[147]: https://www.sphinx-doc.org/
[148]: https://alabaster.readthedocs.io
[149]: _sources/index.rst.txt
