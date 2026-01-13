* Test Plugins
* [ View page source][1]

# Test Plugins[][2]

Bandit supports many different tests to detect various security issues in python code. These tests
are created as plugins and new ones can be created to extend the functionality offered by bandit
today.

## Writing Tests[][3]

*To write a test:*
  * Identify a vulnerability to build a test for, and create a new file in examples/ that contains
    one or more cases of that vulnerability.
  * Create a new Python source file to contain your test, you can reference existing tests for
    examples.
  * Consider the vulnerability you’re testing for, mark the function with one or more of the
    appropriate decorators:
  
  > * @checks(‘Call’)
  > * @checks(‘Import’, ‘ImportFrom’)
  > * @checks(‘Str’)
  
  * Register your plugin using the bandit.plugins entry point, see example.
  * The function that you create should take a parameter “context” which is an instance of the
    context class you can query for information about the current element being examined. You can
    also get the raw AST node for more advanced use cases. Please see the context.py file for more.
  * Extend your Bandit configuration file as needed to support your new test.
  * Execute Bandit against the test file you defined in examples/ and ensure that it detects the
    vulnerability. Consider variations on how this vulnerability might present itself and extend the
    example file and the test function accordingly.

## Config Generation[][4]

In Bandit 1.0+ config files are optional. Plugins that need config settings are required to
implement a module global gen_config function. This function is called with a single parameter, the
test plugin name. It should return a dictionary with keys being the config option names and values
being the default settings for each option. An example gen_config might look like the following:

def gen_config(name):
    if name == 'try_except_continue':
        return {'check_typed_exception': False}

When no config file is specified, or when the chosen file has no section pertaining to a given
plugin, gen_config will be called to provide defaults.

The config file generation tool bandit-config-generator will also call gen_config on all discovered
plugins to produce template config blocks. If the defaults are acceptable then these blocks may be
deleted to create a minimal configuration, or otherwise edited as needed. The above example would
produce the following config snippet.

try_except_continue: {check_typed_exception: false}

## Example Test Plugin[][5]

@bandit.checks('Call')
def prohibit_unsafe_deserialization(context):
    if 'unsafe_load' in context.call_function_name_qual:
        return bandit.Issue(
            severity=bandit.HIGH,
            confidence=bandit.HIGH,
            text="Unsafe deserialization detected."
        )

To register your plugin, you have two options:

1. If you’re using setuptools directly, add something like the following to your setup call:
   
   # If you have an imaginary bson formatter in the bandit_bson module
   # and a function called `formatter`.
   entry_points={'bandit.formatters': ['bson = bandit_bson:formatter']}
   # Or a check for using mako templates in bandit_mako that
   entry_points={'bandit.plugins': ['mako = bandit_mako']}
2. If you’re using pbr, add something like the following to your setup.cfg file:
   
   [entry_points]
   bandit.formatters =
       bson = bandit_bson:formatter
   bandit.plugins =
       mako = bandit_mako

## Plugin ID Groupings[][6]

────┬──────────────────────────────────────
ID  │Description                           
────┼──────────────────────────────────────
B1xx│misc tests                            
────┼──────────────────────────────────────
B2xx│application/framework misconfiguration
────┼──────────────────────────────────────
B3xx│blacklists (calls)                    
────┼──────────────────────────────────────
B4xx│blacklists (imports)                  
────┼──────────────────────────────────────
B5xx│cryptography                          
────┼──────────────────────────────────────
B6xx│injection                             
────┼──────────────────────────────────────
B7xx│XSS                                   
────┴──────────────────────────────────────

## Complete Test Plugin Listing[][7]

* [B101: assert_used][8]
* [B102: exec_used][9]
* [B103: set_bad_file_permissions][10]
* [B104: hardcoded_bind_all_interfaces][11]
* [B105: hardcoded_password_string][12]
* [B106: hardcoded_password_funcarg][13]
* [B107: hardcoded_password_default][14]
* [B108: hardcoded_tmp_directory][15]
* [B109: password_config_option_not_marked_secret][16]
* [B110: try_except_pass][17]
* [B111: execute_with_run_as_root_equals_true][18]
* [B112: try_except_continue][19]
* [B113: request_without_timeout][20]
* [B201: flask_debug_true][21]
* [B202: tarfile_unsafe_members][22]
* [B324: hashlib][23]
* [B501: request_with_no_cert_validation][24]
* [B502: ssl_with_bad_version][25]
* [B503: ssl_with_bad_defaults][26]
* [B504: ssl_with_no_version][27]
* [B505: weak_cryptographic_key][28]
* [B506: yaml_load][29]
* [B507: ssh_no_host_key_verification][30]
* [B508: snmp_insecure_version][31]
* [B509: snmp_weak_cryptography][32]
* [B601: paramiko_calls][33]
* [B602: subprocess_popen_with_shell_equals_true][34]
* [B603: subprocess_without_shell_equals_true][35]
* [B604: any_other_function_with_shell_equals_true][36]
* [B605: start_process_with_a_shell][37]
* [B606: start_process_with_no_shell][38]
* [B607: start_process_with_partial_path][39]
* [B608: hardcoded_sql_expressions][40]
* [B609: linux_commands_wildcard_injection][41]
* [B610: django_extra_used][42]
* [B611: django_rawsql_used][43]
* [B612: logging_config_insecure_listen][44]
* [B613: trojansource][45]
* [B614: pytorch_load][46]
* [B615: huggingface_unsafe_download][47]
* [B701: jinja2_autoescape_false][48]
* [B702: use_of_mako_templates][49]
* [B703: django_mark_safe][50]
* [B704: markupsafe_markup_xss][51]
[ Previous][52] [Next ][53]

© Copyright 2025, Bandit Developers.

Built with [Sphinx][54] using a [theme][55] provided by [Read the Docs][56].

[1]: ../_sources/plugins/index.rst.txt
[2]: #test-plugins
[3]: #writing-tests
[4]: #config-generation
[5]: #example-test-plugin
[6]: #plugin-id-groupings
[7]: #complete-test-plugin-listing
[8]: b101_assert_used.html
[9]: b102_exec_used.html
[10]: b103_set_bad_file_permissions.html
[11]: b104_hardcoded_bind_all_interfaces.html
[12]: b105_hardcoded_password_string.html
[13]: b106_hardcoded_password_funcarg.html
[14]: b107_hardcoded_password_default.html
[15]: b108_hardcoded_tmp_directory.html
[16]: b109_password_config_option_not_marked_secret.html
[17]: b110_try_except_pass.html
[18]: b111_execute_with_run_as_root_equals_true.html
[19]: b112_try_except_continue.html
[20]: b113_request_without_timeout.html
[21]: b201_flask_debug_true.html
[22]: b202_tarfile_unsafe_members.html
[23]: b324_hashlib.html
[24]: b501_request_with_no_cert_validation.html
[25]: b502_ssl_with_bad_version.html
[26]: b503_ssl_with_bad_defaults.html
[27]: b504_ssl_with_no_version.html
[28]: b505_weak_cryptographic_key.html
[29]: b506_yaml_load.html
[30]: b507_ssh_no_host_key_verification.html
[31]: b508_snmp_insecure_version.html
[32]: b509_snmp_weak_cryptography.html
[33]: b601_paramiko_calls.html
[34]: b602_subprocess_popen_with_shell_equals_true.html
[35]: b603_subprocess_without_shell_equals_true.html
[36]: b604_any_other_function_with_shell_equals_true.html
[37]: b605_start_process_with_a_shell.html
[38]: b606_start_process_with_no_shell.html
[39]: b607_start_process_with_partial_path.html
[40]: b608_hardcoded_sql_expressions.html
[41]: b609_linux_commands_wildcard_injection.html
[42]: b610_django_extra_used.html
[43]: b611_django_rawsql_used.html
[44]: b612_logging_config_insecure_listen.html
[45]: b613_trojansource.html
[46]: b614_pytorch_load.html
[47]: b615_huggingface_unsafe_download.html
[48]: b701_jinja2_autoescape_false.html
[49]: b702_use_of_mako_templates.html
[50]: b703_django_mark_safe.html
[51]: b704_markupsafe_markup_xss.html
[52]: ../integrations.html
[53]: b101_assert_used.html
[54]: https://www.sphinx-doc.org/
[55]: https://github.com/readthedocs/sphinx_rtd_theme
[56]: https://readthedocs.org
