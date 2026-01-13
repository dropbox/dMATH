v1.14.x (latest)

* Terraform
* [v1.15.x (alpha)][1]
* [v1.13.x][2]
* [v1.12.x][3]
* [v1.11.x][4]
* [v1.10.x][5]
* [v1.9.x][6]
* [v1.8.x][7]
* [v1.7.x][8]
* [v1.6.x][9]
* [v1.5.x][10]
* [v1.4.x][11]
* [v1.3.x][12]
* [v1.2.x][13]
* [v1.1.x][14]

# Terraform CLI Overview

This topic provides an overview of the Terraform command line interface.

> **Hands-on:** Try the [Terraform: Get Started][15] tutorials.

## Introduction

The command line interface to Terraform is the `terraform` command, which accepts a variety of
subcommands such as `terraform init` or `terraform plan`.

We refer to the `terraform` command line tool as "Terraform CLI" elsewhere in the documentation.
This terminology is often used to distinguish it from other components you might use in the
Terraform product family, such as [HCP Terraform][16] or the various [Terraform providers][17],
which are developed and released separately from Terraform CLI.

To view a list of the commands available in your current Terraform version, run `terraform` with no
additional arguments:

`Usage: terraform [global options] <subcommand> [args]

The available commands for execution are listed below.
The primary workflow commands are given first, followed by
less common or more advanced commands.

Main commands:
  init          Prepare your working directory for other commands
  validate      Check whether the configuration is valid
  plan          Show changes required by the current configuration
  apply         Create or update infrastructure
  destroy       Destroy previously-created infrastructure

All other commands:
  console       Try Terraform expressions at an interactive command prompt
  fmt           Reformat your configuration in the standard style
  force-unlock  Release a stuck lock on the current workspace
  get           Install or upgrade remote Terraform modules
  graph         Generate a Graphviz graph of the steps in an operation
  import        Associate existing infrastructure with a Terraform resource
  login         Obtain and save credentials for a remote host
  logout        Remove locally-stored credentials for a remote host
  metadata      Metadata related commands
  modules       Show all declared modules in a working directory
  output        Show output values from your root module
  providers     Show the providers required for this configuration
  refresh       Update the state to match remote systems
  show          Show the current state or a saved plan
  state         Advanced state management
  taint         Mark a resource instance as not fully functional
  untaint       Remove the 'tainted' state from a resource instance
  version       Show the current Terraform version
  workspace     Workspace management

Global options (use these before the subcommand, if any):
  -chdir=DIR    Switch to a different working directory before executing the
                given subcommand.
  -help         Show this help output, or the help for a specified subcommand.
  -version      An alias for the "version" subcommand.
`

(The output from your current Terraform version may be different than the above example.)

To get specific help for any specific command, use the `-help` option with the relevant subcommand.
For example, to see help about the "validate" subcommand you can run `terraform validate -help`.

The inline help built in to Terraform CLI describes the most important characteristics of each
command. For more detailed information, refer to each command's page for details.

## Switching working directory with `-chdir`

The usual way to run Terraform is to first switch to the directory containing the `.tf` files for
your root module (for example, using the `cd` command), so that Terraform will find those files
automatically without any extra arguments.

In some cases though — particularly when wrapping Terraform in automation scripts — it can be
convenient to run Terraform from a different directory than the root module directory. To allow
that, Terraform supports a global option `-chdir=...` which you can include before the name of the
subcommand you intend to run:

`terraform -chdir=environments/production apply
`

The `chdir` option instructs Terraform to change its working directory to the given directory before
running the given subcommand. This means that any files that Terraform would normally read or write
in the current working directory will be read or written in the given directory instead.

There are two exceptions where Terraform will use the original working directory even when you
specify `-chdir=...`:

* Settings in the [CLI Configuration][18] are not for a specific subcommand and Terraform processes
  them before acting on the `-chdir` option.
* In case you need to use files from the original working directory as part of your configuration, a
  reference to `path.cwd` in the configuration will produce the original working directory instead
  of the overridden working directory. Use `path.root` to get the root module directory.

## Shell Tab-completion

If you use either `bash` or `zsh` as your command shell, Terraform can provide tab-completion
support for all command names and some command arguments.

To add the necessary commands to your shell profile, run the following command:

`terraform -install-autocomplete
`

After installation, it is necessary to restart your shell or to re-read its profile script before
completion will be activated.

To uninstall the completion hook, assuming that it has not been modified manually in the shell
profile, run the following command:

`terraform -uninstall-autocomplete
`

## Upgrade and Security Bulletin Checks

The Terraform CLI commands interact with the HashiCorp service [Checkpoint][19] to check for the
availability of new versions and for critical security bulletins about the current version.

One place where the effect of this can be seen is in `terraform version`, where it is used by
default to indicate in the output when a newer version is available.

Only anonymous information, which cannot be used to identify the user or host, is sent to
Checkpoint. An anonymous ID is sent which helps de-duplicate warning messages. Both the anonymous id
and the use of checkpoint itself are completely optional and can be disabled.

Checkpoint itself can be entirely disabled for all HashiCorp products by setting the environment
variable `CHECKPOINT_DISABLE` to any non-empty value.

Alternatively, settings in [the CLI configuration file][20] can be used to disable checkpoint
features. The following checkpoint-related settings are supported in this file:

* [`disable_checkpoint`][21] - set to `true` to disable checkpoint calls entirely. This is similar
  to the `CHECKPOINT_DISABLE` environment variable described above.
* [`disable_checkpoint_signature`][22] - set to `true` to disable the use of an anonymous signature
  in checkpoint requests. This allows Terraform to check for security bulletins but does not send
  the anonymous signature in these requests.

[The Checkpoint client code][23] used by Terraform is available for review by any interested party.

[Edit this page on GitHub][24]

[1]: /terraform/cli/v1.15.x/commands
[2]: /terraform/cli/v1.13.x/commands
[3]: /terraform/cli/v1.12.x/commands
[4]: /terraform/cli/v1.11.x/commands
[5]: /terraform/cli/v1.10.x/commands
[6]: /terraform/cli/v1.9.x/commands
[7]: /terraform/cli/v1.8.x/commands
[8]: /terraform/cli/v1.7.x/commands
[9]: /terraform/cli/v1.6.x/commands
[10]: /terraform/cli/v1.5.x/commands
[11]: /terraform/cli/v1.4.x/commands
[12]: /terraform/cli/v1.3.x/commands
[13]: /terraform/cli/v1.2.x/commands
[14]: /terraform/cli/v1.1.x/commands
[15]: /terraform/tutorials/aws-get-started?utm_source=WEBSITE&utm_medium=WEB_IO&utm_offer=ARTICLE_PA
GE&utm_content=DOCS
[16]: /terraform/cloud-docs
[17]: /terraform/language/providers
[18]: /terraform/cli/config/config-file
[19]: https://checkpoint.hashicorp.com/
[20]: /terraform/cli/config/config-file
[21]: /terraform/cli/commands#disable_checkpoint
[22]: /terraform/cli/commands#disable_checkpoint_signature
[23]: https://github.com/hashicorp/go-checkpoint
[24]: https://github.com/hashicorp/web-unified-docs/blob/main/content/terraform/v1.14.x/docs/cli/com
mands/index.mdx
