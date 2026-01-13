The W&B Python SDK, accessible at `wandb`, enables you to train and fine-tune models, and manage
models from experimentation to production.

> After performing your training and fine-tuning operations with this SDK, you can use [the Public
> API][1] to query and analyze the data that was logged, and [the Reports and Workspaces API][2] to
> generate a web-publishable [report][3] summarizing your work.

## [​][4]
## Installation and setup

### [​][5]
### Sign up and create an API key

To authenticate your machine with W&B, you must first generate an API key at
[https://wandb.ai/authorize][6].

### [​][7]
### Install and import packages

Install the W&B library.
Report incorrect code
Copy
Ask AI
`pip install wandb
`

### [​][8]
### Import W&B Python SDK:

Report incorrect code
Copy
Ask AI
`import wandb

# Specify your team entity
entity = "<team_entity>"

# Project that the run is recorded to
project = "my-awesome-project"

with wandb.init(entity=entity, project=project) as run:
   run.log({"accuracy": 0.9, "loss": 0.1})
`

[1]: /models/ref/python/public-api
[2]: /models/ref/wandb_workspaces
[3]: /models/reports
[4]: #installation-and-setup
[5]: #sign-up-and-create-an-api-key
[6]: https://wandb.ai/authorize
[7]: #install-and-import-packages
[8]: #import-w&b-python-sdk:
