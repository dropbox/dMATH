# OpenAI Evals

> You can now configure and run Evals directly in the OpenAI Dashboard. [Get started →][1]

Evals provide a framework for evaluating large language models (LLMs) or systems built using LLMs.
We offer an existing registry of evals to test different dimensions of OpenAI models and the ability
to write your own custom evals for use cases you care about. You can also use your data to build
private evals which represent the common LLMs patterns in your workflow without exposing any of that
data publicly.

If you are building with LLMs, creating high quality evals is one of the most impactful things you
can do. Without evals, it can be very difficult and time intensive to understand how different model
versions might affect your use case. In the words of [OpenAI's President Greg Brockman][2]:

[[https://x.com/gdb/status/1733553161884127435?s=20]][3]

## Setup

To run evals, you will need to set up and specify your [OpenAI API key][4]. After you obtain an API
key, specify it using the [`OPENAI_API_KEY` environment variable][5]. Please be aware of the
[costs][6] associated with using the API when running evals. You can also run and create evals using
[Weights & Biases][7].

**Minimum Required Version: Python 3.9**

### Downloading evals

Our evals registry is stored using [Git-LFS][8]. Once you have downloaded and installed LFS, you can
fetch the evals (from within your local copy of the evals repo) with:

cd evals
git lfs fetch --all
git lfs pull

This will populate all the pointer files under `evals/registry/data`.

You may just want to fetch data for a select eval. You can achieve this via:

git lfs fetch --include=evals/registry/data/${your eval}
git lfs pull

### Making evals

If you are going to be creating evals, we suggest cloning this repo directly from GitHub and
installing the requirements using the following command:

pip install -e .

Using `-e`, changes you make to your eval will be reflected immediately without having to reinstall.

Optionally, you can install the formatters for pre-committing with:

pip install -e .[formatters]

Then run `pre-commit install` to install pre-commit into your git hooks. pre-commit will now run on
every commit.

If you want to manually run all pre-commit hooks on a repository, run `pre-commit run --all-files`.
To run individual hooks use `pre-commit run <hook_id>`.

## Running evals

If you don't want to contribute new evals, but simply want to run them locally, you can install the
evals package via pip:

pip install evals

You can find the full instructions to run existing evals in [`run-evals.md`][9] and our existing
eval templates in [`eval-templates.md`][10]. For more advanced use cases like prompt chains or
tool-using agents, you can use our [Completion Function Protocol][11].

We provide the option for you to log your eval results to a Snowflake database, if you have one or
wish to set one up. For this option, you will further have to specify the `SNOWFLAKE_ACCOUNT`,
`SNOWFLAKE_DATABASE`, `SNOWFLAKE_USERNAME`, and `SNOWFLAKE_PASSWORD` environment variables.

## Writing evals

We suggest getting starting by:

* Walking through the process for building an eval: [`build-eval.md`][12]
* Exploring an example of implementing custom eval logic: [`custom-eval.md`][13]
* Writing your own completion functions: [`completion-fns.md`][14]
* Review our starter guide for writing evals: [Getting Started with OpenAI Evals][15]

Please note that we are currently not accepting evals with custom code! While we ask you to not
submit such evals at the moment, you can still submit model-graded evals with custom model-graded
YAML files.

If you think you have an interesting eval, please open a pull request with your contribution. OpenAI
staff actively review these evals when considering improvements to upcoming models.

## FAQ

Do you have any examples of how to build an eval from start to finish?

* Yes! These are in the `examples` folder. We recommend that you also read through
  [`build-eval.md`][16] in order to gain a deeper understanding of what is happening in these
  examples.

Do you have any examples of evals implemented in multiple different ways?

* Yes! In particular, see `evals/registry/evals/coqa.yaml`. We have implemented small subsets of the
  [CoQA][17] dataset for various eval templates to help illustrate the differences.

When I run an eval, it sometimes hangs at the very end (after the final report). What's going on?

* This is a known issue, but you should be able to interrupt it safely and the eval should finish
  immediately after.

There's a lot of code, and I just want to spin up a quick eval. Help? OR,

I am a world-class prompt engineer. I choose not to code. How can I contribute my wisdom?

* If you follow an existing [eval template][18] to build a basic or model-graded eval, you don't
  need to write any evaluation code at all! Just provide your data in JSON format and specify your
  eval parameters in YAML. [build-eval.md][19] walks you through these steps, and you can supplement
  these instructions with the Jupyter notebooks in the `examples` folder to help you get started
  quickly. Keep in mind, though, that a good eval will inevitably require careful thought and
  rigorous experimentation!

## Disclaimer

By contributing to evals, you are agreeing to make your evaluation logic and data under the same MIT
license as this repository. You must have adequate rights to upload any data used in an eval. OpenAI
reserves the right to use this data in future service improvements to our product. Contributions to
OpenAI evals will be subject to our usual Usage Policies:
[https://platform.openai.com/docs/usage-policies][20].

[1]: https://platform.openai.com/docs/guides/evals
[2]: https://twitter.com/gdb/status/1733553161884127435
[3]: https://private-user-images.githubusercontent.com/35577566/289374940-ce7840ff-43a8-4d88-bb2f-6b
207410333b.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3Lmdp
dGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NjY0NjQ3NTUsIm5iZiI6MTc2NjQ2NDQ1NSwicGF0aCI6
Ii8zNTU3NzU2Ni8yODkzNzQ5NDAtY2U3ODQwZmYtNDNhOC00ZDg4LWJiMmYtNmIyMDc0MTAzMzNiLnBuZz9YLUFtei1BbGdvcml0
aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTEyMjMlMkZ1cy1l
YXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUxMjIzVDA0MzQxNVomWC1BbXotRXhwaXJlcz0zMDAmWC1B
bXotU2lnbmF0dXJlPTg2MWJlMGIxZDQ4YWEzYTkyNDY3ZGQ2YmNiZWMwNGFkNGIzMDljZGNjMTQ3OTYyYThhNjRhOTA4ZmVhOTFm
OWUmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.eQC5E7_ZrSiJDglHLkCnodLxG6qDVnT6Gh8nHxpq8Eo
[4]: https://platform.openai.com/account/api-keys
[5]: https://platform.openai.com/docs/quickstart/step-2-setup-your-api-key
[6]: https://openai.com/pricing
[7]: https://wandb.ai/wandb_fc/openai-evals/reports/OpenAI-Evals-Demo-Using-W-B-Prompts-to-Run-Evalu
ations--Vmlldzo0MTI4ODA3
[8]: https://git-lfs.com/
[9]: /openai/evals/blob/main/docs/run-evals.md
[10]: /openai/evals/blob/main/docs/eval-templates.md
[11]: /openai/evals/blob/main/docs/completion-fns.md
[12]: /openai/evals/blob/main/docs/build-eval.md
[13]: /openai/evals/blob/main/docs/custom-eval.md
[14]: /openai/evals/blob/main/docs/completion-fns.md
[15]: https://cookbook.openai.com/examples/evaluation/getting_started_with_openai_evals
[16]: /openai/evals/blob/main/docs/build-eval.md
[17]: https://stanfordnlp.github.io/coqa/
[18]: /openai/evals/blob/main/docs/eval-templates.md
[19]: /openai/evals/blob/main/docs/build-eval.md
[20]: https://platform.openai.com/docs/usage-policies
