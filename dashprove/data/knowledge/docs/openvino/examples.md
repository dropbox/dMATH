English | [简体中文][1] | [日本語][2]

# 📚 OpenVINO™ Notebooks

[[Apache License Version 2.0]][3] [[CI]][4] [[CI]][5]

A collection of ready-to-run Jupyter notebooks for learning and experimenting with the OpenVINO™
Toolkit. The notebooks provide an introduction to OpenVINO basics and teach developers how to
leverage our API for optimized deep learning inference.

🚀 Checkout interactive GitHub pages application for navigation between OpenVINO™ Notebooks content:
[OpenVINO™ Notebooks at GitHub Pages][6]

[[notebooks-selector-preview]][7]

List of all notebooks is available in [index file][8].

[[-----------------------------------------------------]][9]

## Table of Contents

* [Table of Contents][10]
* [📝 Installation Guide][11]
* [🚀 Getting Started][12]
* [⚙️ System Requirements][13]
* [💻 Run the Notebooks][14]
  
  * [To Launch a Single Notebook][15]
  * [To Launch all Notebooks][16]
* [🧹 Cleaning Up][17]
* [⚠️ Troubleshooting][18]
* [📊 Telemetry][19]
* [📚 Additional Resources][20]
* [🧑‍💻 Contributors][21]
* [❓ FAQ][22]

[[-----------------------------------------------------]][23]

## 📝 Installation Guide

OpenVINO Notebooks require Python and Git. To get started, select the guide for your operating
system or environment:

───────────┬──────────┬─────────┬───────────┬──────────┬────────────┬──────────┬────────────────────
[Windows][2│[Ubuntu][2│[macOS][2│[Red       │[CentOS][2│[Azure      │[Docker][3│[Amazon             
4]         │5]        │6]       │Hat][27]   │8]        │ML][29]     │0]        │SageMaker][31]      
───────────┴──────────┴─────────┴───────────┴──────────┴────────────┴──────────┴────────────────────

[[-----------------------------------------------------]][32]

## 🚀 Getting Started

Explore Jupyter notebooks using this [page][33], select one related to your needs or give them all a
try. Good Luck!

**NOTE: The main branch of this repository was updated to support the new OpenVINO 2025.4 release.**
To upgrade to the new release version, please run `pip install --upgrade -r requirements.txt` in
your `openvino_env` virtual environment. If you need to install for the first time, see the
[Installation Guide][34] section below. If you wish to use the previous release version of OpenVINO,
please checkout the [2025.3 branch][35]. If you wish to use the previous Long Term Support (LTS)
version of OpenVINO check out the [2023.3 branch][36].

If you need help, please start a GitHub [Discussion][37].

If you run into issues, please check the [troubleshooting section][38], [FAQs][39] or start a GitHub
[discussion][40].

Notebooks with [[binder logo]][41] and [[colab logo]][42] buttons can be run without installing
anything. [Binder][43] and [Google Colab][44] are free online services with limited resources. For
the best performance, please follow the [Installation Guide][45] and run the notebooks locally.

[[-----------------------------------------------------]][46]

## ⚙️ System Requirements

The notebooks run almost anywhere — your laptop, a cloud VM, or even a Docker container. The table
below lists the supported operating systems and Python versions.

────────────────────────────────────────────────────────┬────────────────────────────
Supported Operating System                              │[Python Version             
                                                        │(64-bit)][47]               
────────────────────────────────────────────────────────┼────────────────────────────
Ubuntu 20.04 LTS, 64-bit                                │3.10 - 3.13                 
────────────────────────────────────────────────────────┼────────────────────────────
Ubuntu 22.04 LTS, 64-bit                                │3.10 - 3.13                 
────────────────────────────────────────────────────────┼────────────────────────────
Red Hat Enterprise Linux 8, 64-bit                      │3.10 - 3.13                 
────────────────────────────────────────────────────────┼────────────────────────────
CentOS 7, 64-bit                                        │3.10 - 3.13                 
────────────────────────────────────────────────────────┼────────────────────────────
macOS 10.15.x versions or higher                        │3.10 - 3.13                 
────────────────────────────────────────────────────────┼────────────────────────────
Windows 10, 64-bit Pro, Enterprise or Education editions│3.10 - 3.13                 
────────────────────────────────────────────────────────┼────────────────────────────
Windows Server 2016 or higher                           │3.10 - 3.13                 
────────────────────────────────────────────────────────┴────────────────────────────

[[-----------------------------------------------------]][48]

## 💻 Run the Notebooks

### To Launch a Single Notebook

If you wish to launch only one notebook, like the Monodepth notebook, run the command below (from
the repository root directory):

jupyter lab notebooks/vision-monodepth/vision-monodepth.ipynb

### To Launch all Notebooks

Launch Jupyter Lab with index `README.md` file opened for easier navigation between notebooks
directories and files. Run the following command from the repository root directory:

jupyter lab notebooks/README.md

Alternatively, in your browser select a notebook from the file browser in Jupyter Lab using the left
sidebar. Each tutorial is located in a subdirectory within the `notebooks` directory.

[[-----------------------------------------------------]][49]

## 🧹 Cleaning Up

1. Shut Down Jupyter Kernel
   
   To end your Jupyter session, press `Ctrl-c`. This will prompt you to `Shutdown this Jupyter
   server (y/[n])?` enter `y` and hit `Enter`.

2. Deactivate Virtual Environment
   
   To deactivate your virtualenv, simply run `deactivate` from the terminal window where you
   activated `openvino_env`. This will deactivate your environment.
   
   To reactivate your environment, run `source openvino_env/bin/activate` on Linux or
   `openvino_env\Scripts\activate` on Windows, then type `jupyter lab` or `jupyter notebook` to
   launch the notebooks again.

3. Delete Virtual Environment *(Optional)*
   
   To remove your virtual environment, simply delete the `openvino_env` directory:

* On Linux and macOS:
  
  rm -rf openvino_env

* On Windows:
  
  rmdir /s openvino_env

* Remove `openvino_env` Kernel from Jupyter
  
  jupyter kernelspec remove openvino_env

[[-----------------------------------------------------]][50]

## ⚠️ Troubleshooting

If these tips do not solve your problem, please open a [discussion topic][51] or create an
[issue][52]!

* To check some common installation problems, run `python check_install.py`. This script is located
  in the openvino_notebooks directory. Please run it after activating the `openvino_env` virtual
  environment.
* If you get an `ImportError`, double-check that you installed the Jupyter kernel. If necessary,
  choose the `openvino_env` kernel from the *Kernel->Change Kernel* menu in Jupyter Lab or Jupyter
  Notebook.
* If OpenVINO is installed globally, do not run installation commands in a terminal where
  `setupvars.bat` or `setupvars.sh` are sourced.
* For Windows installation, it is recommended to use *Command Prompt (`cmd.exe`)*, not *PowerShell*.
* If you get `ImportError: cannot import name 'collect_telemetry' from 'notebook_utils'`, make sure
  that you have the latest version of `notebook_utils.py` file downloaded in the notebook directory.
  Try removing outdated `notebook_utils.py` file and re-run the notebook - new utils file will be
  downloaded.

[[-----------------------------------------------------]][53]

## 📊 Telemetry

When you execute a notebook cell that contains `collect_telemetry()` function, telemetry data is
collected to help us improve your experience. This data only indicates that the cell was executed
and does **not** include any personally identifiable information (PII).

By default, anonymous telemetry data is collected, limited solely to the execution of the notebook.
This telemetry does **not** extend to any Intel software, hardware, websites, or products.

If you prefer to disable telemetry, you can do so at any time by commenting out the specific line
responsible for data collection in the notebook:

# collect_telemetry(...)

Also you can disable telemetry collection by setting `SCARF_NO_ANALYTICS` or `DO_NOT_TRACK`
environment variable to `1`:

export SCARF_NO_ANALYTICS=1
# or
export DO_NOT_TRACK=1

Scarf is used for telemetry purposes. Refer to [Scarf documentation][54] to understand how the data
is collected and processed.

[[-----------------------------------------------------]][55]

## 📚 Additional Resources

* [OpenVINO Blog][56] - a collection of technical articles with OpenVINO best practices, interesting
  use cases and tutorials.
* [Awesome OpenVINO][57] - a curated list of OpenVINO based AI projects.
* [OpenVINO GenAI Samples][58] - collection of OpenVINO GenAI API samples.
* [Edge AI Reference Kit][59] - pre-built components and code samples designed to accelerate the
  development and deployment of production-grade AI applications across various industries, such as
  retail, healthcare, and manufacturing.
* [Open Model Zoo demos][60] - console applications that provide templates to help implement
  specific deep learning inference scenarios. These applications show how to preprocess and
  postprocess data for model inference and organize processing pipelines.
* [oneAPI-samples][61] repository demonstrates the performance and productivity offered by oneAPI
  and its toolkits such as oneDNN in a multiarchitecture environment. OpenVINO™ toolkit takes
  advantage of the discrete GPUs using oneAPI, an open programming model for multi-architecture
  programming.

[[-----------------------------------------------------]][62]

## 🧑‍💻 Contributors

Made with [`contrib.rocks`][63].

[[-----------------------------------------------------]][64]

## ❓ FAQ

* [Which devices does OpenVINO support?][65]
* [What is the first CPU generation you support with OpenVINO?][66]
* [Are there any success stories about deploying real-world solutions with OpenVINO?][67]

* Other names and brands may be claimed as the property of others.

Human Rights Information: “Intel is committed to respecting human rights and avoiding causing or
contributing to adverse impacts on human rights. See Intel’s Global Human Rights Principles at
[https://www.intel.com/content/dam/www/central-libraries/us/en/documents/policy-human-rights.pdf][68
]. Intel’s products and software are intended only to be used in applications that do not cause or
contribute to adverse impacts on human rights.

[1]: /openvinotoolkit/openvino_notebooks/blob/latest/README_cn.md
[2]: /openvinotoolkit/openvino_notebooks/blob/latest/README_ja.md
[3]: https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/LICENSE
[4]: https://github.com/openvinotoolkit/openvino_notebooks/actions/workflows/treon_precommit.yml?que
ry=event%3Apush
[5]: https://github.com/openvinotoolkit/openvino_notebooks/actions/workflows/docker.yml?query=event%
3Apush
[6]: https://openvinotoolkit.github.io/openvino_notebooks/
[7]: https://openvinotoolkit.github.io/openvino_notebooks/
[8]: /openvinotoolkit/openvino_notebooks/blob/latest/notebooks/README.md
[9]: /openvinotoolkit/openvino_notebooks/blob/latest
[10]: #table-of-contents
[11]: #-installation-guide
[12]: #-getting-started
[13]: #%EF%B8%8F-system-requirements
[14]: #-run-the-notebooks
[15]: #to-launch-a-single-notebook
[16]: #to-launch-all-notebooks
[17]: #-cleaning-up
[18]: #%EF%B8%8F-troubleshooting
[19]: #-telemetry
[20]: #-additional-resources
[21]: #-contributors
[22]: #-faq
[23]: /openvinotoolkit/openvino_notebooks/blob/latest
[24]: https://github.com/openvinotoolkit/openvino_notebooks/wiki/Windows
[25]: https://github.com/openvinotoolkit/openvino_notebooks/wiki/Ubuntu
[26]: https://github.com/openvinotoolkit/openvino_notebooks/wiki/macOS
[27]: https://github.com/openvinotoolkit/openvino_notebooks/wiki/Red-Hat-and-CentOS
[28]: https://github.com/openvinotoolkit/openvino_notebooks/wiki/Red-Hat-and-CentOS
[29]: https://github.com/openvinotoolkit/openvino_notebooks/wiki/AzureML
[30]: https://github.com/openvinotoolkit/openvino_notebooks/wiki/Docker
[31]: https://github.com/openvinotoolkit/openvino_notebooks/wiki/SageMaker
[32]: /openvinotoolkit/openvino_notebooks/blob/latest
[33]: https://openvinotoolkit.github.io/openvino_notebooks/
[34]: #-installation-guide
[35]: https://github.com/openvinotoolkit/openvino_notebooks/tree/2025.2
[36]: https://github.com/openvinotoolkit/openvino_notebooks/tree/2023.3
[37]: https://github.com/openvinotoolkit/openvino_notebooks/discussions
[38]: #-troubleshooting
[39]: #-faq
[40]: https://github.com/openvinotoolkit/openvino_notebooks/discussions
[41]: https://camo.githubusercontent.com/c351dded6463f29bc0945aaaa25a1ab9556d5f7e02c99d2c74dd641a5eb
7dee9/68747470733a2f2f6d7962696e6465722e6f72672f62616467655f6c6f676f2e737667
[42]: https://camo.githubusercontent.com/eff96fda6b2e0fff8cdf2978f89d61aa434bb98c00453ae23dd0aab8d14
51633/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d
62616467652e737667
[43]: https://mybinder.org/
[44]: https://colab.research.google.com/
[45]: #-installation-guide
[46]: /openvinotoolkit/openvino_notebooks/blob/latest
[47]: https://www.python.org/
[48]: #
[49]: /openvinotoolkit/openvino_notebooks/blob/latest
[50]: /openvinotoolkit/openvino_notebooks/blob/latest
[51]: https://github.com/openvinotoolkit/openvino_notebooks/discussions
[52]: https://github.com/openvinotoolkit/openvino_notebooks/issues
[53]: /openvinotoolkit/openvino_notebooks/blob/latest
[54]: https://docs.scarf.sh/
[55]: /openvinotoolkit/openvino_notebooks/blob/latest
[56]: https://blog.openvino.ai/
[57]: https://github.com/openvinotoolkit/awesome-openvino
[58]: https://github.com/openvinotoolkit/openvino.genai?tab=readme-ov-file#openvino-genai-samples
[59]: https://github.com/openvinotoolkit/openvino_build_deploy
[60]: https://github.com/openvinotoolkit/open_model_zoo/blob/master/demos/README.md
[61]: https://github.com/oneapi-src/oneAPI-samples
[62]: #-contributors
[63]: https://contrib.rocks
[64]: /openvinotoolkit/openvino_notebooks/blob/latest
[65]: https://docs.openvino.ai/2024/openvino-workflow/running-inference/inference-devices-and-modes.
html
[66]: https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/system-requirements.h
tml
[67]: https://www.intel.com/content/www/us/en/internet-of-things/ai-in-production/success-stories.ht
ml
[68]: https://www.intel.com/content/dam/www/central-libraries/us/en/documents/policy-human-rights.pd
f
