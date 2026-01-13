[https://static.scarf.sh/a.png?x-pxid=bc3c57b0-9a65-49fe-b8ea-f711c4d35b82]

[https://i.imgur.com/nv33goV.png]

# The open standard for data logging

### [Documentation][1] • [Slack Community][2] • [Python Quickstart][3] • [WhyLabs Quickstart][4]

[ [License] ][5] [ [PyPi Version] ][6] [ [Code style: black] ][7] [ [PyPi Downloads] ][8] [ [CI]
][9] [ [Maintainability] ][10]

# What is whylogs[#][11]

whylogs is an open source library for logging any kind of data. With whylogs, users are able to
generate summaries of their datasets (called *whylogs profiles*) which they can use to:

1. Track changes in their dataset
2. Create *data constraints* to know whether their data looks the way it should
3. Quickly visualize key summary statistics about their datasets

These three functionalities enable a variety of use cases for data scientists, machine learning
engineers, and data engineers:

* Detect data drift in model input features
* Detect training-serving skew, concept drift, and model performance degradation
* Validate data quality in model inputs or in a data pipeline
* Perform exploratory data analysis of massive datasets
* Track data distributions & data quality for ML experiments
* Enable data auditing and governance across the organization
* Standardize data documentation practices across the organization
* And more
[ [WhyLabs Signup] ][12]

If you have any questions, comments, or just want to hang out with us, please join [our Slack
Community][13]. In addition to joining the Slack Community, you can also help this project by giving
us a ⭐ in the upper right corner of this page.

# Python Quickstart[#][14]

Installing whylogs using the pip package manager is as easy as running `pip install whylogs` in your
terminal.

From here, you can quickly log a dataset:

import whylogs as why
import pandas as pd

#dataframe
df = pd.read_csv("path/to/file.csv")
results = why.log(df)

And there you have it, you now have a whylogs profile. To learn more about what a whylogs profile is
and what you can do with it, read on.

# Table of Contents[#][15]

* whylogs Profiles
* Data Constraints
* Profile Visualization
* Integrations
* Supported Data Types
* Examples
* Usage Statistics
* Community
* Contribute

# whylogs Profiles[#][16]

## What are profiles[#][17]

whylogs profiles are the core of the whylogs library. They capture key statistical properties of
data, such as the distribution (far beyond simple mean, median, and standard deviation measures),
the number of missing values, and a wide range of configurable custom metrics. By capturing these
summary statistics, we are able to accurately represent the data and enable all of the use cases
described in the introduction.

whylogs profiles have three properties that make them ideal for data logging: they are
**efficient**, **customizable**, and **mergeable**.

[https://user-images.githubusercontent.com/7946482/171064257-26bf727e-3480-4ec3-9c9d-5d8a79567bca.pn
g]

**Efficient**: whylogs profiles efficiently describe the dataset that they represent. This high
fidelity representation of datasets is what enables whylogs profiles to be effective snapshots of
the data. They are better at capturing the characteristics of a dataset than a sample would be—as
discussed in our [Data Logging: Sampling versus Profiling][18] blog post—and are very compact.

[https://user-images.githubusercontent.com/7946482/171064575-72ee0f76-7365-4fd1-9cab-4debb673baa8.pn
g]

**Customizable**: The statistics that whylogs profiles collect are easily configured and
customizable. This is useful because different data types and use cases require different metrics,
and whylogs users need to be able to easily define custom trackers for those metrics. It’s the
customizability of whylogs that enables our text, image, and other complex data trackers.

[https://user-images.githubusercontent.com/7946482/171064525-2d314534-6cdb-4c07-9d9f-5c74d5c03029.pn
g]

**Mergeable**: One of the most powerful features of whylogs profiles is their mergeability.
Mergeability means that whylogs profiles can be combined together to form new profiles which
represent the aggregate of their constituent profiles. This enables logging for distributed and
streaming systems, and allows users to view aggregated data across any time granularity.

## How do you generate profiles[#][19]

Once whylogs is installed, it’s easy to generate profiles in both Python and Java environments.

To generate a profile from a Pandas dataframe in Python, simply run:

import whylogs as why
import pandas as pd

#dataframe
df = pd.read_csv("path/to/file.csv")
results = why.log(df)

## What can you do with profiles[#][20]

Once you’ve generated whylogs profiles, a few things can be done with them:

In your local Python environment, you can set data constraints or visualize your profiles. Setting
data constraints on your profiles allows you to get notified when your data don’t match your
expectations, allowing you to do data unit testing and some baseline data monitoring. With the
Profile Visualizer, you can visually explore your data, allowing you to understand it and ensure
that your ML models are ready for production.

In addition, you can send whylogs profiles to the SaaS ML monitoring and AI observability platform
[WhyLabs][21]. With WhyLabs, you can automatically set up monitoring for your machine learning
models, getting notified on both data quality and data change issues (such as data drift). If you’re
interested in trying out WhyLabs, check out the always free [Starter edition][22], which allows you
to experience the entire platform’s capabilities with no credit card required.

# WhyLabs[#][23]

WhyLabs is a managed service offering built for helping users make the most of their whylogs
profiles. With WhyLabs, users can ingest profiles and set up automated monitoring as well as gain
full observability into their data and ML systems. With WhyLabs, users can ensure the reliability of
their data and models, and debug any problems that arise with them.

Ingesting whylogs profiles into WhyLabs is easy. After obtaining your access credentials from the
platform, you’ll need to set them in your Python environment, log a dataset, and write it to
WhyLabs, like so:

import whylogs as why
import os

os.environ["WHYLABS_API_KEY"] = "YOUR-API-KEY"
os.environ["WHYLABS_DEFAULT_DATASET_ID"] = "model-0" # The selected model project "MODEL-NAME" is "m
odel-0"

results = why.log(df)

results.writer("whylabs").write()

[image]

If you’re interested in trying out WhyLabs, check out the always free [Starter edition][24], which
allows you to experience the entire platform’s capabilities with no credit card required.

# Data Constraints[#][25]

Constraints are a powerful feature built on top of whylogs profiles that enable you to quickly and
easily validate that your data looks the way that it should. There are numerous types of constraints
that you can set on your data (that numerical data will always fall within a certain range, that
text data will always be in a JSON format, etc) and, if your dataset fails to satisfy a constraint,
you can fail your unit tests or your CI/CD pipeline.

A simple example of setting and testing a constraint is:

import whylogs as why
from whylogs.core.constraints import Constraints, ConstraintsBuilder
from whylogs.core.constraints.factories import greater_than_number

profile_view = why.log(df).view()
builder = ConstraintsBuilder(profile_view)
builder.add_constraint(greater_than_number(column_name="col_name", number=0.15))

constraints = builder.build()
constraints.report()

To learn more about constraints, check out: the [Constraints Example][26].

# Profile Visualization[#][27]

In addition to being able to automatically get notified about potential issues in data, it’s also
useful to be able to inspect your data manually. With the profile visualizer, you can generate
interactive reports about your profiles (either a single profile or comparing profiles against each
other) directly in your Jupyter notebook environment. This enables exploratory data analysis, data
drift detection, and data observability.

To access the profile visualizer, install the `[viz]` module of whylogs by running `pip install
"whylogs[viz]"` in your terminal. One type of profile visualization that we can create is a drift
report; here’s a simple example of how to analyze the drift between two profiles:

import whylogs as why

from whylogs.viz import NotebookProfileVisualizer

result = why.log(pandas=df_target)
prof_view = result.view()

result_ref = why.log(pandas=df_reference)
prof_view_ref = result_ref.view()

visualization = NotebookProfileVisualizer()
visualization.set_profiles(target_profile_view=prof_view, reference_profile_view=prof_view_ref)

visualization.summary_drift_report()

[image]

To learn more about visualizing your profiles, check out: the [Visualizer Example][28]

# Data Types[#][29]

whylogs supports both structured and unstructured data, specifically:

────────┬────┬──────────────────────────────────────────────────────────────────────────────────────
Data    │Feat│Notebook Example                                                                      
type    │ures│                                                                                      
────────┼────┼──────────────────────────────────────────────────────────────────────────────────────
Tabular │✅  │[Getting started with structured data][30]                                            
Data    │    │                                                                                      
────────┼────┼──────────────────────────────────────────────────────────────────────────────────────
Image   │✅  │[Getting started with images][31]                                                     
Data    │    │                                                                                      
────────┼────┼──────────────────────────────────────────────────────────────────────────────────────
Text    │✅  │[String Features][32]                                                                 
Data    │    │                                                                                      
────────┼────┼──────────────────────────────────────────────────────────────────────────────────────
Embeddin│✅  │[Embeddings Distance Logging][33]                                                     
gs      │    │                                                                                      
────────┼────┼──────────────────────────────────────────────────────────────────────────────────────
Other   │✋  │Do you have a request for a data type that you don’t see listed here? Raise an issue  
Data    │    │or join our Slack community and make a request! We’re always happy to help            
Types   │    │                                                                                      
────────┴────┴──────────────────────────────────────────────────────────────────────────────────────

# Integrations[#][34]

[current integration]

whylogs can seamslessly interact with different tooling along your Data and ML pipelines. We have
currently built integrations with:

* AWS S3
* Apache Airflow
* Apache Spark
* Mlflow
* GCS

and much more!

If you want to check out our complete list, please refer to our [integrations examples][35] page.

# Examples[#][36]

For a full set of our examples, please check out the [examples folder][37].

# Benchmarks of whylogs[#][38]

By design, whylogs run directly in the data pipeline or in a sidecar container and use highly
scalable streaming algorithms to compute statistics. Since data logging with whylogs happens in the
same infrastructure where the raw data is being processed, it’s important to think about the compute
overhead. For the majority of use cases, the overhead is minimal, usually under 1%. For very large
data volumes with thousands of features and 10M+ QPS it can add ~5% overhead. However, for large
data volumes, customers are typically in a distributed environment, such as Ray or Apache Spark.
This means they benefit from whylogs parallelization—and the map-reducible property of the whylogs
profiles keeping the compute overhead to a minimum. Below are benchmarks to demonstrate how
efficient whylogs is at processing tabular data with default configurations (tracking distributions,
missing values, counts, cardinality, and schema). Two important advantages of this approach are that
parallelization speeds up the calculation and whylogs scales with the number of features, rather
than the number of rows. Learn more about how whylogs scales here.

──────────────┬────────────────────┬────────────────────────────┬─────┬─────────────────────────────
DATA VOLUME   │TOTAL COST OF       │INSTANCE TYPE               │CLUST│PROCESSING TIME              
              │RUNNING WHYLOGS     │                            │ER   │                             
              │                    │                            │SIZE │                             
──────────────┼────────────────────┼────────────────────────────┼─────┼─────────────────────────────
10 GB ~34M    │~ \( 0.026 per 10   │c5a.2xlarge, 8 CPU 16GB RAM,│2    │2.6 minutes of profiling time
rows x 43     │GB, or \)2.45 per TB│$0.308 on demand price per  │insta│per instance (running in     
columns       │                    │hour                        │nces │parallel)                    
──────────────┼────────────────────┼────────────────────────────┼─────┼─────────────────────────────
10 GB, ~34M   │~ \(0.016 per 10 GB,│c6g.2xlarge, 8 CPU 16GB RAM,│2    │1.7 minutes of profiling time
rows x 43     │estimated \)1.60 per│$0.272 on demand price per  │insta│per instance (running in     
columns       │TB                  │hour                        │nces │parallel)                    
──────────────┼────────────────────┼────────────────────────────┼─────┼─────────────────────────────
10 GB ~34M    │~ $ 0.045 per 10 GB │c5a.2xlarge, 8 CPU 16GB RAM,│16   │33 seconds of profiling time 
rows x 43     │                    │$0.308 on demand price per  │insta│per instance (running in     
columns       │                    │hour                        │nces │parallel)                    
──────────────┼────────────────────┼────────────────────────────┼─────┼─────────────────────────────
80 GB, 83M    │~ $0.139 per 80 GB  │c5a.2xlarge, 8 CPU 16GB RAM,│16   │1.7 minutes of profiling time
rows x 119    │                    │$0.308 on demand price per  │insta│per instance (running in     
columns       │                    │hour                        │nces │parallel)                    
──────────────┼────────────────────┼────────────────────────────┼─────┼─────────────────────────────
100 GB, 290M  │~ $0.221 per 100 GB │c5a.2xlarge, 8 CPU 16GB RAM,│16   │2.7 minutes of profiling time
rows x 43     │                    │$0.308 on demand price per  │insta│per instance (running in     
columns       │                    │hour                        │nces │parallel)                    
──────────────┴────────────────────┴────────────────────────────┴─────┴─────────────────────────────

# Usage Statistics[#][39]

Starting with whylogs v1.0.0, whylogs by default collects anonymous information about a user’s
environment. These usage statistics do not include any information about the users or the data they
are profiling, only the environment in which the user is running whylogs.

To read more about what usage statistics whylogs collects, check out the relevant
[documentation][40].

To turn off Usage Statistics, simply set the `WHYLOGS_NO_ANALYTICS` environment variable to True,
like so:

import os
os.environ['WHYLOGS_NO_ANALYTICS']='True'

# Community[#][41]

If you have any questions, comments, or just want to hang out with us, please join [our Slack
channel][42].

# Contribute[#][43]

## How to Contribute[#][44]

We welcome contributions to whylogs. Please see our [contribution guide][45] and our [development
guide][46] for details.

## Contributors[#][47]

Made with [contrib.rocks][48].

* [Index][49]
* [Search Page][50]
* [Module Index][51]

[1]: https://whylogs.readthedocs.io/
[2]: https://bit.ly/whylogsslack
[3]: https://github.com/whylabs/whylogs#python-quickstart
[4]: https://whylogs.readthedocs.io/en/latest/examples/integrations/writers/Writing_to_WhyLabs.html
[5]: https://github.com/whylabs/whylogs-python/blob/mainline/LICENSE
[6]: https://badge.fury.io/py/whylogs
[7]: https://github.com/python/black
[8]: https://pepy.tech/project/whylogs
[9]: bit.ly/whylogs
[10]: https://codeclimate.com/github/whylabs/whylogs-python/maintainability
[11]: #what-is-whylogs
[12]: https://hub.whylabsapp.com/signup
[13]: https://bit.ly/rsqrd-slack
[14]: #python-quickstart-a-name-python-quickstart
[15]: #table-of-contents
[16]: #whylogs-profiles-a-name-whylogs-profiles
[17]: #what-are-profiles
[18]: https://whylabs.ai/blog/posts/data-logging-sampling-versus-profiling
[19]: #how-do-you-generate-profiles
[20]: #what-can-you-do-with-profiles
[21]: https://whylabs.ai
[22]: https://whylabs.ai/free
[23]: #whylabs-a-name-whylabs
[24]: https://hub.whylabsapp.com/signup
[25]: #data-constraints-a-name-data-constraints
[26]: https://bit.ly/whylogsconstraintsexample
[27]: #profile-visualization-a-name-profile-visualization
[28]: https://bit.ly/whylogsvisualizerexample
[29]: #data-types-a-name-data-types
[30]: https://github.com/whylabs/whylogs/blob/mainline/python/examples/basic/Getting_Started.ipynb
[31]: https://github.com/whylabs/whylogs/blob/mainline/python/examples/advanced/Image_Logging.ipynb
[32]: https://github.com/whylabs/whylogs/blob/maintenance/0.7.x/examples/String_Features.ipynb
[33]: https://github.com/whylabs/whylogs/blob/mainline/python/examples/experimental/embeddings/Embed
dings_Distance_Logging.ipynb
[34]: #integrations
[35]: https://github.com/whylabs/whylogs/tree/mainline/python/examples/integrations
[36]: #examples
[37]: https://github.com/whylabs/whylogs/tree/mainline/python/examples
[38]: #benchmarks-of-whylogs
[39]: #usage-statistics-a-name-whylogs-profiles
[40]: https://docs.whylabs.ai/docs/usage-statistics/
[41]: #community
[42]: http://join.slack.whylabs.ai/
[43]: #contribute
[44]: #how-to-contribute
[45]: https://github.com/whylabs/whylogs/blob/mainline/.github/CONTRIBUTING.md
[46]: https://github.com/whylabs/whylogs/blob/mainline/.github/DEVELOPMENT.md
[47]: #contributors
[48]: https://contrib.rocks
[49]: genindex.html
[50]: search.html
[51]: py-modindex.html
