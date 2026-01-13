* [Documentation][1]
* Python API

# Python API

The MLflow Python API is organized into the following modules. The most common functions are exposed
in the [`mlflow`][2] module, so we recommend starting there.

* [mlflow][3]
* [MLflow Tracing APIs][4]
* [MLflow Logged Model APIs][5]
* [mlflow.ag2][6]
* [mlflow.agno][7]
* [mlflow.anthropic][8]
* [mlflow.artifacts][9]
* [mlflow.autogen][10]
* [mlflow.bedrock][11]
* [mlflow.catboost][12]
* [mlflow.client][13]
* [mlflow.config][14]
* [mlflow.crewai][15]
* [mlflow.data][16]
* [mlflow.deployments][17]
* [mlflow.dspy][18]
* [mlflow.entities][19]
* [mlflow.environment_variables][20]
* [mlflow.gateway][21]
* [mlflow.gemini][22]
* [mlflow.genai][23]
* [mlflow.groq][24]
* [mlflow.h2o][25]
* [mlflow.haystack][26]
* [mlflow.johnsnowlabs][27]
* [mlflow.keras][28]
* [mlflow.langchain][29]
* [mlflow.lightgbm][30]
* [mlflow.litellm][31]
* [mlflow.llama_index][32]
* [mlflow.metrics][33]
* [mlflow.mistral][34]
* [mlflow.models][35]
* [mlflow.onnx][36]
* [mlflow.openai][37]
* [mlflow.paddle][38]
* [mlflow.pmdarima][39]
* [mlflow.projects][40]
* [mlflow.prophet][41]
* [mlflow.pydantic_ai][42]
* [mlflow.pyfunc][43]
* [mlflow.pyspark.ml][44]
* [mlflow.pytorch][45]
* [mlflow.sagemaker][46]
* [mlflow.sentence_transformers][47]
* [mlflow.server][48]
* [mlflow.shap][49]
* [mlflow.sklearn][50]
* [mlflow.smolagents][51]
* [mlflow.spacy][52]
* [mlflow.spark][53]
* [mlflow.statsmodels][54]
* [mlflow.strands][55]
* [mlflow.system_metrics][56]
* [mlflow.tensorflow][57]
* [mlflow.tracing][58]
* [mlflow.transformers][59]
* [mlflow.types][60]
* [mlflow.utils][61]
* [mlflow.webhooks][62]
* [mlflow.xgboost][63]

See also the [index of all functions and classes][64].

## Log Levels

MLflow Python APIs log information during execution using the Python Logging API. You can configure
the log level for MLflow logs using the following code snippet. Learn more about Python log levels
at the [Python language logging guide][65].

import logging

logger = logging.getLogger("mlflow")

# Set log level to debugging
logger.setLevel(logging.DEBUG)
[ Previous][66] [Next ][67]

© MLflow Project, a Series of LF Projects, LLC. All rights reserved.

[1]: ../index.html
[2]: mlflow.html#module-mlflow
[3]: mlflow.html
[4]: mlflow.html#mlflow-tracing-apis
[5]: mlflow.html#mlflow-logged-model-apis
[6]: mlflow.ag2.html
[7]: mlflow.agno.html
[8]: mlflow.anthropic.html
[9]: mlflow.artifacts.html
[10]: mlflow.autogen.html
[11]: mlflow.bedrock.html
[12]: mlflow.catboost.html
[13]: mlflow.client.html
[14]: mlflow.config.html
[15]: mlflow.crewai.html
[16]: mlflow.data.html
[17]: mlflow.deployments.html
[18]: mlflow.dspy.html
[19]: mlflow.entities.html
[20]: mlflow.environment_variables.html
[21]: mlflow.gateway.html
[22]: mlflow.gemini.html
[23]: mlflow.genai.html
[24]: mlflow.groq.html
[25]: mlflow.h2o.html
[26]: mlflow.haystack.html
[27]: mlflow.johnsnowlabs.html
[28]: mlflow.keras.html
[29]: mlflow.langchain.html
[30]: mlflow.lightgbm.html
[31]: mlflow.litellm.html
[32]: mlflow.llama_index.html
[33]: mlflow.metrics.html
[34]: mlflow.mistral.html
[35]: mlflow.models.html
[36]: mlflow.onnx.html
[37]: mlflow.openai.html
[38]: mlflow.paddle.html
[39]: mlflow.pmdarima.html
[40]: mlflow.projects.html
[41]: mlflow.prophet.html
[42]: mlflow.pydantic_ai.html
[43]: mlflow.pyfunc.html
[44]: mlflow.pyspark.ml.html
[45]: mlflow.pytorch.html
[46]: mlflow.sagemaker.html
[47]: mlflow.sentence_transformers.html
[48]: mlflow.server.html
[49]: mlflow.shap.html
[50]: mlflow.sklearn.html
[51]: mlflow.smolagents.html
[52]: mlflow.spacy.html
[53]: mlflow.spark.html
[54]: mlflow.statsmodels.html
[55]: mlflow.strands.html
[56]: mlflow.system_metrics.html
[57]: mlflow.tensorflow.html
[58]: mlflow.tracing.html
[59]: mlflow.transformers.html
[60]: mlflow.types.html
[61]: mlflow.utils.html
[62]: mlflow.webhooks.html
[63]: mlflow.xgboost.html
[64]: ../genindex.html
[65]: https://docs.python.org/3/howto/logging.html
[66]: ../index.html
[67]: mlflow.html
