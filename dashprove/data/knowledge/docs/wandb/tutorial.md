Install W&B to track, visualize, and manage machine learning experiments of any size.
Are you looking for information on W&B Weave? See the [Weave Python SDK quickstart][1] or [Weave
TypeScript SDK quickstart][2].

## [​][3]
## Sign up and create an API key

To authenticate your machine with W&B, generate an API key from your user profile or at
[wandb.ai/authorize][4]. Copy the API key and store it securely.

## [​][5]
## Install the `wandb` library and log in

* Command Line
* Python
* Python notebook

1. Set the `WANDB_API_KEY` [environment variable][6].
   Report incorrect code
   Copy
   Ask AI
   `export WANDB_API_KEY=<your_api_key>
   `
2. Install the `wandb` library and log in.
   Report incorrect code
   Copy
   Ask AI
   `pip install wandb
   wandb login
   `
Report incorrect code
Copy
Ask AI
`pip install wandb
`
Report incorrect code
Copy
Ask AI
`import wandb

wandb.login()
`
Report incorrect code
Copy
Ask AI
`!pip install wandb
import wandb
wandb.login()
`

## [​][7]
## Initialize a run and track hyperparameters

In your Python script or notebook, initialize a W&B run object with [`wandb.init()`][8]. Use a
dictionary for the `config` parameter to specify hyperparameter names and values. Within the `with`
statement, you can log metrics and other information to W&B.
Report incorrect code
Copy
Ask AI
`import wandb

wandb.login()

# Project that the run is recorded to
project = "my-awesome-project"

# Dictionary with hyperparameters
config = {
    'epochs' : 10,
    'lr' : 0.01
}

with wandb.init(project=project, config=config) as run:
    # Training code here
    # Log values to W&B with run.log()
    run.log({"accuracy": 0.9, "loss": 0.1})
`
See the next section for a complete example that simulates a training run and logs accuracy and loss
metrics to W&B.
A [run][9] is a core element of W&B. You use runs to [track metrics][10], [create logs][11], track
artifacts, and more.

## [​][12]
## Create a machine learning training experiment

This mock training script logs simulated accuracy and loss metrics to W&B. Copy and paste the
following code into a Python script or notebook cell and run it:
Report incorrect code
Copy
Ask AI
`import wandb
import random

wandb.login()

# Project that the run is recorded to
project = "my-awesome-project"

# Dictionary with hyperparameters
config = {
    'epochs' : 10,
    'lr' : 0.01
}

with wandb.init(project=project, config=config) as run:
    offset = random.random() / 5
    print(f"lr: {config['lr']}")
    
    # Simulate a training run
    for epoch in range(2, config['epochs']):
        acc = 1 - 2**-config['epochs'] - random.random() / config['epochs'] - offset
        loss = 2**-config['epochs'] + random.random() / config['epochs'] + offset
        print(f"epoch={config['epochs']}, accuracy={acc}, loss={loss}")
        run.log({"accuracy": acc, "loss": loss})
`
Visit [wandb.ai/home][13] to view recorded metrics such as accuracy and loss and how they changed
during each training step. The following image shows the loss and accuracy tracked from each run.
Each run object appears in the **Runs** column with generated names.
[Shows loss and accuracy tracked from each run.]

## [​][14]
## Next steps

Explore more features of the W&B ecosystem:

1. Read the [W&B Integration tutorials][15] that combine W&B with frameworks like PyTorch, libraries
   like Hugging Face, and services like SageMaker.
2. Organize runs, automate visualizations, summarize findings, and share updates with collaborators
   using [W&B Reports][16].
3. Create [W&B Artifacts][17] to track datasets, models, dependencies, and results throughout your
   machine learning pipeline.
4. Automate hyperparameter searches and optimize models with [W&B Sweeps][18].
5. Analyze runs, visualize model predictions, and share insights on a [central dashboard][19].
6. Visit [W&B AI Academy][20] to learn about LLMs, MLOps, and W&B Models through hands-on courses.
7. Visit [weave-docs.wandb.ai][21] to learn how to track track, experiment with, evaluate, deploy,
   and improve your LLM-based applications using Weave.

[1]: /weave/quickstart
[2]: /weave/reference/generated_typescript_docs/intro-notebook
[3]: #sign-up-and-create-an-api-key
[4]: https://wandb.ai/authorize
[5]: #install-the-wandb-library-and-log-in
[6]: /models/track/environment-variables
[7]: #initialize-a-run-and-track-hyperparameters
[8]: /models/ref/python/experiments/run
[9]: /models/runs
[10]: /models/track
[11]: /models/track/log
[12]: #create-a-machine-learning-training-experiment
[13]: https://wandb.ai/home
[14]: #next-steps
[15]: /models/integrations
[16]: /models/reports
[17]: /models/artifacts
[18]: /models/sweeps
[19]: /models/tables
[20]: https://wandb.ai/site/courses/
[21]: /weave
