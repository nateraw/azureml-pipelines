# AzureML Pipelines

Example pipelines using AzureML.

## Setup

### Authentication with AzureML

Run the following to create a service principal with contributor access to the resource group where your AzureML workspace resides (replace the values in angle brackets with your own values):

```
az ad sp create-for-rbac --name <service-principal-name> \
                         --role contributor \
                         --scopes /subscriptions/<subscription-id>/resourceGroups/<resource-group> \
                         --sdk-auth
```

**Don't lose this output, you'll need it for this next step!**

This will produce an output like this:

```
{
  "clientId": "<GUID>",
  "clientSecret": "<GUID>",
  "subscriptionId": "<GUID>",
  "tenantId": "<GUID>",
  (...)
}
```

Use the values here to set secrets in your repo. You can do this from the command line if you have [GitHub CLI](https://cli.github.com/) installed:

```
gh secret set AZURE_SUBSCRIPTION_ID
gh secret set AZURE_TENANT_ID
gh secret set AZURE_CLIENT_ID
gh secret set AZURE_CLIENT_SECRET
```

Each line above will prompt you for the value of the secret. You can also set the secrets in the browser by going to the repository settings and clicking on "Secrets" in the left-hand menu.

### Training

The github actions workflow will automatically train a model when you push to the `main` branch or make a pull request to the `main` branch.

Metrics are logged with [MLflow](https://mlflow.org/). You can view the logged metrics in the [AzureML UI](https://ml.azure.com/) by clicking on the "Experiments" tab and then clicking on the experiment you want to view. The one from the CI is called `transformers-pipeline-ci`.

### Deployment

There is a [manual workflow](https://docs.github.com/en/actions/managing-workflow-runs/manually-running-a-workflow) for deployment. In the actions tab, click on the "Deploy" workflow and click "Run workflow". You will be prompted to enter the run ID you want to deploy to.