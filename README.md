# AzureML Pipelines

Example pipelines using AzureML SDKv1. WIP.


## Setup

:pencil2: TODO: Add instructions for setting up a service principal with contributor access to the resource group, github actions setup, etc...

---

# ! ! Ignore the below ! !

### Authentication with AzureML

Run the following to create a service principal with contributor access to the resource group (replace the values in angle brackets with your own values):

```
az ad sp create-for-rbac --name <service-principal-name> \
                         --role contributor \
                         --scopes /subscriptions/<subscription-id>/resourceGroups/<resource-group> \
                         --sdk-auth
```

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

Add this JSON output as a secret with the name AZURE_CREDENTIALS in your GitHub repository.