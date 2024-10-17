# main.tf

# Configure the Azure provider
provider "azurerm" {
  features {}
   subscription_id = var.subscription_id
}

# Get the current Azure client configuration
data "azurerm_client_config" "current" {}

# Create a Resource Group
resource "azurerm_resource_group" "mlops_rg" {
  name     = var.resource_group_name
  location = var.location
}

# Create a Storage Account
resource "azurerm_storage_account" "mlops_storage" {
  name                     = var.storage_account_name
  resource_group_name      = azurerm_resource_group.mlops_rg.name
  location                 = azurerm_resource_group.mlops_rg.location
  account_tier             = "Standard"
  account_replication_type = "LRS"
}

# Create a Key Vault
resource "azurerm_key_vault" "mlops_key_vault" {
  name                = "mlops-pricekey"
  location            = azurerm_resource_group.mlops_rg.location
  resource_group_name = azurerm_resource_group.mlops_rg.name
  sku_name            = "standard"
  tenant_id           = data.azurerm_client_config.current.tenant_id
}

# Create an Application Insights instance
resource "azurerm_application_insights" "mlops_app_insights" {
  name                = "mlops-appinsights"
  location            = azurerm_resource_group.mlops_rg.location
  resource_group_name = azurerm_resource_group.mlops_rg.name
  application_type    = "web"
}

# Create an Azure Machine Learning Workspace
resource "azurerm_machine_learning_workspace" "mlops_ml_workspace" {
  name                = var.ml_workspace_name
  location            = azurerm_resource_group.mlops_rg.location
  resource_group_name = azurerm_resource_group.mlops_rg.name
  key_vault_id        = azurerm_key_vault.mlops_key_vault.id
  application_insights_id = azurerm_application_insights.mlops_app_insights.id
  storage_account_id  = azurerm_storage_account.mlops_storage.id

  identity {
    type = "SystemAssigned"
  }
}

# Create a Databricks Workspace
resource "azurerm_databricks_workspace" "mlops_databricks" {
  name                = var.databricks_workspace_name
  resource_group_name = azurerm_resource_group.mlops_rg.name
  location            = azurerm_resource_group.mlops_rg.location
  sku                 = "standard"
}
