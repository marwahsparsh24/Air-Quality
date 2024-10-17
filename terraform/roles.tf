
# Get the current Azure client configuration
data "azurerm_client_config" "current_roles" {}

# Assign Storage Blob Data Contributor to the current user on the Storage Account
resource "azurerm_role_assignment" "storage_blob_data_contributor" {
  scope                = azurerm_storage_account.mlops_storage.id
  role_definition_name = "Storage Blob Data Contributor"
  principal_id         = data.azurerm_client_config.current.object_id
}

# Assign Contributor to a service principal on the Machine Learning Workspace
resource "azurerm_role_assignment" "ml_workspace_contributor" {
  scope                = azurerm_machine_learning_workspace.mlops_ml_workspace.id
  role_definition_name = "Contributor"
  principal_id         = "var.principal_id"
}

# Assign Key Vault Secrets User to the current user on the Key Vault
resource "azurerm_role_assignment" "key_vault_secrets_user" {
  scope                = azurerm_key_vault.mlops_key_vault.id
  role_definition_name = "Key Vault Secrets User"
  principal_id         = data.azurerm_client_config.current.object_id
}
