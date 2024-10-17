# variables.tf

variable "subscription_id" {
  description = "The subscription ID for the Azure account"
  type        = string
  default     = "var.subscription_id"
}

variable "resource_group_name" {
  description = "Name of the resource group"
  type        = string
  default     = "mlops-airquality-rg"
}

variable "location" {
  description = "Azure region where the resources will be created"
  type        = string
  default     = "East US"
}

variable "storage_account_name" {
  description = "Name of the storage account"
  type        = string
  default     = "airquality"
}

variable "databricks_workspace_name" {
  description = "Name of the Databricks workspace"
  type        = string
  default     = "mlops-airquality-databricks"
}

variable "ml_workspace_name" {
  description = "Name of the Azure Machine Learning workspace"
  type        = string
  default     = "mlops-airquality-workspace"
}

variable "principal_id" {
  type = string
}
