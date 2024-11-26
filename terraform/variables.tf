variable "project_id" {
  description = "The GCP project ID where the resources will be created"
  type        = string
}

variable "region" {
  description = "The GCP region"
  default     = "us-central1"
}

variable "zone" {
  description = "The GCP zone"
  default     = "us-central1-a"
}

variable "machine_type" {
  description = "The machine type for the VM"
  default     = "e2-medium"
}
