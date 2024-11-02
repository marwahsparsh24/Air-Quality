# Provider block for Google Cloud
provider "google" {
  credentials = file("/Users/swapnavippaturi/Downloads/key.json")
  project     = var.project_id  # GCP Project ID
  region      = var.region      # GCP Region
}

# Variables for project and environment configuration
variable "project_id" {
  description = "The GCP project ID"
  type        = string
}

variable "region" {
  description = "The GCP region to deploy resources"
  type        = string
  default     = "us-central1"
}

variable "composer_env_name" {
  description = "The name of the Cloud Composer environment"
  type        = string
}

variable "bucket_name" {
  description = "The name of the Cloud Storage bucket for Composer DAGs"
  type        = string
}

variable "serviceAccount" {
  description = "The service account email"
  type        = string
}

# Create a random ID for uniqueness in bucket name
resource "random_id" "bucket_suffix" {
  byte_length = 4
}

# Create a Cloud Storage bucket for Composer DAGs
resource "google_storage_bucket" "airbucketname" {
  name           = "${var.bucket_name}-${random_id.bucket_suffix.hex}"  # Ensure globally unique name
  location       = var.region
  force_destroy  = true  # Automatically delete bucket contents when destroying
}

# Create IAM service account for Composer
resource "google_service_account" "composer_service_account" {
  account_id   = "composer-service-account"
  display_name = "Composer Service Account"
}

# Grant necessary roles to the Composer service account
resource "google_project_iam_member" "composer_service_account_roles" {
  for_each = toset([
    "roles/composer.admin",      # Admin for Cloud Composer
    "roles/storage.admin",       # Admin for Cloud Storage
    "roles/iam.serviceAccountUser",  # IAM Service Account User role
    "roles/logging.logWriter",   # Logs access
    "roles/monitoring.viewer"    # Monitoring access
  ])

  project = var.project_id
  member  = "serviceAccount:${google_service_account.composer_service_account.email}"
  role    = each.value
}

resource "google_composer_environment" "airquality-composer" {
  name   = "airquality-composer"
  region = "us-central1"

  config {
    software_config {
      image_version = "composer-2.9.7-airflow-2.9.3"  # Use a valid image version
      pypi_packages = {
      "gs://airquality-mlops-rg/composer_requirements/requirements.txt" = ""
      } 
    }
       
  }
}
# Optional: Set up the network and subnetwork (if required)
resource "google_compute_network" "composer_network" {
  name                    = "composer-network"
  auto_create_subnetworks = false
}

resource "google_compute_subnetwork" "composer_subnetwork" {
  name          = "composer-subnetwork"
  network       = google_compute_network.composer_network.name
  ip_cidr_range = "10.0.0.0/24"
  region        = var.region
}
