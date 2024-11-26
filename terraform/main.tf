# Provider configuration
terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 4.0"
    }
  }
  required_version = ">= 1.0"
}

provider "google" {
  credentials = file("/Users/swapnavippaturi/Downloads/key.json")
  project     = var.project_id
  region      = var.region
  zone        = var.zone
}

# Firewall Rule for Airflow Webserver
resource "google_compute_firewall" "allow_airflow" {
  name    = "allow-airflow-web"
  network = "default"
  project = var.project_id

  allow {
    protocol = "tcp"
    ports    = ["8080", "5555"] # Include Flower UI if required
  }

  source_ranges = ["0.0.0.0/0"] # Restrict to your IP if needed
  target_tags   = ["airflow"]
}

# Compute Engine VM
resource "google_compute_instance" "airflow_vm" {
  name         = "airflow-docker-vm"
  machine_type = var.machine_type
  zone         = var.zone
  project      = var.project_id

  boot_disk {
    initialize_params {
      image = "projects/ubuntu-os-cloud/global/images/family/ubuntu-2204-lts"
      size  = 30
    }
  }

  network_interface {
    network = "default"
    access_config {}
  }

  metadata = {
    startup-script = <<-EOT
      #!/bin/bash
      sudo apt update && sudo apt install -y docker.io docker-compose
      sudo usermod -aG docker $USER
      sudo systemctl start docker
      sudo systemctl enable docker

      # Create directories for Airflow
      mkdir -p ~/airflow/dags ~/airflow/logs ~/airflow/plugins ~/airflow/config
      chmod -R 777 ~/airflow

      # Write Docker Compose file
      cat <<EOF > ~/airflow/docker-compose.yaml
      ${file("./docker-compose.yaml")}
      EOF

      # Start Airflow services
      cd ~/airflow
      docker-compose up -d
    EOT
  }

  tags = ["airflow"]

  service_account {
    email = "airquality@airquality-438719.iam.gserviceaccount.com"
    scopes = ["https://www.googleapis.com/auth/cloud-platform"]
  }
}
