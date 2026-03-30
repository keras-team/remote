# Troubleshooting

## Common Issues

### "Project must be specified" error

```bash
export KINETIC_PROJECT="your-project-id"
```

### "404 Requested entity was not found" error

Enable required APIs and create the Artifact Registry repository:

```bash
gcloud services enable compute.googleapis.com \
    cloudbuild.googleapis.com artifactregistry.googleapis.com \
    storage.googleapis.com container.googleapis.com \
    --project=$KINETIC_PROJECT

gcloud artifacts repositories create kinetic \
    --repository-format=docker \
    --location=us \
    --project=$KINETIC_PROJECT
```

### Permission denied errors

Grant required IAM roles:

```bash
gcloud projects add-iam-policy-binding $KINETIC_PROJECT \
    --member="user:your-email@example.com" \
    --role="roles/storage.admin"
```

### Container build failures

Check Cloud Build logs:

```bash
gcloud builds list --project=$KINETIC_PROJECT --limit=5
```

## Automated Diagnostics

If you're encountering issues and aren't sure where to start, use the `kinetic doctor` command. It runs a comprehensive suite of health checks on your local environment and cloud infrastructure:

```bash
kinetic doctor
```

The doctor checks:
- **Local Tools**: Ensures `gcloud` and `kubectl` are installed and on your PATH.
- **Authentication**: Verifies active GCP accounts and Application Default Credentials (ADC).
- **Project Access**: Confirms your GCP project exists and has billing enabled.
- **APIs**: Checks that all required GCP services (GKE, Cloud Build, Artifact Registry) are enabled.
- **Infrastructure**: Validates that your GKE cluster is running and your node pools are healthy.
- **Kubernetes**: Verifies your `kubeconfig` context and cluster connectivity.

If any checks fail, the command will provide specific **fix suggestions** to help you resolve the issue.

## Verify Setup

Run `kinetic status` to check the current state of your provisioned infrastructure. For manual verification:

```bash
# Check authentication
gcloud auth list

# Check project
echo $KINETIC_PROJECT

# Check APIs
gcloud services list --enabled --project=$KINETIC_PROJECT \
    | grep -E "(cloudbuild|artifactregistry|storage|container)"
```

