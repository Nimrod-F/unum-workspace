# Unum Setup and Configuration Guide

## Current Status ✅

Based on the environment check, here's what's already configured:

- ✅ **Python 3.13.3** installed (via `py` launcher)
- ✅ **unum-cli dependencies** installed (cfn-flip, coloredlogs, PyGithub)
- ✅ **AWS CLI 2.17.61** installed
- ✅ **AWS credentials** configured (region: eu-central-1)
- ❌ **AWS SAM CLI** NOT installed (required for deployment)

## Prerequisites Installation

### 1. Install AWS SAM CLI (REQUIRED)

AWS SAM CLI is needed to build and deploy Unum applications.

**Windows Installation:**

```powershell
# Download the MSI installer from:
# https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/install-sam-cli.html

# Or using Chocolatey:
choco install aws-sam-cli

# Or using Winget:
winget install Amazon.SAM-CLI
```

**Verify installation:**

```powershell
sam --version
```

### 2. Install/Configure Python Dependencies

The unum-cli dependencies are already installed, but for completeness:

```powershell
cd c:\Users\NimrodFoldvari\Downloads\DagL\unum_project\unum\unum-cli
py -m pip install -r requirements.txt
```

### 3. Configure AWS Credentials (Already Done ✅)

Your AWS credentials are configured in `~/.aws/credentials`. No action needed.

To view configuration:

```powershell
aws configure list
```

## Setting Up unum-cli

### Option 1: Direct Execution (Recommended for Testing)

```powershell
cd c:\Users\NimrodFoldvari\Downloads\DagL\unum_project\unum\unum-cli
py unum-cli.py --help
```

### Option 2: Add to PATH (Recommended for Regular Use)

Create a wrapper script in a directory that's in your PATH:

**Create `unum-cli.bat`:**

```batch
@echo off
py "c:\Users\NimrodFoldvari\Downloads\DagL\unum_project\unum\unum-cli\unum-cli.py" %*
```

**Or add the unum-cli directory to PATH:**

```powershell
# Temporary (current session only):
$env:Path += ";c:\Users\NimrodFoldvari\Downloads\DagL\unum_project\unum\unum-cli"

# Permanent (add to system/user PATH via Control Panel or):
[Environment]::SetEnvironmentVariable(
    "Path",
    $env:Path + ";c:\Users\NimrodFoldvari\Downloads\DagL\unum_project\unum\unum-cli",
    [EnvironmentVariableTarget]::User
)
```

### Option 3: Install as Python Package

```powershell
cd c:\Users\NimrodFoldvari\Downloads\DagL\unum_project\unum
py -m pip install -e .
```

## Building and Deploying Unum Applications

### Example: Deploy Hello World Application

```powershell
# Navigate to an example application
cd c:\Users\NimrodFoldvari\Downloads\DagL\unum_project\unum-appstore\hello-world

# First, compile the workflow (if needed)
py c:\Users\NimrodFoldvari\Downloads\DagL\unum_project\unum\unum-cli\unum-cli.py compile -p step-functions -w unum-step-functions.json -t unum-template.yaml

# Build with automatic template generation
py c:\Users\NimrodFoldvari\Downloads\DagL\unum_project\unum\unum-cli\unum-cli.py build -g -p aws

# Deploy
py c:\Users\NimrodFoldvari\Downloads\DagL\unum_project\unum\unum-cli\unum-cli.py deploy -b
```

### Step-by-Step Build and Deploy

1. **Compile workflow definition to unum IR (first time only):**

```powershell
py c:\Users\NimrodFoldvari\Downloads\DagL\unum_project\unum\unum-cli\unum-cli.py compile -p step-functions -w unum-step-functions.json -t unum-template.yaml
```

2. **Generate AWS SAM template from Unum template (optional):**

```powershell
py c:\Users\NimrodFoldvari\Downloads\DagL\unum_project\unum\unum-cli\unum-cli.py template -p aws
```

3. **Build the application (with auto-generate template):**

```powershell
py c:\Users\NimrodFoldvari\Downloads\DagL\unum_project\unum\unum-cli\unum-cli.py build -g -p aws
```

4. **Deploy to AWS:**

```powershell
py c:\Users\NimrodFoldvari\Downloads\DagL\unum_project\unum\unum-cli\unum-cli.py deploy
```

### Combined Build and Deploy:

```powershell
py c:\Users\NimrodFoldvari\Downloads\DagL\unum_project\unum\unum-cli\unum-cli.py deploy -b
```

## Setting Up a New Unum Application

### Application Structure

```
myapp/
 ├── unum-template.yaml          # Application configuration
 ├── unum-step-functions.json    # Workflow definition
 ├── function1/
 │   ├── app.py                  # User function code
 │   └── requirements.txt        # Python dependencies
 └── function2/
     ├── app.py
     └── requirements.txt
```

### Minimal unum-template.yaml

```yaml
Globals:
  ApplicationName: my-app
  WorkflowType: step-functions
  WorkflowDefinition: unum-step-functions.json
  FaaSPlatform: aws
  UnumIntermediaryDataStoreType: dynamodb
  UnumIntermediaryDataStoreName: unum-intermediate-datastore
  Checkpoint: true
  GC: true
  Debug: false

Functions:
  Function1:
    Properties:
      CodeUri: function1/
      Runtime: python3.10
      Start: true

  Function2:
    Properties:
      CodeUri: function2/
      Runtime: python3.10
```

### Minimal unum-step-functions.json

```json
{
  "Comment": "A simple workflow",
  "StartAt": "Function1",
  "States": {
    "Function1": {
      "Type": "Task",
      "Resource": "Function1",
      "Next": "Function2"
    },
    "Function2": {
      "Type": "Task",
      "Resource": "Function2",
      "End": true
    }
  }
}
```

### Function app.py Template

```python
def lambda_handler(event, context):
    # Your function logic here
    print(f"Input: {event}")

    # Process data
    result = {"status": "success", "data": event}

    return result
```

## Enabling Future-based Partial Application

To enable the new future-based lazy evaluation feature:

### 1. Enable in Global Configuration (unum-template.yaml)

```yaml
Globals:
  # ... other settings ...
  EnableFutures: true
  FutureInvocationStrategy: "first_ready" # Options: first_ready, threshold, all
  FutureTimeout: 30
```

### 2. Configure Per-Function (unum_config.json)

For fan-in functions, add configuration:

```json
{
  "Name": "aggregate_function",
  "Next": {
    "Name": "next_function",
    "InputType": "Scalar"
  },
  "Checkpoint": true,
  "EnableFutures": true,
  "FutureParams": {
    "param_a": {
      "priority": 1,
      "timeout": 5.0,
      "required": true
    },
    "param_b": {
      "priority": 2,
      "timeout": 10.0,
      "required": true
    }
  }
}
```

### 3. Use the Decorator in Your Function

```python
from function_transformer import unum_lazy_eval

@unum_lazy_eval
def lambda_handler(sensor_a, sensor_b, sensor_c):
    # Function can start as soon as ANY sensor is ready
    # Will automatically await when accessing parameters
    result = sensor_a + sensor_b + sensor_c
    return result
```

## AWS Resources Created by Unum

When you deploy a Unum application, the following AWS resources are created:

1. **Lambda Functions** - One for each function in your application
2. **DynamoDB Table** - For storing intermediate data (checkpoints, synchronization)
3. **IAM Roles** - For Lambda execution permissions
4. **CloudFormation Stack** - Manages all the resources

### DynamoDB Table Setup

The table name is specified in `unum-template.yaml`:

```yaml
UnumIntermediaryDataStoreName: unum-intermediate-datastore
```

**Note:** You may need to create this DynamoDB table manually if it doesn't exist:

```powershell
aws dynamodb create-table `
    --table-name unum-intermediate-datastore `
    --attribute-definitions AttributeName=Name,AttributeType=S `
    --key-schema AttributeName=Name,KeyType=HASH `
    --billing-mode PAY_PER_REQUEST `
    --region eu-central-1
```

## Testing Your Deployment

### 1. Invoke via AWS CLI

```powershell
# Create a test event
$testEvent = @'
{
  "Data": {
    "Source": "http",
    "Value": {
      "test": "data"
    }
  }
}
'@

# Invoke the function
aws lambda invoke `
    --function-name my-app-Function1 `
    --payload $testEvent `
    --region eu-central-1 `
    output.json

# View the output
Get-Content output.json
```

### 2. View Logs

```powershell
# Get log streams
aws logs describe-log-streams `
    --log-group-name /aws/lambda/my-app-Function1 `
    --region eu-central-1 `
    --order-by LastEventTime `
    --descending `
    --max-items 1

# View recent logs
aws logs tail /aws/lambda/my-app-Function1 --follow --region eu-central-1
```

## Troubleshooting

### Issue: SAM CLI not found

**Solution:** Install AWS SAM CLI (see Prerequisites section)

### Issue: "No changes to deploy"

**Solution:** Run build before deploy:

```powershell
py unum-cli.py deploy -b
```

### Issue: DynamoDB table doesn't exist

**Solution:** Create the table:

```powershell
aws dynamodb create-table `
    --table-name unum-intermediate-datastore `
    --attribute-definitions AttributeName=Name,AttributeType=S `
    --key-schema AttributeName=Name,KeyType=HASH `
    --billing-mode PAY_PER_REQUEST
```

### Issue: Permission denied errors

**Solution:** Ensure your AWS credentials have sufficient permissions for:

- Lambda (create, update, invoke functions)
- CloudFormation (create, update stacks)
- DynamoDB (create tables, read/write items)
- IAM (create roles and policies)
- S3 (for SAM deployment artifacts)

### Issue: Python import errors in runtime

**Solution:** Ensure all dependencies are in `requirements.txt` for each function

## Next Steps

1. ✅ Install AWS SAM CLI
2. ✅ Verify installation: `sam --version`
3. ✅ Create or navigate to a Unum application directory
4. ✅ Build: `py unum-cli.py build -t -w unum-step-functions.json -p aws`
5. ✅ Deploy: `py unum-cli.py deploy -b`
6. ✅ Test your deployed application
7. ✅ Monitor logs and metrics in AWS Console

## Additional Resources

- [Unum Documentation](https://github.com/LedgeDash/unum/tree/main/docs)
- [AWS SAM Documentation](https://docs.aws.amazon.com/serverless-application-model/)
- [AWS Lambda Documentation](https://docs.aws.amazon.com/lambda/)
- [Unum Application Examples](https://github.com/LedgeDash/unum-appstore)

## Quick Reference Commands

```powershell
# Compile workflow definition (first time or when workflow changes)
py unum-cli.py compile -p step-functions -w unum-step-functions.json -t unum-template.yaml

# Build with automatic template generation
py unum-cli.py build -g -p aws

# Build only (assumes template.yaml exists)
py unum-cli.py build

# Generate SAM template only
py unum-cli.py template -p aws

# Deploy (with build)
py unum-cli.py deploy -b

# Deploy (without build)
py unum-cli.py deploy

# Clean build artifacts
py unum-cli.py build -c
py unum-cli.py template -c

# View help
py unum-cli.py --help
py unum-cli.py build --help
py unum-cli.py deploy --help
py unum-cli.py compile --help
```
