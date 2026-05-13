# Standalone Greeter Function

This is an **individual Lambda function** deployed independently of any unum workflow.

## Deploy

```bash
cd standalone-greeter
sam build
sam deploy --guided --stack-name standalone-greeter
```

After deployment, note the output ARN:
```
Key: GreeterFunctionArn
Value: arn:aws:lambda:<region>:<account-id>:function:standalone-greeter-GreeterFunction-XXXX
```

## Use in a workflow

Once deployed, use the ARN in a DAGL workflow with `@import`:

```
@import("Greeter", "arn:aws:lambda:<region>:<account-id>:function:standalone-greeter-GreeterFunction-XXXX")
```

The unum compiler will generate a proxy Lambda that calls this function —
the original Greeter function is **never modified**.
