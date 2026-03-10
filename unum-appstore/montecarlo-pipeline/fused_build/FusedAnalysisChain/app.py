import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import modules.Transform.app as step_0
import modules.Estimate.app as step_1
import modules.Validate.app as step_2

def lambda_handler(event, context):
    # Step 0: Transform
    val_0 = step_0.lambda_handler(event, context)
    # Step 1: Estimate
    val_1 = step_1.lambda_handler(val_0, context)
    # Step 2: Validate
    val_2 = step_2.lambda_handler(val_1, context)
    return val_2
