import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import modules.D.app as step_0
import modules.E.app as step_1

def lambda_handler(event, context):
    # Step 0: D
    val_0 = step_0.lambda_handler(event, context)
    # Step 1: E
    val_1 = step_1.lambda_handler(val_0, context)
    return val_1
