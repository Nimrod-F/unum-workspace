import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import modules.Windowing.app as step_0
import modules.ComputeFFT.app as step_1

def lambda_handler(event, context):
    # Step 0: Windowing
    val_0 = step_0.lambda_handler(event, context)
    # Step 1: ComputeFFT
    val_1 = step_1.lambda_handler(val_0, context)
    return val_1
