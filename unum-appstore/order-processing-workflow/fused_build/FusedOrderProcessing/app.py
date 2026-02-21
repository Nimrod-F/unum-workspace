import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import modules.SlowChainStart.app as step_0
import modules.SlowChainMid.app as step_1
import modules.SlowChainEnd.app as step_2

def lambda_handler(event, context):
    # Step 0: SlowChainStart
    val_0 = step_0.lambda_handler(event, context)
    # Step 1: SlowChainMid
    val_1 = step_1.lambda_handler(val_0, context)
    # Step 2: SlowChainEnd
    val_2 = step_2.lambda_handler(val_1, context)
    return val_2
