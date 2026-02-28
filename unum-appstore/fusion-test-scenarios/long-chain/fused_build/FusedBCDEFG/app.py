import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import modules.B.app as step_0
import modules.C.app as step_1
import modules.D.app as step_2
import modules.E.app as step_3
import modules.F.app as step_4
import modules.G.app as step_5

def lambda_handler(event, context):
    # Step 0: B
    val_0 = step_0.lambda_handler(event, context)
    # Step 1: C
    val_1 = step_1.lambda_handler(val_0, context)
    # Step 2: D
    val_2 = step_2.lambda_handler(val_1, context)
    # Step 3: E
    val_3 = step_3.lambda_handler(val_2, context)
    # Step 4: F
    val_4 = step_4.lambda_handler(val_3, context)
    # Step 5: G
    val_5 = step_5.lambda_handler(val_4, context)
    return val_5
