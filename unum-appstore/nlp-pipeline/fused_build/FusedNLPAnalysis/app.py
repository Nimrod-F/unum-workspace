import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import modules.Analyzer.app as step_0
import modules.Classifier.app as step_1
import modules.Summarizer.app as step_2

def lambda_handler(event, context):
    # Step 0: Analyzer
    val_0 = step_0.lambda_handler(event, context)
    # Step 1: Classifier
    val_1 = step_1.lambda_handler(val_0, context)
    # Step 2: Summarizer
    val_2 = step_2.lambda_handler(val_1, context)
    return val_2
