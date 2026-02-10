#!/usr/bin/env python
import json, os, sys, subprocess, time
import argparse
import shutil
import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

from cfn_tools import load_yaml, dump_yaml


def generate_sam_template(unum_template):
    ''' Given an unum template, return an AWS SAM template as a python dict

        @param unum_template python dict

        @return sam_template python dict
    '''

    # boilerplate SAM template fields
    sam_template = {"AWSTemplateFormatVersion": '2010-09-09',
                    "Transform": "AWS::Serverless-2016-10-31"}

    # save workflow-wide configurations as environment variables.
    # Globals:
    #   Function:
    #       Environment:
    #           Variables:
    # These variables will be accessible by Lambda code as environment variables.
    sam_template["Globals"] = {
            "Function": {
                "Environment": {
                    "Variables":{
                        "UNUM_INTERMEDIARY_DATASTORE_TYPE": unum_template["Globals"]["UnumIntermediaryDataStoreType"],
                        "UNUM_INTERMEDIARY_DATASTORE_NAME": unum_template["Globals"]["UnumIntermediaryDataStoreName"],
                        "FAAS_PLATFORM": unum_template["Globals"]["FaaSPlatform"],
                        "CHECKPOINT":unum_template["Globals"]["Checkpoint"],
                        "GC":unum_template["Globals"]["GC"],
                        "EAGER": unum_template["Globals"].get("Eager", False)
                    }
                }
            }
        }
    # Set all Lambda timeouts to 900 sec
    sam_template["Globals"]["Function"]["Timeout"] = 900

    # Copy other global settings from unum-template to sam template
    if "MemorySize" in unum_template["Globals"]:
        sam_template["Globals"]["Function"]["MemorySize"] = unum_template["Globals"]["MemorySize"]

    # For each unum function, create a AWS::Serverless::Function resource in
    # the SAM template under the "Resources" field.
    # All unum functions "Handler" is wrapper.lambda_handler
    # Copy over "CodeUri", "Runtime"
    # Add 
    #   + "AmazonDynamoDBFullAccess"
    #   + "AmazonS3FullAccess"
    #   + "AWSLambdaRole"
    #   + "AWSLambdaBasicExecutionRole"
    # if any is not listed already in the unum template
    unum_function_needed_policies = ["AmazonDynamoDBFullAccess","AmazonS3FullAccess","AWSLambdaRole","AWSLambdaBasicExecutionRole"]
    sam_template["Resources"]={}
    sam_template["Outputs"] = {}

    for f in unum_template["Functions"]:
        unum_function_policies = []
        if "Policies" in unum_template["Functions"][f]["Properties"]:
            unum_function_policies = unum_template["Functions"][f]["Properties"]["Policies"]

        sam_resource = {
                "Type":"AWS::Serverless::Function",
                "Properties": {
                    "Handler":"main.lambda_handler",
                    "Runtime": unum_template["Functions"][f]["Properties"]["Runtime"],
                    "CodeUri": unum_template["Functions"][f]["Properties"]["CodeUri"],
                    "Policies": list(set(unum_function_needed_policies) | set(unum_function_policies))
                }
            }
        
        # Add function-level environment variables if specified
        if "Environment" in unum_template["Functions"][f]["Properties"]:
            sam_resource["Properties"]["Environment"] = {
                "Variables": unum_template["Functions"][f]["Properties"]["Environment"]
            }
        
        sam_template["Resources"][f'{f}Function'] = sam_resource

        # Add command to acquired the deployed Lambda's ARN to the "Outputs"
        # fields of the SAM template
        arn = f"!GetAtt {f}Function.Arn"
        sam_template["Outputs"][f'{f}Function'] = {"Value": f"!GetAtt {f}Function.Arn"}

    return sam_template

def sam_build_clean(args):

    if args.platform_template == None:
        # default AWS SAM template filename to template.yaml
        args.platform_template = 'template.yaml'

    try:
        with open(args.platform_template) as f:
            platform_template = load_yaml(f.read())
    except Exception as e:
        print(f'\033[31m\n Build Clean Failed!\n\n Make sure a platform template file exists\033[0m')
        raise e

    # remove unum runtime files from each function's directory
    runtime_file_basename = os.listdir("common")
    for f in platform_template["Resources"]:
        app_dir = platform_template["Resources"][f]["Properties"]["CodeUri"]
        runtime_files = [app_dir+e for e in runtime_file_basename]
        try:
            subprocess.run(['rm', '-f']+runtime_files, check=True)
        except Exception as e:
            raise e

    # remove the .aws-sam build directory
    try:
        ret = subprocess.run(["rm", "-rf", ".aws-sam"], check = True, capture_output=True)
    except Exception as e:
        raise e

    return

def populate_common_directory():
    """Copy Unum runtime files to the common directory, always overwriting existing files"""
    # Find the unum runtime directory relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    runtime_dir = os.path.join(script_dir, '..', 'runtime')
    runtime_dir = os.path.abspath(runtime_dir)
    
    # Create common directory if it doesn't exist
    common_dir = 'common'
    if not os.path.exists(common_dir):
        os.makedirs(common_dir)
    
    # List of runtime files to copy
    # unum_streaming.py provides partial parameter streaming support
    runtime_files = ['unum.py', 'ds.py', 'main.py', 'faas_invoke_backend.py', 'unum_streaming.py']
    
    # Copy runtime files to common directory (always overwrite to ensure latest version)
    for filename in runtime_files:
        src = os.path.join(runtime_dir, filename)
        dst = os.path.join(common_dir, filename)
        
        if os.path.exists(src):
            print(f'Copying {filename} to common directory')
            shutil.copy2(src, dst)
        else:
            print(f'\033[33mWarning: Runtime file {filename} not found at {src}\033[0m')


def strip_streaming_code(source):
    """
    Strip all injected streaming code from a transformed app.py source.
    
    Removes:
    - from unum_streaming import ... lines
    - # Streaming: ... comment lines
    - _streaming_session = ... lines
    - _streaming_publisher = StreamingPublisher(...) multi-line blocks
    - _streaming_publisher.publish(...) calls
    - if _streaming_publisher.should_invoke_next(): ... blocks
    
    Returns the clean source string.
    """
    import re
    lines = source.split('\n')
    result = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # 1. Skip import line
        if stripped.startswith('from unum_streaming import'):
            i += 1
            continue

        # 2. Skip # Streaming: comment lines
        if stripped.startswith('# Streaming:'):
            i += 1
            continue

        # 3. Skip _streaming_session = ...
        if re.match(r'\s*_streaming_session\s*=', line):
            i += 1
            continue

        # 4. Skip _streaming_publisher = StreamingPublisher(...) block (may span multiple lines)
        if re.match(r'\s*_streaming_publisher\s*=\s*StreamingPublisher\(', line):
            # Skip until the closing paren
            while i < len(lines):
                if ')' in lines[i]:
                    i += 1
                    break
                i += 1
            continue

        # 5. Skip _streaming_publisher.publish(...)
        if re.match(r'\s*_streaming_publisher\.publish\(', line):
            i += 1
            continue

        # 6. Skip early-invoke block:
        #    if _streaming_publisher.should_invoke_next():
        #        _streaming_payload = ...
        #        # optional comment
        #        set_streaming_output(...)
        #        _streaming_publisher.mark_next_invoked()
        if '_streaming_publisher.should_invoke_next()' in stripped:
            i += 1  # skip the if line
            # Skip indented body lines
            while i < len(lines):
                inner = lines[i].strip()
                if inner.startswith('_streaming_payload') or \
                   inner.startswith('set_streaming_output') or \
                   inner.startswith('_streaming_publisher.mark_next_invoked') or \
                   inner.startswith('# Store payload') or \
                   inner.startswith('# Signal') or \
                   inner == '':
                    i += 1
                    # Stop after blank line (block separator)
                    if inner == '':
                        break
                else:
                    break
            continue

        # 7. Skip _streaming_payload = ... (standalone, outside if block)
        if re.match(r'\s*_streaming_payload\s*=', line):
            i += 1
            continue

        # 8. Skip standalone set_streaming_output(...)
        if re.match(r'\s*set_streaming_output\(', line):
            i += 1
            continue

        # 9. Skip standalone _streaming_publisher.mark_next_invoked()
        if re.match(r'\s*_streaming_publisher\.mark_next_invoked\(', line):
            i += 1
            continue

        result.append(line)
        i += 1

    return '\n'.join(result)


def apply_streaming_transform(platform_template):
    """
    Apply AST transformation for Partial Parameter Streaming.
    
    This analyzes each function's app.py to find return value construction,
    and injects code to:
    1. Publish each field to datastore as soon as it's computed
    2. Invoke next function early with futures for pending fields
    3. Allow receiver to resolve futures on-demand
    
    If a source file was already manually edited with streaming code,
    an .original backup is created by stripping the streaming code.
    """
    try:
        from streaming_transformer import StreamingAnalyzer, StreamingTransformer
    except ImportError:
        print('\033[33mWarning: streaming_transformer not found, skipping streaming optimization\033[0m')
        return
    
    print('\n\033[36mApplying Partial Parameter Streaming...\033[0m')
    
    for func_name in platform_template["Resources"]:
        resource = platform_template["Resources"][func_name]
        if resource.get("Type") != "AWS::Lambda::Function" and resource.get("Type") != "AWS::Serverless::Function":
            continue
        
        app_dir = resource["Properties"]["CodeUri"]
        app_path = os.path.join(app_dir, 'app.py')
        
        if not os.path.exists(app_path):
            print(f'  [{func_name}] No app.py found, skipping')
            continue
        
        # Check if this function has a Next - if not, skip streaming (it's the last function)
        unum_config_path = os.path.join(app_dir, 'unum_config.json')
        if os.path.exists(unum_config_path):
            with open(unum_config_path, 'r') as f:
                unum_config = json.load(f)
            if 'Next' not in unum_config or unum_config.get('Next') is None:
                print(f'  [{func_name}] Last function (no Next), skipping streaming')
                continue
        
        # Analyze the file
        try:
            with open(app_path, 'r', encoding='utf-8') as f:
                source = f.read()
            
            backup_path = os.path.join(app_dir, 'app.py.original')
            
            # Check if already transformed (has streaming imports)
            if 'from unum_streaming import StreamingPublisher' in source:
                # Ensure .original backup exists even for manually-edited files
                if not os.path.exists(backup_path):
                    clean_source = strip_streaming_code(source)
                    with open(backup_path, 'w', encoding='utf-8') as f:
                        f.write(clean_source)
                    print(f'  [{func_name}] Already transformed — created .original backup by stripping streaming code')
                else:
                    print(f'  [{func_name}] Already transformed, skipping')
                continue
            
            analyzer = StreamingAnalyzer()
            analysis = analyzer.analyze(source)
            
            if not analysis.can_stream:
                print(f'  [{func_name}] {analysis.reason}')
                continue
            
            # Backup original
            if not os.path.exists(backup_path):
                shutil.copy2(app_path, backup_path)
            
            # Transform the source
            transformer = StreamingTransformer(analysis, func_name)
            new_source, messages = transformer.transform(source)
            
            # Write transformed source
            with open(app_path, 'w', encoding='utf-8') as f:
                f.write(new_source)
            
            for msg in messages:
                print(f'  [{func_name}] \033[32m{msg}\033[0m')
                
        except Exception as e:
            import traceback
            print(f'  [{func_name}] \033[33mError during transformation: {e}\033[0m')
            traceback.print_exc()
    
    print('')


def restore_original_files(platform_template):
    """
    Restore app.py files from .original backups if they exist.
    
    This is called when building in normal mode (without -s) to ensure
    the source files don't contain injected streaming code.
    """
    restored_count = 0
    
    for func_name in platform_template["Resources"]:
        resource = platform_template["Resources"][func_name]
        if resource.get("Type") != "AWS::Lambda::Function" and resource.get("Type") != "AWS::Serverless::Function":
            continue
        
        app_dir = resource["Properties"]["CodeUri"]
        app_path = os.path.join(app_dir, 'app.py')
        backup_path = os.path.join(app_dir, 'app.py.original')
        
        if os.path.exists(backup_path):
            # Restore from backup
            shutil.copy2(backup_path, app_path)
            restored_count += 1
            print(f'  [{func_name}] Restored app.py from backup')
    
    if restored_count > 0:
        print(f'\n\033[36mRestored {restored_count} app.py file(s) from .original backups\033[0m\n')


def sam_build(platform_template, args):

    if args.clean:
        sam_build_clean(platform_template)
        return

    # Ensure common directory has runtime files
    populate_common_directory()
    
    # Handle streaming mode vs normal mode
    if getattr(args, 'streaming', False):
        # Apply streaming transformation to app.py files
        apply_streaming_transform(platform_template)
    else:
        # Normal mode: restore original files if backups exist
        # This ensures we don't accidentally deploy transformed code
        restore_original_files(platform_template)

    # copy files from common to each functions directory
    for f in platform_template["Resources"]:
        app_dir = platform_template["Resources"][f]["Properties"]["CodeUri"]
        # Use cross-platform file operations
        common_dir = 'common'
        if os.path.exists(common_dir):
            for item in os.listdir(common_dir):
                src = os.path.join(common_dir, item)
                dst = os.path.join(app_dir, item)
                if os.path.isfile(src):
                    shutil.copy2(src, dst)
                elif os.path.isdir(src):
                    if os.path.exists(dst):
                        shutil.rmtree(dst)
                    shutil.copytree(src, dst)

    try:
        ret = subprocess.run(["sam", "build", "-t", args.platform_template],
            capture_output=True, check=True, shell=True)
        print(f'\033[32mBuild Succeeded\033[0m\n')
        print(f'\033[33mBuilt Artifacts  : .aws-sam/build\033[0m')
        print(f'\033[33mBuilt Template   : .aws-sam/build/template.yaml\033[0m\n')
        print(f'\033[33mCommands you can use next\n=========================\033[0m')
        print(f'\033[33m[*] Deploy: unum-cli deploy\033[0m\n')
    except subprocess.CalledProcessError as e:
        print(f'\033[31m \n Build Failed!\n\n AWS SAM failed to build due to:\033[0m')
        if e.stderr:
            print(e.stderr.decode())
        print(e)
    except Exception as e:
        print(f'\033[31m \n Build Failed!\n\n AWS SAM failed to build due to:\033[0m')
        print(e)
        import traceback
        traceback.print_exc()
    

def build(args):

    if args.clean:
        if args.platform == 'aws':
            sam_build_clean(args)
        elif args.platform == None:
            sam_build_clean(args)
        elif args.platform == 'azure':
            pass
        else:
            pass
        return

    if args.generate:
        print("\033[33mGenerating platform template...........\033[0m\n")
        template(args)

    if args.platform == None:
        print(f'No target platform specified.\nDefault to \033[33m\033[1mAWS\033[0m.')
        print(f'If AWS is not the desirable target, specify a target platform with -p or --platform.\nSee unum-cli build -h for details.\n')
        args.platform='aws'

    if args.platform == 'aws':
        # Default to AWS
        if args.platform_template == None:
            print(f'No platform template file specified.\nDefault to\033[33m\033[1m template.yaml \033[0m')
            print(f'You can specify a platform template file with -s or --platform_template.\nSee unum-cli build -h for details.\n')
            args.platform_template = "template.yaml"

        try:
            with open(args.platform_template) as f:
                platform_template = load_yaml(f.read())
        except Exception as e:
            print(f'\033[31m \n Build Failed!\n\n Make sure the platform template file exists\033[0m')
            print(f'\033[31m You can specify a platform template file with -s/--platform_template\033[0m')
            print(f'\033[31m Or generate a platform template from your unum template with "unum-cli template" or "unum-cli build -g"\033[0m')
            print(f'\033[31m See unum-cli -h for more details\033[0m\n')
            raise e

        sam_build(platform_template, args)
    else:
        pass

def deploy_sam_first():
    # Deploy the functions as is, get each function's arn, update each
    # function's unum_config.json with the arn, store function name to arn
    # mapping in function-arn.yaml

    # First deployment. Deploy functions as is
    with open("unum-template.yaml") as f:
        app_template = yaml.load(f.read(),Loader=Loader)

    app_name = app_template["Globals"]["ApplicationName"]
    
    # Use region from unum-template or default to eu-central-1
    deploy_region = app_template["Globals"].get("Region", "eu-central-1")

    try:
        ret = subprocess.run(["sam", "deploy",
                          "--stack-name", app_name,
                          "--region", deploy_region,
                          "--no-fail-on-empty-changeset",
                          "--no-confirm-changeset",
                          "--resolve-s3",
                          "--capabilities",
                          "CAPABILITY_IAM"],
                          capture_output=True, shell=True)
    except Exception as e:
        raise e

    # grep for the functions' arn
    stdout = ret.stdout.decode("utf-8")
    print(stdout)
    print(ret.stderr.decode("utf-8"))
    try:
        deploy_output = stdout.split("Outputs")[1]
    except:
        raise IOError(f'SAM stack with the same name already exists')
    
    deploy_output = deploy_output.split('-------------------------------------------------------------------------------------------------')[1]
    
    deploy_output = deploy_output.split()
    function_to_arn_mapping = {}

    i = 0
    while True:
        while deploy_output[i] != "Key":
            i = i+1

        function_name = deploy_output[i+1].replace("Function","")

        while deploy_output[i] != "Value":
            i = i+1
        function_arn = deploy_output[i+1] + deploy_output[i+2]
        function_to_arn_mapping[function_name] = function_arn

        if len(app_template["Functions"]) == len(function_to_arn_mapping.keys()):
            break

    # store function name to arn mapping in function-arn.yaml
    with open("function-arn.yaml", 'w') as f:
        d = yaml.dump(function_to_arn_mapping, Dumper=Dumper)
        f.write(d)

    print(f'function-arn.yaml created')

    # update each function's unum_config.json by replacing function names with
    # arns in the continuation
    for f in app_template["Functions"]:
        app_dir = app_template["Functions"][f]["Properties"]["CodeUri"]
        print(f'Updating function {f} in {app_dir}')

        with open(f'{app_dir}unum_config.json', 'r+') as c:
            config = json.loads(c.read())
            print(f'Overwriting {app_dir}unum_config.json')
            if "Next" in config:
                if isinstance(config["Next"],dict):
                    config["Next"]["Name"] = function_to_arn_mapping[config["Next"]["Name"]]
                if isinstance(config["Next"], list):
                    for cnt in config["Next"]:
                        cnt["Name"] = function_to_arn_mapping[cnt["Name"]]
                c.seek(0)
                c.write(json.dumps(config, indent=4))
                c.truncate()
                print(f'{app_dir}unum_config.json Updated')


def create_function_arn_mapping_from_cloudformation(stack_name, unum_template):
    ''' Create function-arn.yaml by querying CloudFormation stack outputs directly.
    
    This is a more reliable method than parsing SAM deploy stdout, as it uses
    the AWS CLI to query the stack outputs directly.
    '''
    print(f'Querying CloudFormation stack {stack_name} for function ARNs...')
    
    try:
        ret = subprocess.run(
            ["aws", "cloudformation", "describe-stacks",
             "--stack-name", stack_name,
             "--query", "Stacks[0].Outputs",
             "--output", "json"],
            capture_output=True, shell=True, check=True
        )
        
        outputs = json.loads(ret.stdout.decode("utf-8"))
        function_to_arn_mapping = {}
        
        for output in outputs:
            # OutputKey is like "PageRankFunction", we need to strip "Function"
            function_name = output["OutputKey"].replace("Function", "")
            function_arn = output["OutputValue"]
            function_to_arn_mapping[function_name] = function_arn
        
        # Verify we got all functions
        if len(function_to_arn_mapping) < len(unum_template["Functions"]):
            print(f'\033[33mWarning: Only found {len(function_to_arn_mapping)} ARNs, expected {len(unum_template["Functions"])}\033[0m')
        
        # Store function name to arn mapping in function-arn.yaml
        with open("function-arn.yaml", 'w') as f:
            d = yaml.dump(function_to_arn_mapping, Dumper=Dumper)
            f.write(d)
        
        print(f'\033[32mfunction-arn.yaml Created from CloudFormation outputs\033[0m')
        return function_to_arn_mapping
        
    except subprocess.CalledProcessError as e:
        print(f'\033[31mFailed to query CloudFormation stack: {e.stderr.decode("utf-8") if e.stderr else str(e)}\033[0m')
        raise
    except Exception as e:
        print(f'\033[31mFailed to create function-arn.yaml from CloudFormation: {e}\033[0m')
        raise


def create_function_arn_mapping(sam_stdout, unum_template, stack_name=None):
    ''' create a function-arn.yaml file and return the mapping as dict
    '''
    # grep for the functions' arn This method relies on string processing sam
    # deploy stdout to get Lambda ARNs. The obvious downside is that if sam
    # deploy's output format changes, the following code won't work. Still
    # looking for a more reliable/programmable way to get this information.

    try:
        deploy_output = sam_stdout.split("Outputs")[1]
    except:
        # SAM output doesn't contain "Outputs" section - this happens when:
        # 1. Stack already exists and there are no changes to deploy
        # 2. SAM output format has changed
        # Fallback to querying CloudFormation directly
        print(f'\033[33mSAM output does not contain Outputs section.\033[0m')
        print(f'\033[33mFalling back to CloudFormation query...\033[0m')
        
        if stack_name:
            return create_function_arn_mapping_from_cloudformation(stack_name, unum_template)
        else:
            print(f'Failed to create function-arn.yaml.')
            print(f'SAM stack with the same name already exists or SAM output format changed')
            raise IOError(f'Cannot parse SAM output and no stack_name provided for CloudFormation fallback')
    
    deploy_output = deploy_output.split('-------------------------------------------------------------------------------------------------')[1]
    
    deploy_output = deploy_output.split()
    function_to_arn_mapping = {}

    i = 0
    while True:
        while deploy_output[i] != "Key":
            i = i+1

        function_name = deploy_output[i+1].replace("Function","")

        while deploy_output[i] != "Value":
            i = i+1
        function_arn = deploy_output[i+1] + deploy_output[i+2]
        function_to_arn_mapping[function_name] = function_arn

        if len(unum_template["Functions"]) == len(function_to_arn_mapping.keys()):
            break

    # store function name to arn mapping in function-arn.yaml
    with open("function-arn.yaml", 'w') as f:
        d = yaml.dump(function_to_arn_mapping, Dumper=Dumper)
        f.write(d)

    print(f'function-arn.yaml Created')

    return function_to_arn_mapping

def update_unum_config_continuation_to_arn(platform_template, function_to_arn_mapping):
    ''' Given a workflow and its function-to-arn mapping, update each
    function's continuation in unum_config.json with the Lambda's ARN

    This function changes the unum_config.json files in the build artifacts,
    i.e., .aws-sam. It does not modify the source code.
    '''
    base_dir = f'.aws-sam/build'
    for f in platform_template["Resources"]:

        if platform_template["Resources"][f]["Type"] == 'AWS::Serverless::Function':
            function_artifact_dir = f'{base_dir}/{f}'

            print(f'[*] Updating unun_config.json in {function_artifact_dir}')

            try:
                c = open(f'{function_artifact_dir}/unum_config.json', 'r+')
                config = json.loads(c.read())
                print(f'Current config: {config}')

                if "Next" in config:
                    if isinstance(config["Next"],dict):
                        config["Next"]["Name"] = function_to_arn_mapping[config["Next"]["Name"]]
                    if isinstance(config["Next"], list):
                        for cnt in config["Next"]:
                            cnt["Name"] = function_to_arn_mapping[cnt["Name"]]
                    c.seek(0)
                    c.write(json.dumps(config, indent=4))
                    c.truncate()

                print(f'\033[32m {function_artifact_dir}/unum_config.json Updated\033[0m')
                c.close()

            except Exception as e:
                print(f'\033[31m Exceptions updating {function_artifact_dir}/unum_config.json:\033[0m')
                print(f'\033[31m {e} \033[0m')
                return False

    return True



def deploy_sam(args):
    # check if AWS_PROFILE is set
    if os.getenv("AWS_PROFILE") == None:
        print(f'\033[31m \n Deploy Failed!\n\n Make sure AWS_PROFILE is set\033[0m')
        raise OSError(f'Environment variable $AWS_PROFILE must exist')

    # read unum template file
    try:
        with open(args.template) as f:
            unum_template = load_yaml(f.read())
            stack_name = unum_template["Globals"]["ApplicationName"]
    except Exception as e:
        print(f'\033[31m \n Deploy Failed!\n\n Failed to find unum template file: {args.template}\033[0m\n')
        print(f'\033[31m Make sure the unum template file exists\033[0m')
        print(f'\033[31m You can specify a platform template file with -t/--template\033[0m')
        print(f'\033[31m See unum-cli deploy -h for more details\033[0m\n')
        raise e

    # read platform template file (i.e., sam template)
    try:
        with open(args.platform_template) as f:
            platform_template = load_yaml(f.read())
    except Exception as e:
        print(f'\033[31m \n Deploy Failed!\n\n Failed to find platform template file: {args.platform_template}\033[0m\n')
        print(f'\033[31m Make sure the platform template file exists\033[0m')
        print(f'\033[31m You can specify a platform template file with -s/--platform_template\033[0m')
        print(f'\033[31m See unum-cli deploy -h for more details\033[0m\n')
        raise e



    def rollback_first_deployment():
        print(f'\033[31mRemoving function-arn.yaml\033[0m\n')
        if os.path.isfile('function-arn.yaml'):
            try:
                ret = subprocess.run(["rm", "function-arn.yaml"], check = True, capture_output=True)
            except Exception as e:
                print(f'Failed to delete function-arn.yaml')


        # check if the stack is deployed
        ret = subprocess.run(["aws", "cloudformation", "describe-stacks",
                      "--stack-name", stack_name],
                      capture_output=True)

        if ret.returncode == 0:
            stack_info = json.loads(ret.stdout.decode("utf-8"))

            if "Stacks" in stack_info and len(stack_info["Stacks"])>0:
                # if stack indeed exists on AWS, delete it
                print(f'\033[31mRolling back trial deployment\033[0m\n')
                ret = subprocess.run(["aws", "cloudformation", "delete-stack",
                              "--stack-name", stack_name],
                              capture_output=True)
                if ret.returncode != 0:
                    print(f'\033[31mFailed to delete AWS stack {stack_name}\033[0m')

    first_deploy = False
    if os.path.isfile('function-arn.yaml') == False:
        # Need to do a trial deployment to create the Lambda resources and get
        # their arn. With the arns, replace the `Name` field of the
        # continuation of each unum_config.json with the arn of the deployed
        # Lambda, and then deploy again. Note that we modify the
        # unum_config.json in the build artifacts not the source code.

        first_deploy = True

        # trial deployment
        ret, sam_output = sam_deploy_wrapper(stack_name)
        if ret == False:
            print(f'\033[31m Trial Deployment Failed!\033[0m\n')
            raise OSError(f'Failed to deploy to AWS')

        print(sam_output)
        print(f'\033[32m Lambda resources created\033[0m\n')
        print(f'Creating function-to-arn mapping ......')
        # create the function to arn mapping
        function_to_arn_mapping = create_function_arn_mapping(sam_output, unum_template, stack_name)
        print(f'\033[32m\n Function-to-arn Mapping Created\033[0m\n')

    # copy function-arn.yaml into .aws-sam/build/[function_name]/
    base_dir = f'.aws-sam/build'

    for f in platform_template["Resources"]:
        if platform_template["Resources"][f]["Type"] == 'AWS::Serverless::Function':
            function_artifact_dir = f'{base_dir}/{f}'

            print(f'[*] Copying function-arn.yaml into {function_artifact_dir}')

            shutil.copy('function-arn.yaml', function_artifact_dir)


        # # update the unum_config.json in all functions (in the build artifacts, not source code)
        # print(f'Updating unum configuration ......')
        # if update_unum_config_continuation_to_arn(platform_template, function_to_arn_mapping) == False:

        #     # If updating unum configuration fails at this point, rollback
        #     print(f'\033[31m\nFailed to update unum configuration\033[0m\n')
        #     rollback_first_deployment()
        #     print(f'\033[31m\nTrial deployment rolled back\033[0m\n')
        #     print(f'\033[31m\nDeployment Failed\033[0m\n')
        #     exit(1)

        # print(f'\033[32m \nAll unum configuration updated\033[0m\n')
        # time.sleep(5)

    # Validate build artifacts first
    # User might have run unum-cli build asynchronously.
    # Additionally, we need to make sure that all unum configurations have
    # ARNs in their continuations, not function names
    if validate_sam_build_artifacts(platform_template) == False:
        print(f'\033[31m \n Deploy Failed!\n\n Invalid build artifacts\033[0m\n')

        if first_deploy:
            # rollback if this is the first time deploying
            rollback_first_deployment()

        raise ValueError(f'Invalid build artifacts')

    # if validate_sam_build_artifacts_unum_config() == False:
    #     if os.path.isfile('function-arn.yaml'):
    #         with open('function-arn.yaml') as f:
    #             function_to_arn_mapping = load_yaml(f.read())

            # update_unum_config_continuation_to_arn(platform_template, function_to_arn_mapping)

    #     else:
    #         print(f'\033[31m \nDeploy Failed!\n\n Invalid build artifacts\033[0m\n')
    #         print(f'\033[31m unum configurations do not contain ARNs and function-arn.yaml does not exist.\033[0m\n')
    #         if first_deploy:
    #             # rollback if this is the first time deploying
    #             rollback_first_deployment()

    #         raise ValueError(f'Invalid build artifacts')


    # deploy
    ret, sam_output = sam_deploy_wrapper(stack_name)
    if ret == False:
        print(f'\033[31m Deploy Failed!\033[0m\n')
        raise OSError(f'Failed to deploy to AWS')
    else:
        print(sam_output)
        print(f'\033[32m\nDeploy Succeeded!\033[0m')

def sam_deploy_wrapper(stack_name):
    ''' Wrapper around a sam deploy subprocess Note that unum-cli deploy will
    always use .aws-sam/build/template.yaml as the sam deploy template (i.e.,
    sam deploy -t .aws-sam/build/template.yaml), because unum-cli piggybacks
    on the sam build artifacts
    '''
    ret = subprocess.run(["sam", "deploy",
                          "--stack-name", stack_name,
                          "--template-file", ".aws-sam/build/template.yaml",
                          "--no-fail-on-empty-changeset",
                          "--no-confirm-changeset",
                          "--resolve-s3",
                          "--capabilities",
                          "CAPABILITY_IAM"],
                          capture_output=True, shell=True)

    if ret.returncode != 0:
        print(f'\033[31msam deploy Failed! Error message from sam:\033[0m')
        print(f'\033[31m {ret.stderr.decode("utf-8")} \033[0m')
        return False, ret.stderr.decode("utf-8")

    print(f'\033[32m\nsam deploy Succeeded\033[0m')
    return True, ret.stdout.decode("utf-8")

def validate_sam_build_artifacts_unum_config():
    ''' Making sure continuations in unum_config.json have ARns.

    Check all function artifacts in .aws-sam.
    '''
    built_functions = [d for d in os.listdir('.aws-sam/build') if d.endswith('Function')]
    for f in built_functions:
        try:
            with open(f'.aws-sam/build/{f}/unum_config.json') as c:
                config = json.loads(c.read())
        except Exception as e:
            print(f'\033[31m .aws-sam/build/{f}/unum_config.json failed to open \033[0m')
            raise e

        if "Next" in config:
            if isinstance(config["Next"],dict):
                if config["Next"]["Name"].startswith('arn:aws:lambda')== False:
                    return False
            if isinstance(config["Next"], list):
                for cnt in config["Next"]:
                    if cnt["Name"].startswith('arn:aws:lambda') == False:
                        return False

    return True

def validate_sam_build_artifacts(platform_template):
    '''
    AWS: check if all functions in the .aws-sam directory has a unum-config.json file
    '''

    def check_subset(l1, l2):
        ''' return if l1 is a subset of l2
        Return True if all elements of l1 are in l2. Otherwise False
        '''
        for e in l1:
            if e not in l2:
                return False
        return True
    
    # check if .aws-sam/ and .aws-sam/build exists
    if os.path.isdir('.aws-sam/build') == False:
        print(f'\033[31m \n No build artifacts detected\033[0m\n')
        print('''\033[31m For AWS deployment, make sure you have the build artifacts under .aws-sam/build.
 To build an unum workflow, use the unum-cli build command.
 See unum-cli build -h for more details.\033[0m''')
        return False

    # check if the number of directories under .aws-sam/build/ match the
    # number of functions in platform_template
    # Functions should have a directory that ends with the word 'Function', e.g., HelloFunction.
    built_functions = [d for d in os.listdir('.aws-sam/build') if d.endswith('Function')]
    print(f'Built functions detected:')
    for f in built_functions:
        print(f' [*] {f}')

    template_resources = platform_template['Resources'].keys()
    if check_subset(built_functions, template_resources) == False:
        print(f'\033[31m \n Function artifacts do not match the template\033[0m\n')
        return False

    # check if each function directory in .aws-sam/build/ has
    #    + app.py
    #    + unum.py
    #    + ds.py
    #    + unum_config.json
    expected_files_list = [
        'app.py',
        'unum.py',
        'ds.py',
        'unum_config.json'
    ]

    for f in built_functions:
        print(f'Checking {f} ......')
        if check_subset(expected_files_list, os.listdir(f'.aws-sam/build/{f}')):
            print(f'\033[32m Success\033[0m')
        else:
            print(f'\033[31m Failed!\033[0m')
            print(f"\033[31m Make sure you have the following files in each function's build directory:\033[0m")
            print(f'\033[31m {expected_files_list}\033[0m')
            return False

    return True

def validate_build_artifacts(platform_template, platform):

    if platform == 'aws':
        return validate_sam_build_artifacts(platform_template)
    else:
        raise OSError(f'Only AWS SAM deployment supported')

    pass


def deploy(args):

    # Make sure args has all names with valid values
    if args.platform_template == None:
        # platform_template not specified, default to AWS template.yaml
        print(f'No platform template file specified.\nDefault to\033[33m\033[1m template.yaml \033[0m\n')
        args.platform_template = 'template.yaml'

    if args.template == None:
        # unum template not specified, default to unum-template.yaml
        print(f'No unum template file specified.\nDefault to\033[33m\033[1m unum-template.yaml \033[0m\n')
        args.template = 'unum-template.yaml'

    try:
        with open(args.platform_template) as f:
            platform_template = load_yaml(f.read())
    except Exception as e:
        print(f'\033[31m \n Deploy Failed!\n\n Failed to find platform template file: {args.platform_template}\033[0m\n')
        print(f'\033[31m Make sure the platform template file exists\033[0m')
        print(f'\033[31m You can specify a platform template file with -s/--platform_template\033[0m')
        print(f'\033[31m Or generate a platform template from your unum template with "unum-cli template" or "unum-cli build -g"\033[0m')
        print(f'\033[31m See unum-cli -h for more details\033[0m\n')
        raise e

    if "AWSTemplateFormatVersion" in platform_template:
        if args.platform == None:
            # platform not specified
            args.platform = 'aws'

        # platform specified, make sure that it's the same as the platform_template
        elif args.platform =='aws':
            pass
        else: 
            print(f'\033[31m \n Deploy Failed!\n\n Specified platform failed to match template\033[0m\n')
            print(f'\033[31m Specified platform: {args.platform}, template: aws.\033[0m\n')
            raise ValueError(f'Specified platform failed to match template')

    elif "AZure" in platform_template:
        raise OSError(f'AZure deployment not supported yet')
    else:
        raise OSError(f'Other deployment not supported yet')
    
    # build if -b option
    if args.build:
        args.clean=False
        args.generate=False
        build(args)
    
    # validates build artifacts before deploying
    if validate_build_artifacts(platform_template, args.platform) == False:
        print(f'\033[31m \n Deploy Failed!\n\n Invalid build artifacts\033[0m\n')
        raise ValueError(f'Invalid build artifacts')

    print(f'\033[32m\nBuild artifacts validation passed\033[0m\n')

    if args.platform == 'aws':
        print(f'\033[33m\033[1mDeploying to AWS ......\033[0m\n')
        deploy_sam(args)
    elif args.platform == 'azure':
        raise OSError(f'AZure deployment not supported yet')
    else:
        raise OSError(f'Other deployment not supported yet')


def template(args):

    # unum-cli template -c/--clean
    if args.clean:
        try:
            subprocess.run(['rm', '-f', 'template.yaml'], check=True)
        except Exception as e:
            raise e
        return

    # if platform is not specified
    if args.platform == None:
        print(f'No target platform specified.\nDefault to \033[33m\033[1mAWS\033[0m.')
        print(f'If AWS is not the desirable target, specify a target platform with -p or --platform.\nSee unum-cli template -h for details.\n')
        args.platform='aws'

    # if a unum-template file is not specified
    if args.template == None:
        print(f'No unum template file specified.\nDefault to\033[33m\033[1m unum-template.yaml \033[0m')
        print(f'You can specify a template file with -t or --template.\nSee unum-cli template -h for details.\n')
        args.template = 'unum-template.yaml'

    try:
        with open(args.template) as f:
            unum_template = yaml.load(f.read(), Loader=Loader)
    except Exception as e:
        print(f'\033[31m \n Build Failed!\n\n Make sure the template file exists\033[0m')
        raise e

    if args.platform == 'aws':
        platform_template = generate_sam_template(unum_template)

        # Save the AWS SAM template as 'template.yaml'
        print(f'\033[32mPlatform Template Generation Succeeded\033[0m\n')
        print(f'\033[33mAWS SAM Template: template.yaml\033[0m\n')
        try:
            with open('template.yaml','w') as f:
                f.write(dump_yaml(platform_template))
        except Exception as e:
            raise e

        # AWS-specific template post-processing
        # YAML dumpper (even the AWS-provided one) doesn't correctly recognize
        # Cloudformation tags and results in !GetAtt being saved as a string.
        with open('template.yaml','r+') as f:
            cnt = f.read()
            # YAML dumpper (even the AWS-provided one) doesn't correctly recognize
            # Cloudformation tags and results in !GetAtt being saved as a string.
            cnt = cnt.replace("Value: '!GetAtt", "Value: !GetAtt").replace("Function.Arn'","Function.Arn")
            f.seek(0)
            f.write(cnt)
            f.truncate()

    elif args.platform == 'azure':
        # platform_template = generate_azure_template(app_template)
        return
    elif args.platform ==None:
        print(f'Failed to generate platform template due to missing target')
        raise ValueError(f'Specify target platform with -p or --platform. See unum-cli template -h for details.')
    else:
        raise ValueError(f'Unknown platform: {args.platform}')


def compile_step_functions_workflow(workflow, unum_template, functions_info):
    """
    Compile an AWS Step Functions workflow definition to unum_config.json files.
    
    Args:
        workflow: The Step Functions workflow definition (dict)
        unum_template: The unum template with function definitions
        functions_info: Dict mapping function names to their CodeUri paths
    
    Returns:
        Dict mapping function names to their unum_config.json content
    """
    configs = {}
    
    def get_function_code_uri(func_name):
        """Get the CodeUri for a function from the unum template"""
        if func_name in functions_info:
            return functions_info[func_name].get('Properties', {}).get('CodeUri', f'{func_name.lower()}/')
        return f'{func_name.lower()}/'
    
    def find_terminal_functions_in_branch(states, start_at):
        """
        Find the terminal function(s) in a branch by following the chain.
        Returns the name of the last function (the one with "End": true or no "Next").
        """
        current = start_at
        visited = set()
        
        while current and current not in visited:
            visited.add(current)
            state = states.get(current, {})
            
            if state.get('End', False):
                return current
            
            next_state = state.get('Next')
            if not next_state:
                return current
            
            # Check if Next points to a state in this branch
            if next_state in states:
                current = next_state
            else:
                # Next points outside this branch
                return current
        
        return current
    
    def process_parallel_state(parallel_state, next_after_parallel):
        """
        Process a Parallel state and generate configs for all functions in branches.
        Returns a list of (terminal_function_name, branch_index) tuples.
        """
        branches = parallel_state.get('Branches', [])
        terminal_functions = []  # List of (function_name, branch_index)
        
        for branch_idx, branch in enumerate(branches):
            branch_start = branch.get('StartAt')
            branch_states = branch.get('States', {})
            
            # Find the terminal function of this branch
            terminal_func = find_terminal_functions_in_branch(branch_states, branch_start)
            terminal_functions.append((terminal_func, branch_idx))
            
            # Process each state in the branch
            for state_name, state_def in branch_states.items():
                if state_def.get('Type') != 'Task':
                    continue
                
                func_name = state_def.get('Resource', state_name)
                is_start = (state_name == branch_start)
                
                config = {
                    "Name": func_name,
                    "Start": is_start,
                    "Checkpoint": True,
                    "Debug": True
                }
                
                # Determine what comes next for this function
                if state_def.get('End', False):
                    # This is the terminal function of the branch
                    # It should have fan-in config pointing to the state after Parallel
                    pass  # Will be filled in later with fan-in info
                elif state_def.get('Next'):
                    next_in_branch = state_def.get('Next')
                    if next_in_branch in branch_states:
                        # Next is within the same branch - simple chain
                        next_state = branch_states.get(next_in_branch, {})
                        next_func_name = next_state.get('Resource', next_in_branch)
                        config["Next"] = {
                            "Name": next_func_name,
                            "InputType": "Scalar"
                        }
                
                configs[func_name] = config
        
        return terminal_functions
    
    def generate_fan_in_config(terminal_functions, fan_in_target):
        """
        Generate the fan-in Next configuration for terminal functions of parallel branches.
        """
        # Build the Values array for fan-in
        fan_in_values = []
        for func_name, branch_idx in terminal_functions:
            fan_in_values.append(f"{func_name}-unumIndex-{branch_idx}")
        
        fan_in_next = {
            "Name": fan_in_target,
            "InputType": {
                "Fan-in": {
                    "Values": fan_in_values
                }
            },
            "Fan-in-Group": True
        }
        
        return fan_in_next
    
    def process_states(states, start_at, is_top_level=True):
        """
        Process states in a workflow/branch.
        """
        # First pass: find all parallel states and their successors
        parallel_info = {}  # Maps parallel state name to (terminal_functions, next_state)
        
        for state_name, state_def in states.items():
            if state_def.get('Type') == 'Parallel':
                terminal_funcs = process_parallel_state(state_def, state_def.get('Next'))
                parallel_info[state_name] = (terminal_funcs, state_def.get('Next'))
        
        # Second pass: configure fan-in for terminal functions
        for parallel_name, (terminal_funcs, next_state) in parallel_info.items():
            if next_state and next_state in states:
                next_state_def = states[next_state]
                fan_in_target = next_state_def.get('Resource', next_state)
                fan_in_config = generate_fan_in_config(terminal_funcs, fan_in_target)
                
                # Update each terminal function with the fan-in config
                for func_name, branch_idx in terminal_funcs:
                    if func_name in configs:
                        configs[func_name]["Next"] = fan_in_config
        
        # Third pass: process non-parallel Task states at this level
        for state_name, state_def in states.items():
            if state_def.get('Type') == 'Parallel':
                continue
            
            if state_def.get('Type') != 'Task':
                continue
            
            func_name = state_def.get('Resource', state_name)
            
            # Skip if already processed (from parallel branches)
            if func_name in configs:
                # But we may need to add Next if it's missing
                if 'Next' not in configs[func_name] and state_def.get('Next'):
                    next_state = state_def.get('Next')
                    if next_state in states:
                        next_state_def = states[next_state]
                        next_func_name = next_state_def.get('Resource', next_state)
                        configs[func_name]["Next"] = {
                            "Name": next_func_name,
                            "InputType": "Scalar"
                        }
                continue
            
            is_start = False  # Top-level tasks after Parallel are not start
            
            config = {
                "Name": func_name,
                "Start": is_start,
                "Checkpoint": True,
                "Debug": True
            }
            
            if state_def.get('Next'):
                next_state = state_def.get('Next')
                if next_state in states:
                    next_state_def = states[next_state]
                    next_func_name = next_state_def.get('Resource', next_state)
                    config["Next"] = {
                        "Name": next_func_name,
                        "InputType": "Scalar"
                    }
            
            configs[func_name] = config
    
    # Start processing from the top level
    states = workflow.get('States', {})
    start_at = workflow.get('StartAt')
    
    process_states(states, start_at, is_top_level=True)
    
    # Set Start flag for the entry function
    if start_at:
        start_state = states.get(start_at, {})
        start_func_name = start_state.get('Resource', start_at)
        if start_func_name in configs:
            configs[start_func_name]["Start"] = True
    
    return configs


def compile_workflow(args):
    """
    Compile workflow definitions to unum_config.json files for each function.
    
    Supported platforms:
    - step-functions: AWS Step Functions workflow definition
    """
    print(f'\n\033[33m\033[1mCompiling workflow...\033[0m\n')
    
    # Load workflow definition
    try:
        with open(args.workflow, 'r') as f:
            workflow = json.loads(f.read())
        print(f'\033[32mLoaded workflow definition: {args.workflow}\033[0m')
    except Exception as e:
        print(f'\033[31mFailed to load workflow file: {args.workflow}\033[0m')
        raise e
    
    # Load unum template
    try:
        with open(args.template, 'r') as f:
            unum_template = yaml.load(f.read(), Loader=Loader)
        print(f'\033[32mLoaded unum template: {args.template}\033[0m')
    except Exception as e:
        print(f'\033[31mFailed to load unum template file: {args.template}\033[0m')
        raise e
    
    # Extract function info from template
    functions_info = unum_template.get('Functions', {})
    
    if args.platform == 'step-functions':
        configs = compile_step_functions_workflow(workflow, unum_template, functions_info)
    else:
        raise ValueError(f'Unsupported workflow platform: {args.platform}')
    
    # Write unum_config.json files
    print(f'\n\033[33mGenerating unum_config.json files...\033[0m\n')
    
    for func_name, config in configs.items():
        # Get the CodeUri from template
        if func_name in functions_info:
            code_uri = functions_info[func_name].get('Properties', {}).get('CodeUri', f'{func_name.lower()}/')
        else:
            code_uri = f'{func_name.lower()}/'
        
        # Ensure directory exists
        code_dir = code_uri.rstrip('/')
        if not os.path.exists(code_dir):
            print(f'\033[33mWarning: Directory {code_dir} does not exist, creating it...\033[0m')
            os.makedirs(code_dir, exist_ok=True)
        
        config_path = os.path.join(code_dir, 'unum_config.json')
        
        try:
            with open(config_path, 'w') as f:
                f.write(json.dumps(config, indent=2))
            print(f'\033[32m  ✓ {config_path}\033[0m')
        except Exception as e:
            print(f'\033[31m  ✗ Failed to write {config_path}: {e}\033[0m')
            raise e
    
    print(f'\n\033[32m\033[1mCompilation succeeded! Generated {len(configs)} unum_config.json files.\033[0m\n')
    
    # Print summary
    print(f'\033[36mWorkflow Summary:\033[0m')
    start_functions = [name for name, cfg in configs.items() if cfg.get('Start', False)]
    end_functions = [name for name, cfg in configs.items() if 'Next' not in cfg]
    
    print(f'  Start functions: {", ".join(start_functions)}')
    print(f'  End functions: {", ".join(end_functions)}')
    print(f'  Total functions: {len(configs)}')
    print()



def main():
    parser = argparse.ArgumentParser(description='unum CLI utility for creating, building and deploying unum applications',
        # usage = "unum-cli [options] <command> <subcommand> [<subcommand> ...] [parameters]",
        epilog="To see help text for a specific command, use unum-cli <command> -h")

    subparsers = parser.add_subparsers(title='command', dest="command", required=True)

    # init command parser
    init_parser = subparsers.add_parser("init", description="create unum application")

    # template command parser
    template_parser = subparsers.add_parser("template", description="generate platform specific template")
    template_parser.add_argument('-p', '--platform', choices=['aws', 'azure'],
        help="target platform", required=False)
    template_parser.add_argument('-t', '--template',
        help="unum template file", required=False)
    template_parser.add_argument("-c", "--clean", help="Remove build artifacts",
        required=False, action="store_true")

    # build command parser
    build_parser = subparsers.add_parser("build", description="build unum application in the current directory")
    build_parser.add_argument('-p', '--platform', choices=['aws', 'azure'],
        help="target platform", required=False)
    build_parser.add_argument("-g", "--generate", help="Generate a platform template before buliding",
        required = False, action="store_true")
    build_parser.add_argument('-t', '--template',
        help="unum template file", required=False)
    build_parser.add_argument('-s', '--platform_template',
        help="platform template file", required=False)
    build_parser.add_argument("-c", "--clean", help="Remove build artifacts",
        required=False, action="store_true")
    build_parser.add_argument("--streaming", dest="streaming",
        help="Enable Partial Parameter Streaming (invoke next function as params become ready)",
        required=False, action="store_true")

    # deploy command parser
    deploy_parser = subparsers.add_parser("deploy", description="deploy unum application")
    deploy_parser.add_argument('-b', '--build', help="build before deploying. Note: does NOT generate new platform template as in unum-cli build -g",
        required=False, action="store_true")
    deploy_parser.add_argument('-p', '--platform', choices=['aws', 'azure'],
        help="target platform", required=False)
    deploy_parser.add_argument('-t', '--template',
        help="unum template file", required=False)
    deploy_parser.add_argument('-s', '--platform_template',
        help="platform template file", required=False)

    # compile commmand parser
    compile_parser = subparsers.add_parser("compile", description="compile workflow definitions to unum functions")
    compile_parser.add_argument('-p', '--platform', choices=['step-functions'],
        help='workflow definition type', required=True)
    compile_parser.add_argument('-w', '--workflow', required=True, help="workflow file")
    compile_parser.add_argument('-t', '--template', required=True, help="unum template file")
    compile_parser.add_argument('-o', '--optimize', required=False, choices=['trim'], help="optimizations")

    args = parser.parse_args()

    if args.command == 'build':
        build(args)
    elif args.command == 'deploy':
        deploy(args)
    elif args.command == 'template':
        template(args)
    elif args.command =='init':
        init(args)
    elif args.command == 'compile':
        compile_workflow(args)
    else:
        raise IOError(f'Unknown command: {args.command}')
        
if __name__ == '__main__':
    main()