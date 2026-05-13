def lambda_handler(event, context):
    """@input {counts: object}
    @output {result: object}"""
    ret = {}
    for d in event:
        ret.update(d)
    return ret
