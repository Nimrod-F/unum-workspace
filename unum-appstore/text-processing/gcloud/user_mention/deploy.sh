gcloud functions deploy text-processing-user-mention \
--runtime python313 \
--trigger-topic text-processing-user-mention \
--entry-point lambda_handler \
--env-vars-file env.yaml 