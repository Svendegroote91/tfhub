# TF-hub to AI-Platform

This example showcases how to use a model from TF-hub and deploy it on to 
AI-platform. 

The model used is the [Universal sentence encoder](https://tfhub.dev/google/universal-sentence-encoder/4).

## SavedModel
To extract a SavedModel from the TF-hub model, run:

```
python tfhub_to_savedmodel.py
```

## Deploying to AI-platform

First, we need to copy to model to a Google Cloud Storage bucket.

In order to create a bucket:

```
BUCKET_NAME=[YOUR-BUCKET-NAME]
gsutil mb gs://$BUCKET_NAME
```

To copy over the SavedModel:

```
MODEL_DIR=gs://$BUCKET_NAME/model1/
gsutil -m cp -r saved_model/* $MODEL_DIR
```

Define a model name and create a model on AI-platform:

```
MODEL_NAME=sentence
gcloud ai-platform models create $MODEL_NAME --regions europe-west1
```

Define a model version:

```
VERSION_NAME=v1
FRAMEWORK=TENSORFLOW
```

Deploy the model to AI-platform:

```
gcloud ai-platform versions create $VERSION_NAME \
  --model $MODEL_NAME \
  --origin $MODEL_DIR \
  --runtime-version=1.15 \
  --framework $FRAMEWORK \
  --python-version=3.7
```


## Online predictions from AI-platform

In order to use the google-api-client, we need to reference a service account.
So the first step is to make sure you can reference a valid service account 
in the `aiplatform_request.py`. In the example code, the service account is 
called `sa.json` and is at the root directory of the repository.

Subsequently, run:

```
PROJECT=[YOUR-PROJECT-ID]
python aiplatform_request.py input.json --project=$PROJECT --model=sentence --version=v1
```

Please note that the first request will take signicantly more time as the 
model is in a 'cold start'.


