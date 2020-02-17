import argparse
import json
import os

import googleapiclient.discovery


# Create the AI Platform service object.
# To authenticate set the environment variable
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'sa.json'
service = googleapiclient.discovery.build('ml', 'v1')

parser = argparse.ArgumentParser()

parser.add_argument('input',
                    type=str)

parser.add_argument('--project',
                    type=str,
                    required=True)

parser.add_argument('--model',
                    type=str,
                    required=True)

parser.add_argument('--version',
                    type=str,
                    required=True)


def main(input_path, project, model, version=None):
    """Main procedure."""

    with open(input_path, 'r') as f:
        lines = f.read().splitlines()

    instances = [json.loads(line) for line in lines]

    predictions = predict_json(project, model, instances, version)

    for prediction in predictions:
        print(prediction['embedding_output'])


def predict_json(project, model, instances, version=None):
    """Send json data to a deployed model for prediction.

    Args:
        project (str): project where the AI Platform Model is deployed.
        model (str): model name.
        instances ([Mapping[str: Any]]): Keys should be the names of Tensors
            your deployed model expects as inputs. Values should be datatypes
            convertible to Tensors, or (potentially nested) lists of datatypes
            convertible to tensors.
        version: str, version of the model to target.
    Returns:
        Mapping[str: any]: dictionary of prediction results defined by the
            model.
    """

    name = 'projects/{}/models/{}'.format(project, model)

    if version is not None:
        name += '/versions/{}'.format(version)

    response = service.projects().predict(
        name=name,
        body={'instances': instances}
    ).execute()

    if 'error' in response:
        raise RuntimeError(response['error'])

    return response['predictions']


if __name__ == '__main__':
    args = parser.parse_args()
    main(args.input, project=args.project, model=args.model,
         version=args.version)
