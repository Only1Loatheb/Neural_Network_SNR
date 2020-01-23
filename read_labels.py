import yaml

with open("labels.yaml", 'r') as stream:
    try:
        labels = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)